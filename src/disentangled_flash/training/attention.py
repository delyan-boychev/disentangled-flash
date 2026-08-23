"""Differentiable DeBERTa-v2/v3 attention backed by DisentangledFlash kernels."""

from __future__ import annotations

import math
from typing import Any

import torch

from .._reference import DebertaAttentionConfig, OriginalDisentangledSelfAttention
from ..position import SharedPositionPlanCache
from ._kernels import training_attention


class TritonTrainingDisentangledSelfAttention(OriginalDisentangledSelfAttention):
    """Trainable exact disentangled self-attention for CUDA.

    Parameter-derived inference caches are deliberately absent. Q/K/V and the
    projected relative-position tables are rebuilt through ordinary PyTorch
    operators on every forward so gradients flow through all original DeBERTa
    parameters. Only immutable relative-position geometry is cached.

    Attention-probability dropout is intentionally unsupported in the first
    training kernel. Hidden-state dropout and positional-embedding dropout stay
    ordinary PyTorch modules and continue to work normally.
    """

    def __init__(
        self,
        config: DebertaAttentionConfig | Any,
        *,
        position_plan_cache: SharedPositionPlanCache | None = None,
        fp32_precision: str = "strict",
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__(config)
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = tuple(
                part.strip().lower() for part in self.pos_att_type.split("|") if part.strip()
            )
        if fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be 'strict' or 'fast'")
        self.fp32_precision = fp32_precision
        self.assume_unpadded = assume_unpadded
        self.attention_probability_dropout = float(config.attention_probs_dropout_prob)

        uses_position_bias = self.relative_attention and bool(
            {"c2p", "p2c"}.intersection(self.pos_att_type)
        )
        self.position_plan_cache = position_plan_cache or SharedPositionPlanCache(
            position_buckets=self.position_buckets,
            max_relative_positions=self.max_relative_positions,
            position_embedding_size=self.pos_ebd_size,
            uses_position_bias=uses_position_bias,
        )

    def set_position_plan_cache(
        self,
        cache: SharedPositionPlanCache,
    ) -> TritonTrainingDisentangledSelfAttention:
        self.position_plan_cache = cache
        return self

    def _reshape_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        # Keep the same no-copy BHLD layout used by the inference Triton path.
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).permute(0, 2, 1, 3)

    def _project_active_positions(
        self,
        rel_embeddings: torch.Tensor,
        active_slots: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Project only active slots while preserving DeBERTa dropout semantics."""

        att_span = self.pos_ebd_size
        # HF applies positional dropout before projection. Apply it to the full
        # 2*att_span table first so the stochastic operation has the same shape;
        # only the small projection GEMMs are pruned to active slots.
        relative = self.pos_dropout(rel_embeddings[: att_span * 2])
        relative = relative.index_select(0, active_slots).unsqueeze(0)

        pos_key = None
        if self.relative_attention and "c2p" in self.pos_att_type:
            projection = self.key_proj if self.share_att_key else self.pos_key_proj
            pos_key = self.transpose_for_scores(
                projection(relative),
                self.num_attention_heads,
            )

        pos_query = None
        if self.relative_attention and "p2c" in self.pos_att_type:
            projection = self.query_proj if self.share_att_key else self.pos_query_proj
            pos_query = self.transpose_for_scores(
                projection(relative),
                self.num_attention_heads,
            )
        return pos_key, pos_query

    def _normalize_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> tuple[torch.Tensor, bool]:
        if attention_mask.dim() != 2 or attention_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("training attention requires attention_mask with shape [B, L]")
        if self.assume_unpadded:
            return attention_mask, False
        if attention_mask.dtype == torch.bool and attention_mask.is_contiguous():
            return attention_mask, True
        return attention_mask.bool().contiguous(), True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the training Triton path")
        if query_states is not None:
            raise ValueError("the training Triton path currently supports self-attention only")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the training path")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("TritonTrainingDisentangledSelfAttention requires CUDA")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("training attention supports FP16, BF16, and FP32")
        if self.attention_head_size not in {32, 64, 128}:
            raise ValueError("training attention supports head dimensions 32, 64, and 128")
        if self.training and self.attention_probability_dropout != 0.0:
            raise NotImplementedError(
                "attention_probs_dropout_prob must be 0 for the first training kernel; "
                "deterministic fused attention dropout is a follow-up milestone"
            )

        batch_size, sequence_length = hidden_states.shape[:2]
        mask, has_padding = self._normalize_mask(
            attention_mask,
            batch_size,
            sequence_length,
        )

        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        query_layer = self._reshape_heads(query, batch_size, sequence_length)
        key_layer = self._reshape_heads(key, batch_size, sequence_length)
        value_layer = self._reshape_heads(value, batch_size, sequence_length)

        plan = self.position_plan_cache.compact(sequence_length, hidden_states.device)
        pos_key = None
        pos_query = None
        if self.relative_attention and {"c2p", "p2c"}.intersection(self.pos_att_type):
            if rel_embeddings is None:
                raise ValueError("rel_embeddings is required for relative attention")
            pos_key, pos_query = self._project_active_positions(
                rel_embeddings,
                plan.active_slots,
            )

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        scale_factor = 1 + int(has_c2p) + int(has_p2c)
        score_scale = 1.0 / math.sqrt(self.attention_head_size * scale_factor)

        output = training_attention(
            query_layer,
            key_layer,
            value_layer,
            pos_key,
            pos_query,
            plan.delta_to_local,
            mask,
            score_scale=score_scale,
            has_padding=has_padding,
            strict_fp32=self.fp32_precision == "strict",
        )
        return output, None


__all__ = ["TritonTrainingDisentangledSelfAttention"]
