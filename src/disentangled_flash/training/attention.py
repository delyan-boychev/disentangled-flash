"""Differentiable DeBERTa-v2/v3 attention backed by DisentangledFlash kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from .._reference import DebertaAttentionConfig
from .._torch import TorchTrainingDisentangledSelfAttention
from ..packed import PackedSequenceInfo, resolve_packed_info
from ..position import SharedPositionPlanCache
from ..tuning import (
    CompilerSpec,
    HardwareSpec,
    KernelConfig,
    KernelTuningOptions,
    ProfileRegistry,
    WorkloadKey,
)
from ._kernels import training_attention, training_attention_packed


class TritonTrainingDisentangledSelfAttention(TorchTrainingDisentangledSelfAttention):
    """Trainable exact disentangled self-attention for CUDA.

    Parameter-derived inference caches are deliberately absent. Q/K/V and the
    projected relative-position tables are rebuilt through ordinary PyTorch
    operators on every forward so gradients flow through all original DeBERTa
    parameters. Only immutable relative-position geometry is cached.

    Attention-probability dropout runs inside the fused kernels in training
    mode: the Philox mask is regenerated in backward rather than stored, as in
    FlashAttention. Hidden-state dropout and positional-embedding dropout stay
    ordinary PyTorch modules.
    """

    def __init__(
        self,
        config: DebertaAttentionConfig | Any,
        *,
        position_plan_cache: SharedPositionPlanCache | None = None,
        fp32_precision: str = "strict",
        tuning: KernelTuningOptions | None = None,
        profile_registry: ProfileRegistry | None = None,
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__(
            config,
            position_plan_cache=position_plan_cache,
            assume_unpadded=assume_unpadded,
        )
        if fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be 'strict' or 'fast'")
        self.fp32_precision = fp32_precision
        self.tuning = tuning or KernelTuningOptions()
        self._profile_registry = (
            profile_registry
            if profile_registry is not None
            else (
                ProfileRegistry.from_options(self.tuning)
                if self.tuning.mode in {"auto", "profile_only"}
                else ProfileRegistry()
            )
        )
        self._resolved_kernel_configs: dict[tuple[int, WorkloadKey], KernelConfig | None] = {}
        # Training-forward workloads whose saved schedules failed to compile or
        # launch; every phase of such a workload falls back to bounded autotuning.
        self._failed_profile_workloads: set[WorkloadKey] = set()
        self.attention_probability_dropout = float(config.attention_probs_dropout_prob)

    def _resolve_kernel_config(
        self,
        hidden_states: torch.Tensor,
        *,
        sequence_length: int,
        batch_heads: int,
        active_slots: int,
        has_c2p: bool,
        has_p2c: bool,
        layout: str,
        uses_padding_mask: bool,
        has_dropout: bool,
        phase: str,
    ) -> KernelConfig | None:
        if self.tuning.mode == "autotune":
            return None
        if self.tuning.mode == "fixed":
            return self.tuning.fixed_config
        if torch.compiler.is_compiling():
            if self.tuning.mode == "profile_only":
                raise RuntimeError(
                    "profile_only training tuning cannot resolve a dynamic workload inside "
                    "torch.compile; select the profile configuration with fixed mode"
                )
            return None
        workload = WorkloadKey(
            sequence_length=sequence_length,
            head_dim=self.attention_head_size,
            batch_heads=batch_heads,
            active_slots=active_slots,
            dtype=str(hidden_states.dtype).removeprefix("torch."),
            has_c2p=has_c2p,
            has_p2c=has_p2c,
            fp32_precision=self.fp32_precision,
            layout=layout,
            uses_padding_mask=uses_padding_mask,
            phase=phase,
            has_dropout=has_dropout,
        )
        device_index = hidden_states.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cache_key = device_index, workload
        if cache_key not in self._resolved_kernel_configs:
            self._resolved_kernel_configs[cache_key] = self._profile_registry.resolve(
                HardwareSpec.current(device_index),
                CompilerSpec.current(),
                workload,
            )
        config = self._resolved_kernel_configs[cache_key]
        if replace(workload, phase="training_forward") in self._failed_profile_workloads:
            config = None
        if config is None and self.tuning.mode == "profile_only":
            raise RuntimeError(
                self._profile_registry.explain_miss(
                    HardwareSpec.current(device_index),
                    CompilerSpec.current(),
                    workload,
                )
            )
        return config

    def _launch_with_profile_fallback(
        self,
        launch: Callable[[KernelConfig | None, KernelConfig | None, KernelConfig | None], Any],
        config_options: dict[str, Any],
    ) -> Any:
        forward_config = self._resolve_kernel_config(**config_options, phase="training_forward")
        dq_config = None
        dkv_config = None
        if torch.is_grad_enabled():
            dq_config = self._resolve_kernel_config(**config_options, phase="backward_dq")
            dkv_config = self._resolve_kernel_config(**config_options, phase="backward_dkv")
        if forward_config is None and dq_config is None and dkv_config is None:
            return launch(None, None, None)
        try:
            return launch(forward_config, dq_config, dkv_config)
        except Exception as error:
            if self.tuning.mode == "profile_only":
                raise RuntimeError(
                    "saved training kernel configuration failed to launch: "
                    f"forward={forward_config}, dQ={dq_config}, dK/dV={dkv_config}"
                ) from error
            hidden_states = config_options["hidden_states"]
            self._failed_profile_workloads.add(
                WorkloadKey(
                    sequence_length=config_options["sequence_length"],
                    head_dim=self.attention_head_size,
                    batch_heads=config_options["batch_heads"],
                    active_slots=config_options["active_slots"],
                    dtype=str(hidden_states.dtype).removeprefix("torch."),
                    has_c2p=config_options["has_c2p"],
                    has_p2c=config_options["has_p2c"],
                    fp32_precision=self.fp32_precision,
                    layout=config_options["layout"],
                    uses_padding_mask=config_options["uses_padding_mask"],
                    phase="training_forward",
                    has_dropout=config_options["has_dropout"],
                )
            )
            return launch(None, None, None)

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
        position_ids: torch.Tensor | None = None,
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

        if position_ids is not None:
            if position_ids.shape != (batch_size, sequence_length):
                raise ValueError("position_ids must have shape [B, L]")
            if position_ids.device != hidden_states.device:
                raise ValueError("position_ids must be on the hidden_states device")
            plan = self.position_plan_cache.physical(position_ids)
        else:
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
        active_slots = int(plan.active_slots.numel())
        dropout_p = self.attention_probability_dropout if self.training else 0.0
        config_options = {
            "hidden_states": hidden_states,
            "sequence_length": sequence_length,
            "batch_heads": batch_size * self.num_attention_heads,
            "active_slots": active_slots,
            "has_c2p": has_c2p,
            "has_p2c": has_p2c,
            "layout": "padded",
            "uses_padding_mask": has_padding,
            "has_dropout": dropout_p > 0.0,
        }
        output = self._launch_with_profile_fallback(
            lambda forward_config, dq_config, dkv_config: training_attention(
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
                forward_config=forward_config,
                dq_config=dq_config,
                dkv_config=dkv_config,
                autotune_candidates=self.tuning.candidates,
                dropout_p=dropout_p,
            ),
            config_options,
        )
        return output, None

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        *,
        rel_embeddings: torch.Tensor | None = None,
        packed_info: PackedSequenceInfo | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run differentiable unpadded attention without materializing padding."""

        if hidden_states.device.type != "cuda":
            raise RuntimeError("packed Triton training attention requires CUDA")
        if hidden_states.ndim != 2 or hidden_states.size(-1) != self.all_head_size:
            raise ValueError("packed hidden_states must have shape [total_tokens, hidden_size]")
        info = resolve_packed_info(
            cu_seqlens,
            hidden_states.size(0),
            max_seqlen,
            packed_info,
        )
        total_tokens = hidden_states.size(0)

        def packed_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                total_tokens,
                self.num_attention_heads,
                self.attention_head_size,
            ).permute(1, 0, 2)

        query_layer = packed_heads(self.query_proj(hidden_states))
        key_layer = packed_heads(self.key_proj(hidden_states))
        value_layer = packed_heads(self.value_proj(hidden_states))
        plan = self.position_plan_cache.compact(info.max_seqlen, hidden_states.device)
        pos_key = None
        pos_query = None
        if self.relative_attention and {"c2p", "p2c"}.intersection(self.pos_att_type):
            if rel_embeddings is None:
                raise ValueError("rel_embeddings is required for relative attention")
            pos_key, pos_query = self._project_active_positions(rel_embeddings, plan.active_slots)
        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        scale_factor = 1 + int(has_c2p) + int(has_p2c)
        score_scale = 1.0 / math.sqrt(self.attention_head_size * scale_factor)
        dropout_p = self.attention_probability_dropout if self.training else 0.0
        config_options = {
            "hidden_states": hidden_states,
            "sequence_length": info.max_seqlen,
            "batch_heads": len(info.lengths) * self.num_attention_heads,
            "active_slots": int(plan.active_slots.numel()),
            "has_c2p": has_c2p,
            "has_p2c": has_p2c,
            "layout": "packed",
            "uses_padding_mask": False,
            "has_dropout": dropout_p > 0.0,
        }
        output = self._launch_with_profile_fallback(
            lambda forward_config, dq_config, dkv_config: training_attention_packed(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                plan.delta_to_local,
                cu_seqlens,
                max_seqlen=info.max_seqlen,
                score_scale=score_scale,
                strict_fp32=self.fp32_precision == "strict",
                forward_config=forward_config,
                dq_config=dq_config,
                dkv_config=dkv_config,
                autotune_candidates=self.tuning.candidates,
                dropout_p=dropout_p,
            ),
            config_options,
        )
        return output, None


__all__ = ["TritonTrainingDisentangledSelfAttention"]
