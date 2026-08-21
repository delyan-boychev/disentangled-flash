"""Prepared inference-only DeBERTa-v2/v3 disentangled attention.

The hot path keeps the exact eager attention equation, while moving relative
projection and position-index work into explicit preparation.  Position-index
plans can be shared across every layer of an encoder; only the projected
relative keys/queries remain layer-specific.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import torch
from torch.nn import functional as F

from ._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
    build_rpos,
)
from .position import (
    SharedPositionPlanCache,
    canonical_device,
)


class PreparedPositionPlan(NamedTuple):
    """Layer projections plus shared gather indices for one sequence length."""

    sequence_length: int
    active_slots: torch.Tensor
    c2p_local: torch.Tensor
    p2c_local: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


def _invalidate_attention_cache_after_load(
    module: torch.nn.Module,
    _incompatible_keys: Any,
) -> None:
    module.clear_inference_cache()


class InferenceDisentangledSelfAttention(OriginalDisentangledSelfAttention):
    """Cached, active-slot-pruned DeBERTa attention for inference only.

    ``forward_prepared`` performs no cache lookup or mutation and is the entry
    point intended for ``torch.compile(..., fullgraph=True)``.  Pass the same
    :class:`SharedPositionPlanCache` to every encoder layer to avoid duplicating
    the sequence-specific gather map.
    """

    def __init__(
        self,
        config: DebertaAttentionConfig | Any,
        *,
        position_plan_cache: SharedPositionPlanCache | None = None,
    ) -> None:
        super().__init__(config)
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = tuple(
                part.strip().lower()
                for part in self.pos_att_type.split("|")
                if part.strip()
            )
        uses_position_bias = self.relative_attention and bool(
            {"c2p", "p2c"}.intersection(self.pos_att_type)
        )
        self.position_plan_cache = position_plan_cache or SharedPositionPlanCache(
            position_buckets=self.position_buckets,
            max_relative_positions=self.max_relative_positions,
            position_embedding_size=self.pos_ebd_size,
            uses_position_bias=uses_position_bias,
        )
        self.register_buffer("_cached_pos_key", None, persistent=False)
        self.register_buffer("_cached_pos_query", None, persistent=False)
        self.register_buffer("_cached_qkv_weight", None, persistent=False)
        self.register_buffer("_cached_qkv_bias", None, persistent=False)
        self._position_projection_cache: dict[
            tuple[int, str], PreparedPositionPlan
        ] = {}
        self.register_load_state_dict_post_hook(_invalidate_attention_cache_after_load)

    def set_position_plan_cache(
        self,
        cache: SharedPositionPlanCache,
    ) -> "InferenceDisentangledSelfAttention":
        """Attach a model-wide index cache and discard layer-derived plans."""

        self.position_plan_cache = cache
        self._position_projection_cache.clear()
        return self

    def clear_inference_cache(self) -> None:
        """Invalidate every tensor derived from this layer's parameters."""

        self._cached_pos_key = None
        self._cached_pos_query = None
        self._cached_qkv_weight = None
        self._cached_qkv_bias = None
        self._position_projection_cache.clear()

    def train(self, mode: bool = True) -> "InferenceDisentangledSelfAttention":
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn: Any, recurse: bool = True) -> "InferenceDisentangledSelfAttention":
        result = super()._apply(fn, recurse=recurse)
        self.clear_inference_cache()
        return result

    @torch.no_grad()
    def _prepare_fused_qkv(self) -> None:
        self._cached_qkv_weight = torch.cat(
            (self.query_proj.weight, self.key_proj.weight, self.value_proj.weight),
            dim=0,
        ).contiguous()
        biases = (self.query_proj.bias, self.key_proj.bias, self.value_proj.bias)
        self._cached_qkv_bias = (
            torch.cat(biases, dim=0).contiguous()
            if all(bias is not None for bias in biases)
            else None
        )

    @torch.no_grad()
    def prepare_for_inference(
        self,
        rel_embeddings: torch.Tensor | None,
    ) -> "InferenceDisentangledSelfAttention":
        """Cache layer-dependent relative projections and the fused QKV projection."""

        if self.training:
            raise RuntimeError("prepare_for_inference() requires module.eval()")

        self.clear_inference_cache()
        self._prepare_fused_qkv()

        if not self.relative_attention:
            return self
        if rel_embeddings is None:
            raise ValueError("rel_embeddings is required for relative attention")

        att_span = self.pos_ebd_size
        relative = self.pos_dropout(rel_embeddings[: att_span * 2]).unsqueeze(0)

        if "c2p" in self.pos_att_type:
            projection = self.key_proj if self.share_att_key else self.pos_key_proj
            self._cached_pos_key = self.transpose_for_scores(
                projection(relative),
                self.num_attention_heads,
            )

        if "p2c" in self.pos_att_type:
            projection = self.query_proj if self.share_att_key else self.pos_query_proj
            self._cached_pos_query = self.transpose_for_scores(
                projection(relative),
                self.num_attention_heads,
            )
        return self

    def _plan_device(self) -> torch.device:
        if self._cached_pos_key is not None:
            return self._cached_pos_key.device
        if self._cached_pos_query is not None:
            return self._cached_pos_query.device
        return self.query_proj.weight.device

    def _scale(self, scale_factor: int) -> float:
        return math.sqrt(self.attention_head_size * scale_factor)

    def _project_active_positions(
        self,
        active_slots: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pos_key = None
        if self.relative_attention and "c2p" in self.pos_att_type:
            if self._cached_pos_key is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            pos_key = self._cached_pos_key.index_select(1, active_slots).contiguous()

        pos_query = None
        if self.relative_attention and "p2c" in self.pos_att_type:
            if self._cached_pos_query is None:
                raise RuntimeError("call prepare_for_inference() before prepare_shape()")
            pos_query = self._cached_pos_query.index_select(1, active_slots).contiguous()
        return pos_key, pos_query

    @torch.no_grad()
    def prepare_shape(
        self,
        sequence_length: int,
        device: torch.device | str | None = None,
    ) -> PreparedPositionPlan:
        """Prepare layer projections using a model-wide shared index plan."""

        if self.training:
            raise RuntimeError("prepare_shape() requires module.eval()")
        resident_device = self._plan_device()
        resolved_device = canonical_device(device, resident_device)
        cache_key = sequence_length, str(resolved_device)
        cached = self._position_projection_cache.get(cache_key)
        if cached is not None:
            return cached

        indices = self.position_plan_cache.dense(sequence_length, resolved_device)
        pos_key, pos_query = self._project_active_positions(indices.active_slots)
        plan = PreparedPositionPlan(
            sequence_length=sequence_length,
            active_slots=indices.active_slots,
            # For square self-attention DeBERTa's post-transpose p2c lookup is
            # exactly the same delta map as c2p.  Both fields deliberately
            # reference one shared tensor rather than allocating two maps.
            c2p_local=indices.pair_to_local,
            p2c_local=indices.pair_to_local,
            pos_key=pos_key,
            pos_query=pos_query,
        )
        self._position_projection_cache[cache_key] = plan
        return plan

    def _dynamic_position_plan(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor,
    ) -> PreparedPositionPlan:
        """Compatibility path for caller-provided relative-position tensors."""

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"relative_pos must have 2, 3, or 4 dimensions, got {relative_pos.dim()}"
            )

        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
        att_span = self.pos_ebd_size
        c2p_slots = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        r_pos = build_rpos(
            query_layer,
            key_layer,
            relative_pos,
            self.max_relative_positions,
            self.position_buckets,
        )
        p2c_slots = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1).transpose(
            -1,
            -2,
        )
        used_slots = []
        if self.relative_attention and "c2p" in self.pos_att_type:
            used_slots.append(c2p_slots.reshape(-1))
        if self.relative_attention and "p2c" in self.pos_att_type:
            used_slots.append(p2c_slots.reshape(-1))
        if used_slots:
            active_slots = torch.unique(torch.cat(used_slots), sorted=True)
            c2p_local = torch.searchsorted(active_slots, c2p_slots)
            p2c_local = torch.searchsorted(active_slots, p2c_slots.contiguous())
        else:
            active_slots = torch.empty(0, dtype=torch.long, device=query_layer.device)
            c2p_local = c2p_slots
            p2c_local = p2c_slots
        pos_key, pos_query = self._project_active_positions(active_slots)
        return PreparedPositionPlan(
            sequence_length=query_layer.size(-2),
            active_slots=active_slots,
            c2p_local=c2p_local,
            p2c_local=p2c_local,
            pos_key=pos_key,
            pos_query=pos_query,
        )

    def _validate_inference_call(self) -> None:
        if self.training:
            raise RuntimeError("InferenceDisentangledSelfAttention requires module.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "InferenceDisentangledSelfAttention requires torch.no_grad() or "
                "torch.inference_mode()"
            )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_qkv_weight is None:
            raise RuntimeError(
                "fused QKV projection is not prepared; call prepare_for_inference() first"
            )
        projected = F.linear(
            hidden_states,
            self._cached_qkv_weight,
            self._cached_qkv_bias,
        )
        return projected.split(self.all_head_size, dim=-1)

    def _reshape_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).permute(0, 2, 1, 3).contiguous()

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plan: PreparedPositionPlan,
    ) -> tuple[torch.Tensor, None]:
        """Pure tensor forward for a plan created outside the compiled graph."""

        self._validate_inference_call()
        batch_size, sequence_length = hidden_states.shape[:2]
        if sequence_length != plan.sequence_length:
            raise ValueError(
                f"prepared length {plan.sequence_length} does not match input length "
                f"{sequence_length}"
            )

        query, key, value = self._project_qkv(hidden_states)
        query_layer = self._reshape_heads(query, batch_size, sequence_length)
        key_layer = self._reshape_heads(key, batch_size, sequence_length)
        value_layer = self._reshape_heads(value, batch_size, sequence_length)

        scale_factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
        scale = self._scale(scale_factor)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) / scale)

        if self.relative_attention and "c2p" in self.pos_att_type:
            if plan.pos_key is None:
                raise ValueError("prepared plan has no content-to-position keys")
            c2p_raw = torch.matmul(query_layer, plan.pos_key.transpose(-1, -2)) / scale
            attention_scores = attention_scores + torch.gather(
                c2p_raw,
                dim=-1,
                index=plan.c2p_local.expand(
                    batch_size,
                    self.num_attention_heads,
                    sequence_length,
                    sequence_length,
                ),
            )

        if self.relative_attention and "p2c" in self.pos_att_type:
            if plan.pos_query is None:
                raise ValueError("prepared plan has no position-to-content queries")
            p2c_raw = torch.matmul(key_layer, plan.pos_query.transpose(-1, -2)) / scale
            attention_scores = attention_scores + torch.gather(
                p2c_raw.transpose(-1, -2),
                dim=-2,
                index=plan.p2c_local.expand(
                    batch_size,
                    self.num_attention_heads,
                    sequence_length,
                    sequence_length,
                ),
            )

        mask = _prepare_attention_mask(
            attention_mask,
            sequence_length,
            sequence_length,
        ).bool()
        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(query_layer.dtype).min,
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = (
            context_layer.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        return context_layer, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Convenience wrapper; preparation remains outside ``forward_prepared``."""

        self._validate_inference_call()
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the inference path")
        if query_states is not None:
            raise ValueError("the optimized inference path supports self-attention only")

        needs_key = self.relative_attention and "c2p" in self.pos_att_type and self._cached_pos_key is None
        needs_query = self.relative_attention and "p2c" in self.pos_att_type and self._cached_pos_query is None
        needs_qkv = self._cached_qkv_weight is None
        if needs_key or needs_query or needs_qkv:
            self.prepare_for_inference(rel_embeddings)

        sequence_length = hidden_states.size(1)
        if relative_pos is None:
            plan = self.prepare_shape(sequence_length, hidden_states.device)
        else:
            query_layer = hidden_states.view(
                hidden_states.size(0),
                1,
                sequence_length,
                hidden_states.size(-1),
            )
            plan = self._dynamic_position_plan(query_layer, query_layer, relative_pos)
        return self.forward_prepared(hidden_states, attention_mask, plan)


__all__ = ["InferenceDisentangledSelfAttention", "PreparedPositionPlan"]
