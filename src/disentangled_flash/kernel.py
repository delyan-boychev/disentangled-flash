"""CUDA Triton implementation of exact inference-only DeBERTa attention.

QK, relative-score lookup, a factorized padding mask, online softmax, and PV
are fused without constructing ``[B, H, L, L]`` scores/probabilities.  C2P and
P2C remain regular GEMMs over the pruned active relative-position slots.
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple

import torch

from ._reference import DebertaAttentionConfig
from ._torch import TorchInferenceDisentangledSelfAttention
from .position import SharedPositionPlanCache, canonical_device

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is intentionally optional on CPU and macOS.
    triton = None
    tl = None


if triton is not None:
    # Conservative schedules for this DeBERTa kernel.
    #
    # This kernel carries more live state than vanilla FlashAttention:
    #   * Q/K/V tiles
    #   * FP32 online-softmax accumulator
    #   * score tile
    #   * C2P/P2C lookup state
    #   * relative-position indices and masks
    #
    # Keep all schedules at one pipeline stage. Most candidates stay within
    # 64x64; a pair of asymmetric larger tiles is retained for FP16/BF16 and
    # pruned out for heavier FP32/head-dim workloads.
    _AUTOTUNE_CONFIGS = [
        # tiny
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        # short / medium
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        # normal long path
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        # aggressive, but still sane
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ]

    def _prune_autotune_configs(
        configs: list[Any],
        named_args: dict[str, Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Keep only resource-safe candidates useful for the current shape."""

        sequence_length_value = kwargs.get(
            "SEQUENCE_LENGTH",
            named_args.get("SEQUENCE_LENGTH"),
        )
        head_dim_value = kwargs.get(
            "HEAD_DIM",
            named_args.get("HEAD_DIM"),
        )
        is_fp32_value = kwargs.get(
            "IS_FP32",
            named_args.get("IS_FP32"),
        )

        if sequence_length_value is None or head_dim_value is None:
            return configs

        sequence_length = int(sequence_length_value)
        head_dim = int(head_dim_value)
        is_fp32 = bool(is_fp32_value)

        if sequence_length <= 16:
            allowed_shapes = {
                (16, 16),
                (16, 32),
            }

        elif sequence_length <= 32:
            allowed_shapes = {
                (16, 16),
                (16, 32),
                (32, 32),
            }

        elif sequence_length <= 64:
            allowed_shapes = {
                (32, 32),
                (32, 64),
                (64, 32),
                (64, 64),
            }

        else:
            allowed_shapes = {
                (32, 32),
                (32, 64),
                (64, 32),
                (64, 64),
            }

            # Larger tiles are worth testing for FP16/BF16 with normal head sizes.
            # They use only one pipeline stage, and safe configurations above remain
            # available if Triton rejects one for resource usage.
            if not is_fp32 and head_dim <= 64:
                allowed_shapes.update(
                    {
                        (64, 128),
                        (128, 64),
                    }
                )

        kept = [
            config
            for config in configs
            if (
                config.kwargs["BLOCK_M"],
                config.kwargs["BLOCK_N"],
            )
            in allowed_shapes
        ]

        # 32x32 is deliberately present as a conservative fallback for every
        # non-tiny sequence class.
        return kept or configs[:1]

    @triton.jit
    def _deberta_attention_forward_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        attention_mask,
        output,
        ACTIVE_SLOTS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_in_bounds = query_offsets < SEQUENCE_LENGTH

        query_base = query + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        key_base = key + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        value_base = value + batch_head * SEQUENCE_LENGTH * HEAD_DIM
        head = batch_head - batch * NUM_HEADS

        output_base = output + batch * SEQUENCE_LENGTH * NUM_HEADS * HEAD_DIM + head * HEAD_DIM

        query_values = tl.load(
            query_base + query_offsets[:, None] * HEAD_DIM + dimension_offsets[None, :],
            mask=query_in_bounds[:, None],
            other=0.0,
        )
        query_is_kept = tl.load(
            attention_mask + batch * SEQUENCE_LENGTH + query_offsets,
            mask=query_in_bounds,
            other=0,
        ).to(tl.int1)

        row_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        if HAS_C2P:
            c2p_base = c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS

        for key_start in tl.range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            key_in_bounds = key_offsets < SEQUENCE_LENGTH
            key_is_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + key_offsets,
                mask=key_in_bounds,
                other=0,
            ).to(tl.int1)

            key_values = tl.load(
                key_base + key_offsets[:, None] * HEAD_DIM + dimension_offsets[None, :],
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    scores = tl.dot(
                        query_values,
                        tl.trans(key_values),
                        input_precision="ieee",
                    )
                else:
                    scores = tl.dot(
                        query_values,
                        tl.trans(key_values),
                        input_precision="tf32",
                    )
            else:
                scores = tl.dot(query_values, tl.trans(key_values))

            pair_in_bounds = query_in_bounds[:, None] & key_in_bounds[None, :]
            delta_index = query_offsets[:, None] - key_offsets[None, :] + SEQUENCE_LENGTH - 1
            local_slot = tl.load(
                delta_to_local_slot + delta_index,
                mask=pair_in_bounds,
                other=0,
            ).to(tl.int32)

            if HAS_C2P:
                scores += tl.load(
                    c2p_base + query_offsets[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )

            if HAS_P2C:
                scores += tl.load(
                    p2c_base + key_offsets[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )

            scores *= SCORE_SCALE_LOG2
            attended = query_is_kept[:, None] & key_is_kept[None, :] & pair_in_bounds
            scores = tl.where(attended, scores, -float("inf"))

            # HF masks every score of a padded query with one finite minimum;
            # softmax therefore becomes uniform over the complete in-range row.
            padded_query_row = query_in_bounds[:, None] & ~query_is_kept[:, None]
            scores = tl.where(padded_query_row & key_in_bounds[None, :], 0.0, scores)
            scores = tl.where(
                ~query_in_bounds[:, None] & (key_offsets[None, :] == 0),
                0.0,
                scores,
            )

            value_values = tl.load(
                value_base + key_offsets[:, None] * HEAD_DIM + dimension_offsets[None, :],
                mask=key_in_bounds[:, None],
                other=0.0,
            )

            # When the complete K/V sequence fits in one tile, avoid the online
            # softmax recurrence. Both values are constexpr, so Triton removes
            # the unused branch at compile time.
            if SEQUENCE_LENGTH <= BLOCK_N:
                new_row_max = tl.max(scores, axis=1)
                probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = tl.sum(probabilities, axis=1)

                if IS_FP32:
                    if STRICT_FP32:
                        accumulator = tl.dot(
                            probabilities,
                            value_values,
                            input_precision="ieee",
                        )
                    else:
                        accumulator = tl.dot(
                            probabilities,
                            value_values,
                            input_precision="tf32",
                        )
                elif IS_BF16:
                    accumulator = tl.dot(
                        probabilities.to(tl.bfloat16),
                        value_values,
                    )
                else:
                    accumulator = tl.dot(
                        probabilities.to(tl.float16),
                        value_values,
                    )
            else:
                new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
                # NaN bug fix, 0 as a normalization center (mainly left padding affected)
                row_has_scores = new_row_max != -float("inf")
                normalization_center = tl.where(row_has_scores, new_row_max, 0.0)
                correction = tl.where(
                    row_has_scores,
                    tl.math.exp2(row_max - normalization_center),
                    1.0,
                )
                probabilities = tl.math.exp2(scores - normalization_center[:, None])
                new_row_sum = row_sum * correction + tl.sum(probabilities, axis=1)

                accumulator *= correction[:, None]
                if IS_FP32:
                    if STRICT_FP32:
                        accumulator = tl.dot(
                            probabilities,
                            value_values,
                            accumulator,
                            input_precision="ieee",
                        )
                    else:
                        accumulator = tl.dot(
                            probabilities,
                            value_values,
                            accumulator,
                            input_precision="tf32",
                        )
                elif IS_BF16:
                    accumulator = tl.dot(
                        probabilities.to(tl.bfloat16),
                        value_values,
                        accumulator,
                    )
                else:
                    accumulator = tl.dot(
                        probabilities.to(tl.float16),
                        value_values,
                        accumulator,
                    )

            row_max = new_row_max
            row_sum = new_row_sum

        accumulator /= row_sum[:, None]
        tl.store(
            output_base
            + query_offsets[:, None] * (NUM_HEADS * HEAD_DIM)
            + dimension_offsets[None, :],
            accumulator,
            mask=query_in_bounds[:, None],
        )

    _autotune_kwargs: dict[str, Any] = {
        "configs": _AUTOTUNE_CONFIGS,
        "key": [
            "BATCH_SIZE",
            "SEQUENCE_LENGTH",
            "HEAD_DIM",
            "ACTIVE_SLOTS",
            "HAS_C2P",
            "HAS_P2C",
            "IS_BF16",
            "IS_FP32",
            "STRICT_FP32",
        ],
        "prune_configs_by": {"early_config_prune": _prune_autotune_configs},
    }
    if "cache_results" in inspect.signature(triton.autotune).parameters:
        _autotune_kwargs["cache_results"] = True
    _deberta_attention_autotuned_kernel = triton.autotune(**_autotune_kwargs)(
        _deberta_attention_forward_kernel
    )

    def _launch_deberta_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        sequence_length: int,
        active_slots: int,
        score_scale_log2: float,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        head_dim = query.size(-1)

        output = torch.empty(
            (
                batch_size,
                sequence_length,
                num_heads * head_dim,
            ),
            device=query.device,
            dtype=query.dtype,
        )

        def grid(meta: dict[str, Any]) -> tuple[int, int]:
            return (
                triton.cdiv(sequence_length, meta["BLOCK_M"]),
                query.size(0) * num_heads,
            )

        kernel_kwargs = {
            "ACTIVE_SLOTS": active_slots,
            "BATCH_SIZE": batch_size,
            "NUM_HEADS": num_heads,
            "SEQUENCE_LENGTH": sequence_length,
            "HEAD_DIM": query.size(-1),
            "SCORE_SCALE_LOG2": score_scale_log2,
            "HAS_C2P": has_c2p,
            "HAS_P2C": has_p2c,
            "IS_BF16": is_bf16,
            "IS_FP32": is_fp32,
            "STRICT_FP32": strict_fp32,
        }
        torch.library.wrap_triton(_deberta_attention_autotuned_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            output,
            **kernel_kwargs,
        )
        return output

    if hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton"):
        _deberta_attention_op = torch.library.triton_op(
            "gliner2_attention::deberta_attention",
            _launch_deberta_attention,
            mutates_args={},
        )
    else:  # Older PyTorch still supports raw user-authored Triton calls.
        _deberta_attention_op = _launch_deberta_attention


def _require_2d_padding_mask(
    attention_mask: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Validate the fast path's factorized per-token padding-mask contract."""

    if attention_mask.dim() != 2:
        raise ValueError(
            "the Triton fast path requires a factorized padding mask with shape "
            "[B, L]; arbitrary pairwise masks are not supported"
        )
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"[{batch_size}, {sequence_length}]"
        )
    return attention_mask.bool().contiguous()


class TritonPreparedPositionPlan(NamedTuple):
    """Layer projections plus one shared compact position LUT."""

    sequence_length: int
    active_slots: torch.Tensor
    delta_to_local: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


class TritonInferenceDisentangledSelfAttention(TorchInferenceDisentangledSelfAttention):
    """Forward-only CUDA Triton DeBERTa-v2/v3 self-attention.

    Head dimensions 32, 64, and 128 have explicit supported paths.  ``strict``
    FP32 uses IEEE dot products; ``fast`` opts into TF32 inside the fused kernel.
    PyTorch's global FP32 matmul setting still governs the separate C2P/P2C and
    projection GEMMs.
    """

    def __init__(
        self,
        config: DebertaAttentionConfig | Any,
        *,
        position_plan_cache: SharedPositionPlanCache | None = None,
        fp32_precision: str = "strict",
    ) -> None:
        super().__init__(
            config,
            position_plan_cache=position_plan_cache,
        )
        if fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be 'strict' or 'fast'")
        self.fp32_precision = fp32_precision
        self._triton_position_projection_cache: dict[
            tuple[int, str], TritonPreparedPositionPlan
        ] = {}

    def clear_inference_cache(self) -> None:
        super().clear_inference_cache()
        if hasattr(self, "_triton_position_projection_cache"):
            self._triton_position_projection_cache.clear()

    @torch.no_grad()
    def prepare_shape(
        self,
        sequence_length: int,
        device: torch.device | str | None = None,
    ) -> TritonPreparedPositionPlan:
        """Prepare layer projections while keeping position indexing ``O(L)``."""

        if self.training:
            raise RuntimeError("prepare_shape() requires module.eval()")
        resident_device = self._plan_device()
        resolved_device = canonical_device(device, resident_device)
        if resolved_device.type != "cuda":
            raise ValueError("Triton shape plans must be prepared on CUDA")
        cache_key = sequence_length, str(resolved_device)
        cached = self._triton_position_projection_cache.get(cache_key)
        if cached is not None:
            return cached

        indices = self.position_plan_cache.compact(sequence_length, resolved_device)
        pos_key, pos_query = self._project_active_positions(indices.active_slots)
        plan = TritonPreparedPositionPlan(
            sequence_length=sequence_length,
            active_slots=indices.active_slots,
            delta_to_local=indices.delta_to_local.to(dtype=torch.int32).contiguous(),
            pos_key=pos_key,
            pos_query=pos_query,
        )
        self._triton_position_projection_cache[cache_key] = plan
        return plan

    def _validate_triton_call(self, hidden_states: torch.Tensor) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed; this backend requires CUDA and Triton")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("TritonInferenceDisentangledSelfAttention requires CUDA")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("the Triton path supports FP16, BF16, and FP32")
        if self.attention_head_size not in {32, 64, 128}:
            raise ValueError("the Triton path supports attention head dimensions 32, 64, and 128")
        self._validate_inference_call()

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plan: TritonPreparedPositionPlan,
    ) -> tuple[torch.Tensor, None]:
        """Pure forward using a plan prepared completely outside the hot path."""

        self._validate_triton_call(hidden_states)
        batch_size, sequence_length = hidden_states.shape[:2]
        if sequence_length != plan.sequence_length:
            raise ValueError(
                f"prepared length {plan.sequence_length} does not match input length "
                f"{sequence_length}"
            )
        base_mask = _require_2d_padding_mask(
            attention_mask,
            batch_size,
            sequence_length,
        )

        query, key, value = self._project_qkv(hidden_states)
        query_layer = self._reshape_heads(query, batch_size, sequence_length)
        key_layer = self._reshape_heads(key, batch_size, sequence_length)
        value_layer = self._reshape_heads(value, batch_size, sequence_length)

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        active_slot_count = plan.active_slots.numel()
        if has_c2p:
            if plan.pos_key is None:
                raise ValueError("prepared Triton plan has no content-to-position keys")
            c2p = torch.matmul(query_layer, plan.pos_key.transpose(-1, -2))
        else:
            c2p = query_layer
        if has_p2c:
            if plan.pos_query is None:
                raise ValueError("prepared Triton plan has no position-to-content queries")
            p2c = torch.matmul(key_layer, plan.pos_query.transpose(-1, -2))
        else:
            p2c = key_layer

        scale_factor = 1 + int(has_c2p) + int(has_p2c)
        score_scale_log2 = self._scale(scale_factor) ** -1 * 1.4426950408889634
        output = _deberta_attention_op(
            query_layer,
            key_layer,
            value_layer,
            c2p,
            p2c,
            plan.delta_to_local,
            base_mask,
            self.num_attention_heads,
            sequence_length,
            active_slot_count,
            score_scale_log2,
            has_c2p,
            has_p2c,
            hidden_states.dtype == torch.bfloat16,
            hidden_states.dtype == torch.float32,
            self.fp32_precision == "strict",
        )

        return output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Convenience wrapper with lazy preparation outside the pure hot path."""

        self._validate_triton_call(hidden_states)
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the Triton path")
        if query_states is not None:
            raise ValueError("the Triton path supports self-attention only")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the Triton path")

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type
        needs_key = has_c2p and self._cached_pos_key is None
        needs_query = has_p2c and self._cached_pos_query is None
        needs_qkv = self._cached_qkv_weight is None
        if needs_key or needs_query or needs_qkv:
            self.prepare_for_inference(rel_embeddings)

        plan = self.prepare_shape(hidden_states.size(1), hidden_states.device)
        return self.forward_prepared(hidden_states, attention_mask, plan)


DisentangledFlashAttention = TritonInferenceDisentangledSelfAttention

__all__ = [
    "DisentangledFlashAttention",
    "TritonInferenceDisentangledSelfAttention",
    "TritonPreparedPositionPlan",
]
