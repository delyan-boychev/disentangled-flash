"""Triton kernels for exact DeBERTa disentangled attention.

Scores are never materialized as [B, H, L, L]. C2P and P2C stay regular GEMMs
over the active relative-position slots. Training reuses the same forward with
STORE_LSE enabled.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any, NamedTuple

import torch

from ._torch import TorchInferenceDisentangledSelfAttention, TorchPositionPlan
from .packed import PackedSequenceInfo, resolve_packed_info
from .position import canonical_device
from .tuning import (
    DEFAULT_KERNEL_CONFIGS,
    KernelConfig,
    KernelTuningOptions,
    ProfileRegistry,
    WorkloadKey,
    length_family,
    occupancy_family,
    register_profile_registry,
    resolve_launch_config,
    tuning_sequence_length,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is intentionally optional on CPU and macOS.
    triton = None
    tl = None


AUTOTUNE_SPECIALIZATION_KEY = (
    "LENGTH_REGIME",
    "HEAD_DIM",
    "HAS_C2P",
    "HAS_P2C",
    "USE_PADDING_MASK",
    "IS_BF16",
    "IS_FP32",
    "STRICT_FP32",
    "STORE_LSE",
    "PHYSICAL_PAIRS",
)


if triton is not None:
    # This kernel keeps more live state than plain FlashAttention (C2P/P2C
    # lookups, position indices, masks), so schedules stay small.
    def _as_triton_config(config: KernelConfig) -> Any:
        return triton.Config(
            {"BLOCK_M": config.block_m, "BLOCK_N": config.block_n},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    _AUTOTUNE_CONFIGS = [_as_triton_config(config) for config in DEFAULT_KERNEL_CONFIGS]

    def _prune_autotune_configs(
        configs: list[Any],
        named_args: dict[str, Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Keep only resource-safe candidates useful for the current shape."""

        sequence_length_value = kwargs.get(
            "LENGTH_REGIME",
            named_args.get("LENGTH_REGIME"),
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

            # Larger tiles only pay off for FP16/BF16 with normal head sizes.
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
            and not (is_fp32 and config.num_stages > 1)
            and not (config.num_stages > 2 and (sequence_length < 384 or head_dim != 64))
        ]

        # 32x32 is always allowed as a safe fallback.
        return kept or configs[:1]

    # Dropout mask shared by forward and backward: one Philox key per
    # (batch or sequence, head), one counter per (row, col). Backward
    # regenerates it instead of storing it.
    @triton.jit
    def _dropout_keep(dropout_seed, stream, rows, cols, DROPOUT_P):
        seed = tl.load(dropout_seed) + stream
        counters = rows.to(tl.uint32)[:, None] * 65536 + cols.to(tl.uint32)[None, :]
        return tl.rand(seed, counters) >= DROPOUT_P

    @triton.jit(
        do_not_specialize=[
            "ACTIVE_SLOTS",
            "NUM_HEADS",
            "SEQUENCE_LENGTH",
            "POSITION_OFFSET",
        ]
    )
    def _deberta_attention_forward_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        attention_mask,
        output,
        lse_log2,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        ACTIVE_SLOTS,
        NUM_HEADS,
        SEQUENCE_LENGTH,
        POSITION_OFFSET,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE_LOG2,
        LENGTH_REGIME: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        USE_PADDING_MASK: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        STORE_LSE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PHYSICAL_PAIRS: tl.constexpr = False,
        dropout_seed=None,
        DROPOUT_P=0.0,
        DROPOUT_SCALE=1.0,
        HAS_DROPOUT: tl.constexpr = False,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_in_bounds = query_offsets < SEQUENCE_LENGTH

        head = batch_head - batch * NUM_HEADS

        query_base = query + batch * stride_qb + head * stride_qh
        key_base = key + batch * stride_kb + head * stride_kh
        value_base = value + batch * stride_vb + head * stride_vh

        output_base = output + batch * SEQUENCE_LENGTH * NUM_HEADS * HEAD_DIM + head * HEAD_DIM
        if STORE_LSE:
            lse_base = lse_log2 + batch_head * SEQUENCE_LENGTH

        query_values = tl.load(
            query_base
            + query_offsets[:, None] * stride_ql
            + dimension_offsets[None, :] * stride_qd,
            mask=query_in_bounds[:, None],
            other=0.0,
        )
        if USE_PADDING_MASK:
            query_is_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + query_offsets,
                mask=query_in_bounds,
                other=0,
            ).to(tl.int1)
        else:
            query_is_kept = query_in_bounds

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
            if USE_PADDING_MASK:
                key_is_kept = tl.load(
                    attention_mask + batch * SEQUENCE_LENGTH + key_offsets,
                    mask=key_in_bounds,
                    other=0,
                ).to(tl.int1)
            else:
                key_is_kept = key_in_bounds

            key_values = tl.load(
                key_base
                + key_offsets[:, None] * stride_kl
                + dimension_offsets[None, :] * stride_kd,
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
            if PHYSICAL_PAIRS:
                delta_index = (
                    batch * SEQUENCE_LENGTH + query_offsets[:, None]
                ) * SEQUENCE_LENGTH + key_offsets[None, :]
            else:
                delta_index = query_offsets[:, None] - key_offsets[None, :] + POSITION_OFFSET
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
            if USE_PADDING_MASK:
                attended = query_is_kept[:, None] & key_is_kept[None, :] & pair_in_bounds
                scores = tl.where(attended, scores, -float("inf"))

                # Preserve Hugging Face semantics for padded query rows.
                padded_query_row = query_in_bounds[:, None] & ~query_is_kept[:, None]
                scores = tl.where(
                    padded_query_row & key_in_bounds[None, :],
                    0.0,
                    scores,
                )
            else:
                scores = tl.where(
                    pair_in_bounds,
                    scores,
                    -float("inf"),
                )

            # Keep out-of-range rows finite; they are never stored.
            scores = tl.where(
                ~query_in_bounds[:, None] & (key_offsets[None, :] == 0),
                0.0,
                scores,
            )

            value_values = tl.load(
                value_base
                + key_offsets[:, None] * stride_vl
                + dimension_offsets[None, :] * stride_vd,
                mask=key_in_bounds[:, None],
                other=0.0,
            )

            # Single-tile rows skip the online-softmax rescaling.
            if LENGTH_REGIME <= BLOCK_N:
                new_row_max = tl.max(scores, axis=1)
                probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = tl.sum(probabilities, axis=1)
                # Row sums and LSE stay undropped, as in FlashAttention.
                if HAS_DROPOUT:
                    keep = _dropout_keep(
                        dropout_seed, batch_head, query_offsets, key_offsets, DROPOUT_P
                    )
                    probabilities = tl.where(keep, probabilities, 0.0)

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
                if USE_PADDING_MASK:
                    # Left padding can mask a whole K/V tile; avoid -inf - -inf.
                    row_has_scores = new_row_max != -float("inf")
                    normalization_center = tl.where(row_has_scores, new_row_max, 0.0)
                    correction = tl.where(
                        row_has_scores,
                        tl.math.exp2(row_max - normalization_center),
                        1.0,
                    )
                    probabilities = tl.math.exp2(scores - normalization_center[:, None])
                else:
                    correction = tl.math.exp2(row_max - new_row_max)
                    probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
                if HAS_DROPOUT:
                    keep = _dropout_keep(
                        dropout_seed, batch_head, query_offsets, key_offsets, DROPOUT_P
                    )
                    probabilities = tl.where(keep, probabilities, 0.0)

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
        if HAS_DROPOUT:
            accumulator *= DROPOUT_SCALE
        tl.store(
            output_base
            + query_offsets[:, None] * (NUM_HEADS * HEAD_DIM)
            + dimension_offsets[None, :],
            accumulator,
            mask=query_in_bounds[:, None],
        )
        if STORE_LSE:
            lse = row_max + tl.log(row_sum) * 1.4426950408889634
            tl.store(lse_base + query_offsets, lse, mask=query_in_bounds)

    @triton.jit(
        do_not_specialize=[
            "ACTIVE_SLOTS",
            "MAX_SEQLEN",
            "NUM_HEADS",
            "POSITION_OFFSET",
            "TOTAL_TOKENS",
        ]
    )
    def _deberta_attention_packed_forward_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        cu_seqlens,
        output,
        lse_log2,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vh,
        stride_vl,
        stride_vd,
        ACTIVE_SLOTS,
        MAX_SEQLEN,
        NUM_HEADS,
        POSITION_OFFSET,
        TOTAL_TOKENS,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE_LOG2,
        LENGTH_REGIME: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        STORE_LSE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        dropout_seed=None,
        DROPOUT_P=0.0,
        DROPOUT_SCALE=1.0,
        HAS_DROPOUT: tl.constexpr = False,
    ):
        query_block = tl.program_id(0)
        sequence_head = tl.program_id(1)
        sequence = sequence_head // NUM_HEADS
        head = sequence_head - sequence * NUM_HEADS

        sequence_start = tl.load(cu_seqlens + sequence).to(tl.int64)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        sequence_length = sequence_end - sequence_start

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        query_in_bounds = query_offsets < sequence_length
        query_tokens = sequence_start + query_offsets
        dimension_offsets = tl.arange(0, HEAD_DIM)

        query_base = query + head * stride_qh
        key_base = key + head * stride_kh
        value_base = value + head * stride_vh

        query_values = tl.load(
            query_base + query_tokens[:, None] * stride_ql + dimension_offsets[None, :] * stride_qd,
            mask=query_in_bounds[:, None],
            other=0.0,
        )
        row_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        if HAS_C2P:
            c2p_base = c2p + head * TOTAL_TOKENS * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + head * TOTAL_TOKENS * ACTIVE_SLOTS

        for key_start in tl.range(0, sequence_length, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            key_in_bounds = key_offsets < sequence_length
            key_tokens = sequence_start + key_offsets

            key_values = tl.load(
                key_base + key_tokens[:, None] * stride_kl + dimension_offsets[None, :] * stride_kd,
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
            delta_index = query_offsets[:, None] - key_offsets[None, :] + POSITION_OFFSET
            local_slot = tl.load(
                delta_to_local_slot + delta_index,
                mask=pair_in_bounds,
                other=0,
            ).to(tl.int32)

            if HAS_C2P:
                scores += tl.load(
                    c2p_base + query_tokens[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
            if HAS_P2C:
                scores += tl.load(
                    p2c_base + key_tokens[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )

            scores *= SCORE_SCALE_LOG2
            scores = tl.where(pair_in_bounds, scores, -float("inf"))
            scores = tl.where(
                ~query_in_bounds[:, None] & (key_offsets[None, :] == 0),
                0.0,
                scores,
            )

            value_values = tl.load(
                value_base
                + key_tokens[:, None] * stride_vl
                + dimension_offsets[None, :] * stride_vd,
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            if LENGTH_REGIME <= BLOCK_N:
                new_row_max = tl.max(scores, axis=1)
                probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = tl.sum(probabilities, axis=1)
                if HAS_DROPOUT:
                    keep = _dropout_keep(
                        dropout_seed, sequence_head, query_offsets, key_offsets, DROPOUT_P
                    )
                    probabilities = tl.where(keep, probabilities, 0.0)
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
                    accumulator = tl.dot(probabilities.to(tl.bfloat16), value_values)
                else:
                    accumulator = tl.dot(probabilities.to(tl.float16), value_values)
            else:
                new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
                row_has_scores = new_row_max != -float("inf")
                normalization_center = tl.where(row_has_scores, new_row_max, 0.0)
                correction = tl.where(
                    row_has_scores,
                    tl.math.exp2(row_max - normalization_center),
                    1.0,
                )
                probabilities = tl.math.exp2(scores - normalization_center[:, None])
                new_row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
                if HAS_DROPOUT:
                    keep = _dropout_keep(
                        dropout_seed, sequence_head, query_offsets, key_offsets, DROPOUT_P
                    )
                    probabilities = tl.where(keep, probabilities, 0.0)
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
        if HAS_DROPOUT:
            accumulator *= DROPOUT_SCALE
        tl.store(
            output
            + query_tokens[:, None] * (NUM_HEADS * HEAD_DIM)
            + head * HEAD_DIM
            + dimension_offsets[None, :],
            accumulator,
            mask=query_in_bounds[:, None],
        )
        if STORE_LSE:
            lse = row_max + tl.log(row_sum) * 1.4426950408889634
            tl.store(
                lse_log2 + head * TOTAL_TOKENS + query_tokens,
                lse,
                mask=query_in_bounds,
            )

    _AUTOTUNE_KEY = list(AUTOTUNE_SPECIALIZATION_KEY)
    _PACKED_AUTOTUNE_KEY = [
        "LENGTH_REGIME",
        "HEAD_DIM",
        "HAS_C2P",
        "HAS_P2C",
        "HAS_PADDING",
        "IS_BF16",
        "IS_FP32",
        "STRICT_FP32",
        "STORE_LSE",
    ]

    def _make_autotuned_kernel(
        configs: tuple[KernelConfig, ...],
        extra_key: tuple[str, ...] = (),
    ) -> Any:
        autotune_kwargs: dict[str, Any] = {
            "configs": [_as_triton_config(config) for config in configs],
            "key": [*_AUTOTUNE_KEY, *extra_key],
            "prune_configs_by": {"early_config_prune": _prune_autotune_configs},
        }
        if "cache_results" in inspect.signature(triton.autotune).parameters:
            autotune_kwargs["cache_results"] = True
        return triton.autotune(**autotune_kwargs)(_deberta_attention_forward_kernel)

    _deberta_attention_autotuned_kernel = _make_autotuned_kernel(DEFAULT_KERNEL_CONFIGS)

    def _make_packed_autotuned_kernel(
        configs: tuple[KernelConfig, ...],
        extra_key: tuple[str, ...] = (),
    ) -> Any:
        autotune_kwargs: dict[str, Any] = {
            "configs": [_as_triton_config(config) for config in configs],
            "key": [*_PACKED_AUTOTUNE_KEY, *extra_key],
            "prune_configs_by": {"early_config_prune": _prune_autotune_configs},
        }
        if "cache_results" in inspect.signature(triton.autotune).parameters:
            autotune_kwargs["cache_results"] = True
        return triton.autotune(**autotune_kwargs)(_deberta_attention_packed_forward_kernel)

    _deberta_attention_packed_autotuned_kernel = _make_packed_autotuned_kernel(
        DEFAULT_KERNEL_CONFIGS
    )

    def _launch_autotuned_kernel(
        autotuned_kernel: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        sequence_length: int,
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        use_padding_mask: bool,
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
            "NUM_HEADS": num_heads,
            "SEQUENCE_LENGTH": sequence_length,
            "POSITION_OFFSET": position_offset,
            "HEAD_DIM": query.size(-1),
            "SCORE_SCALE_LOG2": score_scale_log2,
            "LENGTH_REGIME": length_regime,
            "HAS_C2P": has_c2p,
            "HAS_P2C": has_p2c,
            "USE_PADDING_MASK": use_padding_mask,
            "IS_BF16": is_bf16,
            "IS_FP32": is_fp32,
            "STRICT_FP32": strict_fp32,
            "STORE_LSE": False,
            "PHYSICAL_PAIRS": False,
        }
        torch.library.wrap_triton(autotuned_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            output,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            **kernel_kwargs,
        )
        return output

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
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        has_padding: bool,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
    ) -> torch.Tensor:
        return _launch_autotuned_kernel(
            _deberta_attention_autotuned_kernel,
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            num_heads,
            sequence_length,
            length_regime,
            active_slots,
            position_offset,
            score_scale_log2,
            has_padding,
            has_c2p,
            has_p2c,
            is_bf16,
            is_fp32,
            strict_fp32,
        )

    def _launch_deberta_attention_configured(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        sequence_length: int,
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        has_padding: bool,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
        block_m: int,
        block_n: int,
        num_warps: int,
        num_stages: int,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        head_dim = query.size(-1)
        output = torch.empty(
            (batch_size, sequence_length, num_heads * head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        grid = (triton.cdiv(sequence_length, block_m), batch_size * num_heads)
        torch.library.wrap_triton(_deberta_attention_forward_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            output,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            ACTIVE_SLOTS=active_slots,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            POSITION_OFFSET=position_offset,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale_log2,
            LENGTH_REGIME=length_regime,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            USE_PADDING_MASK=has_padding,
            IS_BF16=is_bf16,
            IS_FP32=is_fp32,
            STRICT_FP32=strict_fp32,
            STORE_LSE=False,
            PHYSICAL_PAIRS=False,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

    def _launch_packed_autotuned_kernel(
        autotuned_kernel: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
    ) -> torch.Tensor:
        num_heads, total_tokens, head_dim = query.shape
        batch_size = cu_seqlens.numel() - 1
        output = torch.empty(
            (total_tokens, num_heads * head_dim),
            device=query.device,
            dtype=query.dtype,
        )

        def grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(max_seqlen, meta["BLOCK_M"]), batch_size * num_heads

        torch.library.wrap_triton(autotuned_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            output,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            ACTIVE_SLOTS=active_slots,
            MAX_SEQLEN=max_seqlen,
            NUM_HEADS=num_heads,
            POSITION_OFFSET=position_offset,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale_log2,
            LENGTH_REGIME=length_regime,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=False,
            IS_BF16=is_bf16,
            IS_FP32=is_fp32,
            STRICT_FP32=strict_fp32,
            STORE_LSE=False,
        )
        return output

    def _launch_deberta_attention_packed(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
    ) -> torch.Tensor:
        return _launch_packed_autotuned_kernel(
            _deberta_attention_packed_autotuned_kernel,
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            max_seqlen,
            length_regime,
            active_slots,
            position_offset,
            score_scale_log2,
            has_c2p,
            has_p2c,
            is_bf16,
            is_fp32,
            strict_fp32,
        )

    def _launch_deberta_attention_packed_configured(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        c2p: torch.Tensor,
        p2c: torch.Tensor,
        delta_to_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        length_regime: int,
        active_slots: int,
        position_offset: int,
        score_scale_log2: float,
        has_c2p: bool,
        has_p2c: bool,
        is_bf16: bool,
        is_fp32: bool,
        strict_fp32: bool,
        block_m: int,
        block_n: int,
        num_warps: int,
        num_stages: int,
    ) -> torch.Tensor:
        num_heads, total_tokens, head_dim = query.shape
        batch_size = cu_seqlens.numel() - 1
        output = torch.empty(
            (total_tokens, num_heads * head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        grid = (triton.cdiv(max_seqlen, block_m), batch_size * num_heads)
        torch.library.wrap_triton(_deberta_attention_packed_forward_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            output,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            ACTIVE_SLOTS=active_slots,
            MAX_SEQLEN=max_seqlen,
            NUM_HEADS=num_heads,
            POSITION_OFFSET=position_offset,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale_log2,
            LENGTH_REGIME=length_regime,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=False,
            IS_BF16=is_bf16,
            IS_FP32=is_fp32,
            STRICT_FP32=strict_fp32,
            STORE_LSE=False,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

    if hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton"):
        _deberta_attention_op = torch.library.triton_op(
            "gliner2_attention::deberta_attention",
            _launch_deberta_attention,
            mutates_args={},
        )
        _deberta_attention_configured_op = torch.library.triton_op(
            "gliner2_attention::deberta_attention_configured",
            _launch_deberta_attention_configured,
            mutates_args={},
        )
        _deberta_attention_packed_op = torch.library.triton_op(
            "gliner2_attention::deberta_attention_packed",
            _launch_deberta_attention_packed,
            mutates_args={},
        )
        _deberta_attention_packed_configured_op = torch.library.triton_op(
            "gliner2_attention::deberta_attention_packed_configured",
            _launch_deberta_attention_packed_configured,
            mutates_args={},
        )
    else:  # Older PyTorch still supports raw user-authored Triton calls.
        _deberta_attention_op = _launch_deberta_attention
        _deberta_attention_configured_op = _launch_deberta_attention_configured
        _deberta_attention_packed_op = _launch_deberta_attention_packed
        _deberta_attention_packed_configured_op = _launch_deberta_attention_packed_configured

    @cache
    def _custom_autotune_operator(configs: tuple[KernelConfig, ...]) -> Any:
        if configs == DEFAULT_KERNEL_CONFIGS:
            return _deberta_attention_op
        autotuned_kernel = _make_autotuned_kernel(configs)

        def launch(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            c2p: torch.Tensor,
            p2c: torch.Tensor,
            delta_to_local: torch.Tensor,
            attention_mask: torch.Tensor,
            num_heads: int,
            sequence_length: int,
            length_regime: int,
            active_slots: int,
            position_offset: int,
            score_scale_log2: float,
            has_padding: bool,
            has_c2p: bool,
            has_p2c: bool,
            is_bf16: bool,
            is_fp32: bool,
            strict_fp32: bool,
        ) -> torch.Tensor:
            return _launch_autotuned_kernel(
                autotuned_kernel,
                query,
                key,
                value,
                c2p,
                p2c,
                delta_to_local,
                attention_mask,
                num_heads,
                sequence_length,
                length_regime,
                active_slots,
                position_offset,
                score_scale_log2,
                has_padding,
                has_c2p,
                has_p2c,
                is_bf16,
                is_fp32,
                strict_fp32,
            )

        # Register an operator so torch.compile can trace custom candidates.
        if hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton"):
            suffix = abs(hash(configs))
            return torch.library.triton_op(
                f"gliner2_attention::deberta_attention_custom_{suffix}",
                launch,
                mutates_args={},
            )
        return launch

    @cache
    def _custom_packed_autotune_operator(configs: tuple[KernelConfig, ...]) -> Any:
        if configs == DEFAULT_KERNEL_CONFIGS:
            return _deberta_attention_packed_op
        autotuned_kernel = _make_packed_autotuned_kernel(configs)

        def launch(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            c2p: torch.Tensor,
            p2c: torch.Tensor,
            delta_to_local: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_seqlen: int,
            length_regime: int,
            active_slots: int,
            position_offset: int,
            score_scale_log2: float,
            has_c2p: bool,
            has_p2c: bool,
            is_bf16: bool,
            is_fp32: bool,
            strict_fp32: bool,
        ) -> torch.Tensor:
            return _launch_packed_autotuned_kernel(
                autotuned_kernel,
                query,
                key,
                value,
                c2p,
                p2c,
                delta_to_local,
                cu_seqlens,
                max_seqlen,
                length_regime,
                active_slots,
                position_offset,
                score_scale_log2,
                has_c2p,
                has_p2c,
                is_bf16,
                is_fp32,
                strict_fp32,
            )

        if hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton"):
            suffix = abs(hash(configs))
            return torch.library.triton_op(
                f"gliner2_attention::deberta_attention_packed_custom_{suffix}",
                launch,
                mutates_args={},
            )
        return launch


def _validate_2d_padding_mask(
    attention_mask: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Validate only the shape of the factorized padding mask."""

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


def _require_2d_padding_mask(
    attention_mask: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Return the normalized bool padding mask used by the padded kernel."""

    _validate_2d_padding_mask(
        attention_mask,
        batch_size,
        sequence_length,
    )

    if attention_mask.dtype == torch.bool and attention_mask.is_contiguous():
        return attention_mask

    return attention_mask.bool().contiguous()


class TritonPreparedPositionPlan(NamedTuple):
    """Layer projections plus one shared compact position LUT."""

    sequence_length: int
    active_slots: torch.Tensor
    delta_to_local: torch.Tensor
    position_offset: int
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None


class InferenceDisentangledSelfAttention(TorchInferenceDisentangledSelfAttention):
    """DeBERTa-v2/v3 inference attention with a PyTorch or Triton backend."""

    def __init__(
        self,
        config,
        *,
        backend: str = "triton",
        position_plan_cache=None,
        fp32_precision: str = "strict",
        tuning: KernelTuningOptions | None = None,
        profile_registry: ProfileRegistry | None = None,
        assume_unpadded: bool = False,
    ):
        super().__init__(
            config,
            position_plan_cache=position_plan_cache,
            assume_unpadded=assume_unpadded,
        )

        if backend not in {"torch", "triton"}:
            raise ValueError("backend must be 'torch' or 'triton'")

        if fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be 'strict' or 'fast'")

        self.backend = backend
        self.fp32_precision = fp32_precision
        self.assume_unpadded = assume_unpadded
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
        self._profile_registry_key = register_profile_registry(self._profile_registry)
        self._failed_profile_workloads: set[WorkloadKey] = set()
        if triton is not None:
            candidates = self.tuning.candidates or DEFAULT_KERNEL_CONFIGS
            self._autotune_operator = _custom_autotune_operator(candidates)
            self._packed_autotune_operator = _custom_packed_autotune_operator(candidates)
        self._triton_position_projection_cache: dict[
            tuple[int, str], TritonPreparedPositionPlan
        ] = {}

    def _reshape_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        """Return a BHLD view without materializing a contiguous copy."""
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).permute(0, 2, 1, 3)

    def clear_inference_cache(self) -> None:
        super().clear_inference_cache()
        if hasattr(self, "_failed_profile_workloads"):
            self._failed_profile_workloads.clear()
        if hasattr(self, "_triton_position_projection_cache"):
            self._triton_position_projection_cache.clear()

    def _resolve_kernel_config(
        self,
        hidden_states: torch.Tensor,
        *,
        active_slots: int,
        has_c2p: bool,
        has_p2c: bool,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        layout: str = "padded",
        uses_padding_mask: bool = True,
    ) -> tuple[KernelConfig | None, WorkloadKey]:
        """Return a direct-launch config plus its finite workload identity."""

        batch_heads = (
            hidden_states.size(0) if batch_size is None else batch_size
        ) * self.num_attention_heads
        # Families, not exact sizes, so the fields stay constant under torch.compile.
        fields = (
            length_family(hidden_states.size(1) if sequence_length is None else sequence_length),
            self.attention_head_size,
            occupancy_family(batch_heads),
            active_slots,
            str(hidden_states.dtype).removeprefix("torch."),
            has_c2p,
            has_p2c,
            self.fp32_precision,
            layout,
            uses_padding_mask,
            "inference",
            False,
        )
        workload = WorkloadKey(*fields)
        if self.tuning.mode == "autotune":
            return None, workload
        if self.tuning.mode == "fixed":
            return self.tuning.fixed_config, workload
        compiling = torch.compiler.is_compiling()
        if not compiling and workload in self._failed_profile_workloads:
            return None, workload
        device_index = hidden_states.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        config = resolve_launch_config(
            self._profile_registry_key,
            self.tuning.mode,
            device_index,
            fields,
            None if compiling else batch_heads,
        )
        return KernelConfig(*config), workload

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        *,
        rel_embeddings: torch.Tensor | None = None,
        packed_info: PackedSequenceInfo | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run packed attention with the selected inference backend."""

        if self.backend == "torch":
            return super().forward_packed(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                rel_embeddings=rel_embeddings,
                packed_info=packed_info,
            )

        self._validate_triton_call(hidden_states)
        if hidden_states.ndim != 2 or hidden_states.size(-1) != self.all_head_size:
            raise ValueError("packed hidden_states must have shape [total_tokens, hidden_size]")
        if cu_seqlens.device != hidden_states.device:
            raise ValueError("cu_seqlens must be on the hidden_states device")
        info = resolve_packed_info(
            cu_seqlens,
            hidden_states.size(0),
            max_seqlen,
            packed_info,
        )

        needs_positions = self.relative_attention and bool(
            {"c2p", "p2c"}.intersection(self.pos_att_type)
        )
        if self._cached_qkv_weight is None or (
            needs_positions
            and self._cached_pos_key is None
            and self._cached_pos_query is None
            and self._get_cached_shape_plan(info.max_seqlen, hidden_states.device) is None
        ):
            self.prepare_for_inference(rel_embeddings)
        plan = self.prepare_shape(info.max_seqlen, hidden_states.device)

        total_tokens = hidden_states.size(0)
        query, key, value = self._project_qkv(hidden_states)

        def packed_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                total_tokens,
                self.num_attention_heads,
                self.attention_head_size,
            ).permute(1, 0, 2)

        query_layer = packed_heads(query)
        key_layer = packed_heads(key)
        value_layer = packed_heads(value)
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
        selected_config, workload = self._resolve_kernel_config(
            hidden_states,
            active_slots=active_slot_count,
            has_c2p=has_c2p,
            has_p2c=has_p2c,
            batch_size=len(info.lengths),
            sequence_length=info.max_seqlen,
            layout="packed",
            uses_padding_mask=False,
        )
        operation_args = (
            query_layer,
            key_layer,
            value_layer,
            c2p,
            p2c,
            plan.delta_to_local,
            cu_seqlens,
            info.max_seqlen,
            tuning_sequence_length(info.max_seqlen),
            active_slot_count,
            plan.position_offset,
            score_scale_log2,
            has_c2p,
            has_p2c,
            hidden_states.dtype == torch.bfloat16,
            hidden_states.dtype == torch.float32,
            self.fp32_precision == "strict",
        )
        if selected_config is None:
            output = self._packed_autotune_operator(*operation_args)
        else:
            try:
                output = _deberta_attention_packed_configured_op(
                    *operation_args,
                    selected_config.block_m,
                    selected_config.block_n,
                    selected_config.num_warps,
                    selected_config.num_stages,
                )
            except Exception as error:
                if self.tuning.mode == "profile_only":
                    raise RuntimeError(
                        f"saved packed kernel configuration failed to launch: {selected_config}"
                    ) from error
                self._failed_profile_workloads.add(workload)
                output = self._packed_autotune_operator(*operation_args)
        return output, None

    @torch.no_grad()
    def prepare_shape(
        self,
        sequence_length: int,
        device: torch.device | str | None = None,
    ) -> TorchPositionPlan | TritonPreparedPositionPlan:
        """Prepare only the position representation required by the backend."""
        if self.backend == "torch":
            return super().prepare_shape(sequence_length, device)
        if self.training:
            raise RuntimeError("prepare_shape() requires module.eval()")
        resident_device = self._plan_device()
        resolved_device = canonical_device(device, resident_device)
        if resolved_device.type != "cuda":
            raise ValueError("Triton shape plans must be prepared on CUDA")
        if sequence_length > 8192:
            raise ValueError(
                "the bounded Triton kernel family supports sequence lengths up to 8192"
            )
        cache_key = sequence_length, str(resolved_device)
        cached = self._triton_position_projection_cache.get(cache_key)
        if cached is not None:
            return cached

        representative = tuning_sequence_length(sequence_length)
        indices = self.position_plan_cache.compact(representative, resolved_device)
        representative_plan = self._triton_position_projection_cache.get(
            (representative, str(resolved_device))
        )
        if representative_plan is None:
            pos_key, pos_query = self._project_active_positions(indices.active_slots)
        else:
            pos_key, pos_query = representative_plan.pos_key, representative_plan.pos_query
        plan = TritonPreparedPositionPlan(
            sequence_length=sequence_length,
            active_slots=indices.active_slots,
            delta_to_local=indices.delta_to_local.to(dtype=torch.int32).contiguous(),
            position_offset=representative - 1,
            pos_key=pos_key,
            pos_query=pos_query,
        )
        self._triton_position_projection_cache[cache_key] = plan
        return plan

    def _get_cached_shape_plan(
        self,
        sequence_length: int,
        device: torch.device | str,
    ) -> TorchPositionPlan | TritonPreparedPositionPlan | None:
        if self.backend == "torch":
            return super()._get_cached_shape_plan(sequence_length, device)
        resolved_device = canonical_device(
            device,
            self._plan_device(),
        )
        return self._triton_position_projection_cache.get((sequence_length, str(resolved_device)))

    def _validate_triton_call(self, hidden_states: torch.Tensor) -> None:
        if triton is None:
            raise RuntimeError("Triton is not installed; this backend requires CUDA and Triton")
        if hidden_states.device.type != "cuda":
            raise RuntimeError("the Triton inference backend requires CUDA")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("the Triton path supports FP16, BF16, and FP32")
        if self.attention_head_size not in {32, 64, 128}:
            raise ValueError("the Triton path supports attention head dimensions 32, 64, and 128")
        self._validate_inference_call()

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plan: TorchPositionPlan | TritonPreparedPositionPlan,
    ) -> tuple[torch.Tensor, None]:
        """Pure backend-dispatched forward using an already prepared plan."""

        if self.backend == "torch":
            if not isinstance(plan, TorchPositionPlan):
                raise TypeError("the Torch inference backend requires a TorchPositionPlan")
            return super().forward_prepared(
                hidden_states,
                attention_mask,
                plan,
            )

        if not isinstance(plan, TritonPreparedPositionPlan):
            raise TypeError("the Triton inference backend requires a TritonPreparedPositionPlan")

        self._validate_triton_call(hidden_states)
        batch_size, sequence_length = hidden_states.shape[:2]
        if sequence_length != plan.sequence_length:
            raise ValueError(
                f"prepared length {plan.sequence_length} does not match input length "
                f"{sequence_length}"
            )
        if self.assume_unpadded:
            # The mask is unused here, so only check its shape.
            _validate_2d_padding_mask(
                attention_mask,
                batch_size,
                sequence_length,
            )
            base_mask = attention_mask
        else:
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
        selected_config, workload = self._resolve_kernel_config(
            hidden_states,
            active_slots=active_slot_count,
            has_c2p=has_c2p,
            has_p2c=has_p2c,
            layout="padded",
            uses_padding_mask=not self.assume_unpadded,
        )
        operation = self._autotune_operator
        operation_args = (
            query_layer,
            key_layer,
            value_layer,
            c2p,
            p2c,
            plan.delta_to_local,
            base_mask,
            self.num_attention_heads,
            sequence_length,
            tuning_sequence_length(plan.sequence_length),
            active_slot_count,
            plan.position_offset,
            score_scale_log2,
            not self.assume_unpadded,
            has_c2p,
            has_p2c,
            hidden_states.dtype == torch.bfloat16,
            hidden_states.dtype == torch.float32,
            self.fp32_precision == "strict",
        )
        if selected_config is None:
            output = operation(*operation_args)
        else:
            try:
                output = _deberta_attention_configured_op(
                    *operation_args,
                    selected_config.block_m,
                    selected_config.block_n,
                    selected_config.num_warps,
                    selected_config.num_stages,
                )
            except Exception as error:
                if self.tuning.mode == "profile_only":
                    raise RuntimeError(
                        f"saved padded kernel configuration failed to launch: {selected_config}"
                    ) from error
                self._failed_profile_workloads.add(workload)
                output = operation(*operation_args)

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
        """Convenience wrapper with lazy backend-specific preparation."""

        if self.backend == "torch":
            return super().forward(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        self._validate_triton_call(hidden_states)
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the Triton path")
        if query_states is not None:
            raise ValueError("the Triton path supports self-attention only")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the Triton path")

        sequence_length = hidden_states.size(1)

        cached_plan = self._get_cached_shape_plan(
            sequence_length,
            hidden_states.device,
        )
        if cached_plan is not None and self._cached_qkv_weight is not None:
            return self.forward_prepared(
                hidden_states,
                attention_mask,
                cached_plan,
            )

        has_c2p = self.relative_attention and "c2p" in self.pos_att_type
        has_p2c = self.relative_attention and "p2c" in self.pos_att_type

        needs_key = has_c2p and self._cached_pos_key is None
        needs_query = has_p2c and self._cached_pos_query is None
        needs_qkv = self._cached_qkv_weight is None

        if needs_key or needs_query or needs_qkv:
            self.prepare_for_inference(rel_embeddings)

        plan = self.prepare_shape(
            sequence_length,
            hidden_states.device,
        )

        return self.forward_prepared(
            hidden_states,
            attention_mask,
            plan,
        )


DisentangledFlashAttention = InferenceDisentangledSelfAttention

__all__ = [
    "AUTOTUNE_SPECIALIZATION_KEY",
    "DisentangledFlashAttention",
    "InferenceDisentangledSelfAttention",
    "TritonPreparedPositionPlan",
]
