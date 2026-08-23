"""Triton forward/backward kernels for trainable disentangled attention.

This module is intentionally separate from :mod:`disentangled_flash.kernel`,
which remains the proven inference-only implementation.  The training forward
uses the same tiled QK + compact relative lookup + online-softmax + PV
algorithm, but additionally stores one FP32 log-sum-exp value per query row.
Backward recomputes score/probability tiles and never materializes an L x L
attention matrix.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only CI.
    triton = None
    tl = None

_LOG2E = 1.4426950408889634


def _require_training_runtime() -> None:
    if triton is None:
        raise RuntimeError("training attention requires Triton and an NVIDIA CUDA GPU")
    if not hasattr(torch.library, "triton_op") or not hasattr(torch.library, "wrap_triton"):
        raise RuntimeError("training attention requires torch>=2.7 with torch.library.triton_op")


if triton is not None:
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ]

    def _prune_fwd_configs(
        configs: list[Any],
        named_args: dict[str, Any],
        **kwargs: Any,
    ) -> list[Any]:
        length_value = kwargs.get("SEQUENCE_LENGTH", named_args.get("SEQUENCE_LENGTH"))
        head_dim_value = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM"))
        is_fp32_value = kwargs.get("IS_FP32", named_args.get("IS_FP32"))
        if length_value is None or head_dim_value is None:
            return configs

        length = int(length_value)
        head_dim = int(head_dim_value)
        is_fp32 = bool(is_fp32_value)
        if length <= 16:
            allowed = {(16, 16), (16, 32)}
        elif length <= 32:
            allowed = {(16, 16), (16, 32), (32, 32)}
        elif length <= 64:
            allowed = {(32, 32), (32, 64), (64, 32), (64, 64)}
        else:
            allowed = {(32, 32), (32, 64), (64, 32), (64, 64)}
            if not is_fp32 and head_dim <= 64:
                allowed.update({(64, 128), (128, 64)})
        kept = [
            config
            for config in configs
            if (config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]) in allowed
        ]
        return kept or configs[:1]

    @triton.jit
    def _training_forward_kernel(
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
        ACTIVE_SLOTS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS
        head = batch_head - batch * NUM_HEADS

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_lane = tl.arange(0, BLOCK_N)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_in_bounds = query_offsets < SEQUENCE_LENGTH

        query_base = query + batch * stride_qb + head * stride_qh
        key_base = key + batch * stride_kb + head * stride_kh
        value_base = value + batch * stride_vb + head * stride_vh
        output_base = output + batch * SEQUENCE_LENGTH * NUM_HEADS * HEAD_DIM + head * HEAD_DIM
        lse_base = lse_log2 + batch_head * SEQUENCE_LENGTH

        q = tl.load(
            query_base
            + query_offsets[:, None] * stride_ql
            + dimension_offsets[None, :] * stride_qd,
            mask=query_in_bounds[:, None],
            other=0.0,
        )

        if HAS_PADDING:
            query_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + query_offsets,
                mask=query_in_bounds,
                other=0,
            ).to(tl.int1)
        else:
            query_kept = query_in_bounds

        row_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        if HAS_C2P:
            c2p_base = c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS

        for key_start in tl.range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            key_offsets = key_start + key_lane
            key_in_bounds = key_offsets < SEQUENCE_LENGTH

            if HAS_PADDING:
                key_kept = tl.load(
                    attention_mask + batch * SEQUENCE_LENGTH + key_offsets,
                    mask=key_in_bounds,
                    other=0,
                ).to(tl.int1)
            else:
                key_kept = key_in_bounds

            k = tl.load(
                key_base
                + key_offsets[:, None] * stride_kl
                + dimension_offsets[None, :] * stride_kd,
                mask=key_in_bounds[:, None],
                other=0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    scores = tl.dot(q, tl.trans(k), input_precision="ieee")
                else:
                    scores = tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores = tl.dot(q, tl.trans(k))

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
            if HAS_PADDING:
                attended = query_kept[:, None] & key_kept[None, :] & pair_in_bounds
                scores = tl.where(attended, scores, -float("inf"))
                padded_query = query_in_bounds[:, None] & ~query_kept[:, None]
                # Match HF masked_fill(min) -> softmax semantics for padded rows:
                # all in-bounds keys receive equal probability.
                scores = tl.where(
                    padded_query & key_in_bounds[None, :],
                    0.0,
                    scores,
                )
            else:
                scores = tl.where(pair_in_bounds, scores, -float("inf"))

            # Keep unused rows in the final partial query tile numerically valid.
            scores = tl.where(
                ~query_in_bounds[:, None] & (key_offsets[None, :] == 0),
                0.0,
                scores,
            )

            v = tl.load(
                value_base
                + key_offsets[:, None] * stride_vl
                + dimension_offsets[None, :] * stride_vd,
                mask=key_in_bounds[:, None],
                other=0.0,
            )

            if SEQUENCE_LENGTH <= BLOCK_N:
                new_row_max = tl.max(scores, axis=1)
                probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = tl.sum(probabilities, axis=1)
                if IS_FP32:
                    if STRICT_FP32:
                        accumulator = tl.dot(probabilities, v, input_precision="ieee")
                    else:
                        accumulator = tl.dot(probabilities, v, input_precision="tf32")
                elif IS_BF16:
                    accumulator = tl.dot(probabilities.to(tl.bfloat16), v)
                else:
                    accumulator = tl.dot(probabilities.to(tl.float16), v)
            else:
                new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
                correction = tl.math.exp2(row_max - new_row_max)
                probabilities = tl.math.exp2(scores - new_row_max[:, None])
                new_row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
                accumulator *= correction[:, None]
                if IS_FP32:
                    if STRICT_FP32:
                        accumulator = tl.dot(
                            probabilities,
                            v,
                            accumulator,
                            input_precision="ieee",
                        )
                    else:
                        accumulator = tl.dot(
                            probabilities,
                            v,
                            accumulator,
                            input_precision="tf32",
                        )
                elif IS_BF16:
                    accumulator = tl.dot(probabilities.to(tl.bfloat16), v, accumulator)
                else:
                    accumulator = tl.dot(probabilities.to(tl.float16), v, accumulator)

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
        lse = row_max + tl.log(row_sum) * 1.4426950408889634
        tl.store(lse_base + query_offsets, lse, mask=query_in_bounds)

    _FWD_AUTOTUNE_KWARGS: dict[str, Any] = {
        "configs": _FWD_CONFIGS,
        "key": [
            "BATCH_SIZE",
            "SEQUENCE_LENGTH",
            "HEAD_DIM",
            "ACTIVE_SLOTS",
            "HAS_C2P",
            "HAS_P2C",
            "HAS_PADDING",
            "IS_BF16",
            "IS_FP32",
            "STRICT_FP32",
        ],
        "prune_configs_by": {"early_config_prune": _prune_fwd_configs},
    }
    if "cache_results" in inspect.signature(triton.autotune).parameters:
        _FWD_AUTOTUNE_KWARGS["cache_results"] = True
    _training_forward_autotuned = triton.autotune(**_FWD_AUTOTUNE_KWARGS)(_training_forward_kernel)

    @triton.jit
    def _backward_preprocess_kernel(
        output,
        grad_output,
        delta,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        stride_gob,
        stride_goh,
        stride_gol,
        stride_god,
        BATCH_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS
        head = batch_head - batch * NUM_HEADS
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        row_mask = rows < SEQUENCE_LENGTH
        o = tl.load(
            output
            + batch * stride_ob
            + head * stride_oh
            + rows[:, None] * stride_ol
            + dims[None, :] * stride_od,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            grad_output
            + batch * stride_gob
            + head * stride_goh
            + rows[:, None] * stride_gol
            + dims[None, :] * stride_god,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        value = tl.sum(o * do, axis=1)
        tl.store(
            delta + batch_head * SEQUENCE_LENGTH + rows,
            value,
            mask=row_mask,
        )

    @triton.jit
    def _backward_dq_dc_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        attention_mask,
        grad_output,
        lse_log2,
        delta,
        grad_query,
        grad_c2p,
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
        stride_gob,
        stride_goh,
        stride_gol,
        stride_god,
        stride_dqb,
        stride_dqh,
        stride_dql,
        stride_dqd,
        ACTIVE_SLOTS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS
        head = batch_head - batch * NUM_HEADS
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_lane = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, HEAD_DIM)
        row_mask = rows < SEQUENCE_LENGTH

        q_base = query + batch * stride_qb + head * stride_qh
        k_base = key + batch * stride_kb + head * stride_kh
        v_base = value + batch * stride_vb + head * stride_vh
        do_base = grad_output + batch * stride_gob + head * stride_goh
        dq_base = grad_query + batch * stride_dqb + head * stride_dqh
        lse_base = lse_log2 + batch_head * SEQUENCE_LENGTH
        delta_base = delta + batch_head * SEQUENCE_LENGTH

        q = tl.load(
            q_base + rows[:, None] * stride_ql + dims[None, :] * stride_qd,
            mask=row_mask[:, None],
            other=0.0,
        )
        do = tl.load(
            do_base + rows[:, None] * stride_gol + dims[None, :] * stride_god,
            mask=row_mask[:, None],
            other=0.0,
        )
        lse = tl.load(lse_base + rows, mask=row_mask, other=0.0)
        row_delta = tl.load(delta_base + rows, mask=row_mask, other=0.0)

        if HAS_PADDING:
            query_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + rows,
                mask=row_mask,
                other=0,
            ).to(tl.int1)
        else:
            query_kept = row_mask

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        if HAS_C2P:
            c2p_base = c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
            dc_base = grad_c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS

        for key_start in tl.range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            cols = key_start + key_lane
            col_mask = cols < SEQUENCE_LENGTH
            if HAS_PADDING:
                key_kept = tl.load(
                    attention_mask + batch * SEQUENCE_LENGTH + cols,
                    mask=col_mask,
                    other=0,
                ).to(tl.int1)
            else:
                key_kept = col_mask

            k = tl.load(
                k_base + cols[:, None] * stride_kl + dims[None, :] * stride_kd,
                mask=col_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + cols[:, None] * stride_vl + dims[None, :] * stride_vd,
                mask=col_mask[:, None],
                other=0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    scores = tl.dot(q, tl.trans(k), input_precision="ieee")
                else:
                    scores = tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores = tl.dot(q, tl.trans(k))

            pair_in_bounds = row_mask[:, None] & col_mask[None, :]
            delta_index = rows[:, None] - cols[None, :] + SEQUENCE_LENGTH - 1
            local_slot = tl.load(
                delta_to_local_slot + delta_index,
                mask=pair_in_bounds,
                other=0,
            ).to(tl.int32)
            if HAS_C2P:
                scores += tl.load(
                    c2p_base + rows[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
            if HAS_P2C:
                scores += tl.load(
                    p2c_base + cols[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
            scores *= SCORE_SCALE_LOG2

            if HAS_PADDING:
                attended = query_kept[:, None] & key_kept[None, :] & pair_in_bounds
                padded_query = row_mask[:, None] & ~query_kept[:, None]
                padded_pair = padded_query & col_mask[None, :]
                scores = tl.where(attended, scores, -float("inf"))
                scores = tl.where(padded_pair, 0.0, scores)
                active_pair = attended | padded_pair
            else:
                scores = tl.where(pair_in_bounds, scores, -float("inf"))
                active_pair = pair_in_bounds

            p = tl.math.exp2(scores - lse[:, None])
            p = tl.where(active_pair, p, 0.0)

            if IS_FP32:
                if STRICT_FP32:
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                else:
                    dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            else:
                dp = tl.dot(do, tl.trans(v))
            ds_raw = p * (dp.to(tl.float32) - row_delta[:, None]) * SCORE_SCALE
            ds_raw = tl.where(active_pair, ds_raw, 0.0)

            if IS_FP32:
                if STRICT_FP32:
                    dq += tl.dot(ds_raw, k, input_precision="ieee")
                else:
                    dq += tl.dot(ds_raw, k, input_precision="tf32")
            elif IS_BF16:
                dq += tl.dot(ds_raw.to(tl.bfloat16), k)
            else:
                dq += tl.dot(ds_raw.to(tl.float16), k)

            if HAS_C2P:
                tl.atomic_add(
                    dc_base + rows[:, None] * ACTIVE_SLOTS + local_slot,
                    ds_raw,
                    mask=active_pair,
                )

        tl.store(
            dq_base + rows[:, None] * stride_dql + dims[None, :] * stride_dqd,
            dq,
            mask=row_mask[:, None],
        )

    @triton.jit
    def _backward_dkv_dt_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        attention_mask,
        grad_output,
        lse_log2,
        delta,
        grad_key,
        grad_value,
        grad_p2c,
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
        stride_gob,
        stride_goh,
        stride_gol,
        stride_god,
        stride_dkb,
        stride_dkh,
        stride_dkl,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvl,
        stride_dvd,
        ACTIVE_SLOTS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // NUM_HEADS
        head = batch_head - batch * NUM_HEADS
        cols = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        query_lane = tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        col_mask = cols < SEQUENCE_LENGTH

        q_base = query + batch * stride_qb + head * stride_qh
        k_base = key + batch * stride_kb + head * stride_kh
        v_base = value + batch * stride_vb + head * stride_vh
        do_base = grad_output + batch * stride_gob + head * stride_goh
        dk_base = grad_key + batch * stride_dkb + head * stride_dkh
        dv_base = grad_value + batch * stride_dvb + head * stride_dvh
        lse_base = lse_log2 + batch_head * SEQUENCE_LENGTH
        delta_base = delta + batch_head * SEQUENCE_LENGTH

        k = tl.load(
            k_base + cols[:, None] * stride_kl + dims[None, :] * stride_kd,
            mask=col_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + cols[:, None] * stride_vl + dims[None, :] * stride_vd,
            mask=col_mask[:, None],
            other=0.0,
        )

        if HAS_PADDING:
            key_kept = tl.load(
                attention_mask + batch * SEQUENCE_LENGTH + cols,
                mask=col_mask,
                other=0,
            ).to(tl.int1)
        else:
            key_kept = col_mask

        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        if HAS_C2P:
            c2p_base = c2p + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS
            dt_base = grad_p2c + batch_head * SEQUENCE_LENGTH * ACTIVE_SLOTS

        for query_start in tl.range(0, SEQUENCE_LENGTH, BLOCK_M):
            query_start = tl.multiple_of(query_start, BLOCK_M)
            rows = query_start + query_lane
            row_mask = rows < SEQUENCE_LENGTH
            if HAS_PADDING:
                query_kept = tl.load(
                    attention_mask + batch * SEQUENCE_LENGTH + rows,
                    mask=row_mask,
                    other=0,
                ).to(tl.int1)
            else:
                query_kept = row_mask

            q = tl.load(
                q_base + rows[:, None] * stride_ql + dims[None, :] * stride_qd,
                mask=row_mask[:, None],
                other=0.0,
            )
            do = tl.load(
                do_base + rows[:, None] * stride_gol + dims[None, :] * stride_god,
                mask=row_mask[:, None],
                other=0.0,
            )
            lse = tl.load(lse_base + rows, mask=row_mask, other=0.0)
            row_delta = tl.load(delta_base + rows, mask=row_mask, other=0.0)

            if IS_FP32:
                if STRICT_FP32:
                    scores = tl.dot(q, tl.trans(k), input_precision="ieee")
                else:
                    scores = tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores = tl.dot(q, tl.trans(k))

            pair_in_bounds = row_mask[:, None] & col_mask[None, :]
            delta_index = rows[:, None] - cols[None, :] + SEQUENCE_LENGTH - 1
            local_slot = tl.load(
                delta_to_local_slot + delta_index,
                mask=pair_in_bounds,
                other=0,
            ).to(tl.int32)
            if HAS_C2P:
                scores += tl.load(
                    c2p_base + rows[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
            if HAS_P2C:
                scores += tl.load(
                    p2c_base + cols[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_in_bounds,
                    other=0.0,
                )
            scores *= SCORE_SCALE_LOG2

            if HAS_PADDING:
                attended = query_kept[:, None] & key_kept[None, :] & pair_in_bounds
                padded_query = row_mask[:, None] & ~query_kept[:, None]
                padded_pair = padded_query & col_mask[None, :]
                scores = tl.where(attended, scores, -float("inf"))
                scores = tl.where(padded_pair, 0.0, scores)
                active_pair = attended | padded_pair
            else:
                scores = tl.where(pair_in_bounds, scores, -float("inf"))
                active_pair = pair_in_bounds

            p = tl.math.exp2(scores - lse[:, None])
            p = tl.where(active_pair, p, 0.0)

            if IS_FP32:
                if STRICT_FP32:
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                else:
                    dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            else:
                dp = tl.dot(do, tl.trans(v))
            ds_raw = p * (dp.to(tl.float32) - row_delta[:, None]) * SCORE_SCALE
            ds_raw = tl.where(active_pair, ds_raw, 0.0)

            if IS_FP32:
                if STRICT_FP32:
                    dk += tl.dot(tl.trans(ds_raw), q, input_precision="ieee")
                    dv += tl.dot(tl.trans(p), do, input_precision="ieee")
                else:
                    dk += tl.dot(tl.trans(ds_raw), q, input_precision="tf32")
                    dv += tl.dot(tl.trans(p), do, input_precision="tf32")
            elif IS_BF16:
                dk += tl.dot(tl.trans(ds_raw.to(tl.bfloat16)), q)
                dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            else:
                dk += tl.dot(tl.trans(ds_raw.to(tl.float16)), q)
                dv += tl.dot(tl.trans(p.to(tl.float16)), do)

            if HAS_P2C:
                tl.atomic_add(
                    dt_base + cols[None, :] * ACTIVE_SLOTS + local_slot,
                    ds_raw,
                    mask=active_pair,
                )

        tl.store(
            dk_base + cols[:, None] * stride_dkl + dims[None, :] * stride_dkd,
            dk,
            mask=col_mask[:, None],
        )
        tl.store(
            dv_base + cols[:, None] * stride_dvl + dims[None, :] * stride_dvd,
            dv,
            mask=col_mask[:, None],
        )


def _backward_tile_config(sequence_length: int, head_dim: int) -> tuple[int, int, int, int]:
    """Conservative first-generation backward schedule.

    Backward has more live state than forward, so use resource-safe tiles until
    measured training profiles justify a larger autotune search.
    """

    if sequence_length <= 32:
        return 32, 32, 4, 1
    if sequence_length <= 64:
        return 32, 64, 4, 1
    if head_dim <= 64:
        return 64, 64, 4, 1
    return 32, 64, 4, 1


def _empty_optional(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_empty((0,))


if (
    triton is not None
    and hasattr(torch.library, "triton_op")
    and hasattr(torch.library, "wrap_triton")
):

    @torch.library.triton_op(
        "disentangled_flash::training_attention_forward",
        mutates_args={},
    )
    def _training_attention_forward_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        delta_to_local: torch.Tensor,
        attention_mask: torch.Tensor,
        score_scale: float,
        has_padding: bool,
        has_c2p: bool,
        has_p2c: bool,
        strict_fp32: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, sequence_length, head_dim = query.shape
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2) if has_p2c else 0
        c2p = torch.matmul(query, pos_key.transpose(-1, -2)) if has_c2p else _empty_optional(query)
        p2c = torch.matmul(key, pos_query.transpose(-1, -2)) if has_p2c else _empty_optional(key)
        output = torch.empty(
            (batch_size, sequence_length, num_heads * head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        lse_log2 = torch.empty(
            (batch_size, num_heads, sequence_length),
            device=query.device,
            dtype=torch.float32,
        )

        def grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(sequence_length, meta["BLOCK_M"]), batch_size * num_heads

        torch.library.wrap_triton(_training_forward_autotuned)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            output,
            lse_log2,
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
            BATCH_SIZE=batch_size,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=has_padding,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
        )
        return output, lse_log2

    @torch.library.triton_op(
        "disentangled_flash::training_attention_backward",
        mutates_args={},
    )
    def _training_attention_backward_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        delta_to_local: torch.Tensor,
        attention_mask: torch.Tensor,
        output: torch.Tensor,
        lse_log2: torch.Tensor,
        grad_output: torch.Tensor,
        score_scale: float,
        has_padding: bool,
        has_c2p: bool,
        has_p2c: bool,
        strict_fp32: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, sequence_length, head_dim = query.shape
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2) if has_p2c else 0
        c2p = torch.matmul(query, pos_key.transpose(-1, -2)) if has_c2p else _empty_optional(query)
        p2c = torch.matmul(key, pos_query.transpose(-1, -2)) if has_p2c else _empty_optional(key)

        output_heads = output.view(batch_size, sequence_length, num_heads, head_dim).permute(
            0, 2, 1, 3
        )
        grad_output_heads = grad_output.view(
            batch_size, sequence_length, num_heads, head_dim
        ).permute(0, 2, 1, 3)

        delta = torch.empty(
            (batch_size, num_heads, sequence_length),
            device=query.device,
            dtype=torch.float32,
        )
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_c2p = (
            torch.zeros(
                (batch_size, num_heads, sequence_length, active_slots),
                device=query.device,
                dtype=torch.float32,
            )
            if has_c2p
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )
        grad_p2c = (
            torch.zeros(
                (batch_size, num_heads, sequence_length, active_slots),
                device=query.device,
                dtype=torch.float32,
            )
            if has_p2c
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )

        block_m, block_n, num_warps, num_stages = _backward_tile_config(sequence_length, head_dim)
        preprocess_grid = (triton.cdiv(sequence_length, block_m), batch_size * num_heads)
        torch.library.wrap_triton(_backward_preprocess_kernel)[preprocess_grid](
            output_heads,
            grad_output_heads,
            delta,
            output_heads.stride(0),
            output_heads.stride(1),
            output_heads.stride(2),
            output_heads.stride(3),
            grad_output_heads.stride(0),
            grad_output_heads.stride(1),
            grad_output_heads.stride(2),
            grad_output_heads.stride(3),
            BATCH_SIZE=batch_size,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dq_grid = (triton.cdiv(sequence_length, block_m), batch_size * num_heads)
        torch.library.wrap_triton(_backward_dq_dc_kernel)[dq_grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            grad_output_heads,
            lse_log2,
            delta,
            grad_query,
            grad_c2p,
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
            grad_output_heads.stride(0),
            grad_output_heads.stride(1),
            grad_output_heads.stride(2),
            grad_output_heads.stride(3),
            grad_query.stride(0),
            grad_query.stride(1),
            grad_query.stride(2),
            grad_query.stride(3),
            ACTIVE_SLOTS=active_slots,
            BATCH_SIZE=batch_size,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            SCORE_SCALE=score_scale,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=has_padding,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dkv_grid = (triton.cdiv(sequence_length, block_n), batch_size * num_heads)
        torch.library.wrap_triton(_backward_dkv_dt_kernel)[dkv_grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            grad_output_heads,
            lse_log2,
            delta,
            grad_key,
            grad_value,
            grad_p2c,
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
            grad_output_heads.stride(0),
            grad_output_heads.stride(1),
            grad_output_heads.stride(2),
            grad_output_heads.stride(3),
            grad_key.stride(0),
            grad_key.stride(1),
            grad_key.stride(2),
            grad_key.stride(3),
            grad_value.stride(0),
            grad_value.stride(1),
            grad_value.stride(2),
            grad_value.stride(3),
            ACTIVE_SLOTS=active_slots,
            BATCH_SIZE=batch_size,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            SCORE_SCALE=score_scale,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=has_padding,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return grad_query, grad_key, grad_value, grad_c2p, grad_p2c

    def _setup_training_attention_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        (
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            attention_mask,
            score_scale,
            has_padding,
            has_c2p,
            has_p2c,
            strict_fp32,
        ) = inputs
        attention_output, lse_log2 = output
        ctx.save_for_backward(
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            attention_mask,
            attention_output,
            lse_log2,
        )
        ctx.score_scale = score_scale
        ctx.has_padding = has_padding
        ctx.has_c2p = has_c2p
        ctx.has_p2c = has_p2c
        ctx.strict_fp32 = strict_fp32

    def _training_attention_autograd_backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_lse: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        del grad_lse
        (
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            attention_mask,
            output,
            lse_log2,
        ) = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_c2p, grad_p2c = _training_attention_backward_op(
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            attention_mask,
            output,
            lse_log2,
            grad_output.contiguous(),
            ctx.score_scale,
            ctx.has_padding,
            ctx.has_c2p,
            ctx.has_p2c,
            ctx.strict_fp32,
        )

        grad_pos_key = None
        if ctx.has_c2p:
            grad_c2p_typed = grad_c2p.to(dtype=query.dtype)
            grad_query = grad_query + torch.matmul(grad_c2p_typed, pos_key)
            grad_pos_key = torch.matmul(grad_c2p_typed.transpose(-1, -2), query).sum(dim=0)

        grad_pos_query = None
        if ctx.has_p2c:
            grad_p2c_typed = grad_p2c.to(dtype=key.dtype)
            grad_key = grad_key + torch.matmul(grad_p2c_typed, pos_query)
            grad_pos_query = torch.matmul(grad_p2c_typed.transpose(-1, -2), key).sum(dim=0)

        return (
            grad_query,
            grad_key,
            grad_value,
            grad_pos_key,
            grad_pos_query,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    _training_attention_forward_op.register_autograd(
        _training_attention_autograd_backward,
        setup_context=_setup_training_attention_context,
    )

else:
    _training_attention_forward_op = None
    _training_attention_backward_op = None


def training_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    delta_to_local: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    score_scale: float,
    has_padding: bool,
    strict_fp32: bool,
) -> torch.Tensor:
    """Differentiable fused attention entry point.

    ``query``, ``key`` and ``value`` are BHLD views.  ``pos_key`` and
    ``pos_query`` are HRD compact projected relative-position tables.  The
    custom operator saves Q/K/V/O/LSE and the tiny position tables, but not the
    BHLR C2P/P2C intermediates; those are recomputed in backward.
    """

    _require_training_runtime()
    if query.device.type != "cuda":
        raise RuntimeError("training attention requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("training attention supports FP16, BF16, and FP32")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical BHLD shapes")
    if query.size(-1) not in {32, 64, 128}:
        raise ValueError("training attention supports head dimensions 32, 64, and 128")
    if delta_to_local.dtype != torch.int32 or not delta_to_local.is_contiguous():
        raise ValueError("delta_to_local must be a contiguous int32 tensor")
    batch_size, num_heads, sequence_length, head_dim = query.shape
    del num_heads, head_dim
    if delta_to_local.numel() != 2 * sequence_length - 1:
        raise ValueError("delta_to_local must contain 2*L-1 entries")
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask must have shape [B, L]")
    if has_padding and (attention_mask.dtype != torch.bool or not attention_mask.is_contiguous()):
        raise ValueError("padded training requires a contiguous bool attention_mask")

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None
    if not has_c2p and not has_p2c:
        pos_key_tensor = _empty_optional(query)
        pos_query_tensor = _empty_optional(query)
    else:
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2)
        if has_c2p:
            if pos_key.dim() != 3 or pos_key.shape[0] != query.shape[1]:
                raise ValueError("pos_key must have shape [H, R, D]")
            if pos_key.shape[-1] != query.shape[-1]:
                raise ValueError("pos_key head dimension must match query")
            pos_key_tensor = pos_key
        else:
            pos_key_tensor = _empty_optional(query)
        if has_p2c:
            if pos_query.dim() != 3 or pos_query.shape[0] != query.shape[1]:
                raise ValueError("pos_query must have shape [H, R, D]")
            if pos_query.shape[-1] != query.shape[-1]:
                raise ValueError("pos_query head dimension must match query")
            if pos_query.size(-2) != active_slots:
                raise ValueError("pos_key and pos_query must use the same active-slot count")
            pos_query_tensor = pos_query
        else:
            pos_query_tensor = _empty_optional(query)

    output, _lse = _training_attention_forward_op(
        query,
        key,
        value,
        pos_key_tensor,
        pos_query_tensor,
        delta_to_local,
        attention_mask,
        float(score_scale),
        bool(has_padding),
        bool(has_c2p),
        bool(has_p2c),
        bool(strict_fp32),
    )
    return output


__all__ = ["training_attention"]
