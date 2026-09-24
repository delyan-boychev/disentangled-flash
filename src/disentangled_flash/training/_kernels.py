"""Autograd integration and Triton backward kernels for disentangled attention.

The canonical tiled forward lives in :mod:`disentangled_flash.kernel` and is
shared by inference and training.  This module enables its ``STORE_LSE``
specialization when gradients are enabled and owns the recompute-based
backward.  Under ``no_grad()``/``inference_mode()`` it launches the same forward
with ``STORE_LSE=False`` and allocates no LSE tensor.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any

import torch

from ..tuning import (
    DEFAULT_BACKWARD_KERNEL_CONFIGS,
    DEFAULT_KERNEL_CONFIGS,
    KernelConfig,
    tuning_sequence_length,
)

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
    from ..kernel import (
        _deberta_attention_forward_kernel,
        _deberta_attention_packed_forward_kernel,
        _make_autotuned_kernel,
        _make_packed_autotuned_kernel,
    )

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
        PHYSICAL_PAIRS: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        LENGTH_REGIME: tl.constexpr,
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
            if PHYSICAL_PAIRS:
                delta_index = (batch * SEQUENCE_LENGTH + rows[:, None]) * SEQUENCE_LENGTH + cols[
                    None, :
                ]
            else:
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
                score_grad_pair = attended
            else:
                scores = tl.where(pair_in_bounds, scores, -float("inf"))
                active_pair = pair_in_bounds
                score_grad_pair = pair_in_bounds

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
            ds_raw = tl.where(score_grad_pair, ds_raw, 0.0)

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
                    mask=score_grad_pair,
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
        PHYSICAL_PAIRS: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        LENGTH_REGIME: tl.constexpr,
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
            if PHYSICAL_PAIRS:
                delta_index = (batch * SEQUENCE_LENGTH + rows[:, None]) * SEQUENCE_LENGTH + cols[
                    None, :
                ]
            else:
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
                score_grad_pair = attended
            else:
                scores = tl.where(pair_in_bounds, scores, -float("inf"))
                active_pair = pair_in_bounds
                score_grad_pair = pair_in_bounds

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
            ds_raw = tl.where(score_grad_pair, ds_raw, 0.0)

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
                    mask=score_grad_pair,
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

    @triton.jit
    def _packed_backward_preprocess_kernel(
        output,
        grad_output,
        cu_seqlens,
        delta,
        NUM_HEADS,
        TOTAL_TOKENS,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        sequence_head = tl.program_id(1)
        sequence = sequence_head // NUM_HEADS
        head = sequence_head - sequence * NUM_HEADS
        sequence_start = tl.load(cu_seqlens + sequence).to(tl.int64)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        sequence_length = sequence_end - sequence_start
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        tokens = sequence_start + rows
        dims = tl.arange(0, HEAD_DIM)
        row_mask = rows < sequence_length
        offsets = tokens[:, None] * (NUM_HEADS * HEAD_DIM) + head * HEAD_DIM + dims[None, :]
        o = tl.load(output + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(grad_output + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        tl.store(
            delta + head * TOTAL_TOKENS + tokens,
            tl.sum(o * do, axis=1),
            mask=row_mask,
        )

    @triton.jit
    def _packed_backward_dq_dc_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        cu_seqlens,
        grad_output,
        lse_log2,
        delta,
        grad_query,
        grad_c2p,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_dqh,
        stride_dql,
        stride_dqd,
        ACTIVE_SLOTS,
        MAX_SEQLEN,
        NUM_HEADS,
        POSITION_OFFSET,
        TOTAL_TOKENS,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        LENGTH_REGIME: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        sequence_head = tl.program_id(1)
        sequence = sequence_head // NUM_HEADS
        head = sequence_head - sequence * NUM_HEADS
        sequence_start = tl.load(cu_seqlens + sequence).to(tl.int64)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        sequence_length = sequence_end - sequence_start
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        query_tokens = sequence_start + rows
        key_lane = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, HEAD_DIM)
        row_mask = rows < sequence_length

        q_base = query + head * stride_qh
        k_base = key + head * stride_kh
        v_base = value + head * stride_vh
        dq_base = grad_query + head * stride_dqh
        q = tl.load(
            q_base + query_tokens[:, None] * stride_ql + dims[None, :] * stride_qd,
            mask=row_mask[:, None],
            other=0.0,
        )
        output_offsets = (
            query_tokens[:, None] * (NUM_HEADS * HEAD_DIM)
            + head * HEAD_DIM
            + dims[None, :]
        )
        do = tl.load(grad_output + output_offsets, mask=row_mask[:, None], other=0.0)
        lse = tl.load(
            lse_log2 + head * TOTAL_TOKENS + query_tokens,
            mask=row_mask,
            other=0.0,
        )
        row_delta = tl.load(
            delta + head * TOTAL_TOKENS + query_tokens,
            mask=row_mask,
            other=0.0,
        )
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        if HAS_C2P:
            c2p_base = c2p + head * TOTAL_TOKENS * ACTIVE_SLOTS
            dc_base = grad_c2p + head * TOTAL_TOKENS * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + head * TOTAL_TOKENS * ACTIVE_SLOTS

        for key_start in tl.range(0, sequence_length, BLOCK_N):
            key_start = tl.multiple_of(key_start, BLOCK_N)
            cols = key_start + key_lane
            key_tokens = sequence_start + cols
            col_mask = cols < sequence_length
            k = tl.load(
                k_base + key_tokens[:, None] * stride_kl + dims[None, :] * stride_kd,
                mask=col_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + key_tokens[:, None] * stride_vl + dims[None, :] * stride_vd,
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
            pair_mask = row_mask[:, None] & col_mask[None, :]
            local_slot = tl.load(
                delta_to_local_slot + rows[:, None] - cols[None, :] + POSITION_OFFSET,
                mask=pair_mask,
                other=0,
            ).to(tl.int32)
            if HAS_C2P:
                scores += tl.load(
                    c2p_base + query_tokens[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_mask,
                    other=0.0,
                )
            if HAS_P2C:
                scores += tl.load(
                    p2c_base + key_tokens[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_mask,
                    other=0.0,
                )
            scores = tl.where(pair_mask, scores * SCORE_SCALE_LOG2, -float("inf"))
            p = tl.where(pair_mask, tl.math.exp2(scores - lse[:, None]), 0.0)
            if IS_FP32:
                if STRICT_FP32:
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                else:
                    dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            else:
                dp = tl.dot(do, tl.trans(v))
            ds = tl.where(
                pair_mask,
                p * (dp.to(tl.float32) - row_delta[:, None]) * SCORE_SCALE,
                0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    dq += tl.dot(ds, k, input_precision="ieee")
                else:
                    dq += tl.dot(ds, k, input_precision="tf32")
            elif IS_BF16:
                dq += tl.dot(ds.to(tl.bfloat16), k)
            else:
                dq += tl.dot(ds.to(tl.float16), k)
            if HAS_C2P:
                tl.atomic_add(
                    dc_base + query_tokens[:, None] * ACTIVE_SLOTS + local_slot,
                    ds,
                    mask=pair_mask,
                )

        tl.store(
            dq_base + query_tokens[:, None] * stride_dql + dims[None, :] * stride_dqd,
            dq,
            mask=row_mask[:, None],
        )

    @triton.jit
    def _packed_backward_dkv_dt_kernel(
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local_slot,
        cu_seqlens,
        grad_output,
        lse_log2,
        delta,
        grad_key,
        grad_value,
        grad_p2c,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_dkh,
        stride_dkl,
        stride_dkd,
        stride_dvh,
        stride_dvl,
        stride_dvd,
        ACTIVE_SLOTS,
        MAX_SEQLEN,
        NUM_HEADS,
        POSITION_OFFSET,
        TOTAL_TOKENS,
        HEAD_DIM: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        SCORE_SCALE_LOG2: tl.constexpr,
        HAS_C2P: tl.constexpr,
        HAS_P2C: tl.constexpr,
        IS_BF16: tl.constexpr,
        IS_FP32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        LENGTH_REGIME: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        sequence_head = tl.program_id(1)
        sequence = sequence_head // NUM_HEADS
        head = sequence_head - sequence * NUM_HEADS
        sequence_start = tl.load(cu_seqlens + sequence).to(tl.int64)
        sequence_end = tl.load(cu_seqlens + sequence + 1).to(tl.int64)
        sequence_length = sequence_end - sequence_start
        cols = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        key_tokens = sequence_start + cols
        query_lane = tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        col_mask = cols < sequence_length

        q_base = query + head * stride_qh
        k_base = key + head * stride_kh
        v_base = value + head * stride_vh
        dk_base = grad_key + head * stride_dkh
        dv_base = grad_value + head * stride_dvh
        k = tl.load(
            k_base + key_tokens[:, None] * stride_kl + dims[None, :] * stride_kd,
            mask=col_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + key_tokens[:, None] * stride_vl + dims[None, :] * stride_vd,
            mask=col_mask[:, None],
            other=0.0,
        )
        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        if HAS_C2P:
            c2p_base = c2p + head * TOTAL_TOKENS * ACTIVE_SLOTS
        if HAS_P2C:
            p2c_base = p2c + head * TOTAL_TOKENS * ACTIVE_SLOTS
            dt_base = grad_p2c + head * TOTAL_TOKENS * ACTIVE_SLOTS

        for query_start in tl.range(0, sequence_length, BLOCK_M):
            query_start = tl.multiple_of(query_start, BLOCK_M)
            rows = query_start + query_lane
            query_tokens = sequence_start + rows
            row_mask = rows < sequence_length
            q = tl.load(
                q_base + query_tokens[:, None] * stride_ql + dims[None, :] * stride_qd,
                mask=row_mask[:, None],
                other=0.0,
            )
            output_offsets = (
                query_tokens[:, None] * (NUM_HEADS * HEAD_DIM)
                + head * HEAD_DIM
                + dims[None, :]
            )
            do = tl.load(grad_output + output_offsets, mask=row_mask[:, None], other=0.0)
            lse = tl.load(
                lse_log2 + head * TOTAL_TOKENS + query_tokens,
                mask=row_mask,
                other=0.0,
            )
            row_delta = tl.load(
                delta + head * TOTAL_TOKENS + query_tokens,
                mask=row_mask,
                other=0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    scores = tl.dot(q, tl.trans(k), input_precision="ieee")
                else:
                    scores = tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores = tl.dot(q, tl.trans(k))
            pair_mask = row_mask[:, None] & col_mask[None, :]
            local_slot = tl.load(
                delta_to_local_slot + rows[:, None] - cols[None, :] + POSITION_OFFSET,
                mask=pair_mask,
                other=0,
            ).to(tl.int32)
            if HAS_C2P:
                scores += tl.load(
                    c2p_base + query_tokens[:, None] * ACTIVE_SLOTS + local_slot,
                    mask=pair_mask,
                    other=0.0,
                )
            if HAS_P2C:
                scores += tl.load(
                    p2c_base + key_tokens[None, :] * ACTIVE_SLOTS + local_slot,
                    mask=pair_mask,
                    other=0.0,
                )
            scores = tl.where(pair_mask, scores * SCORE_SCALE_LOG2, -float("inf"))
            p = tl.where(pair_mask, tl.math.exp2(scores - lse[:, None]), 0.0)
            if IS_FP32:
                if STRICT_FP32:
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                else:
                    dp = tl.dot(do, tl.trans(v), input_precision="tf32")
            else:
                dp = tl.dot(do, tl.trans(v))
            ds = tl.where(
                pair_mask,
                p * (dp.to(tl.float32) - row_delta[:, None]) * SCORE_SCALE,
                0.0,
            )
            if IS_FP32:
                if STRICT_FP32:
                    dk += tl.dot(tl.trans(ds), q, input_precision="ieee")
                    dv += tl.dot(tl.trans(p), do, input_precision="ieee")
                else:
                    dk += tl.dot(tl.trans(ds), q, input_precision="tf32")
                    dv += tl.dot(tl.trans(p), do, input_precision="tf32")
            elif IS_BF16:
                dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
                dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            else:
                dk += tl.dot(tl.trans(ds.to(tl.float16)), q)
                dv += tl.dot(tl.trans(p.to(tl.float16)), do)
            if HAS_P2C:
                tl.atomic_add(
                    dt_base + key_tokens[None, :] * ACTIVE_SLOTS + local_slot,
                    ds,
                    mask=pair_mask,
                )

        tl.store(
            dk_base + key_tokens[:, None] * stride_dkl + dims[None, :] * stride_dkd,
            dk,
            mask=col_mask[:, None],
        )
        tl.store(
            dv_base + key_tokens[:, None] * stride_dvl + dims[None, :] * stride_dvd,
            dv,
            mask=col_mask[:, None],
        )


def _backward_tile_config(sequence_length: int, head_dim: int) -> KernelConfig:
    """Resource-safe fallback used only when autotuning is unavailable."""

    if sequence_length <= 32:
        return KernelConfig(32, 32, 4)
    if sequence_length <= 64:
        return KernelConfig(32, 64, 4)
    if head_dim <= 64:
        return KernelConfig(64, 64, 4)
    return KernelConfig(32, 64, 4)


if triton is not None:
    _BACKWARD_AUTOTUNE_KEY = [
        "LENGTH_REGIME",
        "HEAD_DIM",
        "HAS_C2P",
        "HAS_P2C",
        "HAS_PADDING",
        "PHYSICAL_PAIRS",
        "IS_BF16",
        "IS_FP32",
        "STRICT_FP32",
    ]

    def _prune_backward_configs(
        configs: list[Any],
        named_args: dict[str, Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Prune schedules that predictably spill for the current head size."""

        head_dim = int(kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM", 64)))
        length = int(kwargs.get("LENGTH_REGIME", named_args.get("LENGTH_REGIME", 64)))
        is_fp32 = bool(kwargs.get("IS_FP32", named_args.get("IS_FP32", False)))
        kept = []
        for config in configs:
            block_m = config.kwargs["BLOCK_M"]
            block_n = config.kwargs["BLOCK_N"]
            if length <= 32 and max(block_m, block_n) > 32:
                continue
            if (is_fp32 or head_dim == 128) and block_m * block_n > 2048:
                continue
            if head_dim == 128 and config.num_warps < 4:
                continue
            kept.append(config)
        return kept or configs[:1]

    def _zero_atomic_output(name: str):
        def pre_hook(arguments: dict[str, Any]) -> None:
            arguments[name].zero_()

        return pre_hook

    def _make_backward_autotuned_kernel(
        kernel: Any,
        atomic_output: str,
        *,
        packed: bool = False,
        configs: tuple[KernelConfig, ...] = DEFAULT_BACKWARD_KERNEL_CONFIGS,
    ) -> Any:
        triton_configs = [
            triton.Config(
                {"BLOCK_M": config.block_m, "BLOCK_N": config.block_n},
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                pre_hook=_zero_atomic_output(atomic_output),
            )
            for config in configs
        ]
        autotune_kwargs: dict[str, Any] = {
            "configs": triton_configs,
            "key": [
                name
                for name in _BACKWARD_AUTOTUNE_KEY
                if not packed or name not in {"HAS_PADDING", "PHYSICAL_PAIRS"}
            ],
            "prune_configs_by": {"early_config_prune": _prune_backward_configs},
        }
        if "cache_results" in inspect.signature(triton.autotune).parameters:
            autotune_kwargs["cache_results"] = True
        return triton.autotune(**autotune_kwargs)(kernel)

    @cache
    def _make_training_autotune_bundle(
        configs: tuple[KernelConfig, ...],
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        return (
            _make_autotuned_kernel(configs),
            _make_packed_autotuned_kernel(configs),
            _make_backward_autotuned_kernel(
                _backward_dq_dc_kernel,
                "grad_c2p",
                configs=configs,
            ),
            _make_backward_autotuned_kernel(
                _backward_dkv_dt_kernel,
                "grad_p2c",
                configs=configs,
            ),
            _make_backward_autotuned_kernel(
                _packed_backward_dq_dc_kernel,
                "grad_c2p",
                packed=True,
                configs=configs,
            ),
            _make_backward_autotuned_kernel(
                _packed_backward_dkv_dt_kernel,
                "grad_p2c",
                packed=True,
                configs=configs,
            ),
        )

    _DEFAULT_TRAINING_CONFIGS = tuple(
        dict.fromkeys(DEFAULT_KERNEL_CONFIGS + DEFAULT_BACKWARD_KERNEL_CONFIGS)
    )
    _training_autotune_bundles = [_make_training_autotune_bundle(_DEFAULT_TRAINING_CONFIGS)]
    _training_autotune_bundle_ids = {_DEFAULT_TRAINING_CONFIGS: 0}

    def _register_training_autotune_candidates(configs: tuple[KernelConfig, ...]) -> int:
        existing = _training_autotune_bundle_ids.get(configs)
        if existing is not None:
            return existing
        identifier = len(_training_autotune_bundles)
        _training_autotune_bundles.append(_make_training_autotune_bundle(configs))
        _training_autotune_bundle_ids[configs] = identifier
        return identifier


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
        store_lse: bool,
        autotune_bundle_id: int,
        forward_block_m: int,
        forward_block_n: int,
        forward_num_warps: int,
        forward_num_stages: int,
        dq_block_m: int,
        dq_block_n: int,
        dq_num_warps: int,
        dq_num_stages: int,
        dkv_block_m: int,
        dkv_block_n: int,
        dkv_num_warps: int,
        dkv_num_stages: int,
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
        lse_log2 = (
            torch.empty(
                (batch_size, num_heads, sequence_length),
                device=query.device,
                dtype=torch.float32,
            )
            if store_lse
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )
        lse_storage = lse_log2 if store_lse else output

        def grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(sequence_length, meta["BLOCK_M"]), batch_size * num_heads

        forward_kernel = _deberta_attention_forward_kernel
        if not forward_block_m:
            forward_kernel = _training_autotune_bundles[autotune_bundle_id][0]
        launch_options: dict[str, Any] = {}
        if forward_block_m:
            launch_options.update(
                BLOCK_M=forward_block_m,
                BLOCK_N=forward_block_n,
                num_warps=forward_num_warps,
                num_stages=forward_num_stages,
            )
        torch.library.wrap_triton(forward_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            attention_mask,
            output,
            lse_storage,
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
            POSITION_OFFSET=sequence_length - 1,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            LENGTH_REGIME=tuning_sequence_length(sequence_length),
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            USE_PADDING_MASK=has_padding,
            PHYSICAL_PAIRS=delta_to_local.ndim == 3,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            STORE_LSE=store_lse,
            **launch_options,
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
        autotune_bundle_id: int,
        dq_block_m: int,
        dq_block_n: int,
        dq_num_warps: int,
        dq_num_stages: int,
        dkv_block_m: int,
        dkv_block_n: int,
        dkv_num_warps: int,
        dkv_num_stages: int,
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

        fallback = _backward_tile_config(sequence_length, head_dim)
        preprocess_block = dq_block_m or fallback.block_m
        preprocess_warps = dq_num_warps or fallback.num_warps
        preprocess_stages = dq_num_stages or fallback.num_stages
        preprocess_grid = (
            triton.cdiv(sequence_length, preprocess_block),
            batch_size * num_heads,
        )
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
            BLOCK_M=preprocess_block,
            num_warps=preprocess_warps,
            num_stages=preprocess_stages,
        )

        dq_kernel = _backward_dq_dc_kernel
        if not dq_block_m:
            dq_kernel = _training_autotune_bundles[autotune_bundle_id][2]

        def dq_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(sequence_length, meta["BLOCK_M"]), batch_size * num_heads

        dq_options: dict[str, Any] = {}
        if dq_block_m:
            dq_options.update(
                BLOCK_M=dq_block_m,
                BLOCK_N=dq_block_n,
                num_warps=dq_num_warps,
                num_stages=dq_num_stages,
            )
        torch.library.wrap_triton(dq_kernel)[dq_grid](
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
            PHYSICAL_PAIRS=delta_to_local.ndim == 3,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            LENGTH_REGIME=tuning_sequence_length(sequence_length),
            **dq_options,
        )

        dkv_kernel = _backward_dkv_dt_kernel
        if not dkv_block_m:
            dkv_kernel = _training_autotune_bundles[autotune_bundle_id][3]

        def dkv_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(sequence_length, meta["BLOCK_N"]), batch_size * num_heads

        dkv_options: dict[str, Any] = {}
        if dkv_block_m:
            dkv_options.update(
                BLOCK_M=dkv_block_m,
                BLOCK_N=dkv_block_n,
                num_warps=dkv_num_warps,
                num_stages=dkv_num_stages,
            )
        torch.library.wrap_triton(dkv_kernel)[dkv_grid](
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
            PHYSICAL_PAIRS=delta_to_local.ndim == 3,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            LENGTH_REGIME=tuning_sequence_length(sequence_length),
            **dkv_options,
        )
        return grad_query, grad_key, grad_value, grad_c2p, grad_p2c

    @torch.library.triton_op(
        "disentangled_flash::training_attention_packed_forward",
        mutates_args={},
    )
    def _training_attention_packed_forward_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        delta_to_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        score_scale: float,
        has_c2p: bool,
        has_p2c: bool,
        strict_fp32: bool,
        store_lse: bool,
        autotune_bundle_id: int,
        forward_block_m: int,
        forward_block_n: int,
        forward_num_warps: int,
        forward_num_stages: int,
        dq_block_m: int,
        dq_block_n: int,
        dq_num_warps: int,
        dq_num_stages: int,
        dkv_block_m: int,
        dkv_block_n: int,
        dkv_num_warps: int,
        dkv_num_stages: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_heads, total_tokens, head_dim = query.shape
        batch_size = cu_seqlens.numel() - 1
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2) if has_p2c else 0
        c2p = torch.matmul(query, pos_key.transpose(-1, -2)) if has_c2p else _empty_optional(query)
        p2c = torch.matmul(key, pos_query.transpose(-1, -2)) if has_p2c else _empty_optional(key)
        output = torch.empty(
            (total_tokens, num_heads * head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        lse_log2 = (
            torch.empty((num_heads, total_tokens), device=query.device, dtype=torch.float32)
            if store_lse
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )
        lse_storage = lse_log2 if store_lse else output

        def grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(max_seqlen, meta["BLOCK_M"]), batch_size * num_heads

        forward_kernel = _deberta_attention_packed_forward_kernel
        if not forward_block_m:
            forward_kernel = _training_autotune_bundles[autotune_bundle_id][1]
        launch_options: dict[str, Any] = {}
        if forward_block_m:
            launch_options.update(
                BLOCK_M=forward_block_m,
                BLOCK_N=forward_block_n,
                num_warps=forward_num_warps,
                num_stages=forward_num_stages,
            )
        torch.library.wrap_triton(forward_kernel)[grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            output,
            lse_storage,
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
            POSITION_OFFSET=max_seqlen - 1,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            LENGTH_REGIME=tuning_sequence_length(max_seqlen),
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            HAS_PADDING=False,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            STORE_LSE=store_lse,
            **launch_options,
        )
        return output, lse_log2

    @torch.library.triton_op(
        "disentangled_flash::training_attention_packed_backward",
        mutates_args={},
    )
    def _training_attention_packed_backward_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_key: torch.Tensor,
        pos_query: torch.Tensor,
        delta_to_local: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
        lse_log2: torch.Tensor,
        grad_output: torch.Tensor,
        max_seqlen: int,
        score_scale: float,
        has_c2p: bool,
        has_p2c: bool,
        strict_fp32: bool,
        autotune_bundle_id: int,
        dq_block_m: int,
        dq_block_n: int,
        dq_num_warps: int,
        dq_num_stages: int,
        dkv_block_m: int,
        dkv_block_n: int,
        dkv_num_warps: int,
        dkv_num_stages: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads, total_tokens, head_dim = query.shape
        batch_size = cu_seqlens.numel() - 1
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2) if has_p2c else 0
        c2p = torch.matmul(query, pos_key.transpose(-1, -2)) if has_c2p else _empty_optional(query)
        p2c = torch.matmul(key, pos_query.transpose(-1, -2)) if has_p2c else _empty_optional(key)
        delta = torch.empty((num_heads, total_tokens), device=query.device, dtype=torch.float32)
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_c2p = (
            torch.zeros(
                (num_heads, total_tokens, active_slots),
                device=query.device,
                dtype=torch.float32,
            )
            if has_c2p
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )
        grad_p2c = (
            torch.zeros(
                (num_heads, total_tokens, active_slots),
                device=query.device,
                dtype=torch.float32,
            )
            if has_p2c
            else torch.empty((0,), device=query.device, dtype=torch.float32)
        )
        fallback = _backward_tile_config(max_seqlen, head_dim)
        preprocess_block = dq_block_m or fallback.block_m
        preprocess_grid = (
            triton.cdiv(max_seqlen, preprocess_block),
            batch_size * num_heads,
        )
        torch.library.wrap_triton(_packed_backward_preprocess_kernel)[preprocess_grid](
            output,
            grad_output,
            cu_seqlens,
            delta,
            NUM_HEADS=num_heads,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            BLOCK_M=preprocess_block,
            num_warps=dq_num_warps or fallback.num_warps,
            num_stages=dq_num_stages or fallback.num_stages,
        )

        dq_kernel = _packed_backward_dq_dc_kernel
        if not dq_block_m:
            dq_kernel = _training_autotune_bundles[autotune_bundle_id][4]

        def dq_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(max_seqlen, meta["BLOCK_M"]), batch_size * num_heads

        dq_options: dict[str, Any] = {}
        if dq_block_m:
            dq_options.update(
                BLOCK_M=dq_block_m,
                BLOCK_N=dq_block_n,
                num_warps=dq_num_warps,
                num_stages=dq_num_stages,
            )
        torch.library.wrap_triton(dq_kernel)[dq_grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            grad_output,
            lse_log2,
            delta,
            grad_query,
            grad_c2p,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            grad_query.stride(0),
            grad_query.stride(1),
            grad_query.stride(2),
            ACTIVE_SLOTS=active_slots,
            MAX_SEQLEN=max_seqlen,
            NUM_HEADS=num_heads,
            POSITION_OFFSET=max_seqlen - 1,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            SCORE_SCALE=score_scale,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            LENGTH_REGIME=tuning_sequence_length(max_seqlen),
            **dq_options,
        )

        dkv_kernel = _packed_backward_dkv_dt_kernel
        if not dkv_block_m:
            dkv_kernel = _training_autotune_bundles[autotune_bundle_id][5]

        def dkv_grid(meta: dict[str, Any]) -> tuple[int, int]:
            return triton.cdiv(max_seqlen, meta["BLOCK_N"]), batch_size * num_heads

        dkv_options: dict[str, Any] = {}
        if dkv_block_m:
            dkv_options.update(
                BLOCK_M=dkv_block_m,
                BLOCK_N=dkv_block_n,
                num_warps=dkv_num_warps,
                num_stages=dkv_num_stages,
            )
        torch.library.wrap_triton(dkv_kernel)[dkv_grid](
            query,
            key,
            value,
            c2p,
            p2c,
            delta_to_local,
            cu_seqlens,
            grad_output,
            lse_log2,
            delta,
            grad_key,
            grad_value,
            grad_p2c,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            grad_key.stride(0),
            grad_key.stride(1),
            grad_key.stride(2),
            grad_value.stride(0),
            grad_value.stride(1),
            grad_value.stride(2),
            ACTIVE_SLOTS=active_slots,
            MAX_SEQLEN=max_seqlen,
            NUM_HEADS=num_heads,
            POSITION_OFFSET=max_seqlen - 1,
            TOTAL_TOKENS=total_tokens,
            HEAD_DIM=head_dim,
            SCORE_SCALE=score_scale,
            SCORE_SCALE_LOG2=score_scale * _LOG2E,
            HAS_C2P=has_c2p,
            HAS_P2C=has_p2c,
            IS_BF16=query.dtype == torch.bfloat16,
            IS_FP32=query.dtype == torch.float32,
            STRICT_FP32=strict_fp32,
            LENGTH_REGIME=tuning_sequence_length(max_seqlen),
            **dkv_options,
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
            store_lse,
            autotune_bundle_id,
            forward_block_m,
            forward_block_n,
            forward_num_warps,
            forward_num_stages,
            dq_block_m,
            dq_block_n,
            dq_num_warps,
            dq_num_stages,
            dkv_block_m,
            dkv_block_n,
            dkv_num_warps,
            dkv_num_stages,
        ) = inputs
        del forward_block_m, forward_block_n, forward_num_warps, forward_num_stages
        if not store_lse:
            raise RuntimeError("differentiable training forward requires STORE_LSE=True")
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
        ctx.autotune_bundle_id = autotune_bundle_id
        ctx.dq_config = (dq_block_m, dq_block_n, dq_num_warps, dq_num_stages)
        ctx.dkv_config = (dkv_block_m, dkv_block_n, dkv_num_warps, dkv_num_stages)

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
            ctx.autotune_bundle_id,
            *ctx.dq_config,
            *ctx.dkv_config,
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
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

    def _setup_training_attention_packed_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        (
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            cu_seqlens,
            max_seqlen,
            score_scale,
            has_c2p,
            has_p2c,
            strict_fp32,
            store_lse,
            autotune_bundle_id,
            forward_block_m,
            forward_block_n,
            forward_num_warps,
            forward_num_stages,
            dq_block_m,
            dq_block_n,
            dq_num_warps,
            dq_num_stages,
            dkv_block_m,
            dkv_block_n,
            dkv_num_warps,
            dkv_num_stages,
        ) = inputs
        del forward_block_m, forward_block_n, forward_num_warps, forward_num_stages
        if not store_lse:
            raise RuntimeError("differentiable packed training forward requires STORE_LSE=True")
        attention_output, lse_log2 = output
        ctx.save_for_backward(
            query,
            key,
            value,
            pos_key,
            pos_query,
            delta_to_local,
            cu_seqlens,
            attention_output,
            lse_log2,
        )
        ctx.max_seqlen = max_seqlen
        ctx.score_scale = score_scale
        ctx.has_c2p = has_c2p
        ctx.has_p2c = has_p2c
        ctx.strict_fp32 = strict_fp32
        ctx.autotune_bundle_id = autotune_bundle_id
        ctx.dq_config = (dq_block_m, dq_block_n, dq_num_warps, dq_num_stages)
        ctx.dkv_config = (dkv_block_m, dkv_block_n, dkv_num_warps, dkv_num_stages)

    def _training_attention_packed_autograd_backward(
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
            cu_seqlens,
            output,
            lse_log2,
        ) = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_c2p, grad_p2c = (
            _training_attention_packed_backward_op(
                query,
                key,
                value,
                pos_key,
                pos_query,
                delta_to_local,
                cu_seqlens,
                output,
                lse_log2,
                grad_output.contiguous(),
                ctx.max_seqlen,
                ctx.score_scale,
                ctx.has_c2p,
                ctx.has_p2c,
                ctx.strict_fp32,
                ctx.autotune_bundle_id,
                *ctx.dq_config,
                *ctx.dkv_config,
            )
        )
        grad_pos_key = None
        if ctx.has_c2p:
            grad_c2p_typed = grad_c2p.to(dtype=query.dtype)
            grad_query = grad_query + torch.matmul(grad_c2p_typed, pos_key)
            grad_pos_key = torch.matmul(grad_c2p_typed.transpose(-1, -2), query)
        grad_pos_query = None
        if ctx.has_p2c:
            grad_p2c_typed = grad_p2c.to(dtype=key.dtype)
            grad_key = grad_key + torch.matmul(grad_p2c_typed, pos_query)
            grad_pos_query = torch.matmul(grad_p2c_typed.transpose(-1, -2), key)
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    _training_attention_packed_forward_op.register_autograd(
        _training_attention_packed_autograd_backward,
        setup_context=_setup_training_attention_packed_context,
    )

else:
    _training_attention_forward_op = None
    _training_attention_backward_op = None
    _training_attention_packed_forward_op = None
    _training_attention_packed_backward_op = None


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
    forward_config: KernelConfig | None = None,
    dq_config: KernelConfig | None = None,
    dkv_config: KernelConfig | None = None,
    autotune_candidates: tuple[KernelConfig, ...] | None = None,
) -> torch.Tensor:
    """Differentiable fused attention entry point.

    ``query``, ``key`` and ``value`` are BHLD views.  ``pos_key`` and
    ``pos_query`` are HRD compact projected relative-position tables.  The
    grad-enabled custom operator saves Q/K/V/O/LSE and the tiny position tables,
    but not the BHLR C2P/P2C intermediates; those are recomputed in backward.
    Under ``no_grad()`` the shared forward runs with ``STORE_LSE=False``.
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
    if delta_to_local.device != query.device:
        raise ValueError("position lookup must be on the query device")
    if delta_to_local.ndim == 3:
        if delta_to_local.shape != (batch_size, sequence_length, sequence_length):
            raise ValueError("physical pair lookup must have shape [B, L, L]")
    elif delta_to_local.ndim != 1 or delta_to_local.numel() != 2 * sequence_length - 1:
        raise ValueError("delta_to_local must contain 2*L-1 entries or a [B, L, L] map")
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

    store_lse = torch.is_grad_enabled()
    def config_values(config: KernelConfig | None) -> tuple[int, int, int, int]:
        if config is None:
            return 0, 0, 0, 0
        return config.block_m, config.block_n, config.num_warps, config.num_stages

    candidates = autotune_candidates or _DEFAULT_TRAINING_CONFIGS
    autotune_bundle_id = _register_training_autotune_candidates(tuple(candidates))
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
        bool(store_lse),
        autotune_bundle_id,
        *config_values(forward_config),
        *config_values(dq_config),
        *config_values(dkv_config),
    )
    return output


def training_attention_packed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    delta_to_local: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_seqlen: int,
    score_scale: float,
    strict_fp32: bool,
    forward_config: KernelConfig | None = None,
    dq_config: KernelConfig | None = None,
    dkv_config: KernelConfig | None = None,
    autotune_candidates: tuple[KernelConfig, ...] | None = None,
) -> torch.Tensor:
    """Differentiable packed attention over FlashAttention-style boundaries."""

    _require_training_runtime()
    if query.device.type != "cuda":
        raise RuntimeError("packed training attention requires CUDA tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("packed training attention supports FP16, BF16, and FP32")
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("packed query, key, and value must have identical [H, T, D] shapes")
    if query.size(-1) not in {32, 64, 128}:
        raise ValueError("packed training attention supports head dimensions 32, 64, and 128")
    if cu_seqlens.device != query.device:
        raise ValueError("cu_seqlens must be on the query device")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous int32 vector")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    if delta_to_local.device != query.device:
        raise ValueError("position lookup must be on the query device")
    if (
        delta_to_local.dtype != torch.int32
        or delta_to_local.ndim != 1
        or delta_to_local.numel() != 2 * max_seqlen - 1
        or not delta_to_local.is_contiguous()
    ):
        raise ValueError("delta_to_local must be a contiguous int32 vector with 2*max_seqlen-1 entries")

    has_c2p = pos_key is not None
    has_p2c = pos_query is not None
    if not has_c2p and not has_p2c:
        pos_key_tensor = _empty_optional(query)
        pos_query_tensor = _empty_optional(query)
    else:
        active_slots = pos_key.size(-2) if has_c2p else pos_query.size(-2)
        if has_c2p:
            if pos_key.shape != (query.size(0), active_slots, query.size(-1)):
                raise ValueError("pos_key must have shape [H, R, D]")
            pos_key_tensor = pos_key
        else:
            pos_key_tensor = _empty_optional(query)
        if has_p2c:
            if pos_query.shape != (query.size(0), active_slots, query.size(-1)):
                raise ValueError("pos_query must have shape [H, R, D]")
            pos_query_tensor = pos_query
        else:
            pos_query_tensor = _empty_optional(query)

    def config_values(config: KernelConfig | None) -> tuple[int, int, int, int]:
        if config is None:
            return 0, 0, 0, 0
        return config.block_m, config.block_n, config.num_warps, config.num_stages

    candidates = autotune_candidates or _DEFAULT_TRAINING_CONFIGS
    autotune_bundle_id = _register_training_autotune_candidates(tuple(candidates))
    output, _lse = _training_attention_packed_forward_op(
        query,
        key,
        value,
        pos_key_tensor,
        pos_query_tensor,
        delta_to_local,
        cu_seqlens,
        max_seqlen,
        float(score_scale),
        bool(has_c2p),
        bool(has_p2c),
        bool(strict_fp32),
        bool(torch.is_grad_enabled()),
        autotune_bundle_id,
        *config_values(forward_config),
        *config_values(dq_config),
        *config_values(dkv_config),
    )
    return output


__all__ = ["training_attention", "training_attention_packed"]
