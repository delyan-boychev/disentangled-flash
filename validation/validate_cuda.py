"""Hostile numerical validation for prepared DeBERTa attention on CUDA.

This is deliberately separate from the throughput distribution benchmark.  It
reports raw max/mean absolute errors without imposing pass/fail tolerances.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch

from benchmarks.benchmark_cuda import (
    compile_isolated,
    configure_fp32,
    initialize_parameters,
    parse_csv,
    parse_csv_ints,
)
from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash._torch import TorchInferenceDisentangledSelfAttention
from disentangled_flash._validation import require_finite
from disentangled_flash.kernel import TritonInferenceDisentangledSelfAttention

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
POSITION_MODES = {
    "c2p": ("c2p",),
    "p2c": ("p2c",),
    "both": ("c2p", "p2c"),
}
MASK_PATTERNS = ("none", "right", "left", "heavy", "all")


def make_case(
    embedding_table: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    pattern: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_ids = torch.randint(
        1,
        embedding_table.size(0),
        (batch_size, sequence_length),
        generator=generator,
    )
    mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    if pattern in {"right", "left"}:
        low = max(1, sequence_length // 3)
        lengths = torch.randint(
            low,
            sequence_length + 1,
            (batch_size,),
            generator=generator,
        )
        positions = torch.arange(sequence_length)[None, :]
        if pattern == "right":
            mask = positions < lengths[:, None]
        else:
            mask = positions >= sequence_length - lengths[:, None]
    elif pattern == "heavy":
        kept = max(1, sequence_length // 4)
        mask[:, kept:] = False
    elif pattern == "all":
        mask.zero_()
    elif pattern != "none":
        raise ValueError(f"unknown mask pattern: {pattern}")
    token_ids.masked_fill_(~mask, 0)
    token_ids = token_ids.to(embedding_table.device)
    return embedding_table[token_ids].contiguous(), mask.to(embedding_table.device)


def build_pair(
    backend: str,
    dtype: torch.dtype,
    head_dim: int,
    position_mode: str,
    fp32_precision: str,
    seed: int,
) -> tuple[
    OriginalDisentangledSelfAttention,
    torch.nn.Module,
    torch.Tensor,
    torch.Tensor,
]:
    num_heads = 12
    hidden_size = num_heads * head_dim
    config = DebertaAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_size=head_dim,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=2048,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=POSITION_MODES[position_mode],
        norm_rel_ebd="none",
    )
    reference = OriginalDisentangledSelfAttention(config)
    initialize_parameters(reference, seed)
    if backend == "triton":
        target = TritonInferenceDisentangledSelfAttention(
            config,
            fp32_precision=fp32_precision,
        )
    else:
        target = TorchInferenceDisentangledSelfAttention(config)
    target.load_state_dict(reference.state_dict(), strict=True)

    device = torch.device("cuda")
    reference = reference.to(device=device, dtype=dtype).eval()
    target = target.to(device=device, dtype=dtype).eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    relative = (
        torch.empty(512, hidden_size)
        .normal_(
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        .to(device=device, dtype=dtype)
    )
    embedding = torch.empty(8192, hidden_size).normal_(
        mean=0.0,
        std=0.02,
        generator=generator,
    )
    embedding[0].zero_()
    embedding = embedding.to(device=device, dtype=dtype)
    target.prepare_for_inference(relative)
    return reference, target, relative, embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", type=parse_csv, default=["torch", "triton"])
    parser.add_argument("--dtypes", type=parse_csv, default=["fp16", "bf16", "fp32"])
    parser.add_argument("--executions", type=parse_csv, default=["eager"])
    parser.add_argument("--head-dims", type=parse_csv_ints, default=[64])
    parser.add_argument("--batches", type=parse_csv_ints, default=[1, 2])
    parser.add_argument(
        "--lengths",
        type=parse_csv_ints,
        default=[
            1,
            2,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513,
            1024,
            2048,
        ],
    )
    parser.add_argument("--position-modes", type=parse_csv, default=["both"])
    parser.add_argument("--mask-patterns", type=parse_csv, default=list(MASK_PATTERNS))
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--fp32-precision", choices=["strict", "fast"], default="strict")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", default="attention_cuda_validation.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if set(args.backends) - {"torch", "triton"}:
        raise ValueError("backends must contain only torch and/or triton")
    if set(args.dtypes) - set(DTYPES):
        raise ValueError(f"dtypes must be selected from {sorted(DTYPES)}")
    if set(args.executions) - {"eager", "compile"}:
        raise ValueError("executions must contain only eager and/or compile")
    if set(args.position_modes) - set(POSITION_MODES):
        raise ValueError(f"position modes must be selected from {sorted(POSITION_MODES)}")
    if set(args.mask_patterns) - set(MASK_PATTERNS):
        raise ValueError(f"mask patterns must be selected from {list(MASK_PATTERNS)}")
    if set(args.head_dims) - {32, 64, 128}:
        raise ValueError("head dimensions must be selected from 32, 64, and 128")

    configure_fp32(args.fp32_precision)
    if "triton" in args.backends:
        os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
    rows = []
    for dtype_name in args.dtypes:
        for head_dim in args.head_dims:
            for position_mode in args.position_modes:
                for backend in args.backends:
                    reference, target, relative, embedding = build_pair(
                        backend,
                        DTYPES[dtype_name],
                        head_dim,
                        position_mode,
                        args.fp32_precision,
                        args.seed,
                    )
                    with torch.inference_mode():
                        for sequence_length in args.lengths:
                            plan = target.prepare_shape(sequence_length, "cuda")

                            def eager_call(
                                hidden_states: torch.Tensor,
                                attention_mask: torch.Tensor,
                                target=target,
                                plan=plan,
                            ) -> torch.Tensor:
                                return target.forward_prepared(
                                    hidden_states,
                                    attention_mask,
                                    plan,
                                )[0]

                            for execution in args.executions:
                                call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
                                if execution == "compile":
                                    call = compile_isolated(
                                        eager_call,
                                        mode=args.compile_mode,
                                        fullgraph=True,
                                        dynamic=True,
                                    )
                                else:
                                    call = eager_call
                                for batch_size in args.batches:
                                    for pattern_index, pattern in enumerate(args.mask_patterns):
                                        hidden_states, mask = make_case(
                                            embedding,
                                            batch_size,
                                            sequence_length,
                                            pattern,
                                            args.seed
                                            + sequence_length * 10_000
                                            + batch_size * 100
                                            + pattern_index,
                                        )
                                        reference_output = reference(
                                            hidden_states,
                                            _prepare_attention_mask(
                                                mask,
                                                sequence_length,
                                                sequence_length,
                                            ),
                                            rel_embeddings=relative,
                                        )[0]
                                        target_output = call(hidden_states, mask)
                                        case = (
                                            f"backend={backend}, dtype={dtype_name}, "
                                            f"execution={execution}, head_dim={head_dim}, "
                                            f"position_mode={position_mode}, "
                                            f"batch_size={batch_size}, "
                                            f"sequence_length={sequence_length}, "
                                            f"mask_pattern={pattern}"
                                        )
                                        require_finite(reference_output, "reference", case)
                                        require_finite(target_output, "target", case)
                                        difference = (
                                            reference_output.float() - target_output.float()
                                        ).abs()
                                        row = {
                                            "backend": backend,
                                            "dtype": dtype_name,
                                            "execution": execution,
                                            "head_dim": head_dim,
                                            "position_mode": position_mode,
                                            "batch_size": batch_size,
                                            "sequence_length": sequence_length,
                                            "mask_pattern": pattern,
                                            "max_abs_error": difference.max().item(),
                                            "mean_abs_error": difference.mean().item(),
                                        }
                                        rows.append(row)
                                        print(
                                            f"{backend:>9} {dtype_name:>4} {execution:>7} "
                                            f"D={head_dim:>3} B={batch_size:>2} "
                                            f"L={sequence_length:>4} {position_mode:>4} "
                                            f"{pattern:>5} max={row['max_abs_error']:.7g} "
                                            f"mean={row['mean_abs_error']:.7g}",
                                            flush=True,
                                        )
                    del reference, target, relative, embedding
                    torch.cuda.empty_cache()

    summary = []
    groups = sorted(
        {(row["backend"], row["dtype"], row["execution"], row["head_dim"]) for row in rows}
    )
    print("\nMaximum observed errors")
    print(f"{'backend':>9} {'dtype':>5} {'execution':>9} {'D':>4} {'max abs':>14} {'mean abs':>14}")
    for backend, dtype_name, execution, head_dim in groups:
        selected = [
            row
            for row in rows
            if (row["backend"], row["dtype"], row["execution"], row["head_dim"])
            == (backend, dtype_name, execution, head_dim)
        ]
        item = {
            "backend": backend,
            "dtype": dtype_name,
            "execution": execution,
            "head_dim": head_dim,
            "max_abs_error": max(row["max_abs_error"] for row in selected),
            "max_mean_abs_error": max(row["mean_abs_error"] for row in selected),
        }
        summary.append(item)
        print(
            f"{backend:>9} {dtype_name:>5} {execution:>9} {head_dim:>4} "
            f"{item['max_abs_error']:>14.7g} {item['max_mean_abs_error']:>14.7g}"
        )

    output = Path(args.output).resolve()
    output.write_text(
        json.dumps({"configuration": vars(args), "summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved validation results to {output}")


if __name__ == "__main__":
    main()
