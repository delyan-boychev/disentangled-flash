"""Deferred process-isolated DeBERTa-v3-base end-to-end training benchmark.

The default matrix keeps the number of tokens per step approximately constant
as sequence length changes. A fixed batch can still be requested for capacity
studies and for reproducing older benchmark runs. This is not part of the
release-facing kernel benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

RESULT_PREFIX = "__TRAIN_RESULT__="
VARIANTS = ("deberta_hf", "deberta_flash", "deberta_df")
DEFAULT_LENGTHS = (128, 512, 2048, 8192)
DEFAULT_TOTAL_TOKENS = 16_384


def batch_size_for_length(length: int, total_tokens: int, fixed_batch_size: int | None) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if fixed_batch_size is not None:
        if fixed_batch_size <= 0:
            raise ValueError("batch size must be positive")
        return fixed_batch_size
    if total_tokens <= 0:
        raise ValueError("total tokens must be positive")
    return max(1, total_tokens // length)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def make_config(max_length: int, dropout: float):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=128_100,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max(8192, max_length),
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=512,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        conv_kernel_size=0,
    )


def build_model(
    variant: str,
    length: int,
    dtype: torch.dtype,
    device: str,
    *,
    dropout: float = 0.1,
    tuning_mode: str = "auto",
    profile_paths: tuple[str, ...] = (),
):
    from transformers import DebertaV2Model

    config = make_config(length, dropout)
    if variant == "deberta_flash":
        try:
            from flashdeberta import FlashDebertaV2Model
        except ImportError as exc:
            raise RuntimeError("install FlashDeBERTa with: pip install flashdeberta -U") from exc
        model = FlashDebertaV2Model(config)
    else:
        model = DebertaV2Model(config)

    model.to(device=device, dtype=dtype).train()
    if variant == "deberta_df":
        from disentangled_flash import KernelTuningOptions, enable_deberta_training

        enable_deberta_training(
            model,
            assume_unpadded=True,
            fp32_precision="strict",
            tuning=KernelTuningOptions(
                mode=tuning_mode,
                profile_paths=profile_paths,
            ),
        )
    return model


def train_step(model, input_ids, attention_mask):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state
    loss = output.float().square().mean()
    loss.backward()
    return loss


def worker(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    device_name = f"cuda:{device_index}"
    dtype = dtype_from_name(args.dtype)

    gc.collect()
    torch.cuda.empty_cache()
    model = build_model(
        args.variant,
        args.length,
        dtype,
        device_name,
        dropout=args.dropout,
        tuning_mode=args.tuning_mode,
        profile_paths=tuple(args.profile),
    )
    if args.execution == "compile":
        model = torch.compile(model, mode=args.compile_mode, fullgraph=args.fullgraph)
    input_ids = torch.randint(
        10,
        30_000,
        (args.batch_size, args.length),
        device=device_name,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    for _ in range(args.warmup):
        train_step(model, input_ids, attention_mask)
    torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    losses = []
    for index in range(args.iters):
        starts[index].record()
        loss = train_step(model, input_ids, attention_mask)
        ends[index].record()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()

    timings = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    median_ms = statistics.median(timings)
    peak = torch.cuda.max_memory_allocated()
    return {
        "variant": args.variant,
        "length": args.length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "dropout": args.dropout,
        "execution": args.execution,
        "tuning_mode": args.tuning_mode,
        "profiles": list(args.profile),
        "status": "ok",
        "median_fwd_bwd_ms": median_ms,
        "mean_fwd_bwd_ms": statistics.fmean(timings),
        "tokens_per_s": args.batch_size * args.length * 1000.0 / median_ms,
        "baseline_gib": baseline / 1024**3,
        "peak_gib": peak / 1024**3,
        "incremental_peak_gib": max(0, peak - baseline) / 1024**3,
        "loss": losses[-1],
        "gpu": torch.cuda.get_device_name(device_index),
    }


def run_subprocess(args: argparse.Namespace, variant: str, length: int) -> dict[str, object]:
    batch_size = batch_size_for_length(length, args.total_tokens, args.batch_size)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        variant,
        "--length",
        str(length),
        "--batch-size",
        str(batch_size),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        args.device,
        "--dropout",
        str(args.dropout),
        "--execution",
        args.execution,
        "--compile-mode",
        args.compile_mode,
        "--tuning-mode",
        args.tuning_mode,
    ]
    cmd.append("--fullgraph" if args.fullgraph else "--no-fullgraph")
    for profile in args.profile:
        cmd.extend(("--profile", profile))
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    marker = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            marker = line[len(RESULT_PREFIX) :]
    if marker is None:
        status = "oom" if "out of memory" in proc.stderr.lower() else "worker_crash"
        return {
            "variant": variant,
            "length": length,
            "batch_size": batch_size,
            "dtype": args.dtype,
            "dropout": args.dropout,
            "execution": args.execution,
            "tuning_mode": args.tuning_mode,
            "profiles": list(args.profile),
            "status": status,
            "error": (proc.stderr or proc.stdout)[-4000:],
        }
    return json.loads(marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=DEFAULT_TOTAL_TOKENS,
        help="Approximate tokens per step when --batch-size is omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Use a fixed batch instead of constant-token scheduling.",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--execution", choices=("eager", "compile"), default="eager")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="deberta_training_scaling.json")
    parser.add_argument(
        "--tuning-mode",
        choices=("auto", "heuristic", "autotune", "profile_only"),
        default="auto",
        help="DisentangledFlash tuning policy (default: auto)",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Kernel profile for DisentangledFlash; repeat to provide multiple profiles",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, default="deberta_df", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, default=512, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.profile = [str(Path(profile).expanduser().resolve()) for profile in args.profile]
    missing_profiles = [profile for profile in args.profile if not Path(profile).is_file()]
    if missing_profiles:
        raise SystemExit(f"profile does not exist: {missing_profiles[0]}")

    if args.worker:
        try:
            result = worker(args)
        except torch.cuda.OutOfMemoryError as exc:
            result = {
                "variant": args.variant,
                "length": args.length,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "dropout": args.dropout,
                "execution": args.execution,
                "tuning_mode": args.tuning_mode,
                "profiles": list(args.profile),
                "status": "oom",
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - worker must serialize arbitrary failures.
            result = {
                "variant": args.variant,
                "length": args.length,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "dropout": args.dropout,
                "execution": args.execution,
                "tuning_mode": args.tuning_mode,
                "profiles": list(args.profile),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(RESULT_PREFIX + json.dumps(result))
        return

    print(f"DisentangledFlash tuning mode: {args.tuning_mode}")
    if args.profile:
        for profile in args.profile:
            print(f"DisentangledFlash profile: {profile}")
    results = []
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    total = len(args.variants) * len(args.lengths)
    count = 0
    for length in args.lengths:
        for variant in args.variants:
            count += 1
            batch_size = batch_size_for_length(length, args.total_tokens, args.batch_size)
            print(
                f"[{count:02d}/{total:02d}] {variant} B={batch_size} L={length}",
                flush=True,
            )
            result = run_subprocess(args, variant, length)
            results.append(result)
            if result["status"] == "ok":
                print(
                    f"  {result['median_fwd_bwd_ms']:.2f} ms  "
                    f"{result['tokens_per_s']:.0f} tok/s  "
                    f"peak={result['peak_gib']:.3f} GiB",
                    flush=True,
                )
            else:
                print(f"  {result['status']}: {result.get('error', '')[:300]}")

    Path(args.output).write_text(
        json.dumps(
            {
                "configuration": {
                    "lengths": args.lengths,
                    "variants": args.variants,
                    "batch_size": args.batch_size,
                    "batch_schedule": (
                        "fixed" if args.batch_size is not None else "constant_tokens"
                    ),
                    "total_tokens": args.total_tokens,
                    "dtype": args.dtype,
                    "dropout": args.dropout,
                    "execution": args.execution,
                    "compile_mode": args.compile_mode,
                    "fullgraph": args.fullgraph,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "device": args.device,
                    "tuning_mode": args.tuning_mode,
                    "profiles": args.profile,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
