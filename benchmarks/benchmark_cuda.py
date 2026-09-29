"""Paper-oriented single-layer disentangled-attention benchmark.

The benchmark follows the constant-token convention used by FlashAttention:
each length uses ``max(1, total_tokens // length)`` sequences.  Every shape,
backend, pass, and dropout setting runs in a fresh process so an OOM or an
unavailable optional backend remains a result instead of aborting the matrix.

All inputs are active tokens, so there is no padded/packed layout axis.
FlashDeBERTa is optional and is exercised through its public single-attention-
layer class when installed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash._torch import (
    TorchInferenceDisentangledSelfAttention,
    TorchTrainingDisentangledSelfAttention,
)
from disentangled_flash.kernel import InferenceDisentangledSelfAttention
from disentangled_flash.training import TritonTrainingDisentangledSelfAttention
from disentangled_flash.tuning import KernelTuningOptions

RESULT_PREFIX = "__ATTENTION_RESULT__="
IMPLEMENTATIONS = ("base", "torch", "triton", "flashdeberta")
PASS_MODES = ("forward", "forward_backward")
DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)
DEFAULT_TOTAL_TOKENS = 16_384


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "triton", "transformers", "flashdeberta", "disentangled-flash"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_commit = None
    git_dirty = None
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_commit = commit.stdout.strip()
        git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    cuda = {
        "available": torch.cuda.is_available(),
        "runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "command": sys.argv,
    }


def batch_size_for_length(length: int, total_tokens: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    return max(1, total_tokens // length)


def effective_attention_flops(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    pass_mode: str,
) -> int:
    """Return conventional dense-attention FLOPs, excluding relative bias work."""

    forward = 4 * batch_size * num_heads * sequence_length**2 * head_dim
    multiplier = {
        "forward": 1.0,
        "forward_backward": 3.5,
    }[pass_mode]
    return int(forward * multiplier)


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_config(head_dim: int, dropout: float) -> DebertaAttentionConfig:
    num_heads = 12
    hidden_size = num_heads * head_dim
    return DebertaAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_size=head_dim,
        attention_probs_dropout_prob=dropout,
        hidden_dropout_prob=0.0,
        max_relative_positions=512,
        max_position_embeddings=8192,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd="none",
    )


class AttentionCall(nn.Module):
    """Normalize the mask contract of benchmark implementations."""

    def __init__(self, module: nn.Module, *, expand_mask: bool = False) -> None:
        super().__init__()
        self.module = module
        self.expand_mask = expand_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask
        if self.expand_mask:
            length = hidden_states.size(1)
            mask = _prepare_attention_mask(mask, length, length)
        return self.module(
            hidden_states,
            mask,
            rel_embeddings=rel_embeddings,
        )[0]


def _flashdeberta_config_kwargs(config: DebertaAttentionConfig) -> dict[str, Any]:
    values = dict(config.__dict__)
    pos_att_type = values.get("pos_att_type")
    if isinstance(pos_att_type, tuple):
        values["pos_att_type"] = list(pos_att_type)
    return values


def _make_flashdeberta(config: DebertaAttentionConfig) -> nn.Module:
    try:
        from flashdeberta.model import FlashDisentangledSelfAttention
        from transformers import DebertaV2Config
    except ImportError as exc:
        raise ModuleNotFoundError(
            "FlashDeBERTa is not installed; install the benchmark extra"
        ) from exc
    flash_config = DebertaV2Config(**_flashdeberta_config_kwargs(config))
    return FlashDisentangledSelfAttention(flash_config)


def make_module(
    implementation: str,
    config: DebertaAttentionConfig,
    reference_state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    pass_mode: str,
    tuning_mode: str,
    profile_paths: tuple[str, ...],
) -> AttentionCall:
    expand_mask = False
    if implementation == "base":
        module = OriginalDisentangledSelfAttention(config)
        expand_mask = True
    elif implementation == "torch":
        module = (
            TorchInferenceDisentangledSelfAttention(config, assume_unpadded=True)
            if pass_mode == "forward"
            else TorchTrainingDisentangledSelfAttention(config, assume_unpadded=True)
        )
    elif implementation == "triton":
        options = KernelTuningOptions(mode=tuning_mode, profile_paths=profile_paths)
        module = (
            InferenceDisentangledSelfAttention(
                config,
                backend="triton",
                assume_unpadded=True,
                tuning=options,
            )
            if pass_mode == "forward"
            else TritonTrainingDisentangledSelfAttention(
                config,
                assume_unpadded=True,
                tuning=options,
            )
        )
    elif implementation == "flashdeberta":
        module = _make_flashdeberta(config)
    else:
        raise ValueError(f"unknown implementation: {implementation}")
    module.load_state_dict(reference_state, strict=True)
    return AttentionCall(module, expand_mask=expand_mask).to(device=device, dtype=dtype)


def _operation(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    rel_embeddings: torch.Tensor,
    pass_mode: str,
):
    if pass_mode == "forward":
        with torch.no_grad():
            return module(hidden_states, attention_mask, rel_embeddings)
    if pass_mode == "forward_backward":
        module.zero_grad(set_to_none=True)
        if hidden_states.grad is not None:
            hidden_states.grad = None
        if rel_embeddings.grad is not None:
            rel_embeddings.grad = None
        output = module(hidden_states, attention_mask, rel_embeddings)
        output.float().square().mean().backward()
        return output
    raise ValueError(f"operation does not directly handle {pass_mode}")


def _measure_events(operation, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        operation()
        ends[index].record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def worker(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    dtype = dtype_from_name(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")
    if args.implementation == "flashdeberta" and args.dropout:
        return {
            "status": "unsupported",
            "reason": "FlashDeBERTa does not apply attention-probability dropout",
        }
    batch_size = batch_size_for_length(args.length, args.total_tokens)
    config = make_config(args.head_dim, args.dropout)
    torch.manual_seed(args.seed)
    reference = OriginalDisentangledSelfAttention(config)
    reference_state = reference.state_dict()
    module = make_module(
        args.implementation,
        config,
        reference_state,
        device=device,
        dtype=dtype,
        pass_mode=args.pass_mode,
        tuning_mode=args.tuning_mode,
        profile_paths=tuple(args.profile),
    )
    module.train(args.pass_mode != "forward")
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    hidden_states = torch.randn(
        batch_size,
        args.length,
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=args.pass_mode != "forward",
    )
    attention_mask = torch.ones(batch_size, args.length, device=device, dtype=torch.bool)
    rel_embeddings = torch.randn(
        config.position_buckets * 2,
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=args.pass_mode != "forward",
    )

    def operation():
        return _operation(
            module,
            hidden_states,
            attention_mask,
            rel_embeddings,
            args.pass_mode,
        )

    timings = _measure_events(operation, args.warmup, args.iters)
    module.zero_grad(set_to_none=True)
    hidden_states.grad = None
    rel_embeddings.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device_index)
    median_ms = statistics.median(timings)
    flops = effective_attention_flops(
        batch_size,
        config.num_attention_heads,
        args.length,
        args.head_dim,
        args.pass_mode,
    )
    return {
        "status": "ok",
        "implementation": args.implementation,
        "equivalent_math": True,
        "pass": args.pass_mode,
        "dropout": args.dropout,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "sequence_length": args.length,
        "total_tokens": batch_size * args.length,
        "head_dim": args.head_dim,
        "num_heads": config.num_attention_heads,
        "p50_ms": median_ms,
        "mean_ms": statistics.fmean(timings),
        "p90_ms": sorted(timings)[max(0, math.ceil(0.9 * len(timings)) - 1)],
        "timings_ms": timings,
        "tokens_per_second": batch_size * args.length * 1000.0 / median_ms,
        "effective_attention_tflops": flops / (median_ms * 1e9),
        "effective_flops_definition": "dense QK+PV only; relative-bias work excluded",
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": max(0, peak - baseline),
        "gpu": torch.cuda.get_device_name(device_index),
        "tuning_mode": args.tuning_mode if args.implementation == "triton" else None,
        "profiles": list(args.profile) if args.implementation == "triton" else [],
    }


def run_subprocess(
    args: argparse.Namespace,
    implementation: str,
    pass_mode: str,
    length: int,
    dropout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_cuda",
        "--worker",
        "--implementation",
        implementation,
        "--pass-mode",
        pass_mode,
        "--length",
        str(length),
        "--total-tokens",
        str(args.total_tokens),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--dropout",
        str(dropout),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--tuning-mode",
        args.tuning_mode,
    ]
    for profile in args.profile:
        command.extend(("--profile", profile))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    marker = next(
        (
            line[len(RESULT_PREFIX) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if marker is not None:
        return json.loads(marker)
    error = completed.stderr or completed.stdout
    lowered = error.lower()
    if "out of memory" in lowered:
        status = "oom"
    elif "not installed" in lowered or "no module named" in lowered:
        status = "unavailable"
    else:
        status = "worker_crash"
    return {
        "status": status,
        "implementation": implementation,
        "pass": pass_mode,
        "dropout": dropout,
        "dtype": args.dtype,
        "batch_size": batch_size_for_length(length, args.total_tokens),
        "sequence_length": length,
        "error": error[-4000:],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--implementations", nargs="+", choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS)
    )
    parser.add_argument("--passes", nargs="+", choices=PASS_MODES, default=list(PASS_MODES))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--training-dropouts",
        nargs="+",
        type=float,
        default=[0.0],
        help="Dropout values for training passes; inference forward always uses zero.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="attention_kernel_results.json")
    parser.add_argument(
        "--tuning-mode",
        choices=("auto", "autotune", "profile_only"),
        default="auto",
    )
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--pass-mode", choices=PASS_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.profile = [str(Path(path).expanduser().resolve()) for path in args.profile]
    missing = [path for path in args.profile if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"profile does not exist: {missing[0]}")
    if args.worker:
        if args.implementation is None or args.pass_mode is None or args.length is None:
            raise SystemExit("worker mode requires implementation, pass-mode, and length")
        try:
            result = worker(args)
        except torch.cuda.OutOfMemoryError as exc:
            result = {"status": "oom", "error": str(exc)}
        except ModuleNotFoundError as exc:
            result = {"status": "unavailable", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - preserve optional-backend failures.
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("implementation", args.implementation)
        result.setdefault("pass", args.pass_mode)
        result.setdefault("dropout", args.dropout)
        result.setdefault("sequence_length", args.length)
        result.setdefault("batch_size", batch_size_for_length(args.length, args.total_tokens))
        print(RESULT_PREFIX + json.dumps(result))
        return

    if any(not 0.0 <= value < 1.0 for value in args.training_dropouts):
        raise SystemExit("training dropout values must be in [0, 1)")
    if args.total_tokens < 1 or any(length < 1 for length in args.lengths):
        raise SystemExit("lengths and --total-tokens must be positive")
    results: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "environment": collect_environment(),
        "configuration": {
            "implementations": args.implementations,
            "passes": args.passes,
            "lengths": args.lengths,
            "batch_schedule": "constant_tokens",
            "total_tokens": args.total_tokens,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "training_dropouts": args.training_dropouts,
            "warmup": args.warmup,
            "iters": args.iters,
            "device": args.device,
            "seed": args.seed,
            "tuning_mode": args.tuning_mode,
            "profiles": args.profile,
        },
        "results": results,
    }
    write_json(output, payload)
    cases = []
    for implementation in args.implementations:
        for pass_mode in args.passes:
            dropouts = [0.0] if pass_mode == "forward" else args.training_dropouts
            for dropout in dropouts:
                for length in args.lengths:
                    cases.append((implementation, pass_mode, length, dropout))

    for index, (implementation, pass_mode, length, dropout) in enumerate(cases, 1):
        batch_size = batch_size_for_length(length, args.total_tokens)
        print(
            f"[{index:03d}/{len(cases):03d}] {implementation} {pass_mode} "
            f"B={batch_size} L={length} dropout={dropout:g}",
            flush=True,
        )
        result = run_subprocess(args, implementation, pass_mode, length, dropout)
        results.append(result)
        write_json(output, payload)
        if result["status"] == "ok":
            print(
                f"  {result['p50_ms']:.3f} ms  "
                f"{result['effective_attention_tflops']:.2f} effective TFLOP/s  "
                f"peak={result['peak_allocated_bytes'] / 1024**3:.3f} GiB",
                flush=True,
            )
        else:
            print(f"  {result['status']}: {result.get('error', result.get('reason', ''))[:240]}")

    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
