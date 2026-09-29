"""Paper-oriented single-layer disentangled-attention benchmark.

Follows FlashAttention's constant-token convention: each length uses
max(1, total_tokens // length) sequences, all tokens active. Every
configuration runs in its own process, so an OOM or a missing optional backend
is recorded instead of aborting the matrix, and the matrix is repeated with the
implementation order rotated so slow periods on the node don't favor anyone.
The default timer is Triton's do_bench, which flushes L2 before each iteration.

Setup a real encoder does once for all layers (the Hugging Face [B, 1, L, L]
mask and relative-position matrix) happens before timing. The reported TFLOP/s
counts only dense QK and PV work, so it is a QK+PV-equivalent rate for the whole
layer, not hardware utilization.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
    build_relative_position,
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

    def prepare(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the per-encoder mask and relative positions once, outside timing."""

        if not self.expand_mask:
            return attention_mask, None
        length = hidden_states.size(1)
        relative_pos = build_relative_position(
            hidden_states,
            hidden_states,
            bucket_size=self.module.position_buckets,
            max_position=self.module.max_relative_positions,
        )
        return _prepare_attention_mask(attention_mask, length, length), relative_pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rel_embeddings: torch.Tensor,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.module(
            hidden_states,
            attention_mask,
            relative_pos=relative_pos,
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
    relative_pos: torch.Tensor | None = None,
):
    if pass_mode == "forward":
        with torch.no_grad():
            return module(hidden_states, attention_mask, rel_embeddings, relative_pos)
    if pass_mode == "forward_backward":
        module.zero_grad(set_to_none=True)
        if hidden_states.grad is not None:
            hidden_states.grad = None
        if rel_embeddings.grad is not None:
            rel_embeddings.grad = None
        output = module(hidden_states, attention_mask, rel_embeddings, relative_pos)
        output.float().square().mean().backward()
        return output
    raise ValueError(f"operation does not directly handle {pass_mode}")


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "p50_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "samples": len(ordered),
    }


def _prepare_case(
    implementation: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    batch_size = batch_size_for_length(args.length, args.total_tokens)
    config = make_config(args.head_dim, args.dropout)
    torch.manual_seed(args.seed)
    reference_state = OriginalDisentangledSelfAttention(config).state_dict()
    module = make_module(
        implementation,
        config,
        reference_state,
        device=device,
        dtype=dtype,
        pass_mode=args.pass_mode,
        tuning_mode=args.tuning_mode,
        profile_paths=tuple(args.profile),
    )
    module.train(args.pass_mode != "forward")
    # Same seed for every implementation, so all of them see identical inputs.
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    training = args.pass_mode != "forward"
    hidden_states = torch.randn(
        batch_size,
        args.length,
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=training,
    )
    attention_mask = torch.ones(batch_size, args.length, device=device, dtype=torch.bool)
    rel_embeddings = torch.randn(
        config.position_buckets * 2,
        config.hidden_size,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=training,
    )
    attention_mask, relative_pos = module.prepare(hidden_states, attention_mask)

    def operation():
        return _operation(
            module,
            hidden_states,
            attention_mask,
            rel_embeddings,
            args.pass_mode,
            relative_pos,
        )

    def clear_gradients() -> None:
        module.zero_grad(set_to_none=True)
        hidden_states.grad = None
        rel_embeddings.grad = None

    return {
        "operation": operation,
        "clear_gradients": clear_gradients,
        "batch_size": batch_size,
        "num_heads": config.num_attention_heads,
    }


def _failure(implementation: str, error: BaseException) -> dict[str, Any]:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        status = "oom"
    elif isinstance(error, ModuleNotFoundError):
        status = "unavailable"
    else:
        status = "error"
    torch.cuda.empty_cache()
    return {"status": status, "implementation": implementation, "error": f"{error}"[:4000]}


def _gc_paused(run):
    """Run with the garbage collector paused, as timeit does.

    A collection costs milliseconds and would otherwise land on whichever short
    step happens to trigger it.
    """

    gc.collect()
    collecting = gc.isenabled()
    gc.disable()
    try:
        return run()
    finally:
        if collecting:
            gc.enable()


def _measure_events(operation, iterations: int) -> tuple[list[float], float]:
    """Back-to-back CUDA-event timing on warm caches; also host ms per iteration."""

    def run() -> tuple[list[float], float]:
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        host_started = time.perf_counter()
        for index in range(iterations):
            starts[index].record()
            operation()
            ends[index].record()
        host_ms = (time.perf_counter() - host_started) * 1000.0 / iterations
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)], host_ms

    return _gc_paused(run)


def _measure_do_bench(operation, rep_ms: float) -> list[float]:
    """Triton's do_bench: the L2 cache is flushed before every timed iteration."""

    from triton.testing import do_bench

    samples = _gc_paused(lambda: do_bench(operation, warmup=0, rep=rep_ms, return_mode="all"))
    return [float(sample) for sample in samples]


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Time one implementation for one configuration, in its own process."""

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    dtype = dtype_from_name(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")
    implementation = args.implementation
    if implementation == "flashdeberta" and args.dropout:
        return {
            "status": "unsupported",
            "implementation": implementation,
            "reason": "FlashDeBERTa does not apply attention-probability dropout",
        }

    case = _prepare_case(implementation, args, device, dtype)
    operation = case["operation"]
    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()

    host_ms = None
    before = torch.cuda.memory_stats(device_index)
    if args.timer == "do_bench":
        samples = _measure_do_bench(operation, args.rep_ms)
    else:
        samples, host_ms = _measure_events(operation, args.iters)
    after = torch.cuda.memory_stats(device_index)
    # Retries and fresh cudaMalloc calls while timing mean the cache ran out of
    # fitting blocks; both are slow and synchronize the device.
    retries = after.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0)
    mallocs = after.get("num_device_alloc", 0) - before.get("num_device_alloc", 0)

    case["clear_gradients"]()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device_index)

    summary = summarize_samples(samples)
    batch_size = case["batch_size"]
    flops = effective_attention_flops(
        batch_size, case["num_heads"], args.length, args.head_dim, args.pass_mode
    )
    return {
        "status": "ok",
        "implementation": implementation,
        "equivalent_math": True,
        "pass": args.pass_mode,
        "dropout": args.dropout,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "sequence_length": args.length,
        "total_tokens": batch_size * args.length,
        "head_dim": args.head_dim,
        "num_heads": case["num_heads"],
        "timer": args.timer,
        **summary,
        # Host time to issue one iteration (events timer only). Close to p50 means
        # the step is limited by Python and kernel launches, not by the GPU.
        "host_ms_per_iter": host_ms,
        "allocator_retries": retries,
        "cuda_mallocs_while_timing": mallocs,
        "timings_ms": samples,
        "tokens_per_second": batch_size * args.length * 1000.0 / summary["p50_ms"],
        "effective_attention_tflops": flops / (summary["p50_ms"] * 1e9),
        "effective_flops_definition": (
            "dense QK+PV FLOPs over the whole layer's time; projections and "
            "relative-bias work are timed but not counted"
        ),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": max(0, peak - baseline),
        "gpu": torch.cuda.get_device_name(device_index),
        "tuning_mode": args.tuning_mode if implementation == "triton" else None,
        "profiles": list(args.profile) if implementation == "triton" else [],
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
        "--timer",
        args.timer,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--rep-ms",
        str(args.rep_ms),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--tuning-mode",
        args.tuning_mode,
    ]
    for profile in args.profile:
        command.extend(("--profile", profile))
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    marker = next(
        (
            line[len(RESULT_PREFIX) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    common = {
        "implementation": implementation,
        "pass": pass_mode,
        "dropout": dropout,
        "dtype": args.dtype,
        "batch_size": batch_size_for_length(length, args.total_tokens),
        "sequence_length": length,
    }
    if marker is not None:
        return {**common, **json.loads(marker)}
    error = completed.stderr or completed.stdout
    lowered = error.lower()
    if "out of memory" in lowered:
        status = "oom"
    elif "not installed" in lowered or "no module named" in lowered:
        status = "unavailable"
    else:
        status = "worker_crash"
    return {**common, "status": status, "error": error[-4000:]}


def combine_repeats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge one configuration's repeated runs into a single result."""

    ok = [run for run in runs if run.get("status") == "ok"]
    if not ok:
        return runs[-1]
    medians = [run["p50_ms"] for run in ok]
    best = min(medians)
    combined = dict(min(ok, key=lambda run: abs(run["p50_ms"] - statistics.median(medians))))
    p50 = statistics.median(medians)
    combined.update(
        {
            "p50_ms": p50,
            "min_ms": min(run["min_ms"] for run in ok),
            "repeat_p50_ms": medians,
            "repeat_spread": max(medians) / best - 1.0,
            "repeats": len(runs),
            "failed_repeats": len(runs) - len(ok),
            "tokens_per_second": combined["tokens_per_second"] * combined["p50_ms"] / p50,
            "effective_attention_tflops": combined["effective_attention_tflops"]
            * combined["p50_ms"]
            / p50,
        }
    )
    return combined


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
    parser.add_argument(
        "--timer",
        choices=("do_bench", "events"),
        default="do_bench",
        help="do_bench flushes L2 before each iteration; events times back-to-back warm runs.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Untimed warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations (events timer).")
    parser.add_argument("--rep-ms", type=float, default=500.0, help="Timed ms (do_bench timer).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Run the matrix this many times, rotating the implementation order.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="attention_kernel_results.json")
    parser.add_argument(
        "--tuning-mode",
        choices=("auto", "heuristic", "autotune", "profile_only"),
        default="auto",
    )
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--pass-mode", choices=PASS_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=0.0, help=argparse.SUPPRESS)
    return parser


def _describe(result: dict[str, Any]) -> str:
    if result["status"] != "ok":
        return f"{result['status']}: {result.get('error', result.get('reason', ''))[:240]}"
    flags = []
    if result["allocator_retries"] or result["cuda_mallocs_while_timing"]:
        flags.append(
            f"allocator: {result['allocator_retries']} retries, "
            f"{result['cuda_mallocs_while_timing']} cudaMalloc while timing"
        )
    host = result.get("host_ms_per_iter")
    if host is not None and host >= 0.8 * result["p50_ms"]:
        flags.append(f"host-bound, {host:.2f} ms/iter on the CPU")
    return (
        f"p50 {result['p50_ms']:8.3f} ms  min {result['min_ms']:8.3f}  "
        f"incremental peak {result['incremental_peak_allocated_bytes'] / 1024**3:.3f} GiB"
        + (f"  [{'; '.join(flags)}]" if flags else "")
    )


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
        except Exception as exc:  # noqa: BLE001 - preserve optional-backend failures
            result = _failure(args.implementation, exc)
        print(RESULT_PREFIX + json.dumps(result))
        return

    if args.repeats < 1 or args.iters < 1 or args.rep_ms <= 0:
        raise SystemExit("--repeats, --iters and --rep-ms must be positive")
    if any(not 0.0 <= value < 1.0 for value in args.training_dropouts):
        raise SystemExit("training dropout values must be in [0, 1)")
    if args.total_tokens < 1 or any(length < 1 for length in args.lengths):
        raise SystemExit("lengths and --total-tokens must be positive")
    results: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
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
            "timer": args.timer,
            "warmup": args.warmup,
            "iters": args.iters,
            "rep_ms": args.rep_ms,
            "repeats": args.repeats,
            "device": args.device,
            "seed": args.seed,
            "tuning_mode": args.tuning_mode,
            "profiles": args.profile,
        },
        "results": results,
        "runs": [],
    }
    groups = []
    for pass_mode in args.passes:
        dropouts = [0.0] if pass_mode == "forward" else args.training_dropouts
        for dropout in dropouts:
            for length in args.lengths:
                groups.append((pass_mode, length, dropout))

    # Every configuration gets its own process. Repeats rotate the order of the
    # implementations, so slow periods on the node don't favor any of them.
    runs: dict[tuple[str, str, int, float], list[dict[str, Any]]] = {}
    total = len(groups) * len(args.implementations) * args.repeats
    step = 0
    for repeat in range(args.repeats):
        shift = repeat % len(args.implementations)
        order = args.implementations[shift:] + args.implementations[:shift]
        for pass_mode, length, dropout in groups:
            for implementation in order:
                step += 1
                batch_size = batch_size_for_length(length, args.total_tokens)
                result = run_subprocess(args, implementation, pass_mode, length, dropout)
                runs.setdefault((implementation, pass_mode, length, dropout), []).append(result)
                payload["runs"].append({**result, "repeat": repeat})
                write_json(output, payload)
                print(
                    f"[{step:03d}/{total:03d}] repeat {repeat + 1} {implementation:13} {pass_mode:16} "
                    f"B={batch_size} L={length} dropout={dropout:g}  {_describe(result)}",
                    flush=True,
                )

    for pass_mode, length, dropout in groups:
        for implementation in args.implementations:
            results.append(combine_repeats(runs[(implementation, pass_mode, length, dropout)]))
    write_json(output, payload)
    print()
    for result in results:
        print(
            f"{result['implementation']:13} {result['pass']:16} L={result['sequence_length']:5} "
            f"{_describe(result)}"
            + (
                f"  repeats {', '.join(f'{value:.3f}' for value in result['repeat_p50_ms'])}"
                if result.get("status") == "ok"
                else ""
            )
        )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
