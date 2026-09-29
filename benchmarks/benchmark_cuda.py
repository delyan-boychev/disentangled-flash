"""Paper-oriented single-layer disentangled-attention benchmark.

The benchmark follows the constant-token convention used by FlashAttention:
each length uses ``max(1, total_tokens // length)`` sequences.  Every shape,
backend, pass, and dropout setting runs in a fresh process so an OOM or an
unavailable optional backend remains a result instead of aborting the matrix.

All inputs are active tokens, so there is no padded/packed layout axis. The
implementations for one shape share a process and run in interleaved rounds,
so GPU throttling or drift affects all of them. With nvidia-ml-py installed,
rounds run below 90% of the best observed SM clock are left out of the
statistics.
FlashDeBERTa is optional and is exercised through its public single-attention-
layer class when installed.

Setup that a real encoder does once for all layers (the Hugging Face [B, 1, L, L]
mask and relative-position matrix) is built before timing. The reported TFLOP/s
counts only dense QK and PV work, so it is a QK+PV-equivalent rate for the whole
layer, not hardware utilization.
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


def _measure_events(operation, iterations: int) -> list[float]:
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        operation()
        ends[index].record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


class ClockMonitor:
    """Read the SM clock through NVML; every read returns None without nvidia-ml-py."""

    def __init__(self, device_index: int) -> None:
        self._nvml: Any = None
        self._handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(device_index).uuid)
            uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            except pynvml.NVMLError:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml, self._handle = pynvml, handle
        except Exception as error:  # noqa: BLE001 - clock readings are optional diagnostics
            print(f"SM clock monitoring disabled: {type(error).__name__}: {error}", flush=True)

    def sm_clock(self) -> int | None:
        if self._handle is None:
            return None
        try:
            return int(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM))
        except Exception:  # noqa: BLE001
            return None


def summarize_rounds(
    rounds: list[tuple[list[float], int | None]],
    *,
    clock_floor: float = 0.9,
    drift_limit: float = 0.05,
) -> dict[str, Any]:
    """Summarize interleaved rounds, excluding rounds run below the best observed clock."""

    clocks = [clock for _, clock in rounds if clock is not None]
    reference = max(clocks) if clocks else None
    throttled = [
        reference is not None and clock is not None and clock < clock_floor * reference
        for _, clock in rounds
    ]
    kept = [
        sample for (samples, _), slow in zip(rounds, throttled) if not slow for sample in samples
    ]
    if not kept:
        kept = [sample for samples, _ in rounds for sample in samples]
    ordered = sorted(kept)
    quarter = max(1, len(kept) // 4)
    drift = statistics.median(kept[-quarter:]) / statistics.median(kept[:quarter]) - 1.0

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "p50_ms": statistics.median(kept),
        "mean_ms": statistics.fmean(kept),
        "min_ms": ordered[0],
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "samples_kept": len(kept),
        "rounds": len(rounds),
        "throttled_rounds": sum(throttled),
        "sm_clock_mhz": [clock for _, clock in rounds],
        "drift": drift,
        "stable": abs(drift) <= drift_limit,
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


def worker(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Time every implementation for one (pass, length, dropout) in interleaved rounds.

    Rotating the order each round spreads GPU throttling and drift evenly over
    the implementations instead of penalizing whichever runs last.
    """

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    dtype = dtype_from_name(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")

    results: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    for implementation in args.implementation:
        if implementation == "flashdeberta" and args.dropout:
            results[implementation] = {
                "status": "unsupported",
                "implementation": implementation,
                "reason": "FlashDeBERTa does not apply attention-probability dropout",
            }
            continue
        try:
            case = _prepare_case(implementation, args, device, dtype)
            for _ in range(args.warmup):
                case["operation"]()
            torch.cuda.synchronize()
            cases[implementation] = case
        except Exception as error:  # noqa: BLE001 - keep the other implementations going
            results[implementation] = _failure(implementation, error)

    clocks = ClockMonitor(device_index)
    per_round = max(1, math.ceil(args.iters / args.rounds))
    rounds: dict[str, list[tuple[list[float], int | None]]] = {name: [] for name in cases}
    order = list(cases)
    for round_index in range(args.rounds):
        shift = round_index % max(1, len(order))
        for implementation in order[shift:] + order[:shift]:
            if implementation in results:
                continue
            try:
                samples = _measure_events(cases[implementation]["operation"], per_round)
            except Exception as error:  # noqa: BLE001
                results[implementation] = _failure(implementation, error)
                continue
            rounds[implementation].append((samples, clocks.sm_clock()))

    for implementation, case in cases.items():
        if implementation in results:
            continue
        for other in cases.values():
            other["clear_gradients"]()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
        try:
            case["operation"]()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            results[implementation] = _failure(implementation, error)
            continue
        peak = torch.cuda.max_memory_allocated(device_index)
        summary = summarize_rounds(rounds[implementation])
        batch_size = case["batch_size"]
        flops = effective_attention_flops(
            batch_size,
            case["num_heads"],
            args.length,
            args.head_dim,
            args.pass_mode,
        )
        results[implementation] = {
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
            **summary,
            "timings_ms": [sample for samples, _ in rounds[implementation] for sample in samples],
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
    return [results[implementation] for implementation in args.implementation]


def run_subprocess(
    args: argparse.Namespace,
    implementations: list[str],
    pass_mode: str,
    length: int,
    dropout: float,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_cuda",
        "--worker",
        "--implementation",
        *implementations,
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
        "--rounds",
        str(args.rounds),
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
    batch_size = batch_size_for_length(length, args.total_tokens)
    common = {
        "pass": pass_mode,
        "dropout": dropout,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "sequence_length": length,
    }
    if marker is not None:
        return [{**common, **result} for result in json.loads(marker)]
    error = completed.stderr or completed.stdout
    lowered = error.lower()
    if "out of memory" in lowered:
        status = "oom"
    elif "not installed" in lowered or "no module named" in lowered:
        status = "unavailable"
    else:
        status = "worker_crash"
    return [
        {**common, "status": status, "implementation": implementation, "error": error[-4000:]}
        for implementation in implementations
    ]


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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--iters", type=int, default=100, help="Timed iterations per implementation."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Split the iterations into interleaved rounds so drift hits every implementation.",
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
    parser.add_argument(
        "--implementation", nargs="+", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS
    )
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
            results = worker(args)
        except Exception as exc:  # noqa: BLE001 - report the failure for every implementation
            results = [_failure(implementation, exc) for implementation in args.implementation]
        print(RESULT_PREFIX + json.dumps(results))
        return

    if args.rounds < 1 or args.iters < args.rounds:
        raise SystemExit("--rounds must be positive and at most --iters")
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
            "rounds": args.rounds,
            "device": args.device,
            "seed": args.seed,
            "tuning_mode": args.tuning_mode,
            "profiles": args.profile,
        },
        "results": results,
    }
    write_json(output, payload)
    groups = []
    for pass_mode in args.passes:
        dropouts = [0.0] if pass_mode == "forward" else args.training_dropouts
        for dropout in dropouts:
            for length in args.lengths:
                groups.append((pass_mode, length, dropout))

    unstable = []
    for index, (pass_mode, length, dropout) in enumerate(groups, 1):
        batch_size = batch_size_for_length(length, args.total_tokens)
        print(
            f"[{index:03d}/{len(groups):03d}] {pass_mode} B={batch_size} L={length} "
            f"dropout={dropout:g}: {', '.join(args.implementations)}",
            flush=True,
        )
        group_results = run_subprocess(args, args.implementations, pass_mode, length, dropout)
        results.extend(group_results)
        write_json(output, payload)
        for result in group_results:
            name = result["implementation"]
            if result["status"] != "ok":
                detail = result.get("error", result.get("reason", ""))[:240]
                print(f"  {name:13} {result['status']}: {detail}")
                continue
            flags = []
            if result["throttled_rounds"]:
                flags.append(f"{result['throttled_rounds']}/{result['rounds']} rounds throttled")
            if not result["stable"]:
                flags.append(f"drift {result['drift']:+.1%}")
                unstable.append((name, pass_mode, length))
            print(
                f"  {name:13} p50 {result['p50_ms']:8.3f} ms  min {result['min_ms']:8.3f}  "
                f"incremental peak {result['incremental_peak_allocated_bytes'] / 1024**3:.3f} GiB"
                + (f"  [{'; '.join(flags)}]" if flags else ""),
                flush=True,
            )

    if unstable:
        print(f"Unstable results (rerun these): {unstable}")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
