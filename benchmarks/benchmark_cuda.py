"""Single-layer disentangled-attention benchmark.

Follows FlashAttention's conventions: every length uses
max(1, total_tokens // length) sequences with all tokens active, and backward
is timed on its own by running forward once and replaying
``out.backward(grad, retain_graph=True)``.

Every configuration runs in its own process, so an OOM or a missing optional
backend is recorded instead of aborting the matrix. Timing follows
FlashAttention 3: do_bench with 3 ms warmup and 30 ms of timing, and a 1 s pause
between processes so the GPU doesn't carry power throttling into the next run.

Timing is eager, with the L2 cache flushed before each iteration, so it
includes Python, autograd and launch overhead. Each result also records the CPU
time to issue one iteration; when that reaches the latency, the point is
host-bound and measures the CPU rather than the GPU.

TFLOP/s counts the attention's matmul work: QK and PV, plus the C2P/P2C
position-table GEMMs over the relative positions a length actually uses.
Softmax, bias gathers and the Q/K/V and position projections are left out, as
FlashAttention leaves out softmax and projections.
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
from disentangled_flash.position import SharedPositionPlanCache
from disentangled_flash.training import TritonTrainingDisentangledSelfAttention
from disentangled_flash.tuning import KernelTuningOptions

RESULT_PREFIX = "__ATTENTION_RESULT__="
IMPLEMENTATIONS = ("base", "torch", "triton", "flashdeberta")
PASS_MODES = ("forward", "backward", "forward_backward")
DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)
DEFAULT_TOTAL_TOKENS = 16_384
# Backward recomputes every score term, then forms both operand gradients.
DENSE_FLOPS = {"forward": 4, "backward": 10, "forward_backward": 14}
POSITION_FLOPS = {"forward": 2, "backward": 6, "forward_backward": 8}


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "triton", "transformers", "flashdeberta", "disentangled-flash"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = {"commit": None, "dirty": None}
    root = Path(__file__).resolve().parents[1]
    try:
        run = lambda *command: subprocess.run(
            ["git", *command], cwd=root, check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        git = {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        pass
    devices = []
    if torch.cuda.is_available():
        devices = [
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
        "cuda": {"runtime": torch.version.cuda, "devices": devices},
        "git": git,
        "command": sys.argv,
    }


def batch_size_for_length(length: int, total_tokens: int) -> int:
    if length <= 0 or total_tokens <= 0:
        raise ValueError("length and total_tokens must be positive")
    return max(1, total_tokens // length)


def relative_position_count(config: DebertaAttentionConfig, sequence_length: int) -> int:
    """Distinct relative-position rows that sequences of this length read."""

    embedding_size = (
        config.position_buckets if config.position_buckets > 0 else config.max_relative_positions
    )
    plans = SharedPositionPlanCache(
        position_buckets=config.position_buckets,
        max_relative_positions=config.max_relative_positions,
        position_embedding_size=embedding_size,
    )
    return plans.compact(sequence_length, "cpu").active_slots.numel()


def attention_flops(
    config: DebertaAttentionConfig, batch_size: int, sequence_length: int, pass_mode: str
) -> int:
    """Matmul FLOPs of disentangled attention: QK, PV and the position-table GEMMs."""

    terms = len({"c2p", "p2c"}.intersection(config.pos_att_type or ()))
    terms *= config.relative_attention
    positions = relative_position_count(config, sequence_length) if terms else 0
    per_head = (
        (DENSE_FLOPS[pass_mode] * sequence_length + POSITION_FLOPS[pass_mode] * terms * positions)
        * sequence_length
        * config.attention_head_size
    )
    return batch_size * config.num_attention_heads * per_head


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_config(head_dim: int) -> DebertaAttentionConfig:
    """DeBERTa-v3-base attention: 12 heads, 256 position buckets."""

    num_heads = 12
    return DebertaAttentionConfig(
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        attention_head_size=head_dim,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        max_relative_positions=512,
        max_position_embeddings=8192,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd="none",
    )


class AttentionCall(nn.Module):
    """Give every implementation the same call signature."""

    def __init__(self, module: nn.Module, *, expand_mask: bool = False) -> None:
        super().__init__()
        self.module = module
        self.expand_mask = expand_mask

    def prepare(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
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

    def forward(self, hidden_states, attention_mask, rel_embeddings, relative_pos=None):
        return self.module(
            hidden_states, attention_mask, relative_pos=relative_pos, rel_embeddings=rel_embeddings
        )[0]


def _flashdeberta_config_kwargs(config: DebertaAttentionConfig) -> dict[str, Any]:
    values = dict(config.__dict__)
    if isinstance(values.get("pos_att_type"), tuple):
        values["pos_att_type"] = list(values["pos_att_type"])
    return values


def _make_flashdeberta(config: DebertaAttentionConfig) -> nn.Module:
    try:
        from flashdeberta.model import FlashDisentangledSelfAttention
        from transformers import DebertaV2Config
    except ImportError as exc:
        raise ModuleNotFoundError(
            "FlashDeBERTa is not installed; install the benchmark extra"
        ) from exc
    return FlashDisentangledSelfAttention(DebertaV2Config(**_flashdeberta_config_kwargs(config)))


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
    inference = pass_mode == "forward"
    if implementation == "base":
        module = OriginalDisentangledSelfAttention(config)
    elif implementation == "torch":
        module = (
            TorchInferenceDisentangledSelfAttention(config, assume_unpadded=True)
            if inference
            else TorchTrainingDisentangledSelfAttention(config, assume_unpadded=True)
        )
    elif implementation == "triton":
        options = KernelTuningOptions(mode=tuning_mode, profile_paths=profile_paths)
        cls = (
            InferenceDisentangledSelfAttention
            if inference
            else TritonTrainingDisentangledSelfAttention
        )
        extra = {"backend": "triton"} if inference else {}
        module = cls(config, assume_unpadded=True, tuning=options, **extra)
    elif implementation == "flashdeberta":
        module = _make_flashdeberta(config)
    else:
        raise ValueError(f"unknown implementation: {implementation}")
    module.load_state_dict(reference_state, strict=True)
    call = AttentionCall(module, expand_mask=implementation == "base")
    return call.to(device=device, dtype=dtype)


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "p50_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "samples": len(ordered),
    }


def _prepare_case(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    """Build the module and inputs; return the timed callable and its grad tensors."""

    batch_size = batch_size_for_length(args.length, args.total_tokens)
    config = make_config(args.head_dim)
    torch.manual_seed(args.seed)
    reference_state = OriginalDisentangledSelfAttention(config).state_dict()
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
    training = args.pass_mode != "forward"
    module.train(training)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    def randn(*shape, requires_grad=training):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=dtype, requires_grad=requires_grad
        )

    hidden_states = randn(batch_size, args.length, config.hidden_size)
    rel_embeddings = randn(config.position_buckets * 2, config.hidden_size)
    attention_mask = torch.ones(batch_size, args.length, device=device, dtype=torch.bool)
    attention_mask, relative_pos = module.prepare(hidden_states, attention_mask)
    inputs = (hidden_states, attention_mask, rel_embeddings, relative_pos)
    leaves = [hidden_states, rel_embeddings, *module.parameters()]

    def clear_gradients() -> None:
        for tensor in leaves:
            tensor.grad = None

    if args.pass_mode == "forward":

        def operation():
            with torch.no_grad():
                module(*inputs)

    elif args.pass_mode == "forward_backward":
        grad_output = randn(batch_size, args.length, config.hidden_size, requires_grad=False)

        def operation():
            clear_gradients()
            module(*inputs).backward(grad_output)

    else:
        output = module(*inputs)
        grad_output = randn(*output.shape, requires_grad=False)

        def operation():
            clear_gradients()
            output.backward(grad_output, retain_graph=True)

    return operation, clear_gradients, batch_size, config.num_attention_heads


def _gc_paused(run):
    """Run with the garbage collector paused, as timeit does."""

    gc.collect()
    collecting = gc.isenabled()
    gc.disable()
    try:
        return run()
    finally:
        if collecting:
            gc.enable()


def _measure(operation, warmup_ms: float, rep_ms: float) -> list[float]:
    from triton.testing import do_bench

    samples = _gc_paused(
        lambda: do_bench(operation, warmup=warmup_ms, rep=rep_ms, return_mode="all")
    )
    return [float(sample) for sample in samples]


def _measure_host_issue(operation, iterations: int = 20) -> float:
    """Median CPU time to issue one iteration from an idle GPU.

    The launch queue never fills, so this is the Python, autograd and launch
    cost alone. Close to the eager p50 means the step is host-bound.
    """

    def run() -> float:
        samples = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - started) * 1000.0)
        torch.cuda.synchronize()
        return statistics.median(samples)

    return _gc_paused(run)


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Time one implementation for one configuration, in its own process."""

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = device.index or 0
    torch.cuda.set_device(device_index)
    dtype = dtype_from_name(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")

    operation, clear_gradients, batch_size, num_heads = _prepare_case(args, device, dtype)
    # Exclude compilation, lazy setup, and do_bench's L2 buffer from measurement.
    operation()
    _measure(operation, 0.0, 1.0)
    torch.cuda.synchronize()

    before = torch.cuda.memory_stats(device_index)
    samples = _measure(operation, args.warmup_ms, args.rep_ms)
    after = torch.cuda.memory_stats(device_index)
    host_ms = _measure_host_issue(operation)

    clear_gradients()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device_index)

    summary = summarize_samples(samples)
    flops = attention_flops(make_config(args.head_dim), batch_size, args.length, args.pass_mode)

    def counter(name: str) -> int:
        return after.get(name, 0) - before.get(name, 0)

    return {
        "status": "ok",
        "head_dim": args.head_dim,
        "num_heads": num_heads,
        **summary,
        "timings_ms": samples,
        "host_ms_per_iter": host_ms,
        "allocator_retries": counter("num_alloc_retries"),
        "cuda_mallocs_while_timing": counter("num_device_alloc"),
        "tokens_per_second": batch_size * args.length * 1000.0 / summary["p50_ms"],
        "attention_tflops": flops / (summary["p50_ms"] * 1e9),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": max(0, peak - baseline),
        "gpu": torch.cuda.get_device_name(device_index),
        "tuning_mode": args.tuning_mode if args.implementation == "triton" else None,
    }


def run_subprocess(
    args: argparse.Namespace, implementation: str, pass_mode: str, length: int
) -> dict[str, Any]:
    command = [sys.executable, "-m", "benchmarks.benchmark_cuda", "--worker"]
    for flag, value in (
        ("--implementation", implementation),
        ("--pass-mode", pass_mode),
        ("--length", length),
        ("--total-tokens", args.total_tokens),
        ("--head-dim", args.head_dim),
        ("--dtype", args.dtype),
        ("--warmup-ms", args.warmup_ms),
        ("--rep-ms", args.rep_ms),
        ("--device", args.device),
        ("--seed", args.seed),
        ("--tuning-mode", args.tuning_mode),
        *(("--profile", profile) for profile in args.profile),
    ):
        command.extend((flag, str(value)))
    environment = {**os.environ}
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False, env=environment
    )
    common = {
        "implementation": implementation,
        "pass": pass_mode,
        "dtype": args.dtype,
        "batch_size": batch_size_for_length(length, args.total_tokens),
        "sequence_length": length,
    }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return {**common, **json.loads(line[len(RESULT_PREFIX) :])}
    error = completed.stderr or completed.stdout
    lowered = error.lower()
    if "out of memory" in lowered:
        status = "oom"
    elif "not installed" in lowered or "no module named" in lowered:
        status = "unavailable"
    else:
        status = "worker_crash"
    return {**common, "status": status, "error": error[-4000:]}


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
    # FlashAttention 3's settings: do_bench(warmup=3, rep=30), 1 s between runs.
    parser.add_argument("--warmup-ms", type=float, default=3.0, help="do_bench warmup time.")
    parser.add_argument("--rep-ms", type=float, default=30.0, help="do_bench timed time.")
    parser.add_argument(
        "--cooldown-s", type=float, default=1.0, help="Pause between worker processes."
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
    return parser


def _describe(result: dict[str, Any]) -> str:
    if result["status"] != "ok":
        return f"{result['status']}: {result.get('error', '')[:240]}"
    flags = []
    if result["allocator_retries"] or result["cuda_mallocs_while_timing"]:
        flags.append(
            f"allocator: {result['allocator_retries']} retries, "
            f"{result['cuda_mallocs_while_timing']} cudaMalloc while timing"
        )
    if result["host_ms_per_iter"] >= 0.8 * result["p50_ms"]:
        flags.append(f"host-bound, {result['host_ms_per_iter']:.2f} ms/iter on the CPU")
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
        except Exception as exc:  # noqa: BLE001 - record failures, keep the matrix going
            status = (
                "oom"
                if isinstance(exc, torch.cuda.OutOfMemoryError)
                else "unavailable"
                if isinstance(exc, ModuleNotFoundError)
                else "error"
            )
            result = {"status": status, "error": f"{exc}"[:4000]}
        print(RESULT_PREFIX + json.dumps(result))
        return

    if args.rep_ms <= 0 or args.warmup_ms < 0 or args.cooldown_s < 0:
        raise SystemExit("--rep-ms must be positive, --warmup-ms and --cooldown-s non-negative")
    if args.total_tokens < 1 or any(length < 1 for length in args.lengths):
        raise SystemExit("lengths and --total-tokens must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: getattr(args, key)
        for key in (
            "implementations",
            "passes",
            "lengths",
            "total_tokens",
            "head_dim",
            "dtype",
            "warmup_ms",
            "rep_ms",
            "cooldown_s",
            "device",
            "seed",
            "tuning_mode",
            "profile",
        )
    }
    results: list[dict[str, Any]] = []
    payload = {
        "schema_version": 5,
        "environment": collect_environment(),
        "configuration": configuration,
        "results": results,
    }
    cases = [
        (pass_mode, length, implementation)
        for pass_mode in args.passes
        for length in args.lengths
        for implementation in args.implementations
    ]
    for step, (pass_mode, length, implementation) in enumerate(cases, start=1):
        if step > 1:
            time.sleep(args.cooldown_s)
        result = run_subprocess(args, implementation, pass_mode, length)
        results.append(result)
        write_json(output, payload)
        print(
            f"[{step:03d}/{len(cases):03d}] {implementation:13} {pass_mode:16} "
            f"B={result['batch_size']} L={length}  {_describe(result)}",
            flush=True,
        )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
