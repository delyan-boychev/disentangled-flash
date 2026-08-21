"""Isolated CUDA benchmark for standalone DeBERTa attention/encoder backends.

By default this benchmarks the reference, prepared PyTorch, and Triton
implementations under the four requested execution configurations:

* FP16 eager
* FP16 ``torch.compile``
* FP32 eager
* FP32 ``torch.compile``

Each implementation/dtype/execution combination runs in a fresh process.  All
implementations receive identically seeded embedding-derived inputs, while each
measured iteration receives a different token batch and padding pattern.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch

from disentangled_flash.deberta import DebertaV2InferenceEncoder
from disentangled_flash._prepared import InferenceDisentangledSelfAttention
from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash.kernel import TritonInferenceDisentangledSelfAttention


IMPLEMENTATIONS = {
    "original": OriginalDisentangledSelfAttention,
    "optimized": InferenceDisentangledSelfAttention,
    "triton": TritonInferenceDisentangledSelfAttention,
}
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def configure_fp32(precision: str) -> None:
    torch.set_float32_matmul_precision("highest" if precision == "strict" else "high")
    try:
        torch.backends.cuda.matmul.fp32_precision = (
            "ieee" if precision == "strict" else "tf32"
        )
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = precision == "fast"


def compile_isolated(
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compile one bucket without sharing Dynamo's per-code-object budget.

    PyTorch 2.13 added ``isolate_recompiles=True`` for this exact factory
    pattern.  Older releases use the officially documented code-object cloning
    workaround so separate length buckets cannot exhaust one another's default
    eight-entry recompile limit.
    """

    compile_kwargs: dict[str, Any] = {
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if "isolate_recompiles" in inspect.signature(torch.compile).parameters:
        compile_kwargs["isolate_recompiles"] = True
    elif isinstance(function, types.FunctionType):
        clone = types.FunctionType(
            function.__code__.replace(),
            function.__globals__,
            name=function.__name__,
            argdefs=function.__defaults__,
            closure=function.__closure__,
        )
        clone.__kwdefaults__ = function.__kwdefaults__
        function = clone
    return torch.compile(function, **compile_kwargs)


def initialize_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, (torch.nn.Linear, torch.nn.Conv1d)):
                child.weight.normal_(mean=0.0, std=0.02, generator=generator)
                if child.bias is not None:
                    child.bias.zero_()
            elif isinstance(child, torch.nn.Embedding):
                child.weight.normal_(mean=0.0, std=0.02, generator=generator)
            elif isinstance(child, torch.nn.LayerNorm):
                if child.weight is not None:
                    child.weight.fill_(1.0)
                if child.bias is not None:
                    child.bias.zero_()


def make_embedding_table(
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.empty(vocab_size, hidden_size, dtype=torch.float32, device="cpu")
    table.normal_(mean=0.0, std=0.02, generator=generator)
    table[0].zero_()
    return table.to(device=device, dtype=dtype)


def make_inputs(
    embedding_table: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    minimum_length_fraction: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    minimum_length = max(1, int(sequence_length * minimum_length_fraction))
    for run in range(count):
        generator = torch.Generator(device="cpu").manual_seed(seed + run)
        token_ids = torch.randint(
            1,
            embedding_table.size(0),
            (batch_size, sequence_length),
            generator=generator,
            device="cpu",
        )
        lengths = torch.randint(
            minimum_length,
            sequence_length + 1,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        positions = torch.arange(sequence_length, device="cpu")[None, :]
        mask_cpu = positions < lengths[:, None]
        token_ids.masked_fill_(~mask_cpu, 0)

        token_ids = token_ids.to(device=embedding_table.device)
        mask = mask_cpu.to(device=embedding_table.device)
        hidden_states = embedding_table[token_ids].contiguous()
        batches.append((hidden_states, mask))
    return batches


def make_models(
    scope: str,
    implementation: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    hidden_size: int,
    num_attention_heads: int,
    attention_head_size: int,
    fp32_precision: str,
    sequence_lengths: list[int],
    num_hidden_layers: int,
    intermediate_size: int,
    conv_kernel_size: int,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.Tensor | None,
    float,
]:
    config = DebertaAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_head_size=attention_head_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=-1,
        max_position_embeddings=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        conv_kernel_size=conv_kernel_size,
    )
    if scope == "encoder":
        reference = DebertaV2Encoder(config)
    else:
        reference = OriginalDisentangledSelfAttention(config)
    initialize_parameters(reference, seed)
    if scope == "encoder":
        source = DebertaV2Encoder(config)
        source.load_state_dict(reference.state_dict(), strict=True)
        if implementation == "original":
            target = source
        else:
            target = DebertaV2InferenceEncoder(
                source,
                config,
                backend=implementation,
                fp32_precision=fp32_precision,
            )
    elif implementation == "triton":
        target = TritonInferenceDisentangledSelfAttention(
            config,
            fp32_precision=fp32_precision,
        )
        target.load_state_dict(reference.state_dict(), strict=True)
    elif implementation == "optimized":
        target = InferenceDisentangledSelfAttention(config)
        target.load_state_dict(reference.state_dict(), strict=True)
    else:
        target = OriginalDisentangledSelfAttention(config)
        target.load_state_dict(reference.state_dict(), strict=True)
    reference = reference.to(device=device, dtype=dtype).eval()
    target = target.to(device=device, dtype=dtype).eval()

    rel_embeddings = None
    if scope == "attention":
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        rel_embeddings = torch.empty(
            config.position_buckets * 2,
            config.hidden_size,
            dtype=torch.float32,
            device="cpu",
        )
        rel_embeddings.normal_(mean=0.0, std=0.02, generator=generator)
        rel_embeddings = rel_embeddings.to(device=device, dtype=dtype)

    preparation_ms = 0.0
    if implementation != "original":
        torch.cuda.synchronize()
        started = time.perf_counter()
        if scope == "encoder":
            target.prepare_for_inference(sequence_lengths)
        else:
            target.prepare_for_inference(rel_embeddings)
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - started) * 1000.0
    return reference, target, rel_embeddings, preparation_ms


def make_callable(
    scope: str,
    implementation: str,
    target: torch.nn.Module,
    rel_embeddings: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if scope == "encoder":
        if implementation != "original":
            target.activate_shape(sequence_length)

        def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return target(
                hidden_states,
                attention_mask,
                output_hidden_states=False,
            ).last_hidden_state

        return call

    if implementation == "original":

        def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return target(
                hidden_states,
                _prepare_attention_mask(
                    attention_mask,
                    hidden_states.size(1),
                    hidden_states.size(1),
                ),
                rel_embeddings=rel_embeddings,
            )[0]

        return call

    plan = target.prepare_shape(sequence_length, device)

    def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return target.forward_prepared(hidden_states, attention_mask, plan)[0]

    return call


def measure_calls(
    call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    log_prefix: str,
) -> tuple[list[float], list[int]]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in batches]
    ends = [torch.cuda.Event(enable_timing=True) for _ in batches]
    for index, ((hidden_states, attention_mask), start, end) in enumerate(
        zip(batches, starts, ends),
        start=1,
    ):
        print(
            f"{log_prefix} measured run {index:02d}/{len(batches):02d} "
            f"valid_tokens={int(attention_mask.sum().item())}",
            flush=True,
        )
        start.record()
        call(hidden_states, attention_mask)
        end.record()
    torch.cuda.synchronize()
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    valid_tokens = [int(mask.sum().item()) for _, mask in batches]
    for index, milliseconds in enumerate(timings, start=1):
        print(
            f"{log_prefix} completed run {index:02d}/{len(batches):02d} "
            f"latency={milliseconds:.4f} ms",
            flush=True,
        )
    return timings, valid_tokens


def compare_outputs(
    scope: str,
    reference: torch.nn.Module,
    target_call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rel_embeddings: torch.Tensor | None,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for hidden_states, attention_mask in batches:
        if scope == "encoder":
            reference_output = reference(
                hidden_states,
                attention_mask,
                output_hidden_states=False,
            ).last_hidden_state
        else:
            reference_output = reference(
                hidden_states,
                _prepare_attention_mask(
                    attention_mask,
                    hidden_states.size(1),
                    hidden_states.size(1),
                ),
                rel_embeddings=rel_embeddings,
            )[0]
        target_output = target_call(hidden_states, attention_mask)
        difference = (reference_output.float() - target_output.float()).abs()
        maximum = max(maximum, difference.max().item())
        absolute_sum += difference.sum().item()
        element_count += difference.numel()
    return maximum, absolute_sum / element_count


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation: {args.implementation}")
    if args.dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {args.dtype}")
    if args.hidden_size != args.num_attention_heads * args.attention_head_size:
        raise ValueError(
            "hidden_size must equal num_attention_heads * attention_head_size"
        )
    if args.implementation == "triton":
        # Triton documents this switch as the supported way to report tuning
        # time and the winning configuration for every new tuning key.
        os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

    configure_fp32(args.fp32_precision)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    reference, target, rel_embeddings, preparation_ms = make_models(
        args.scope,
        args.implementation,
        dtype,
        device,
        args.seed,
        args.hidden_size,
        args.num_attention_heads,
        args.attention_head_size,
        args.fp32_precision,
        args.lengths,
        args.num_hidden_layers,
        args.intermediate_size,
        args.conv_kernel_size,
    )
    embedding_table = make_embedding_table(
        args.vocab_size,
        args.hidden_size,
        dtype,
        device,
        args.seed + 2,
    )

    metadata = {
        "implementation": args.implementation,
        "scope": args.scope,
        "dtype": args.dtype,
        "execution": args.execution,
        "compile_mode": args.compile_mode if args.execution == "compile" else None,
        "fullgraph": args.fullgraph if args.execution == "compile" else None,
        "dynamic_batch": args.dynamic if args.execution == "compile" else None,
        "preparation_ms": preparation_ms,
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "attention_head_size": args.attention_head_size,
        "qkv_projection": "fused",
        "fp32_precision": args.fp32_precision,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "conv_kernel_size": args.conv_kernel_size,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print(
        f"worker scope={args.scope} implementation={args.implementation} dtype={args.dtype} "
        f"execution={args.execution} gpu={metadata['gpu']} "
        f"prepare={preparation_ms:.3f} ms",
        flush=True,
    )

    results = []
    with torch.inference_mode():
        for sequence_length in args.lengths:
            eager_call = make_callable(
                args.scope,
                args.implementation,
                target,
                rel_embeddings,
                sequence_length,
                device,
            )
            if args.execution == "compile":
                call = compile_isolated(
                    eager_call,
                    mode=args.compile_mode,
                    fullgraph=args.fullgraph,
                    dynamic=args.dynamic,
                )
            else:
                call = eager_call

            compile_ms = 0.0
            if args.execution == "compile":
                probe_batch = max(args.batches)
                probe = make_inputs(
                    embedding_table,
                    probe_batch,
                    sequence_length,
                    1,
                    args.seed + sequence_length * 100_000 - 1,
                    args.minimum_length_fraction,
                )[0]
                print(
                    f"[{args.implementation}/{args.dtype}/{args.execution} "
                    f"L={sequence_length}] compiling with dynamic batch probe "
                    f"B={probe_batch}",
                    flush=True,
                )
                torch.cuda.synchronize()
                compile_started = time.perf_counter()
                call(*probe)
                torch.cuda.synchronize()
                compile_ms = (time.perf_counter() - compile_started) * 1000.0
                print(
                    f"[{args.implementation}/{args.dtype}/{args.execution} "
                    f"L={sequence_length}] compile completed in {compile_ms:.3f} ms",
                    flush=True,
                )
                del probe

            for batch_size in args.batches:
                shape_seed = args.seed + sequence_length * 100_000 + batch_size * 1_000
                warmup_batches = make_inputs(
                    embedding_table,
                    batch_size,
                    sequence_length,
                    args.warmup,
                    shape_seed,
                    args.minimum_length_fraction,
                )
                measured_batches = make_inputs(
                    embedding_table,
                    batch_size,
                    sequence_length,
                    args.samples,
                    shape_seed + 100,
                    args.minimum_length_fraction,
                )
                log_prefix = (
                    f"[{args.implementation}/{args.dtype}/{args.execution} "
                    f"B={batch_size} L={sequence_length}]"
                )

                torch.cuda.synchronize()
                setup_started = time.perf_counter()
                for index, (hidden_states, attention_mask) in enumerate(
                    warmup_batches,
                    start=1,
                ):
                    print(
                        f"{log_prefix} warmup {index:02d}/{args.warmup:02d}",
                        flush=True,
                    )
                    call(hidden_states, attention_mask)
                torch.cuda.synchronize()
                setup_ms = (time.perf_counter() - setup_started) * 1000.0

                torch.cuda.reset_peak_memory_stats()
                timings, valid_tokens = measure_calls(call, measured_batches, log_prefix)
                peak_allocated_bytes = torch.cuda.max_memory_allocated()
                peak_reserved_bytes = torch.cuda.max_memory_reserved()
                max_error, mean_error = compare_outputs(
                    args.scope,
                    reference,
                    call,
                    rel_embeddings,
                    measured_batches,
                )
                total_seconds = sum(timings) / 1000.0
                result = {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "samples": args.samples,
                    "p50_ms": statistics.median(timings),
                    "p90_ms": percentile(timings, 0.90),
                    "p95_ms": percentile(timings, 0.95),
                    "mean_ms": statistics.mean(timings),
                    "docs_per_second": batch_size * len(timings) / total_seconds,
                    "valid_tokens_per_second": sum(valid_tokens) / total_seconds,
                    "mean_valid_tokens_per_batch": statistics.mean(valid_tokens),
                    "max_abs_error": max_error,
                    "mean_abs_error": mean_error,
                    "compile_ms_for_length": compile_ms,
                    "warmup_ms": setup_ms,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_reserved_bytes": peak_reserved_bytes,
                }
                results.append(result)
                print(
                    f"{log_prefix} summary p50={result['p50_ms']:.4f} ms "
                    f"p90={result['p90_ms']:.4f} ms "
                    f"docs/s={result['docs_per_second']:.2f} "
                    f"valid_tok/s={result['valid_tokens_per_second']:.2f} "
                    f"max_error={max_error:.7g} mean_error={mean_error:.7g}",
                    flush=True,
                )

                del warmup_batches, measured_batches
                torch.cuda.empty_cache()

    payload = {"metadata": metadata, "results": results}
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def print_summary(payloads: list[dict[str, Any]]) -> None:
    baseline = {}
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["implementation"] != "original":
            continue
        for result in payload["results"]:
            key = (
                metadata["dtype"],
                metadata["execution"],
                result["batch_size"],
                result["sequence_length"],
            )
            baseline[key] = result["p50_ms"]

    print()
    scopes = sorted({payload["metadata"]["scope"] for payload in payloads})
    print(f"Final CUDA {'/'.join(scopes)} summary")
    print(
        f"{'dtype':>5} {'execution':>9} {'implementation':>14} {'B':>3} {'L':>4} "
        f"{'p50 ms':>10} {'p90 ms':>10} {'speedup':>9} "
        f"{'docs/s':>12} {'valid tok/s':>14} {'max error':>12}"
    )
    for payload in payloads:
        metadata = payload["metadata"]
        for result in payload["results"]:
            key = (
                metadata["dtype"],
                metadata["execution"],
                result["batch_size"],
                result["sequence_length"],
            )
            reference_ms = baseline.get(key)
            speedup = reference_ms / result["p50_ms"] if reference_ms is not None else 1.0
            print(
                f"{metadata['dtype']:>5} {metadata['execution']:>9} "
                f"{metadata['implementation']:>14} "
                f"{result['batch_size']:>3} {result['sequence_length']:>4} "
                f"{result['p50_ms']:>10.4f} {result['p90_ms']:>10.4f} "
                f"{speedup:>8.2f}x {result['docs_per_second']:>12.2f} "
                f"{result['valid_tokens_per_second']:>14.2f} "
                f"{result['max_abs_error']:>12.6g}"
            )


def run_parent(args: argparse.Namespace) -> None:
    invalid_implementations = set(args.implementations) - set(IMPLEMENTATIONS)
    invalid_dtypes = set(args.dtypes) - set(DTYPES)
    invalid_executions = set(args.executions) - {"eager", "compile"}
    if invalid_implementations or invalid_dtypes or invalid_executions:
        raise ValueError(
            f"invalid selections: implementations={sorted(invalid_implementations)}, "
            f"dtypes={sorted(invalid_dtypes)}, executions={sorted(invalid_executions)}"
        )

    payloads = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="deberta_cuda_benchmark_") as temporary_dir:
        for dtype in args.dtypes:
            for execution in args.executions:
                for implementation in args.implementations:
                    worker_output = Path(temporary_dir) / (
                        f"{implementation}_{dtype}_{execution}.json"
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.benchmark_cuda",
                        "--worker",
                        "--scope",
                        args.scope,
                        "--implementation",
                        implementation,
                        "--dtype",
                        dtype,
                        "--execution",
                        execution,
                        "--worker-output",
                        str(worker_output),
                        "--batches",
                        ",".join(str(value) for value in args.batches),
                        "--lengths",
                        ",".join(str(value) for value in args.lengths),
                        "--warmup",
                        str(args.warmup),
                        "--samples",
                        str(args.samples),
                        "--compile-mode",
                        args.compile_mode,
                        "--vocab-size",
                        str(args.vocab_size),
                        "--minimum-length-fraction",
                        str(args.minimum_length_fraction),
                        "--seed",
                        str(args.seed),
                        "--hidden-size",
                        str(args.hidden_size),
                        "--num-attention-heads",
                        str(args.num_attention_heads),
                        "--attention-head-size",
                        str(args.attention_head_size),
                        "--fp32-precision",
                        args.fp32_precision,
                        "--num-hidden-layers",
                        str(args.num_hidden_layers),
                        "--intermediate-size",
                        str(args.intermediate_size),
                        "--conv-kernel-size",
                        str(args.conv_kernel_size),
                    ]
                    command.append("--fullgraph" if args.fullgraph else "--no-fullgraph")
                    command.append("--dynamic" if args.dynamic else "--static")
                    print()
                    print(
                        f"Starting worker: implementation={implementation} dtype={dtype} "
                        f"execution={execution}",
                        flush=True,
                    )
                    completed = subprocess.run(command, cwd=os.getcwd(), check=False)
                    if completed.returncode != 0:
                        failures.append(
                            {
                                "implementation": implementation,
                                "dtype": dtype,
                                "execution": execution,
                                "returncode": completed.returncode,
                            }
                        )
                        continue
                    payloads.append(json.loads(worker_output.read_text(encoding="utf-8")))

    aggregate = {
        "requested": {
            "implementations": args.implementations,
            "scope": args.scope,
            "dtypes": args.dtypes,
            "executions": args.executions,
            "batches": args.batches,
            "lengths": args.lengths,
            "warmup": args.warmup,
            "samples": args.samples,
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "attention_head_size": args.attention_head_size,
            "qkv_projection": "fused",
            "fp32_precision": args.fp32_precision,
            "num_hidden_layers": args.num_hidden_layers,
            "intermediate_size": args.intermediate_size,
            "conv_kernel_size": args.conv_kernel_size,
        },
        "workers": payloads,
        "failures": failures,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print_summary(payloads)
    print()
    print(f"Saved complete results to {output_path}")
    if failures:
        print(f"Failed workers: {json.dumps(failures)}")
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["attention", "encoder"], default="attention")
    parser.add_argument(
        "--implementations",
        type=parse_csv,
        default=["original", "optimized", "triton"],
    )
    parser.add_argument("--dtypes", type=parse_csv, default=["fp16", "fp32"])
    parser.add_argument("--executions", type=parse_csv, default=["eager", "compile"])
    parser.add_argument("--batches", type=parse_csv_ints, default=[1, 8, 16, 32])
    parser.add_argument(
        "--lengths",
        type=parse_csv_ints,
        default=[16, 32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--attention-head-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument(
        "--fp32-precision",
        choices=["strict", "fast"],
        default="strict",
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--minimum-length-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="attention_cuda_results.json")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=True)
    parser.add_argument("--static", dest="dynamic", action="store_false")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=sorted(IMPLEMENTATIONS), help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=sorted(DTYPES), help=argparse.SUPPRESS)
    parser.add_argument("--execution", choices=["eager", "compile"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if not all([args.implementation, args.dtype, args.execution, args.worker_output]):
            raise SystemExit("worker mode requires implementation, dtype, execution, and output")
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
