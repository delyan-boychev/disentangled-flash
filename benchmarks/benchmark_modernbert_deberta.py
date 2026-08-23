#!/usr/bin/env python3
"""Benchmark ModernBERT vs DeBERTa vs DeBERTa+DisentangledFlash.

This benchmark is intentionally inference-only and process-isolated per
(model variant, sequence length) measurement.  It reports:

- parameter count / theoretical parameter bytes
- median / mean / p10 / p90 latency
- sequences/s and tokens/s
- CUDA baseline allocated/reserved memory
- CUDA peak allocated/reserved memory
- incremental peak allocated/reserved memory above the loaded model+inputs
- OOM / error status

The architectures are:
  1) Hugging Face ModernBERT-base architecture with its default local/global
     attention pattern
  2) the same ModernBERT architecture forced to use full attention in every layer
     (diagnostic only; isolates the effect of ModernBERT's sparse/local pattern)
  3) parameter-matched DeBERTa-v3-style architecture (DebertaV2Model class,
     c2p+p2c, shared attention keys, bucketed relative positions)
  4) exactly the same DeBERTa model as (3), with only its encoder attention
     path replaced by DisentangledFlash.

To make parameter counts comparable, ModernBERT keeps its standard 22-layer,
768-hidden, 12-head, 1152-intermediate architecture. DeBERTa keeps hidden=768,
heads=12 and a DeBERTa-v3-style block, while using 15 layers and automatically
choosing the FFN intermediate width (multiple of 64) that best matches the
ModernBERT parameter count. This is therefore a parameter-matched DeBERTa-v3-style
architecture, not the canonical microsoft/deberta-v3-base checkpoint architecture.

Example:
  python benchmark_modernbert_deberta.py \
      --lengths 256 512 1024 2048 4096 8192 \
      --batch-size 1 \
      --dtype fp16 \
      --modernbert-attn flash_attention_2 \
      --warmup 5 \
      --iters 20 \
      --output-json encoder_scaling.json \
      --output-csv encoder_scaling.csv

For a dependency-light ModernBERT run, use --modernbert-attn sdpa.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


VARIANTS = ("modernbert", "modernbert_global", "deberta_hf", "deberta_df")
DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
VOCAB_SIZE = 50_368
HIDDEN_SIZE = 768
NUM_HEADS = 12
MODERNBERT_LAYERS = 22
MODERNBERT_INTERMEDIATE = 1152
DEBERTA_LAYERS = 15
POSITION_BUCKETS = 256
RESULT_PREFIX = "__BENCH_RESULT__="


def gib(x: int | float) -> float:
    return float(x) / (1024.0**3)


def mib(x: int | float) -> float:
    return float(x) / (1024.0**2)


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def dtype_from_name(torch: Any, name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def make_modernbert_config(max_length: int, *, all_global: bool = False):
    from transformers import ModernBertConfig

    # Use the real ModernBERT-base dimensions while spelling out the main
    # attention settings so results remain self-describing across versions.
    config = ModernBertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MODERNBERT_INTERMEDIATE,
        num_hidden_layers=MODERNBERT_LAYERS,
        num_attention_heads=NUM_HEADS,
        max_position_embeddings=max(8192, max_length),
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        local_attention=128,
    )

    if all_global:
        # Diagnostic only: preserve the complete ModernBERT block while forcing
        # every layer to full attention. This isolates how much of the default
        # ModernBERT speed/memory advantage comes from its local/global pattern.
        config.layer_types = ["full_attention"] * config.num_hidden_layers

    return config


def make_deberta_config(max_length: int, intermediate_size: int):
    from transformers import DebertaV2Config

    # This is DeBERTa-v3-style disentangled attention.  DeBERTa-v3 checkpoints
    # use the DebertaV2Model implementation in Transformers.
    return DebertaV2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=DEBERTA_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max(8192, max_length),
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=True,
        # Preserve DeBERTa-v3-base relative-position geometry while allowing
        # the encoder input itself to extend beyond 512 tokens. Leaving this at
        # -1 would make it inherit max_position_embeddings (8192 here), changing
        # the relative-bucket geometry in this long-context benchmark.
        max_relative_positions=512,
        position_buckets=POSITION_BUCKETS,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        conv_kernel_size=0,
    )


def _construct_on_meta(constructor, config):
    import torch

    try:
        with torch.device("meta"):
            return constructor(config)
    except Exception:
        # Fallback for older PyTorch/Transformers combinations.  This allocates
        # the temporary model on CPU, so meta is strongly preferred.
        return constructor(config)


def resolve_parameter_match(max_length: int, requested_intermediate: int | None) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import DebertaV2Model, ModernBertModel
    except ImportError as exc:
        raise SystemExit(
            "This script requires PyTorch and a Transformers release containing ModernBERT. "
            "Install/upgrade transformers (ModernBERT is available in recent releases)."
        ) from exc

    modern_cfg = make_modernbert_config(max_length)
    modern = _construct_on_meta(ModernBertModel, modern_cfg)
    modern_params = count_parameters(modern)
    del modern
    gc.collect()

    if requested_intermediate is not None:
        candidates = [requested_intermediate]
    else:
        # Keep the DeBERTa FFN close to its canonical 4x width while choosing
        # the closest parameter match.  64-wide steps are GPU-friendly.
        candidates = list(range(2560, 3585, 64))

    best: tuple[int, int] | None = None
    for intermediate in candidates:
        cfg = make_deberta_config(max_length, intermediate)
        model = _construct_on_meta(DebertaV2Model, cfg)
        params = count_parameters(model)
        del model
        gc.collect()
        if best is None or abs(params - modern_params) < abs(best[1] - modern_params):
            best = (intermediate, params)

    assert best is not None
    deberta_intermediate, deberta_params = best
    diff_pct = 100.0 * (deberta_params - modern_params) / modern_params

    return {
        "modernbert_params": modern_params,
        "deberta_params": deberta_params,
        "deberta_intermediate_size": deberta_intermediate,
        "parameter_difference_pct": diff_pct,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def set_modernbert_attention_backend(model: Any, backend: str) -> None:
    # Newer Transformers exposes a public runtime setter.  Older ModernBERT
    # implementations read config._attn_implementation directly.
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(backend)
    else:
        model.config._attn_implementation = backend


def build_model(
    variant: str,
    max_length: int,
    deberta_intermediate: int,
    modernbert_attn: str,
    device: str,
    dtype_name: str,
    seed: int,
):
    import torch
    from transformers import DebertaV2Model, ModernBertModel

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dtype = dtype_from_name(torch, dtype_name)

    if variant in {"modernbert", "modernbert_global"}:
        config = make_modernbert_config(
            max_length,
            all_global=(variant == "modernbert_global"),
        )
        # Set before construction for versions that choose masks/interfaces at init.
        config._attn_implementation = modernbert_attn
        model = ModernBertModel(config)
        set_modernbert_attention_backend(model, modernbert_attn)
    else:
        config = make_deberta_config(max_length, deberta_intermediate)
        model = DebertaV2Model(config)

    model.eval()
    model.to(device=device, dtype=dtype)

    if variant == "deberta_df":
        try:
            from disentangled_flash import optimize_deberta
        except ImportError as exc:
            raise RuntimeError(
                "deberta_df requires the DisentangledFlash package. From the repo, run: pip install -e ."
            ) from exc

        # Important: prepare ONLY this worker's one sequence length.  Preparing
        # all benchmark buckets would make short-length baseline memory include
        # caches belonging to the 8K case.
        optimize_deberta(
            model,
            sequence_lengths=[max_length],
            assume_unpadded=True,
        )
        model.encoder.activate_shape(max_length)

    return model, config


def synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def forward_once(model: Any, input_ids: Any, attention_mask: Any):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )


def worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise RuntimeError("this benchmark currently expects a CUDA device")

    # Bare ``cuda`` has no explicit index and torch.cuda.set_device(torch.device("cuda"))
    # can fail. Resolve it deterministically to logical cuda:0 (which is normally the
    # allocated GPU inside a Slurm job).
    device_index = requested_device.index if requested_device.index is not None else 0
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    # Make the allocator state as clean as possible inside this isolated process.
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(torch, device)

    result: dict[str, Any] = {
        "variant": args.variant,
        "length": args.length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "modernbert_attn": (
            args.modernbert_attn
            if args.variant in {"modernbert", "modernbert_global"}
            else None
        ),
        "status": "ok",
    }

    model = None
    input_ids = None
    attention_mask = None

    try:
        model, config = build_model(
            variant=args.variant,
            max_length=args.length,
            deberta_intermediate=args.deberta_intermediate_resolved,
            modernbert_attn=args.modernbert_attn,
            device=device,
            dtype_name=args.dtype,
            seed=args.seed,
        )

        gc.collect()
        torch.cuda.empty_cache()
        synchronize(torch, device)

        params = count_parameters(model)
        dtype = dtype_from_name(torch, args.dtype)
        element_size = torch.tensor([], dtype=dtype).element_size()

        # Avoid random pad/special IDs; token identity does not affect tensor shapes.
        token_high = min(30_000, VOCAB_SIZE - 1)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 17)
        input_ids = torch.randint(
            low=10,
            high=token_high,
            size=(args.batch_size, args.length),
            generator=gen,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(
            (args.batch_size, args.length), dtype=torch.long, device=device
        )

        result.update(
            {
                "params": params,
                "parameter_memory_gib": gib(params * element_size),
                "hidden_size": int(config.hidden_size),
                "num_layers": int(config.num_hidden_layers),
                "num_heads": int(config.num_attention_heads),
                "intermediate_size": int(config.intermediate_size),
                "gpu_name": torch.cuda.get_device_name(torch.device(device)),
                "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(torch.device(device)))),
                "torch_version": torch.__version__,
            }
        )

        if variant in {"modernbert", "modernbert_global"}:
            layer_types = list(getattr(config, "layer_types", []))
            result.update(
                {
                    "layer_types": layer_types,
                    "num_full_attention_layers": sum(
                        layer_type == "full_attention" for layer_type in layer_types
                    ),
                    "num_sliding_attention_layers": sum(
                        layer_type == "sliding_attention" for layer_type in layer_types
                    ),
                    "local_attention": int(getattr(config, "local_attention", 0)),
                }
            )
        else:
            result.update(
                {
                    "position_buckets": int(config.position_buckets),
                    "max_relative_positions": int(config.max_relative_positions),
                    "relative_attention": bool(config.relative_attention),
                    "share_att_key": bool(config.share_att_key),
                    "pos_att_type": list(config.pos_att_type),
                    "assume_unpadded": variant == "deberta_df",
                }
            )

        # Warm up kernel dispatch, FlashAttention/Triton compilation, and
        # DisentangledFlash autotuning.  None of this is included in timing or peaks.
        with torch.inference_mode():
            for _ in range(args.warmup):
                out = forward_once(model, input_ids, attention_mask)
                del out
        synchronize(torch, device)

        gc.collect()
        torch.cuda.empty_cache()
        synchronize(torch, device)

        # Baseline includes model + persistent implementation caches + input tensors.
        baseline_alloc = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()

        # Dedicated peak-memory pass.
        with torch.inference_mode():
            out = forward_once(model, input_ids, attention_mask)
            synchronize(torch, device)
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            del out
        synchronize(torch, device)

        result.update(
            {
                "baseline_allocated_gib": gib(baseline_alloc),
                "baseline_reserved_gib": gib(baseline_reserved),
                "peak_allocated_gib": gib(peak_alloc),
                "peak_reserved_gib": gib(peak_reserved),
                "incremental_peak_allocated_gib": gib(max(0, peak_alloc - baseline_alloc)),
                "incremental_peak_reserved_gib": gib(max(0, peak_reserved - baseline_reserved)),
            }
        )

        # Timing pass using CUDA events.  Events are created after the memory pass,
        # so event setup is not part of the reported peak allocation.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]

        with torch.inference_mode():
            for i in range(args.iters):
                starts[i].record()
                out = forward_once(model, input_ids, attention_mask)
                ends[i].record()
                del out
        synchronize(torch, device)

        latencies_ms = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
        median_ms = statistics.median(latencies_ms)
        mean_ms = statistics.fmean(latencies_ms)

        result.update(
            {
                "latency_median_ms": median_ms,
                "latency_mean_ms": mean_ms,
                "latency_p10_ms": percentile(latencies_ms, 0.10),
                "latency_p90_ms": percentile(latencies_ms, 0.90),
                "sequences_per_s": args.batch_size * 1000.0 / median_ms,
                "tokens_per_s": args.batch_size * args.length * 1000.0 / median_ms,
                "timings_ms": latencies_ms,
            }
        )

    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        result["status"] = "oom"
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            result["peak_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
            result["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
        except Exception:
            pass
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            del model, input_ids, attention_mask
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result


def run_worker_subprocess(args: argparse.Namespace, variant: str, length: int, deberta_intermediate: int):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        variant,
        "--length",
        str(length),
        "--batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--modernbert-attn",
        args.modernbert_attn,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--deberta-intermediate-resolved",
        str(deberta_intermediate),
    ]
    if args.allow_tf32:
        cmd.append("--allow-tf32")

    proc = subprocess.run(cmd, text=True, capture_output=True)
    marker_line = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            marker_line = line[len(RESULT_PREFIX) :]

    if marker_line is None:
        return {
            "variant": variant,
            "length": length,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "status": "worker_crash",
            "error": (
                f"worker exited with code {proc.returncode}; "
                f"stdout={proc.stdout[-4000:]!r}; stderr={proc.stderr[-4000:]!r}"
            ),
        }

    result = json.loads(marker_line)
    if proc.stderr.strip():
        # Keep warnings available for audit without spamming the main table.
        result["worker_stderr_tail"] = proc.stderr[-4000:]
    return result


def add_pairwise_metrics(results: list[dict[str, Any]]) -> None:
    by_key = {(r["variant"], r["length"]): r for r in results}
    for length in sorted({r["length"] for r in results}):
        hf = by_key.get(("deberta_hf", length))
        df = by_key.get(("deberta_df", length))
        mb = by_key.get(("modernbert", length))
        mb_global = by_key.get(("modernbert_global", length))

        if hf and df and hf.get("status") == "ok" and df.get("status") == "ok":
            hf_ms = hf["latency_median_ms"]
            df_ms = df["latency_median_ms"]
            df["speedup_vs_deberta_hf"] = hf_ms / df_ms

            hf_mem = hf["peak_allocated_gib"]
            df_mem = df["peak_allocated_gib"]
            df["peak_memory_reduction_vs_deberta_hf_pct"] = 100.0 * (hf_mem - df_mem) / hf_mem

            hf_inc = hf["incremental_peak_allocated_gib"]
            df_inc = df["incremental_peak_allocated_gib"]
            if hf_inc > 0:
                df["incremental_memory_reduction_vs_deberta_hf_pct"] = 100.0 * (hf_inc - df_inc) / hf_inc

        if mb and df and mb.get("status") == "ok" and df.get("status") == "ok":
            df["latency_ratio_vs_modernbert"] = df["latency_median_ms"] / mb["latency_median_ms"]
            df["peak_memory_ratio_vs_modernbert"] = df["peak_allocated_gib"] / mb["peak_allocated_gib"]

        if (
            mb
            and mb_global
            and mb.get("status") == "ok"
            and mb_global.get("status") == "ok"
        ):
            mb_global["latency_ratio_vs_modernbert_default"] = (
                mb_global["latency_median_ms"] / mb["latency_median_ms"]
            )
            mb_global["incremental_memory_ratio_vs_modernbert_default"] = (
                mb_global["incremental_peak_allocated_gib"]
                / mb["incremental_peak_allocated_gib"]
                if mb["incremental_peak_allocated_gib"] > 0
                else None
            )

        if (
            mb_global
            and df
            and mb_global.get("status") == "ok"
            and df.get("status") == "ok"
        ):
            df["latency_ratio_vs_modernbert_global"] = (
                df["latency_median_ms"] / mb_global["latency_median_ms"]
            )


def save_csv(path: Path, results: list[dict[str, Any]]) -> None:
    # Exclude the raw per-iteration timing array from CSV; it remains in JSON.
    keys: list[str] = []
    seen: set[str] = set()
    for row in results:
        for key in row:
            if key == "timings_ms":
                continue
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== Inference scaling summary ===")
    header = (
        f"{'variant':<15} {'L':>6} {'status':>8} {'lat(ms)':>10} "
        f"{'tok/s':>12} {'peak GiB':>10} {'inc GiB':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['variant']:<15} {r['length']:>6} {r.get('status','?'):>8} "
            f"{fmt(r.get('latency_median_ms')):>10} "
            f"{fmt(r.get('tokens_per_s'), 0):>12} "
            f"{fmt(r.get('peak_allocated_gib'), 3):>10} "
            f"{fmt(r.get('incremental_peak_allocated_gib'), 3):>10}"
        )

    print("\n=== DisentangledFlash vs HF DeBERTa ===")
    print(f"{'L':>6} {'speedup':>10} {'peak mem red.':>15} {'inc mem red.':>15}")
    print("-" * 50)
    for r in results:
        if r["variant"] != "deberta_df":
            continue
        print(
            f"{r['length']:>6} "
            f"{fmt(r.get('speedup_vs_deberta_hf'), 2):>10} "
            f"{(fmt(r.get('peak_memory_reduction_vs_deberta_hf_pct'), 1) + '%') if r.get('peak_memory_reduction_vs_deberta_hf_pct') is not None else '-':>15} "
            f"{(fmt(r.get('incremental_memory_reduction_vs_deberta_hf_pct'), 1) + '%') if r.get('incremental_memory_reduction_vs_deberta_hf_pct') is not None else '-':>15}"
        )

    print("\n=== ModernBERT all-global diagnostic ===")
    print(f"{'L':>6} {'global/default lat':>20} {'global/default inc mem':>24}")
    print("-" * 54)
    for r in results:
        if r["variant"] != "modernbert_global":
            continue
        print(
            f"{r['length']:>6} "
            f"{fmt(r.get('latency_ratio_vs_modernbert_default'), 2):>20} "
            f"{fmt(r.get('incremental_memory_ratio_vs_modernbert_default'), 2):>24}"
        )


def main(args: argparse.Namespace) -> int:
    if args.worker:
        res = worker(args)
        print(RESULT_PREFIX + json.dumps(res, sort_keys=True))
        return 0

    if not args.lengths:
        raise SystemExit("--lengths must not be empty")
    if min(args.lengths) < 1:
        raise SystemExit("all sequence lengths must be positive")

    max_length = max(args.lengths)
    match = resolve_parameter_match(max_length, args.deberta_intermediate)
    deberta_intermediate = int(match["deberta_intermediate_size"])

    print("=== Parameter match ===")
    print(f"ModernBERT: {match['modernbert_params']:,} params")
    print(
        f"DeBERTa:    {match['deberta_params']:,} params "
        f"({match['parameter_difference_pct']:+.3f}% vs ModernBERT)"
    )
    print(
        f"DeBERTa dimensions: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}, "
        f"layers={DEBERTA_LAYERS}, intermediate={deberta_intermediate}"
    )
    print(
        f"ModernBERT dimensions: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}, "
        f"layers={MODERNBERT_LAYERS}, intermediate={MODERNBERT_INTERMEDIATE}"
    )

    results: list[dict[str, Any]] = []
    total = len(VARIANTS) * len(args.lengths)
    n = 0

    # Length-major ordering makes it easy to compare all three models as results arrive.
    for length in args.lengths:
        for variant in VARIANTS:
            n += 1
            print(f"[{n:02d}/{total:02d}] {variant}  L={length}", flush=True)
            res = run_worker_subprocess(args, variant, length, deberta_intermediate)
            results.append(res)
            if res.get("status") == "ok":
                print(
                    f"    median={res['latency_median_ms']:.2f} ms, "
                    f"peak={res['peak_allocated_gib']:.3f} GiB, "
                    f"incremental={res['incremental_peak_allocated_gib']:.3f} GiB"
                )
            else:
                print(f"    {res.get('status')}: {res.get('error', '')[:300]}")

    add_pairwise_metrics(results)

    payload = {
        "metadata": {
            "created_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "lengths": args.lengths,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "modernbert_attn": args.modernbert_attn,
            "allow_tf32": args.allow_tf32,
            "seed": args.seed,
            "parameter_match": match,
            "note": (
                "Each model/length pair is measured in a fresh subprocess. "
                "DeBERTa-HF and DeBERTa-DF use the same architecture/config and deterministic seed; "
                "DF changes only the inference attention implementation and uses the valid all-token "
                "no-padding specialization because this synthetic benchmark supplies all-ones masks. "
                "ModernBERT-global is a diagnostic variant that forces every ModernBERT layer to full attention."
            ),
        },
        "results": results,
    }

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(csv_path, results)

    print_summary(results)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument(
        "--modernbert-attn",
        choices=("flash_attention_2", "sdpa", "flex_attention", "eager"),
        default="flash_attention_2",
        help="Attention backend used by ModernBERT. flash_attention_2 is the intended primary comparison.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument(
        "--deberta-intermediate",
        type=int,
        default=None,
        help="Override automatic parameter matching for DeBERTa FFN width.",
    )
    p.add_argument("--output-json", default="modernbert_deberta_scaling.json")
    p.add_argument("--output-csv", default="modernbert_deberta_scaling.csv")

    # Internal worker flags.  Users normally do not set these directly.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=VARIANTS, default="modernbert", help=argparse.SUPPRESS)
    p.add_argument("--length", type=int, default=256, help=argparse.SUPPRESS)
    p.add_argument(
        "--deberta-intermediate-resolved", type=int, default=3200, help=argparse.SUPPRESS
    )

    args = p.parse_args()
    if args.warmup < 1:
        p.error("--warmup must be >= 1")
    if args.iters < 1:
        p.error("--iters must be >= 1")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
