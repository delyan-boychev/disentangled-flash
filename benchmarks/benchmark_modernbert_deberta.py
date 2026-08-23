#!/usr/bin/env python3
"""Benchmark ModernBERT and DeBERTa attention implementations.

Variants:
  - modernbert: Hugging Face ModernBERT with its default local/global pattern
  - modernbert_global: same ModernBERT block with full attention in every layer
  - deberta_hf: parameter-matched Hugging Face DeBERTa-v3-style encoder
  - deberta_flash: same DeBERTa config using Knowledgator FlashDeBERTa
  - deberta_df: same DeBERTa config using DisentangledFlash

Each (variant, sequence length) is measured in a fresh subprocess.
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

VARIANTS = (
    "modernbert",
    "modernbert_global",
    "deberta_hf",
    "deberta_flash",
    "deberta_df",
)
DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
VOCAB_SIZE = 50_368
HIDDEN_SIZE = 768
NUM_HEADS = 12
MODERNBERT_LAYERS = 22
MODERNBERT_INTERMEDIATE = 1152
DEBERTA_LAYERS = 15
POSITION_BUCKETS = 256
MAX_RELATIVE_POSITIONS = 512
RESULT_PREFIX = "__BENCH_RESULT__="


def gib(x: float) -> float:
    return float(x) / (1024.0**3)


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
        config.layer_types = ["full_attention"] * config.num_hidden_layers
    return config


def make_deberta_config(max_length: int, intermediate_size: int):
    from transformers import DebertaV2Config

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
        # Keep DeBERTa-v3-base relative-position geometry while extending the
        # sequence length itself.
        max_relative_positions=MAX_RELATIVE_POSITIONS,
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
        return constructor(config)


def resolve_parameter_match(
    max_length: int,
    requested_intermediate: int | None,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import DebertaV2Model, ModernBertModel
    except ImportError as exc:
        raise SystemExit(
            "This benchmark requires PyTorch and a Transformers release containing ModernBERT."
        ) from exc

    modern = _construct_on_meta(ModernBertModel, make_modernbert_config(max_length))
    modern_params = count_parameters(modern)
    del modern
    gc.collect()

    candidates = (
        [requested_intermediate]
        if requested_intermediate is not None
        else list(range(2560, 3585, 64))
    )
    best: tuple[int, int] | None = None
    for intermediate in candidates:
        model = _construct_on_meta(
            DebertaV2Model,
            make_deberta_config(max_length, intermediate),
        )
        params = count_parameters(model)
        del model
        gc.collect()
        if best is None or abs(params - modern_params) < abs(best[1] - modern_params):
            best = (intermediate, params)

    assert best is not None
    deberta_intermediate, deberta_params = best
    return {
        "modernbert_params": modern_params,
        "deberta_params": deberta_params,
        "deberta_intermediate_size": deberta_intermediate,
        "parameter_difference_pct": 100.0 * (deberta_params - modern_params) / modern_params,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def set_modernbert_attention_backend(model: Any, backend: str) -> None:
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
        config._attn_implementation = modernbert_attn
        model = ModernBertModel(config)
        set_modernbert_attention_backend(model, modernbert_attn)
    elif variant == "deberta_flash":
        config = make_deberta_config(max_length, deberta_intermediate)
        try:
            from flashdeberta import FlashDebertaV2Model
        except ImportError as exc:
            raise RuntimeError(
                "deberta_flash requires FlashDeBERTa. Install it with: pip install flashdeberta -U"
            ) from exc
        # Direct construction keeps the exact same synthetic DeBERTa config as
        # the HF and DisentangledFlash variants.
        model = FlashDebertaV2Model(config)
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
                "deberta_df requires DisentangledFlash. From this repo run: pip install -e ."
            ) from exc
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
    device_index = requested_device.index if requested_device.index is not None else 0
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    gc.collect()
    torch.cuda.empty_cache()
    synchronize(torch, device)

    result: dict[str, Any] = {
        "variant": args.variant,
        "length": args.length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "modernbert_attn": (
            args.modernbert_attn if args.variant in {"modernbert", "modernbert_global"} else None
        ),
        "status": "ok",
    }
    model = input_ids = attention_mask = None

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
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 17)
        input_ids = torch.randint(
            10,
            min(30_000, VOCAB_SIZE - 1),
            (args.batch_size, args.length),
            generator=gen,
            dtype=torch.long,
            device=device,
        )
        # This synthetic scaling benchmark deliberately contains no padding.
        attention_mask = torch.ones(
            (args.batch_size, args.length),
            dtype=torch.long,
            device=device,
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
                "cuda_capability": ".".join(
                    map(str, torch.cuda.get_device_capability(torch.device(device)))
                ),
                "torch_version": torch.__version__,
            }
        )

        if args.variant in {"modernbert", "modernbert_global"}:
            layer_types = list(getattr(config, "layer_types", []))
            result.update(
                {
                    "layer_types": layer_types,
                    "num_full_attention_layers": sum(x == "full_attention" for x in layer_types),
                    "num_sliding_attention_layers": sum(
                        x == "sliding_attention" for x in layer_types
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
                    "deberta_implementation": {
                        "deberta_hf": "huggingface",
                        "deberta_flash": "flashdeberta",
                        "deberta_df": "disentangled_flash",
                    }[args.variant],
                    "assume_unpadded": args.variant == "deberta_df",
                }
            )

        # Warmup includes FlashAttention/Triton compilation and autotuning.
        with torch.inference_mode():
            for _ in range(args.warmup):
                out = forward_once(model, input_ids, attention_mask)
                del out
        synchronize(torch, device)
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(torch, device)

        baseline_alloc = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
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

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        with torch.inference_mode():
            for i in range(args.iters):
                starts[i].record()
                out = forward_once(model, input_ids, attention_mask)
                ends[i].record()
                del out
        synchronize(torch, device)

        latencies_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        median_ms = statistics.median(latencies_ms)
        result.update(
            {
                "latency_median_ms": median_ms,
                "latency_mean_ms": statistics.fmean(latencies_ms),
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


def run_worker_subprocess(
    args: argparse.Namespace,
    variant: str,
    length: int,
    deberta_intermediate: int,
):
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
        result["worker_stderr_tail"] = proc.stderr[-4000:]
    return result


def _ok(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("status") == "ok")


def _ratio(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    denom = b.get(key, 0)
    return None if not denom else a[key] / denom


def add_pairwise_metrics(results: list[dict[str, Any]]) -> None:
    by_key = {(r["variant"], r["length"]): r for r in results}
    for length in sorted({r["length"] for r in results}):
        mb = by_key.get(("modernbert", length))
        mb_global = by_key.get(("modernbert_global", length))
        hf = by_key.get(("deberta_hf", length))
        flash = by_key.get(("deberta_flash", length))
        df = by_key.get(("deberta_df", length))

        if _ok(hf) and _ok(flash):
            flash["speedup_vs_deberta_hf"] = hf["latency_median_ms"] / flash["latency_median_ms"]
            if hf["incremental_peak_allocated_gib"] > 0:
                flash["incremental_memory_reduction_vs_deberta_hf_pct"] = (
                    100.0
                    * (
                        hf["incremental_peak_allocated_gib"]
                        - flash["incremental_peak_allocated_gib"]
                    )
                    / hf["incremental_peak_allocated_gib"]
                )

        if _ok(hf) and _ok(df):
            df["speedup_vs_deberta_hf"] = hf["latency_median_ms"] / df["latency_median_ms"]
            df["peak_memory_reduction_vs_deberta_hf_pct"] = (
                100.0
                * (hf["peak_allocated_gib"] - df["peak_allocated_gib"])
                / hf["peak_allocated_gib"]
            )
            if hf["incremental_peak_allocated_gib"] > 0:
                df["incremental_memory_reduction_vs_deberta_hf_pct"] = (
                    100.0
                    * (hf["incremental_peak_allocated_gib"] - df["incremental_peak_allocated_gib"])
                    / hf["incremental_peak_allocated_gib"]
                )

        if _ok(flash) and _ok(df):
            df["speedup_vs_flashdeberta"] = flash["latency_median_ms"] / df["latency_median_ms"]
            df["incremental_memory_ratio_vs_flashdeberta"] = _ratio(
                df, flash, "incremental_peak_allocated_gib"
            )

        if _ok(mb) and _ok(mb_global):
            mb_global["latency_ratio_vs_modernbert_default"] = _ratio(
                mb_global, mb, "latency_median_ms"
            )
            mb_global["incremental_memory_ratio_vs_modernbert_default"] = _ratio(
                mb_global, mb, "incremental_peak_allocated_gib"
            )

        if _ok(mb) and _ok(df):
            df["latency_ratio_vs_modernbert"] = _ratio(df, mb, "latency_median_ms")
            df["incremental_memory_ratio_vs_modernbert"] = _ratio(
                df, mb, "incremental_peak_allocated_gib"
            )

        if _ok(mb_global) and _ok(df):
            df["latency_ratio_vs_modernbert_global"] = _ratio(df, mb_global, "latency_median_ms")
            df["incremental_memory_ratio_vs_modernbert_global"] = _ratio(
                df, mb_global, "incremental_peak_allocated_gib"
            )


def save_csv(path: Path, results: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in results:
        for key in row:
            if key != "timings_ms" and key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== Inference scaling summary ===")
    header = (
        f"{'variant':<18} {'L':>6} {'status':>8} {'lat(ms)':>10} "
        f"{'tok/s':>12} {'peak GiB':>10} {'inc GiB':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['variant']:<18} {r['length']:>6} {r.get('status', '?'):>8} "
            f"{fmt(r.get('latency_median_ms')):>10} "
            f"{fmt(r.get('tokens_per_s'), 0):>12} "
            f"{fmt(r.get('peak_allocated_gib'), 3):>10} "
            f"{fmt(r.get('incremental_peak_allocated_gib'), 3):>10}"
        )

    by_key = {(r["variant"], r["length"]): r for r in results}
    print("\n=== DeBERTa implementation comparison ===")
    print(f"{'L':>6} {'Flash/HF speedup':>18} {'DF/HF speedup':>15} {'DF/Flash speedup':>18}")
    print("-" * 64)
    for length in sorted({r["length"] for r in results}):
        flash = by_key.get(("deberta_flash", length), {})
        df = by_key.get(("deberta_df", length), {})
        print(
            f"{length:>6} {fmt(flash.get('speedup_vs_deberta_hf')):>18} "
            f"{fmt(df.get('speedup_vs_deberta_hf')):>15} "
            f"{fmt(df.get('speedup_vs_flashdeberta')):>18}"
        )

    print("\n=== ModernBERT all-global diagnostic ===")
    print(f"{'L':>6} {'global/default lat':>20} {'global/default inc mem':>24}")
    print("-" * 54)
    for r in results:
        if r["variant"] == "modernbert_global":
            print(
                f"{r['length']:>6} {fmt(r.get('latency_ratio_vs_modernbert_default')):>20} "
                f"{fmt(r.get('incremental_memory_ratio_vs_modernbert_default')):>24}"
            )

    print("\n=== DisentangledFlash vs all-global ModernBERT ===")
    print(f"{'L':>6} {'DF/global lat':>16} {'DF/global inc mem':>20}")
    print("-" * 46)
    for r in results:
        if r["variant"] == "deberta_df":
            print(
                f"{r['length']:>6} {fmt(r.get('latency_ratio_vs_modernbert_global')):>16} "
                f"{fmt(r.get('incremental_memory_ratio_vs_modernbert_global')):>20}"
            )


def main(args: argparse.Namespace) -> int:
    if args.worker:
        print(RESULT_PREFIX + json.dumps(worker(args), sort_keys=True))
        return 0
    if not args.lengths:
        raise SystemExit("--lengths must not be empty")
    if min(args.lengths) < 1:
        raise SystemExit("all sequence lengths must be positive")

    match = resolve_parameter_match(max(args.lengths), args.deberta_intermediate)
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
                print(f"    {res.get('status')}: {res.get('error', '')[:500]}")

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
                "All three DeBERTa variants use the same synthetic architecture/config. "
                "DisentangledFlash uses the no-padding specialization because masks are all ones. "
                "ModernBERT-global forces every ModernBERT layer to full attention."
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
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--deberta-intermediate", type=int, default=None)
    p.add_argument("--output-json", default="modernbert_deberta_scaling.json")
    p.add_argument("--output-csv", default="modernbert_deberta_scaling.csv")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=VARIANTS, default="modernbert", help=argparse.SUPPRESS)
    p.add_argument("--length", type=int, default=256, help=argparse.SUPPRESS)
    p.add_argument(
        "--deberta-intermediate-resolved",
        type=int,
        default=3200,
        help=argparse.SUPPRESS,
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
