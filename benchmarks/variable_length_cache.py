"""Variable-length inference: warm-up, cache memory, output parity and latency.

Replays GLiNER2 2.1.0's forward hook: every new length re-prepares all lengths seen so
far, then activates it. Uses a randomly initialised DeBERTa-v3-base sized encoder (no
downloads). Compare two source trees, e.g. this branch against v1.1.1:

    python benchmarks/variable_length_cache.py --src src --label branch
    git worktree add /tmp/df-v111 v1.1.1
    python benchmarks/variable_length_cache.py --src /tmp/df-v111/src --label v1.1.1
    python benchmarks/variable_length_cache.py --compare v1.1.1 branch

``prepare`` time is the hook's own cost (no kernel compilation); ``forward`` time of
the first pass also includes Triton compilation of each kernel family.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

PARITY_LENGTHS = (8, 37, 128, 129, 300, 512)
LATENCY_CASES = ((1, 32), (1, 128), (8, 128), (1, 512), (8, 384))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace, out: Path) -> None:
    sys.path.insert(0, str(Path(args.src).resolve()))
    from disentangled_flash._reference import DebertaAttentionConfig, DebertaV2Encoder
    from disentangled_flash.deberta import DebertaV2OptimizedEncoder

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    config = DebertaAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        attention_head_size=64,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=512,
        max_position_embeddings=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        conv_kernel_size=0,
    )
    torch.manual_seed(0)
    source = DebertaV2Encoder(config).to(device=device, dtype=dtype)
    encoder = DebertaV2OptimizedEncoder(source, config, backend=args.backend, inference=True).eval()
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    def inputs(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(length * 1000 + batch)
        hidden = torch.randn(batch, length, 768, generator=generator).to(device, dtype)
        mask = torch.ones(batch, length, dtype=torch.long, device=device)
        if batch > 1:
            mask[1, length // 2 + 1 :] = 0
        return hidden, mask

    seen: set[int] = set()
    prepare_seconds = 0.0

    def hook(length: int) -> None:  # GLiNER2 2.1.0, gliner2/models/base.py
        nonlocal prepare_seconds
        start = time.perf_counter()
        if length not in encoder._prepared_plans:
            seen.add(length)
            encoder.prepare_for_inference(tuple(sorted(seen | set(encoder._prepared_plans))))
        encoder.activate_shape(length)
        synchronize(device)
        prepare_seconds += time.perf_counter() - start

    lengths = random.Random(0).sample(range(8, args.max_length + 1), args.lengths)
    passes = []
    with torch.inference_mode():
        for _ in range(2):
            prepare_seconds = 0.0
            synchronize(device)
            start = time.perf_counter()
            for length in lengths:
                hook(length)
                encoder(*inputs(args.batch, length))
            synchronize(device)
            passes.append((time.perf_counter() - start, prepare_seconds))
        cache = torch.cuda.memory_allocated(device) - baseline if device.type == "cuda" else 0

        outputs = {}
        for length in PARITY_LENGTHS:
            hook(length)
            output = encoder(*inputs(2, length)).last_hidden_state
            outputs[length] = output.float().cpu()

        latency = {}
        for batch, length in LATENCY_CASES:
            hook(length)
            hidden, mask = inputs(batch, length)
            for _ in range(3):
                encoder(hidden, mask)
            timings = []
            for _ in range(20):
                synchronize(device)
                start = time.perf_counter()
                encoder(hidden, mask)
                synchronize(device)
                timings.append(time.perf_counter() - start)
            latency[f"B={batch} L={length}"] = statistics.median(timings) * 1e3

    summary = {
        "label": args.label,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "backend": args.backend,
        "dtype": args.dtype,
        "lengths": args.lengths,
        "first_pass_s": passes[0][0],
        "first_pass_prepare_s": passes[0][1],
        "steady_pass_s": passes[1][0],
        "cache_mib": cache / 2**20,
        "latency_ms": latency,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, out / f"{args.label}.pt")
    (out / f"{args.label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def compare(out: Path, old: str, new: str) -> None:
    a, b = (json.loads((out / f"{label}.json").read_text()) for label in (old, new))
    print(f"{a['device']}, {a['backend']}, {a['dtype']}, {a['lengths']} lengths")
    print(f"{'':26}{old:>12}{new:>12}")
    for key in ("first_pass_s", "first_pass_prepare_s", "steady_pass_s", "cache_mib"):
        print(f"{key:26}{a[key]:12.2f}{b[key]:12.2f}")
    for case, value in a["latency_ms"].items():
        new_value = b["latency_ms"][case]
        change = 100 * (new_value / value - 1)
        print(f"latency {case:18}{value:12.2f}{new_value:12.2f}  {change:+.1f}%")
    outputs_a = torch.load(out / f"{old}.pt")
    outputs_b = torch.load(out / f"{new}.pt")
    for length, tensor in outputs_a.items():
        difference = (tensor - outputs_b[length]).abs().max().item()
        print(f"parity L={length:4d}  max |{old} - {new}| = {difference:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="src", help="source tree holding disentangled_flash")
    parser.add_argument("--label", default="branch")
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="triton", choices=("triton", "torch"))
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--lengths", type=int, default=191)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out-dir", default="variable_length_cache_results")
    args = parser.parse_args()
    if args.compare:
        compare(Path(args.out_dir), *args.compare)
    else:
        run(args, Path(args.out_dir))


if __name__ == "__main__":
    main()
