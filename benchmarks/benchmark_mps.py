"""Benchmark the standalone DeBERTa attention implementations on Apple MPS."""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from disentangled_flash._prepared import InferenceDisentangledSelfAttention
from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)


def synchronize() -> None:
    torch.mps.synchronize()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(
    function,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    synchronize()

    timings = []
    for _ in range(samples):
        synchronize()
        started = time.perf_counter()
        for _ in range(inner_iterations):
            function()
        synchronize()
        timings.append((time.perf_counter() - started) * 1000.0 / inner_iterations)
    return statistics.median(timings), percentile(timings, 0.90)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=parse_csv_ints, default=[1, 8, 16, 32])
    parser.add_argument(
        "--lengths",
        type=parse_csv_ints,
        default=[16, 32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--inner-iterations", type=int, default=2)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available to this Python process")

    torch.manual_seed(17)
    device = torch.device("mps")
    dtype = torch.float16
    config = DebertaAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=-1,
        max_position_embeddings=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type="p2c|c2p",
    )

    reference = OriginalDisentangledSelfAttention(config).to(device=device, dtype=dtype).eval()
    optimized = InferenceDisentangledSelfAttention(config).to(device=device, dtype=dtype).eval()
    optimized.load_state_dict(reference.state_dict(), strict=True)
    rel_embeddings = torch.randn(
        config.position_buckets * 2,
        config.hidden_size,
        device=device,
        dtype=dtype,
    ).mul_(0.02)

    synchronize()
    prepare_started = time.perf_counter()
    optimized.prepare_for_inference(rel_embeddings)
    synchronize()
    prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

    print(
        f"device=mps dtype={dtype} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} prepare={prepare_ms:.3f} ms"
    )
    print()
    print(
        f"{'B':>3} {'L':>4} {'slots':>6} "
        f"{'reference p50':>14} {'optimized p50':>14} {'speedup':>9} "
        f"{'reference p90':>14} {'optimized p90':>14} "
        f"{'max error':>12} {'mean error':>12}"
    )

    with torch.inference_mode():
        for batch_size in args.batches:
            for length in args.lengths:
                hidden_states = torch.randn(
                    batch_size,
                    length,
                    config.hidden_size,
                    device=device,
                    dtype=dtype,
                ).mul_(0.02)
                attention_mask = torch.ones(
                    batch_size,
                    length,
                    dtype=torch.bool,
                    device=device,
                )

                reference_output = reference(
                    hidden_states,
                    _prepare_attention_mask(attention_mask, length, length),
                    rel_embeddings=rel_embeddings,
                )[0]
                prepared_plan = optimized.prepare_shape(length, device)
                optimized_output = optimized.forward_prepared(
                    hidden_states,
                    attention_mask,
                    prepared_plan,
                )[0]
                synchronize()

                difference = (reference_output.float() - optimized_output.float()).abs()
                max_error = difference.max().item()
                mean_error = difference.mean().item()

                reference_call = lambda: reference(
                    hidden_states,
                    _prepare_attention_mask(attention_mask, length, length),
                    rel_embeddings=rel_embeddings,
                )
                optimized_call = lambda: optimized.forward_prepared(
                    hidden_states,
                    attention_mask,
                    prepared_plan,
                )
                reference_p50, reference_p90 = measure(
                    reference_call,
                    args.warmup,
                    args.samples,
                    args.inner_iterations,
                )
                optimized_p50, optimized_p90 = measure(
                    optimized_call,
                    args.warmup,
                    args.samples,
                    args.inner_iterations,
                )

                active_slots = prepared_plan.active_slots.numel()
                speedup = reference_p50 / optimized_p50
                print(
                    f"{batch_size:>3} {length:>4} {active_slots:>6} "
                    f"{reference_p50:>11.3f} ms {optimized_p50:>11.3f} ms "
                    f"{speedup:>8.2f}x "
                    f"{reference_p90:>11.3f} ms {optimized_p90:>11.3f} ms "
                    f"{max_error:>12.6g} {mean_error:>12.6g}"
                )


if __name__ == "__main__":
    main()
