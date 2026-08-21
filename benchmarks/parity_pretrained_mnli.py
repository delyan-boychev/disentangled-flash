"""Task-level parity + speed smoke test for the Triton DeBERTa-v2/v3 encoder.

Runs an official DeBERTa-v2 model fine-tuned on MNLI twice:
  1. untouched Hugging Face reference
  2. same checkpoint with only the DeBERTa encoder replaced by our inference backend

It compares:
  * task predictions
  * logits / probabilities
  * final encoder hidden states
  * end-to-end sequence-classification forward latency

Tokenization and model loading are intentionally excluded from timing.

Example:
    python parity_pretrained_mnli.py

Optional:
    python parity_pretrained_mnli.py --dtype fp32
    python parity_pretrained_mnli.py --backend optimized
    python parity_pretrained_mnli.py --warmup 20 --iterations 100
"""

from __future__ import annotations

import argparse
import statistics

import torch

from disentangled_flash.deberta import enable_deberta_inference

EXAMPLES = [
    (
        "A dog is running through a field.",
        "An animal is running.",
        "ENTAILMENT",
    ),
    (
        "A man is sleeping on the couch.",
        "The man is awake and standing.",
        "CONTRADICTION",
    ),
    (
        "A woman is reading a book.",
        "The book is about astronomy.",
        "NEUTRAL",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="microsoft/deberta-v2-xlarge-mnli",
        help="HF sequence-classification checkpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=("triton", "optimized"),
        default="triton",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=64,
        help="Fixed padded sequence length / prepared encoder bucket.",
    )
    parser.add_argument(
        "--fp32-precision",
        choices=("strict", "fast"),
        default="strict",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def enable_backend(
    backbone: torch.nn.Module,
    *,
    backend: str,
    bucket: int,
    fp32_precision: str,
) -> None:
    """Enable one prepared inference backend; QKV fusion is unconditional."""

    enable_deberta_inference(
        backbone,
        backend=backend,
        sequence_lengths=[bucket],
        fp32_precision=fp32_precision,
    )


@torch.inference_mode()
def run_model(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    return (
        output.logits.detach().float().cpu(),
        output.hidden_states[-1].detach().float().cpu(),
    )


@torch.inference_mode()
def benchmark_model(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float]:
    """Return p50_ms, p90_ms, mean_ms for full model forward."""

    # Important: benchmark exactly the production-like forward, without
    # output_hidden_states so hidden-state collection does not distort latency.
    def forward_once() -> None:
        model(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
        )

    for _ in range(warmup):
        forward_once()

    torch.cuda.synchronize()

    timings_ms: list[float] = []

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        forward_once()
        end_events[i].record()

    torch.cuda.synchronize()

    for start, end in zip(start_events, end_events):
        timings_ms.append(float(start.elapsed_time(end)))

    return (
        statistics.median(timings_ms),
        percentile(timings_ms, 0.90),
        statistics.mean(timings_ms),
    )


def load_model(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    return model.to(device=device).eval()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this parity test")

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    device = torch.device("cuda")
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"model:      {args.model}")
    print(f"backend:    {args.backend}")
    print(f"dtype:      {dtype}")
    print(f"gpu:        {torch.cuda.get_device_name(device)}")
    print(f"bucket:     {args.bucket}")
    print(f"batch size: {args.batch_size}")
    print(f"warmup:     {args.warmup}")
    print(f"iterations: {args.iterations}")
    print()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    selected_examples = [EXAMPLES[index % len(EXAMPLES)] for index in range(args.batch_size)]
    premises = [premise for premise, _, _ in selected_examples]
    hypotheses = [hypothesis for _, hypothesis, _ in selected_examples]

    encoded = tokenizer(
        premises,
        hypotheses,
        padding="max_length",
        truncation=True,
        max_length=args.bucket,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device=device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    batch_size = inputs["input_ids"].size(0)

    # ------------------------------------------------------------------
    # Untouched Hugging Face reference.
    # ------------------------------------------------------------------
    print("Loading / running Hugging Face reference...")
    reference = load_model(args.model, device=device, dtype=dtype)

    id2label = {int(index): label for index, label in reference.config.id2label.items()}

    reference_logits, reference_hidden = run_model(reference, inputs)

    reference_p50, reference_p90, reference_mean = benchmark_model(
        reference,
        inputs,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    reference_probs = reference_logits.softmax(dim=-1)

    del reference
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Same checkpoint, replacing only the DeBERTa encoder.
    # ------------------------------------------------------------------
    print(f"Loading / running {args.backend} encoder...")
    candidate = load_model(args.model, device=device, dtype=dtype)

    base_model_prefix = candidate.base_model_prefix
    backbone = getattr(candidate, base_model_prefix)

    enable_backend(
        backbone,
        backend=args.backend,
        bucket=args.bucket,
        fp32_precision=args.fp32_precision,
    )

    candidate_logits, candidate_hidden = run_model(candidate, inputs)

    candidate_p50, candidate_p90, candidate_mean = benchmark_model(
        candidate,
        inputs,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    candidate_probs = candidate_logits.softmax(dim=-1)

    # ------------------------------------------------------------------
    # Task-level parity.
    # ------------------------------------------------------------------
    logit_error = (reference_logits - candidate_logits).abs()
    prob_error = (reference_probs - candidate_probs).abs()
    hidden_error = (reference_hidden - candidate_hidden).abs()

    reference_prediction = reference_logits.argmax(dim=-1)
    candidate_prediction = candidate_logits.argmax(dim=-1)

    print()
    print("=" * 92)
    print("MNLI TASK PREDICTIONS")
    print("=" * 92)

    all_predictions_match = True

    for index, (premise, hypothesis, expected) in enumerate(selected_examples):
        ref_id = int(reference_prediction[index])
        cand_id = int(candidate_prediction[index])

        ref_label = id2label[ref_id]
        cand_label = id2label[cand_id]

        all_predictions_match &= ref_id == cand_id

        ref_conf = float(reference_probs[index, ref_id])
        cand_conf = float(candidate_probs[index, cand_id])

        print(f"\ncase {index + 1}: expected semantic class ~ {expected}")
        print(f"  premise:    {premise}")
        print(f"  hypothesis: {hypothesis}")
        print(f"  reference:  {ref_label:<14} p={ref_conf:.8f}")
        print(f"  {args.backend:<10}: {cand_label:<14} p={cand_conf:.8f}")
        print(f"  max logit delta: {float(logit_error[index].max()):.8g}")
        print(f"  max prob delta:  {float(prob_error[index].max()):.8g}")

    print()
    print("=" * 92)
    print("NUMERICAL PARITY")
    print("=" * 92)
    print(f"predictions identical:       {all_predictions_match}")
    print(f"logits max abs error:        {float(logit_error.max()):.8g}")
    print(f"logits mean abs error:       {float(logit_error.mean()):.8g}")
    print(f"probabilities max abs error: {float(prob_error.max()):.8g}")
    print(f"probabilities mean abs err:  {float(prob_error.mean()):.8g}")
    print(f"hidden max abs error:        {float(hidden_error.max()):.8g}")
    print(f"hidden mean abs error:       {float(hidden_error.mean()):.8g}")

    # ------------------------------------------------------------------
    # Full-task speed.
    # ------------------------------------------------------------------
    reference_eps = batch_size / (reference_p50 / 1000.0)
    candidate_eps = batch_size / (candidate_p50 / 1000.0)

    print()
    print("=" * 92)
    print("FULL MNLI TASK SPEED")
    print("tokenization and model loading excluded")
    print("=" * 92)

    print(
        f"{'reference':<12} "
        f"p50={reference_p50:>9.3f} ms  "
        f"p90={reference_p90:>9.3f} ms  "
        f"mean={reference_mean:>9.3f} ms  "
        f"throughput={reference_eps:>10.2f} examples/s"
    )

    print(
        f"{args.backend:<12} "
        f"p50={candidate_p50:>9.3f} ms  "
        f"p90={candidate_p90:>9.3f} ms  "
        f"mean={candidate_mean:>9.3f} ms  "
        f"throughput={candidate_eps:>10.2f} examples/s"
    )

    p50_speedup = reference_p50 / candidate_p50
    p90_speedup = reference_p90 / candidate_p90

    print()
    print(f"p50 speedup: {p50_speedup:.4f}x")
    print(f"p90 speedup: {p90_speedup:.4f}x")

    if not all_predictions_match:
        raise SystemExit("\nTask-level parity FAILED: at least one predicted MNLI label changed")

    print("\nTask-level parity PASSED: all predicted labels are identical.")


if __name__ == "__main__":
    main()
