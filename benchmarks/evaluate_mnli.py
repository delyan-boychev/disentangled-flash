"""Evaluate DeBERTa inference backends on the real GLUE/MNLI dataset."""

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from disentangled_flash.deberta import enable_deberta_inference
from disentangled_flash.packed import pack_padded_with_info, unpack_packed
from disentangled_flash.tuning import KernelTuningOptions

IMPLEMENTATIONS = ("base", "torch", "triton", "flashdeberta")
LAYOUTS = ("padded", "packed")


@dataclass(frozen=True)
class DatasetBatch:
    inputs: dict[str, torch.Tensor]
    dataset_labels: torch.Tensor


@dataclass(frozen=True)
class Variant:
    implementation: str
    layout: str

    @property
    def name(self) -> str:
        return f"{self.implementation}-{self.layout}"


@dataclass(frozen=True)
class EvaluationResult:
    logits: torch.Tensor
    run_times_ms: tuple[float, ...]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument(
        "--implementations",
        type=parse_csv,
        default=list(IMPLEMENTATIONS),
        help="Comma-separated candidates; base is always included as the parity reference.",
    )
    parser.add_argument(
        "--layouts",
        type=parse_csv,
        default=list(LAYOUTS),
        help="Comma-separated layouts for the DF PyTorch and Triton implementations.",
    )
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--bucket", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("validation_matched", "validation_mismatched"),
        default="validation_matched",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum dataset rows; zero evaluates the complete split.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Measured full-dataset passes. The default processes every row exactly once.",
    )
    parser.add_argument(
        "--fp32-precision",
        choices=("strict", "fast"),
        default="strict",
    )
    parser.add_argument(
        "--tuning-mode",
        choices=("auto", "autotune", "profile_only"),
        default="auto",
    )
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--parity-atol", type=float)
    parser.add_argument("--parity-rtol", type=float)
    parser.add_argument(
        "--require-full-parity",
        action="store_true",
        help="Exit unsuccessfully after reporting every variant if any parity check fails.",
    )
    return parser.parse_args(argv)


def requested_variants(implementations: list[str], layouts: list[str]) -> tuple[Variant, ...]:
    unknown_implementations = set(implementations) - set(IMPLEMENTATIONS)
    unknown_layouts = set(layouts) - set(LAYOUTS)
    if unknown_implementations or unknown_layouts:
        raise ValueError(
            f"invalid selections: implementations={sorted(unknown_implementations)}, "
            f"layouts={sorted(unknown_layouts)}"
        )

    variants = [Variant("base", "padded")]
    for implementation in implementations:
        if implementation == "base":
            continue
        if implementation == "flashdeberta":
            variants.append(Variant("flashdeberta", "packed"))
            continue
        variants.extend(Variant(implementation, layout) for layout in layouts)
    return tuple(dict.fromkeys(variants))


def batch_ranges(total: int, batch_size: int) -> tuple[tuple[int, int], ...]:
    """Return a non-overlapping, exhaustive partition of dataset rows."""

    if total < 1:
        raise ValueError("the selected MNLI split is empty")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    return tuple((start, min(start + batch_size, total)) for start in range(0, total, batch_size))


def prepare_dataset(
    tokenizer,
    *,
    split: str,
    limit: int,
    batch_size: int,
    bucket: int,
    device: torch.device,
) -> tuple[list[DatasetBatch], tuple[str, ...]]:
    """Tokenize and transfer the complete selected split before measurement."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "GLUE/MNLI evaluation requires datasets; install with: pip install -e '.[benchmark,hf]'"
        ) from exc

    if limit < 0:
        raise ValueError("--limit must be >= 0")
    dataset = load_dataset("nyu-mll/glue", "mnli", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    encoded = tokenizer(
        list(dataset["premise"]),
        list(dataset["hypothesis"]),
        padding="max_length",
        truncation=True,
        max_length=bucket,
        return_tensors="pt",
    )
    prepared = {
        key: value.to(device=device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    batches = [
        DatasetBatch(
            inputs={key: value[start:end] for key, value in prepared.items()},
            dataset_labels=labels[start:end],
        )
        for start, end in batch_ranges(len(dataset), batch_size)
    ]
    label_names = tuple(str(name) for name in dataset.features["label"].names)
    return batches, label_names


def labels_for_model(
    batches: list[DatasetBatch],
    label_names: tuple[str, ...],
    label2id: dict[str, int],
) -> torch.Tensor:
    normalized = {str(name).casefold(): int(index) for name, index in label2id.items()}
    try:
        dataset_to_model = torch.tensor(
            [normalized[name.casefold()] for name in label_names],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(
            f"model labels {sorted(label2id)} do not cover MNLI labels {label_names}"
        ) from exc
    return dataset_to_model[torch.cat([batch.dataset_labels for batch in batches])]


def enable_backend(
    backbone: torch.nn.Module,
    *,
    backend: str,
    bucket: int,
    fp32_precision: str,
    layout: str,
    tuning_mode: str,
    profile_paths: tuple[str, ...],
) -> None:
    enable_deberta_inference(
        backbone,
        backend=backend,
        sequence_lengths=None if layout == "packed" else [bucket],
        fp32_precision=fp32_precision,
        tuning=KernelTuningOptions(mode=tuning_mode, profile_paths=profile_paths),
    )


def forward_model(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    layout: str,
) -> torch.Tensor:
    if layout == "padded":
        return model(**inputs, return_dict=True).logits

    backbone = getattr(model, model.base_model_prefix)
    attention_mask = inputs["attention_mask"]
    embedding_output = backbone.embeddings(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs.get("token_type_ids"),
        mask=attention_mask,
    )
    packed_embeddings, cu_seqlens, packed_info = pack_padded_with_info(
        embedding_output,
        attention_mask,
    )
    encoder_output = backbone.encoder.forward_packed(
        packed_embeddings,
        cu_seqlens,
        packed_info.max_seqlen,
        output_hidden_states=False,
        return_dict=True,
        packed_info=packed_info,
    )
    sequence_output, _ = unpack_packed(
        encoder_output.last_hidden_state,
        cu_seqlens,
        attention_mask.size(1),
        packed_info=packed_info,
    )
    return model.classifier(model.dropout(model.pooler(sequence_output)))


def load_hf_model(model_name: str, *, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device=device).eval()


def load_variant_model(
    variant: Variant,
    model_name: str,
    *,
    bucket: int,
    fp32_precision: str,
    tuning_mode: str,
    profile_paths: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
):
    if variant.implementation == "flashdeberta":
        try:
            from flashdeberta import FlashDebertaV2ForSequenceClassification
        except ImportError as exc:
            raise RuntimeError(
                "FlashDeBERTa is required; install with: pip install -e '.[benchmark,hf]'"
            ) from exc
        model = FlashDebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        return model.to(device=device).eval()

    model = load_hf_model(model_name, device=device, dtype=dtype)
    if variant.implementation != "base":
        enable_backend(
            getattr(model, model.base_model_prefix),
            backend=variant.implementation,
            bucket=bucket,
            fp32_precision=fp32_precision,
            layout=variant.layout,
            tuning_mode=tuning_mode,
            profile_paths=profile_paths,
        )
    return model


def execution_layout(variant: Variant) -> str:
    # FlashDeBERTa consumes padded tensors and the mask, then packs internally.
    return "padded" if variant.implementation == "flashdeberta" else variant.layout


@torch.inference_mode()
def evaluate_runs(
    model: torch.nn.Module,
    batches: list[DatasetBatch],
    *,
    description: str,
    layout: str,
    runs: int,
) -> EvaluationResult:
    """Measure full passes from their first forward, with no warmup."""

    from tqdm.auto import tqdm

    run_times_ms: list[float] = []
    first_logits: torch.Tensor | None = None
    for run in range(runs):
        logits: list[torch.Tensor] = []
        progress = tqdm(
            batches,
            desc=f"{description} run {run + 1}/{runs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            miniters=max(1, len(batches) // 100),
        )
        with progress:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for batch in progress:
                logits.append(forward_model(model, batch.inputs, layout=layout).detach())
            torch.cuda.synchronize()
            run_times_ms.append((time.perf_counter() - started) * 1000.0)
        print(f"  run {run + 1:02d}/{runs:02d}: {run_times_ms[-1]:.3f} ms", flush=True)
        if first_logits is None:
            first_logits = torch.cat(logits).float().cpu()

    if first_logits is None:
        raise RuntimeError("evaluation produced no logits")
    return EvaluationResult(first_logits, tuple(run_times_ms))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def has_full_parity(decision_mismatches: int) -> bool:
    """Return whether every classification decision matches the reference."""

    return decision_mismatches == 0


def print_performance(name: str, result: EvaluationResult, *, examples: int) -> None:
    times = list(result.run_times_ms)
    mean_ms = statistics.mean(times)
    print(
        f"{name:<20} first={times[0]:>11.3f} ms  mean={mean_ms:>11.3f} ms  "
        f"p50={statistics.median(times):>11.3f} ms  p90={percentile(times, 0.90):>11.3f} ms  "
        f"throughput={examples / (mean_ms / 1000.0):>10.2f} examples/s"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this MNLI evaluation")
    if args.bucket < 1 or args.batch_size < 1 or args.runs < 1:
        raise ValueError("--bucket, --batch-size, and --runs must be >= 1")

    variants = requested_variants(args.implementations, args.layouts)
    device = torch.device("cuda")
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    atol = (
        args.parity_atol
        if args.parity_atol is not None
        else (2e-2 if args.dtype == "fp16" else 1e-4)
    )
    rtol = (
        args.parity_rtol
        if args.parity_rtol is not None
        else (2e-2 if args.dtype == "fp16" else 1e-4)
    )

    from transformers import AutoTokenizer

    print(f"Preparing GLUE/MNLI {args.split}; all data preparation is outside timing...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    batches, label_names = prepare_dataset(
        tokenizer,
        split=args.split,
        limit=args.limit,
        batch_size=args.batch_size,
        bucket=args.bucket,
        device=device,
    )
    examples = sum(batch.inputs["input_ids"].size(0) for batch in batches)
    valid_tokens = sum(int(batch.inputs["attention_mask"].sum()) for batch in batches)
    padded_tokens = examples * args.bucket
    print(
        f"examples={examples} batches={len(batches)} valid_tokens={valid_tokens} "
        f"padded_tokens={padded_tokens} padding={1.0 - valid_tokens / padded_tokens:.2%}"
    )
    print(f"runs={args.runs}; no warmup; the first forward and any first-use compilation are timed")

    results: dict[Variant, EvaluationResult] = {}
    targets: torch.Tensor | None = None
    for variant in variants:
        print(f"\nLoading and evaluating {variant.name}...")
        model = load_variant_model(
            variant,
            args.model,
            bucket=args.bucket,
            fp32_precision=args.fp32_precision,
            tuning_mode=args.tuning_mode,
            profile_paths=tuple(args.profile),
            device=device,
            dtype=dtype,
        )
        if targets is None:
            targets = labels_for_model(batches, label_names, model.config.label2id)
        results[variant] = evaluate_runs(
            model,
            batches,
            description=variant.name,
            layout=execution_layout(variant),
            runs=args.runs,
        )
        del model
        torch.cuda.empty_cache()

    base_variant = Variant("base", "padded")
    base = results[base_variant]
    if targets is None:
        raise RuntimeError("base evaluation did not produce aligned labels")
    base_predictions = base.logits.argmax(dim=-1)
    base_probabilities = base.logits.softmax(dim=-1)

    print("\n" + "=" * 112)
    print("GLUE/MNLI RESULTS — FULL DATASET PARITY AFTER CLASSIFICATION DECISION")
    print("=" * 112)
    parity_failures: list[str] = []
    for variant, result in results.items():
        predictions = result.logits.argmax(dim=-1)
        mismatches = int((predictions != base_predictions).sum())
        accuracy = float((predictions == targets).float().mean())
        logit_error = (result.logits - base.logits).abs()
        probability_error = (result.logits.softmax(dim=-1) - base_probabilities).abs()
        logits_close = torch.allclose(result.logits, base.logits, atol=atol, rtol=rtol)
        full_parity = has_full_parity(mismatches)

        print_performance(variant.name, result, examples=examples)
        print(
            f"  accuracy={accuracy:.6f}  full_parity={full_parity}  "
            f"decision_mismatches={mismatches}/{examples}  logits_close={logits_close}"
        )
        print(
            f"  logits max/mean abs={float(logit_error.max()):.8g}/"
            f"{float(logit_error.mean()):.8g}  probabilities max/mean abs="
            f"{float(probability_error.max()):.8g}/{float(probability_error.mean()):.8g}"
        )
        if not full_parity:
            parity_failures.append(variant.name)

    if parity_failures:
        print(f"\nVariants without full parity: {', '.join(parity_failures)}")
        if args.require_full_parity:
            raise SystemExit(1)
    else:
        print("\nFull parity passed for every evaluated implementation.")


if __name__ == "__main__":
    main()
