"""Multi-step training parity: Hugging Face DeBERTa vs DisentangledFlash.

Trains both on a small structured task (each target mixes a local symbol with a
sequence-level anchor) so the loss actually falls. RNGs are seeded, but CUDA
algorithms are left non-deterministic, as in normal training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

COMPUTE_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA without deterministic fallbacks."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def make_config(max_length: int, dropout: float = 0.1):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=128_100,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max(8192, max_length),
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=512,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        conv_kernel_size=0,
    )


def tensor_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.detach().float().reshape(-1)
    candidate_float = candidate.detach().float().reshape(-1)
    difference = candidate_float - reference_float

    reference_sq = torch.sum(reference_float * reference_float, dtype=torch.float64).item()
    candidate_sq = torch.sum(candidate_float * candidate_float, dtype=torch.float64).item()
    difference_sq = torch.sum(difference * difference, dtype=torch.float64).item()
    dot = torch.sum(reference_float * candidate_float, dtype=torch.float64).item()
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    difference_norm = math.sqrt(difference_sq)

    if reference_norm == 0.0:
        relative_l2 = 0.0 if difference_norm == 0.0 else math.inf
    else:
        relative_l2 = difference_norm / reference_norm
    denominator = reference_norm * candidate_norm
    if denominator == 0.0:
        cosine = 1.0 if reference_norm == 0.0 and candidate_norm == 0.0 else 0.0
    else:
        cosine = dot / denominator

    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": difference_norm,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def named_tensor_stats(
    reference_items: Iterable[tuple[str, torch.Tensor | None]],
    candidate_items: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    reference = dict(reference_items)
    candidate = dict(candidate_items)
    if reference.keys() != candidate.keys():
        missing_candidate = sorted(reference.keys() - candidate.keys())
        missing_reference = sorted(candidate.keys() - reference.keys())
        raise RuntimeError(
            "parameter-name mismatch: "
            f"missing_candidate={missing_candidate[:8]} "
            f"missing_reference={missing_reference[:8]}"
        )

    reference_sq = 0.0
    candidate_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    worst_abs_name = "none"
    worst_abs_stats: dict[str, float] | None = None
    worst_rel_name = "none"
    worst_rel_stats: dict[str, float] | None = None
    compared = 0

    for name, ref_tensor in reference.items():
        cand_tensor = candidate[name]
        if ref_tensor is None or cand_tensor is None:
            if ref_tensor is not None or cand_tensor is not None:
                raise RuntimeError(f"gradient presence mismatch for {name}")
            continue

        stats = tensor_stats(ref_tensor, cand_tensor)
        compared += 1
        reference_sq += stats["reference_norm"] ** 2
        candidate_sq += stats["candidate_norm"] ** 2
        difference_sq += stats["difference_norm"] ** 2
        dot += stats["cosine"] * stats["reference_norm"] * stats["candidate_norm"]

        if stats["max_abs"] > max_abs:
            max_abs = stats["max_abs"]
            worst_abs_name = name
            worst_abs_stats = stats
        if stats["reference_norm"] > 0.0 and (
            worst_rel_stats is None or stats["relative_l2"] > worst_rel_stats["relative_l2"]
        ):
            worst_rel_name = name
            worst_rel_stats = stats

    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    difference_norm = math.sqrt(difference_sq)
    relative_l2 = difference_norm / reference_norm if reference_norm else math.inf
    denominator = reference_norm * candidate_norm
    cosine = dot / denominator if denominator else 0.0

    return {
        "compared_tensors": compared,
        "max_abs": max_abs,
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "difference_norm": difference_norm,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "worst_abs_name": worst_abs_name,
        "worst_abs": worst_abs_stats,
        "worst_relative_name": worst_rel_name,
        "worst_relative": worst_rel_stats,
    }


def parameter_items(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return list(model.named_parameters())


def gradient_items(model: torch.nn.Module) -> list[tuple[str, torch.Tensor | None]]:
    return [(name, parameter.grad) for name, parameter in model.named_parameters()]


def make_attention_mask(batch_size: int, length: int, pattern: str) -> torch.Tensor:
    mask = torch.ones((batch_size, length), dtype=torch.bool)
    if pattern == "none":
        return mask
    if pattern != "right":
        raise ValueError(f"unsupported mask pattern: {pattern}")
    for batch in range(batch_size):
        keep = max(1, length - (batch + 1) * max(1, length // (batch_size + 2)))
        mask[batch, keep:] = False
    return mask


def build_structured_dataset(
    *,
    generator: torch.Generator,
    dataset_batches: int,
    batch_size: int,
    length: int,
    hidden_size: int,
    mask_pattern: str,
    num_anchors: int,
    symbol_vocab_size: int,
    target_std: float,
    device: torch.device,
) -> tuple[list[Batch], int]:
    """Create a finite learnable local+contextual regression dataset."""

    anchor_table = torch.empty((num_anchors, hidden_size), dtype=torch.float32)
    anchor_table.normal_(mean=0.0, std=target_std, generator=generator)
    symbol_table = torch.empty((symbol_vocab_size, hidden_size), dtype=torch.float32)
    symbol_table.normal_(mean=0.0, std=target_std, generator=generator)

    anchor_span = max(1, length // 64)
    dataset: list[Batch] = []
    for _ in range(dataset_batches):
        anchors = torch.randint(
            0,
            num_anchors,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        )
        symbols = torch.randint(
            0,
            symbol_vocab_size,
            (batch_size, length),
            generator=generator,
            dtype=torch.long,
        )
        symbols[:, :anchor_span] = anchors[:, None] % symbol_vocab_size

        input_ids = 1000 + symbols
        input_ids[:, :anchor_span] = 10 + anchors[:, None]
        attention_mask = make_attention_mask(batch_size, length, mask_pattern)
        input_ids = input_ids.masked_fill(~attention_mask, 0)

        local_target = symbol_table[symbols]
        anchor_target = anchor_table[anchors][:, None, :]
        target = 0.5 * local_target + 0.5 * anchor_target

        dataset.append(
            (
                input_ids.to(device=device, non_blocking=True),
                attention_mask.to(device=device, non_blocking=True),
                target.to(device=device, non_blocking=True),
            )
        )
    return dataset, anchor_span


def autocast_context(dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def forward_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(compute_dtype):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    per_token = (output.float() - target).square().mean(dim=-1)
    weights = attention_mask.float()
    loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
    return output, loss


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    batch: Batch,
    compute_dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    try:
        input_ids, attention_mask, target = batch
        _, loss = forward_loss(model, input_ids, attention_mask, target, compute_dtype)
        return loss.item()
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dataset-batches", type=int, default=4)
    parser.add_argument("--num-anchors", type=int, default=32)
    parser.add_argument("--symbol-vocab-size", type=int, default=64)
    parser.add_argument("--target-std", type=float, default=0.5)
    parser.add_argument("--dtype", choices=tuple(COMPUTE_DTYPES), default="bf16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-pattern", choices=("none", "right"), default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="multistep_training_parity.json")
    parser.add_argument(
        "--tuning-mode",
        choices=("auto", "autotune", "profile_only"),
        default="auto",
    )
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--max-fixed-loss-relative", type=float, default=0.05)
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="Exit unsuccessfully if final fixed-evaluation loss differs beyond the threshold.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.dataset_batches < 1:
        raise ValueError("--dataset-batches must be >= 1")
    if args.num_anchors < 2:
        raise ValueError("--num-anchors must be >= 2")
    if args.symbol_vocab_size < 2:
        raise ValueError("--symbol-vocab-size must be >= 2")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.max_fixed_loss_relative < 0.0:
        raise ValueError("--max-fixed-loss-relative must be non-negative")
    args.profile = [str(Path(profile).expanduser().resolve()) for profile in args.profile]
    missing_profiles = [profile for profile in args.profile if not Path(profile).is_file()]
    if missing_profiles:
        raise ValueError(f"profile does not exist: {missing_profiles[0]}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    compute_dtype = COMPUTE_DTYPES[args.dtype]

    seed_everything(args.seed)

    from transformers import DebertaV2Model
    from transformers import __version__ as transformers_version

    config = make_config(args.length, args.dropout)
    reference = DebertaV2Model(config)
    candidate = copy.deepcopy(reference)

    reference.to(device=device, dtype=torch.float32).train()
    candidate.to(device=device, dtype=torch.float32).train()

    from disentangled_flash import KernelTuningOptions, enable_deberta_training

    enable_deberta_training(
        candidate,
        assume_unpadded=args.mask_pattern == "none",
        fp32_precision="strict",
        tuning=KernelTuningOptions(
            mode=args.tuning_mode,
            profile_paths=tuple(args.profile),
        ),
    )

    reference_names = [name for name, _ in reference.named_parameters()]
    candidate_names = [name for name, _ in candidate.named_parameters()]
    if reference_names != candidate_names:
        raise RuntimeError("reference and candidate parameter names differ after conversion")

    initial_stats = named_tensor_stats(parameter_items(reference), parameter_items(candidate))
    if initial_stats["max_abs"] != 0.0:
        raise RuntimeError(f"models are not identical at step 0: {initial_stats}")

    data_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000)
    dataset, anchor_span = build_structured_dataset(
        generator=data_generator,
        dataset_batches=args.dataset_batches,
        batch_size=args.batch_size,
        length=args.length,
        hidden_size=config.hidden_size,
        mask_pattern=args.mask_pattern,
        num_anchors=args.num_anchors,
        symbol_vocab_size=args.symbol_vocab_size,
        target_std=args.target_std,
        device=device,
    )

    initial_fixed_reference_loss = evaluate_loss(reference, dataset[0], compute_dtype)
    initial_fixed_candidate_loss = evaluate_loss(candidate, dataset[0], compute_dtype)

    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_kwargs)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), **optimizer_kwargs)

    rows: list[dict[str, Any]] = []
    for step in range(args.steps):
        batch_index = step % len(dataset)
        input_ids, attention_mask, target = dataset[batch_index]

        reference_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_output, reference_loss = forward_loss(
            reference,
            input_ids,
            attention_mask,
            target,
            compute_dtype,
        )
        reference_loss.backward()
        torch.cuda.synchronize()

        candidate_output, candidate_loss = forward_loss(
            candidate,
            input_ids,
            attention_mask,
            target,
            compute_dtype,
        )
        candidate_loss.backward()
        torch.cuda.synchronize()

        output_error = tensor_stats(reference_output, candidate_output)
        gradient_error = named_tensor_stats(
            gradient_items(reference),
            gradient_items(candidate),
        )
        loss_abs = abs(candidate_loss.detach().item() - reference_loss.detach().item())
        loss_scale = max(abs(reference_loss.detach().item()), torch.finfo(torch.float32).eps)

        reference_optimizer.step()
        candidate_optimizer.step()
        torch.cuda.synchronize()

        parameter_error = named_tensor_stats(
            parameter_items(reference),
            parameter_items(candidate),
        )
        fixed_reference_loss = evaluate_loss(reference, dataset[0], compute_dtype)
        fixed_candidate_loss = evaluate_loss(candidate, dataset[0], compute_dtype)
        fixed_loss_abs = abs(fixed_candidate_loss - fixed_reference_loss)

        row = {
            "step": step + 1,
            "batch_index": batch_index,
            "reference_loss": reference_loss.detach().item(),
            "candidate_loss": candidate_loss.detach().item(),
            "loss_abs": loss_abs,
            "loss_relative": loss_abs / loss_scale,
            "fixed_reference_loss_after_step": fixed_reference_loss,
            "fixed_candidate_loss_after_step": fixed_candidate_loss,
            "fixed_loss_abs_after_step": fixed_loss_abs,
            "output": output_error,
            "gradients": gradient_error,
            "parameters_after_step": parameter_error,
        }
        rows.append(row)
        print(
            f"step={step + 1:>3}/{args.steps} "
            f"batch={batch_index} "
            f"trainHF={row['reference_loss']:.6f} "
            f"trainDF={row['candidate_loss']:.6f} "
            f"fixedHF={fixed_reference_loss:.6f} "
            f"fixedDF={fixed_candidate_loss:.6f} "
            f"gRel={gradient_error['relative_l2']:.3e} "
            f"gCos={gradient_error['cosine']:.8f} "
            f"pRel={parameter_error['relative_l2']:.3e}",
            flush=True,
        )

    final_fixed_reference = rows[-1]["fixed_reference_loss_after_step"]
    final_fixed_candidate = rows[-1]["fixed_candidate_loss_after_step"]
    final_fixed_relative = abs(final_fixed_candidate - final_fixed_reference) / max(
        abs(final_fixed_reference),
        torch.finfo(torch.float32).eps,
    )
    parity_passed = final_fixed_relative <= args.max_fixed_loss_relative
    result = {
        "configuration": vars(args),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "gpu": torch.cuda.get_device_name(device_index),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "parameter_dtype": "fp32",
            "compute_dtype": args.dtype,
            "reference": "transformers.DebertaV2Model legacy attention",
            "candidate": "DisentangledFlash training attention",
        },
        "task": {
            "name": "anchor_symbol_regression",
            "description": (
                "Each token target mixes a local-symbol code with a sequence-level anchor code; "
                "the anchor is encoded in the first anchor_span tokens."
            ),
            "anchor_span": anchor_span,
            "dataset_batches": args.dataset_batches,
        },
        "initial_parameter_parity": initial_stats,
        "initial_fixed_reference_loss": initial_fixed_reference_loss,
        "initial_fixed_candidate_loss": initial_fixed_candidate_loss,
        "steps": rows,
        "final": rows[-1],
        "parity": {
            "passed": parity_passed,
            "metric": "final_fixed_loss_relative",
            "value": final_fixed_relative,
            "maximum": args.max_fixed_loss_relative,
            "reference_loss_reduction": initial_fixed_reference_loss - final_fixed_reference,
            "candidate_loss_reduction": initial_fixed_candidate_loss - final_fixed_candidate,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print(
        f"Training parity {'passed' if parity_passed else 'failed'}: "
        f"final fixed-loss relative difference={final_fixed_relative:.3e} "
        f"(maximum {args.max_fixed_loss_relative:.3e})"
    )
    if args.require_parity and not parity_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
