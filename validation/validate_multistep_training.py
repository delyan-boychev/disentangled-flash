"""Multi-step full-model training parity: legacy HF DeBERTa vs DisentangledFlash.

This is an integration/trajectory test on top of the kernel-level gradient parity
suite. It seeds ordinary RNGs for reproducibility but intentionally does not
force deterministic CUDA algorithms, so the execution remains representative
of normal training.
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


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and CUDA without deterministic fallbacks."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Do not replace normal training kernels with deterministic alternatives.
    torch.use_deterministic_algorithms(False)


def make_config(max_length: int):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=50_368,
        hidden_size=768,
        num_hidden_layers=15,
        num_attention_heads=12,
        intermediate_size=3200,
        hidden_act="gelu",
        # The initial trajectory test isolates the attention implementation.
        # Attention-probability dropout is not yet supported by the training
        # kernel, and disabling hidden dropout avoids unrelated RNG drift.
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
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
        # Relative error is not informative for an exactly-zero reference.
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


def make_batch(
    *,
    generator: torch.Generator,
    batch_size: int,
    length: int,
    hidden_size: int,
    mask_pattern: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        10,
        30_000,
        (batch_size, length),
        generator=generator,
        dtype=torch.long,
    )
    attention_mask = torch.ones((batch_size, length), dtype=torch.bool)
    if mask_pattern == "right":
        for batch in range(batch_size):
            keep = max(1, length - (batch + 1) * max(1, length // (batch_size + 2)))
            attention_mask[batch, keep:] = False
            input_ids[batch, keep:] = 0
    elif mask_pattern != "none":
        raise ValueError(f"unsupported mask pattern: {mask_pattern}")

    probe = torch.empty((batch_size, length, hidden_size), dtype=torch.float32)
    probe.normal_(mean=0.0, std=0.2, generator=generator)
    return (
        input_ids.to(device=device, non_blocking=True),
        attention_mask.to(device=device, non_blocking=True),
        probe.to(device=device, non_blocking=True),
    )


def autocast_context(dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def forward_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probe: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(compute_dtype):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    # Stable positive objective with gradients at every output position. The
    # same deterministic random target is used by both model trajectories.
    loss = (output.float() - probe).square().mean()
    return output, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=tuple(COMPUTE_DTYPES), default="bf16")
    parser.add_argument("--mask-pattern", choices=("none", "right"), default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="multistep_training_parity.json")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    device_index = device.index if device.index is not None else 0
    torch.cuda.set_device(device_index)
    compute_dtype = COMPUTE_DTYPES[args.dtype]

    seed_everything(args.seed)

    from transformers import DebertaV2Model, __version__ as transformers_version

    config = make_config(args.length)
    # Construct one legacy HF model, then deepcopy it so the two trajectories
    # begin bit-identically before converting only the candidate encoder.
    reference = DebertaV2Model(config)
    candidate = copy.deepcopy(reference)

    reference.to(device=device, dtype=torch.float32).train()
    candidate.to(device=device, dtype=torch.float32).train()

    from disentangled_flash import enable_deberta_training

    enable_deberta_training(
        candidate,
        assume_unpadded=args.mask_pattern == "none",
        fp32_precision="strict",
    )

    reference_names = [name for name, _ in reference.named_parameters()]
    candidate_names = [name for name, _ in candidate.named_parameters()]
    if reference_names != candidate_names:
        raise RuntimeError("reference and candidate parameter names differ after conversion")

    initial_stats = named_tensor_stats(parameter_items(reference), parameter_items(candidate))
    if initial_stats["max_abs"] != 0.0:
        raise RuntimeError(f"models are not identical at step 0: {initial_stats}")

    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_kwargs)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), **optimizer_kwargs)

    data_generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_000)
    rows: list[dict[str, Any]] = []

    for step in range(args.steps):
        input_ids, attention_mask, probe = make_batch(
            generator=data_generator,
            batch_size=args.batch_size,
            length=args.length,
            hidden_size=config.hidden_size,
            mask_pattern=args.mask_pattern,
            device=device,
        )

        reference_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)

        reference_output, reference_loss = forward_loss(
            reference,
            input_ids,
            attention_mask,
            probe,
            compute_dtype,
        )
        reference_loss.backward()
        torch.cuda.synchronize()

        candidate_output, candidate_loss = forward_loss(
            candidate,
            input_ids,
            attention_mask,
            probe,
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

        row = {
            "step": step + 1,
            "reference_loss": reference_loss.detach().item(),
            "candidate_loss": candidate_loss.detach().item(),
            "loss_abs": loss_abs,
            "loss_relative": loss_abs / loss_scale,
            "output": output_error,
            "gradients": gradient_error,
            "parameters_after_step": parameter_error,
        }
        rows.append(row)
        print(
            f"step={step + 1:>3}/{args.steps} "
            f"lossHF={row['reference_loss']:+.8e} "
            f"lossDF={row['candidate_loss']:+.8e} "
            f"dLoss={row['loss_abs']:.3e} "
            f"gRel={gradient_error['relative_l2']:.3e} "
            f"gCos={gradient_error['cosine']:.8f} "
            f"pRel={parameter_error['relative_l2']:.3e} "
            f"pMax={parameter_error['max_abs']:.3e}",
            flush=True,
        )

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
        "initial_parameter_parity": initial_stats,
        "steps": rows,
        "final": rows[-1],
    }
    output = Path(args.output).resolve()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
