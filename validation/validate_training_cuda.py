"""Gradient parity validation for the DisentangledFlash training backend on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash.packed import pack_padded_with_info
from disentangled_flash.training import TritonTrainingDisentangledSelfAttention

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def initialize_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.empty(parameter.shape, dtype=torch.float32)
            values.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.copy_(values.to(dtype=parameter.dtype))


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    difference = candidate_float - reference_float
    absolute_difference = difference.abs()

    reference_flat = reference_float.reshape(-1)
    candidate_flat = candidate_float.reshape(-1)
    difference_flat = difference.reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    difference_norm = torch.linalg.vector_norm(difference_flat)
    eps = torch.finfo(torch.float32).eps

    relative_l2 = difference_norm / torch.clamp(reference_norm, min=eps)
    denominator = reference_norm * candidate_norm
    if denominator.item() == 0.0:
        cosine = 1.0 if reference_norm.item() == 0.0 and candidate_norm.item() == 0.0 else 0.0
    else:
        cosine = torch.dot(reference_flat, candidate_flat) / denominator

    return {
        "max_abs": absolute_difference.max().item(),
        "mean_abs": absolute_difference.mean().item(),
        "relative_l2": relative_l2.item(),
        "cosine": float(cosine),
    }


def make_mask(batch_size: int, length: int, pattern: str, device: torch.device) -> torch.Tensor:
    mask = torch.ones((batch_size, length), dtype=torch.bool, device=device)
    if pattern == "none":
        return mask
    if pattern != "right":
        raise ValueError(f"unsupported mask pattern: {pattern}")
    for batch in range(batch_size):
        keep = max(1, length - (batch + 1) * max(1, length // (batch_size + 2)))
        mask[batch, keep:] = False
    return mask


def run_case(
    *,
    length: int,
    batch_size: int,
    dtype: torch.dtype,
    dtype_name: str,
    mask_pattern: str,
    fp32_precision: str,
    assume_unpadded: bool,
    layout: str,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    num_heads = 4
    head_dim = 64
    hidden_size = num_heads * head_dim
    config = DebertaAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_size=head_dim,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=max(2048, length),
        max_relative_positions=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd="none",
    )

    reference = OriginalDisentangledSelfAttention(config)
    initialize_parameters(reference, seed)
    candidate = TritonTrainingDisentangledSelfAttention(
        config,
        fp32_precision=fp32_precision,
        assume_unpadded=assume_unpadded,
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    reference = reference.to(device=device, dtype=dtype).train()
    candidate = candidate.to(device=device, dtype=dtype).train()

    generator = torch.Generator(device="cpu").manual_seed(seed + length * 17 + batch_size)
    hidden_cpu = torch.empty((batch_size, length, hidden_size), dtype=torch.float32)
    hidden_cpu.normal_(mean=0.0, std=0.2, generator=generator)
    relative_cpu = torch.empty((512, hidden_size), dtype=torch.float32)
    relative_cpu.normal_(mean=0.0, std=0.02, generator=generator)
    probe_cpu = torch.empty((batch_size, length, hidden_size), dtype=torch.float32)
    probe_cpu.normal_(mean=0.0, std=0.2, generator=generator)

    hidden_ref = hidden_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    hidden_candidate_padded = hidden_cpu.to(device=device, dtype=dtype)
    relative_ref = relative_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    relative_candidate = relative_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    probe = probe_cpu.to(device=device, dtype=torch.float32)
    mask = make_mask(batch_size, length, mask_pattern, device)

    reference_output = reference(
        hidden_ref,
        _prepare_attention_mask(mask, length, length),
        rel_embeddings=relative_ref,
    )[0]
    if layout == "packed":
        hidden_candidate, cu_seqlens, packed_info = pack_padded_with_info(
            hidden_candidate_padded,
            mask,
        )
        hidden_candidate = hidden_candidate.detach().requires_grad_(True)
        candidate_output = candidate.forward_packed(
            hidden_candidate,
            cu_seqlens,
            packed_info.max_seqlen,
            rel_embeddings=relative_candidate,
            packed_info=packed_info,
        )[0]
        reference_for_loss = reference_output[mask]
        probe_for_loss = probe[mask]
    else:
        hidden_candidate = hidden_candidate_padded.requires_grad_(True)
        candidate_output = candidate(
            hidden_candidate,
            mask,
            rel_embeddings=relative_candidate,
        )[0]
        reference_for_loss = reference_output
        probe_for_loss = probe

    reference_loss = (reference_for_loss.float() * probe_for_loss).sum()
    candidate_loss = (candidate_output.float() * probe_for_loss).sum()
    reference_loss.backward()
    candidate_loss.backward()
    torch.cuda.synchronize()

    parameter_errors: dict[str, dict[str, float]] = {}
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in reference.named_parameters():
        candidate_parameter = candidate_parameters[name]
        if parameter.grad is None or candidate_parameter.grad is None:
            continue
        parameter_errors[name] = error_stats(parameter.grad, candidate_parameter.grad)

    worst_parameter = max(
        parameter_errors.items(),
        key=lambda item: item[1]["max_abs"],
        default=(
            "none",
            {"max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0, "cosine": 1.0},
        ),
    )
    worst_relative_parameter = max(
        parameter_errors.items(),
        key=lambda item: item[1]["relative_l2"],
        default=(
            "none",
            {"max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0, "cosine": 1.0},
        ),
    )
    row: dict[str, object] = {
        "length": length,
        "batch_size": batch_size,
        "dtype": dtype_name,
        "mask_pattern": mask_pattern,
        "layout": layout,
        "assume_unpadded": assume_unpadded,
        "output": error_stats(reference_for_loss, candidate_output),
        "hidden_grad": error_stats(
            hidden_ref.grad[mask] if layout == "packed" else hidden_ref.grad,
            hidden_candidate.grad,
        ),
        "relative_grad": error_stats(relative_ref.grad, relative_candidate.grad),
        "worst_parameter": worst_parameter[0],
        "worst_parameter_error": worst_parameter[1],
        "worst_relative_parameter": worst_relative_parameter[0],
        "worst_relative_parameter_error": worst_relative_parameter[1],
        "parameter_errors": parameter_errors,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=parse_csv_ints, default=[32, 64, 129, 257, 512])
    parser.add_argument("--batches", type=parse_csv_ints, default=[1, 2])
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp32")
    parser.add_argument("--mask-pattern", choices=("none", "right"), default="none")
    parser.add_argument("--fp32-precision", choices=("strict", "fast"), default="strict")
    parser.add_argument("--assume-unpadded", action="store_true")
    parser.add_argument("--layouts", nargs="+", choices=("padded", "packed"), default=["padded", "packed"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", default="training_cuda_validation.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.assume_unpadded and args.mask_pattern != "none":
        raise ValueError("--assume-unpadded requires --mask-pattern none")

    rows = []
    for length in args.lengths:
        for batch_size in args.batches:
            for layout in args.layouts:
                row = run_case(
                    length=length,
                    batch_size=batch_size,
                    dtype=DTYPES[args.dtype],
                    dtype_name=args.dtype,
                    mask_pattern=args.mask_pattern,
                    fp32_precision=args.fp32_precision,
                    assume_unpadded=args.assume_unpadded,
                    layout=layout,
                    seed=args.seed,
                )
                rows.append(row)
                print(
                    f"L={length:>4} B={batch_size} {layout:>6} "
                    f"{args.dtype:>4} {args.mask_pattern:>5} "
                    f"out={row['output']['max_abs']:.7g} "
                    f"dX={row['hidden_grad']['max_abs']:.7g} "
                    f"dXrel={row['hidden_grad']['relative_l2']:.3e} "
                    f"dXcos={row['hidden_grad']['cosine']:.8f} "
                    f"dRel={row['relative_grad']['max_abs']:.7g} "
                    f"worst={row['worst_parameter']}:"
                    f"{row['worst_parameter_error']['max_abs']:.7g} "
                    f"rel={row['worst_parameter_error']['relative_l2']:.3e} "
                    f"cos={row['worst_parameter_error']['cosine']:.8f} "
                    f"worstRel={row['worst_relative_parameter']}:"
                    f"{row['worst_relative_parameter_error']['relative_l2']:.3e}",
                    flush=True,
                )
                torch.cuda.empty_cache()

    output = Path(args.output).resolve()
    output.write_text(
        json.dumps({"configuration": vars(args), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
