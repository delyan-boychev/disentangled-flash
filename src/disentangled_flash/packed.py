"""Utilities for FlashAttention-style unpadded token batches."""

from __future__ import annotations

from itertools import accumulate, pairwise
from typing import NamedTuple

import torch


class PackedSequenceInfo(NamedTuple):
    """Validated host-side description of a ``cu_seqlens`` token batch."""

    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    max_seqlen: int


def _validate_cu_seqlens_tensor(cu_seqlens: torch.Tensor) -> None:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor with B+1 entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must use int32 or int64")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")


def validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_seqlen: int | None = None,
) -> PackedSequenceInfo:
    """Validate cumulative sequence boundaries used by packed attention.

    Validation is intentionally a setup operation: boundaries are copied to the
    host once so the inference wrapper can dispatch each unpadded sequence
    without ever constructing a dense padded batch.
    """

    _validate_cu_seqlens_tensor(cu_seqlens)
    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at 0 and end at the packed token count")
    lengths = tuple(end - start for start, end in pairwise(offsets))
    if any(length <= 0 for length in lengths):
        raise ValueError("cu_seqlens must be strictly increasing; empty sequences are unsupported")
    actual_max = max(lengths)
    if max_seqlen is not None and max_seqlen != actual_max:
        raise ValueError(
            f"max_seqlen={max_seqlen} does not match the longest packed sequence ({actual_max})"
        )
    return PackedSequenceInfo(offsets, lengths, actual_max)


def resolve_packed_info(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_seqlen: int | None = None,
    packed_info: PackedSequenceInfo | None = None,
) -> PackedSequenceInfo:
    """Use trusted host metadata or validate device boundaries once.

    ``packed_info`` is intended for nested encoder calls after the public entry
    point has already validated ``cu_seqlens``. Structural checks remain on the
    hot path, but the CUDA-to-host boundary copy is not repeated for each layer.
    """

    if packed_info is None:
        return validate_cu_seqlens(cu_seqlens, total_tokens, max_seqlen)

    _validate_cu_seqlens_tensor(cu_seqlens)
    if len(packed_info.offsets) != cu_seqlens.numel():
        raise ValueError("packed_info and cu_seqlens have different batch sizes")
    if packed_info.offsets[0] != 0 or packed_info.offsets[-1] != total_tokens:
        raise ValueError("packed_info does not match the packed token count")
    if len(packed_info.lengths) + 1 != len(packed_info.offsets):
        raise ValueError("packed_info has inconsistent offsets and lengths")
    derived_lengths = tuple(end - start for start, end in pairwise(packed_info.offsets))
    if derived_lengths != packed_info.lengths:
        raise ValueError("packed_info lengths do not match its offsets")
    if any(length <= 0 for length in packed_info.lengths):
        raise ValueError("packed_info contains an empty sequence")
    if packed_info.max_seqlen != max(packed_info.lengths):
        raise ValueError("packed_info has an incorrect maximum sequence length")
    if max_seqlen is not None and packed_info.max_seqlen != max_seqlen:
        raise ValueError(
            f"max_seqlen={max_seqlen} does not match the longest packed sequence "
            f"({packed_info.max_seqlen})"
        )
    return packed_info


def pack_padded_with_info(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, PackedSequenceInfo]:
    """Remove right padding and return reusable host boundary metadata."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, L, D]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [B, L]")
    mask = attention_mask.bool()
    lengths = mask.sum(dim=1, dtype=torch.int32)
    length_values = tuple(int(value) for value in lengths.detach().cpu().tolist())
    if any(length <= 0 for length in length_values):
        raise ValueError("empty sequences are unsupported")
    expected = torch.arange(hidden_states.size(1), device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("pack_padded supports right-padded batches only")
    packed = hidden_states[mask]
    cu_seqlens = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=mask.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = lengths.cumsum(0)
    offsets = tuple(accumulate((0, *length_values)))
    info = PackedSequenceInfo(offsets, length_values, max(length_values))
    return packed, cu_seqlens, info


def pack_padded(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Remove right-padding and return ``(tokens, cu_seqlens, max_seqlen)``."""

    packed, cu_seqlens, info = pack_padded_with_info(hidden_states, attention_mask)
    return packed, cu_seqlens, info.max_seqlen


def unpack_packed(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_length: int,
    *,
    packed_info: PackedSequenceInfo | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore a right-padded tensor and its boolean padding mask."""

    if packed.ndim < 2:
        raise ValueError("packed must have shape [total_tokens, ...]")
    info = resolve_packed_info(
        cu_seqlens,
        packed.size(0),
        packed_info=packed_info,
    )
    if sequence_length < info.max_seqlen:
        raise ValueError("sequence_length is smaller than the longest packed sequence")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    positions = torch.arange(sequence_length, device=packed.device)
    mask = positions[None, :] < lengths[:, None]
    output = packed.new_zeros((len(info.lengths), sequence_length, *packed.shape[1:]))
    output[mask] = packed
    return output, mask


__all__ = [
    "PackedSequenceInfo",
    "pack_padded",
    "pack_padded_with_info",
    "unpack_packed",
    "validate_cu_seqlens",
]
