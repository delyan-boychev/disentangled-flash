"""Utilities for FlashAttention-style unpadded token batches."""

from __future__ import annotations

from itertools import pairwise
from typing import NamedTuple

import torch


class PackedSequenceInfo(NamedTuple):
    """Validated host-side description of a ``cu_seqlens`` token batch."""

    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    max_seqlen: int


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

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor with B+1 entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must use int32 or int64")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
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


def pack_padded(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Remove right-padding and return ``(tokens, cu_seqlens, max_seqlen)``."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, L, D]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [B, L]")
    mask = attention_mask.bool()
    lengths = mask.sum(dim=1, dtype=torch.int32)
    if bool((lengths <= 0).any().item()):
        raise ValueError("empty sequences are unsupported")
    expected = torch.arange(hidden_states.size(1), device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("pack_padded supports right-padded batches only")
    packed = hidden_states[mask]
    cu_seqlens = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=mask.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = lengths.cumsum(0)
    return packed, cu_seqlens, int(lengths.max().item())


def unpack_packed(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore a right-padded tensor and its boolean padding mask."""

    if packed.ndim < 2:
        raise ValueError("packed must have shape [total_tokens, ...]")
    info = validate_cu_seqlens(cu_seqlens, packed.size(0))
    if sequence_length < info.max_seqlen:
        raise ValueError("sequence_length is smaller than the longest packed sequence")
    output = packed.new_zeros((len(info.lengths), sequence_length, *packed.shape[1:]))
    mask = torch.zeros((len(info.lengths), sequence_length), dtype=torch.bool, device=packed.device)
    for batch, (start, end) in enumerate(zip(info.offsets, info.offsets[1:])):
        length = end - start
        output[batch, :length] = packed[start:end]
        mask[batch, :length] = True
    return output, mask


__all__ = [
    "PackedSequenceInfo",
    "pack_padded",
    "unpack_packed",
    "validate_cu_seqlens",
]
