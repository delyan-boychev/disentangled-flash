"""Relative-position index plans for DeBERTa attention.

They depend only on the bucketing config, length, and device, so every layer
can share one cache.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ._reference import make_log_bucket_position

# cuBLAS only uses its fast Hopper GEMMs for 16-byte aligned rows. The active
# slot count is 2L-1 for short sequences, so position tables get zero rows up to
# a multiple of 8; the kernels never index them.
POSITION_SLOT_ALIGNMENT = 8


def aligned_slot_count(slot_count: int) -> int:
    return -(-slot_count // POSITION_SLOT_ALIGNMENT) * POSITION_SLOT_ALIGNMENT


def pad_position_table(table: torch.Tensor | None) -> torch.Tensor | None:
    """Pad a [..., slots, D] projected position table to an aligned slot count."""

    if table is None:
        return None
    extra = aligned_slot_count(table.size(-2)) - table.size(-2)
    if not extra:
        return table
    return torch.nn.functional.pad(table, (0, 0, 0, extra))


class SharedPositionIndexPlan(NamedTuple):
    """Compact ``O(L)`` position plan consumed by the Triton kernel."""

    sequence_length: int
    active_slots: torch.Tensor
    delta_to_local: torch.Tensor


class SharedDensePositionIndexPlan(NamedTuple):
    """One model-wide dense gather map for the prepared PyTorch backend."""

    compact: SharedPositionIndexPlan
    pair_to_local: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.compact.sequence_length

    @property
    def active_slots(self) -> torch.Tensor:
        return self.compact.active_slots


def canonical_device(
    requested: torch.device | str | None,
    resident: torch.device,
) -> torch.device:
    """Resolve devices such as ``cuda``/``mps`` to the resident device."""

    resolved = torch.device(requested) if requested is not None else resident
    if resolved.type == resident.type and resolved.index is None:
        return resident
    return resolved


class SharedPositionPlanCache:
    """Position-index tensors shared by all encoder layers.

    compact() never builds an L x L tensor. dense() builds one int64 map for the
    PyTorch path, shared across layers.
    """

    def __init__(
        self,
        *,
        position_buckets: int,
        max_relative_positions: int,
        position_embedding_size: int,
        uses_position_bias: bool = True,
    ) -> None:
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.position_embedding_size = position_embedding_size
        self.uses_position_bias = uses_position_bias
        self._compact: dict[tuple[int, str], SharedPositionIndexPlan] = {}
        self._dense: dict[tuple[int, str], SharedDensePositionIndexPlan] = {}

    def clear(self) -> None:
        self._compact.clear()
        self._dense.clear()

    def _key(
        self,
        sequence_length: int,
        device: torch.device | str,
    ) -> tuple[int, str, torch.device]:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        resolved = torch.device(device)
        return sequence_length, str(resolved), resolved

    @torch.no_grad()
    def compact(
        self,
        sequence_length: int,
        device: torch.device | str,
    ) -> SharedPositionIndexPlan:
        """Return the compact delta LUT for one sequence length."""

        length, device_key, resolved_device = self._key(sequence_length, device)
        cache_key = length, device_key
        cached = self._compact.get(cache_key)
        if cached is not None:
            return cached

        if self.uses_position_bias:
            deltas = torch.arange(-(length - 1), length, dtype=torch.long)
            if self.position_buckets > 0:
                deltas = make_log_bucket_position(
                    deltas,
                    self.position_buckets,
                    self.max_relative_positions,
                ).to(torch.long)
            global_slots = torch.clamp(
                deltas + self.position_embedding_size,
                0,
                self.position_embedding_size * 2 - 1,
            )
            active_slots = torch.unique(global_slots, sorted=True)
            delta_to_local = torch.searchsorted(active_slots, global_slots).to(torch.int32)
        else:
            active_slots = torch.empty(0, dtype=torch.long)
            delta_to_local = torch.zeros(length * 2 - 1, dtype=torch.int32)

        plan = SharedPositionIndexPlan(
            sequence_length=length,
            active_slots=active_slots.to(device=resolved_device),
            delta_to_local=delta_to_local.to(device=resolved_device),
        )
        self._compact[cache_key] = plan
        return plan

    @torch.no_grad()
    def physical(self, position_ids: torch.Tensor) -> SharedPositionIndexPlan:
        """Build a per-example [B, N, N] slot map from position_ids. Not cached."""
        if position_ids.ndim != 2 or not all(position_ids.shape):
            raise ValueError("position_ids must have nonempty shape [B, N]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must be int32 or int64")
        ids = position_ids.to(torch.int64)
        deltas = ids[:, :, None] - ids[:, None, :]
        if self.uses_position_bias:
            if self.position_buckets > 0:
                deltas = make_log_bucket_position(
                    deltas, self.position_buckets, self.max_relative_positions
                ).to(torch.long)
            slots = (deltas + self.position_embedding_size).clamp(
                0, self.position_embedding_size * 2 - 1
            )
            active, inverse = torch.unique(slots, sorted=True, return_inverse=True)
            lookup = inverse.to(torch.int32).contiguous()
        else:
            active = ids.new_empty(0)
            lookup = torch.zeros_like(deltas, dtype=torch.int32)
        return SharedPositionIndexPlan(position_ids.shape[1], active, lookup)

    @torch.no_grad()
    def dense(
        self,
        sequence_length: int,
        device: torch.device | str,
    ) -> SharedDensePositionIndexPlan:
        """Return the shared dense gather map for prepared PyTorch attention."""

        length, device_key, resolved_device = self._key(sequence_length, device)
        cache_key = length, device_key
        cached = self._dense.get(cache_key)
        if cached is not None:
            return cached

        compact = self.compact(length, resolved_device)
        positions = torch.arange(length, dtype=torch.long)
        delta_indices = positions[:, None] - positions[None, :] + length - 1
        pair_to_local = compact.delta_to_local.cpu()[delta_indices].to(
            device=resolved_device,
            dtype=torch.long,
        )
        plan = SharedDensePositionIndexPlan(
            compact=compact,
            pair_to_local=pair_to_local.unsqueeze(0).unsqueeze(0),
        )
        self._dense[cache_key] = plan
        return plan


__all__ = [
    "SharedDensePositionIndexPlan",
    "SharedPositionIndexPlan",
    "SharedPositionPlanCache",
    "canonical_device",
]
