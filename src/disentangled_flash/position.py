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


class PositionWindow(NamedTuple):
    """An aligned row window of a full projected position table.

    ``delta_to_local`` indexes rows of ``table.narrow(-2, start, width)``, so every
    length reads a view of one table instead of its own padded copy.
    """

    start: int
    width: int
    delta_to_local: torch.Tensor


def position_window(
    active_slots: torch.Tensor,
    delta_to_local: torch.Tensor,
    table_rows: int,
) -> PositionWindow | None:
    """Fit ``active_slots`` into an aligned window of a ``table_rows``-row table.

    The window is ``aligned_slot_count(len(active_slots))`` rows wide and covers the
    active slots; rows past them are real table rows that the kernels never index.
    Returns None when the aligned window does not fit in the table.
    """

    slot_count = int(active_slots.numel())
    width = aligned_slot_count(slot_count)
    if slot_count == 0 or width > table_rows:
        return None
    first = int(active_slots[0])
    start = min(first, table_rows - width)
    shift = first - start
    lookup = delta_to_local if not shift else (delta_to_local + shift).to(delta_to_local.dtype)
    return PositionWindow(start, width, lookup)


def dense_pair_map(delta_to_local: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Expand a compact ``[2L - 1]`` slot lookup to the ``[1, 1, L, L]`` gather map.

    Built on the lookup's device and not cached: it is ``O(L^2)`` and cheap to rebuild,
    so keeping one per sequence length would grow memory with every length seen.
    """

    positions = torch.arange(sequence_length, device=delta_to_local.device)
    delta_indices = positions[:, None] - positions[None, :] + sequence_length - 1
    return delta_to_local.long()[delta_indices][None, None]


class GradientBand(NamedTuple):
    """Distance columns that let backward write position gradients without atomics.

    Distance index u = i - j + L - 1 maps to gradient column u - low. Inside the
    band (0 < column < high - low) every distance has its own column, so each
    program writes each of its (row, distance) gradients once. Distances at or
    past either end share the first or last slot and are summed in registers
    into columns 0 and high - low. column_slots maps every column to its local
    slot so the gradient GEMMs can fold columns back into slots.
    """

    low: int
    high: int
    column_slots: torch.Tensor


def gradient_band(delta_to_local: torch.Tensor, device: torch.device) -> GradientBand:
    """Build the band from a compact [2L - 1] slot lookup (monotone in distance)."""

    slots = delta_to_local.tolist()
    last = len(slots) - 1
    low = 0
    while low < last and slots[low + 1] == slots[0]:
        low += 1
    high = last
    while high - 1 > low and slots[high - 1] == slots[last]:
        high -= 1
    high = max(high, low + 1)
    columns = [slots[0], *slots[low + 1 : high], slots[last]]
    # Padding columns never receive gradient; repeating the last slot keeps the
    # map monotone.
    columns += [columns[-1]] * (aligned_slot_count(len(columns)) - len(columns))
    return GradientBand(low, high, torch.tensor(columns, dtype=torch.long, device=device))


class SharedPositionIndexPlan(NamedTuple):
    """Compact ``O(L)`` position plan consumed by the Triton kernel."""

    sequence_length: int
    active_slots: torch.Tensor
    delta_to_local: torch.Tensor
    # None for per-example position_ids, whose backward keeps FP32 atomics.
    gradient_band: GradientBand | None = None


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

    compact() never builds an L x L tensor and is cached per length (it is O(L)).
    dense() builds the int64 [L, L] map for the PyTorch path on demand and does not
    cache it, so memory does not grow with the number of distinct lengths.
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

    def clear(self) -> None:
        self._compact.clear()

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
            gradient_band=gradient_band(delta_to_local, resolved_device),
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
        """Build the dense gather map for PyTorch attention (not cached)."""

        length, _, resolved_device = self._key(sequence_length, device)
        compact = self.compact(length, resolved_device)
        return SharedDensePositionIndexPlan(
            compact=compact,
            pair_to_local=dense_pair_map(compact.delta_to_local, length),
        )


__all__ = [
    "GradientBand",
    "PositionWindow",
    "SharedDensePositionIndexPlan",
    "SharedPositionIndexPlan",
    "SharedPositionPlanCache",
    "canonical_device",
]
