"""Model-wide relative-position index plans for DeBERTa attention.

The tensors in this module depend only on the DeBERTa position-bucketing
configuration, sequence length, and device.  They do not depend on layer
weights, so one cache can be shared by every attention layer in an encoder.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ._reference import make_log_bucket_position


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
    """Cache immutable position-index tensors shared across encoder layers.

    ``compact()`` never constructs an ``L x L`` tensor.  ``dense()`` adds one
    int64 gather map for the reference prepared-PyTorch path; crucially that map
    is shared by all layers instead of being duplicated per layer.
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
