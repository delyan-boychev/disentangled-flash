"""Validation checks shared by repository tooling and tests."""

from __future__ import annotations

import torch


def require_finite(output: torch.Tensor, label: str, case: str) -> None:
    """Fail immediately instead of allowing NaNs to disappear in max summaries."""

    finite = torch.isfinite(output)
    if not finite.all():
        nonfinite_count = (~finite).sum().item()
        raise RuntimeError(f"{label} produced {nonfinite_count} non-finite value(s) for {case}")
