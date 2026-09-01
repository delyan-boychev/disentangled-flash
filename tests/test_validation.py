import pytest
import torch

from disentangled_flash._validation import require_finite


def test_require_finite_accepts_finite_output():
    require_finite(torch.tensor([0.0, -1.0, 2.0]), "target", "left-padding case")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_require_finite_rejects_nonfinite_output(nonfinite):
    with pytest.raises(
        RuntimeError,
        match=r"target produced 1 non-finite value\(s\) for left-padding case",
    ):
        require_finite(torch.tensor([0.0, nonfinite]), "target", "left-padding case")
