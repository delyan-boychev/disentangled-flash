"""Differentiable training support for DisentangledFlash."""

from .attention import TritonTrainingDisentangledSelfAttention

__all__ = [
    "TritonTrainingDisentangledSelfAttention",
]
