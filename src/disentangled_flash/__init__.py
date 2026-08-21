"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton."""

from .deberta import (
    DebertaV2InferenceEncoder,
    compile_deberta_buckets,
    enable_deberta_inference,
    optimize_deberta,
)
from .kernel import DisentangledFlashAttention

__version__ = "0.1.1"

__all__ = [
    "DebertaV2InferenceEncoder",
    "DisentangledFlashAttention",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "optimize_deberta",
]
