"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton."""

from .deberta import (
    DebertaV2InferenceEncoder,
    compile_deberta_buckets,
    enable_deberta_inference,
    optimize_deberta,
)
from .kernel import DisentangledFlashAttention
from .packed import (
    PackedSequenceInfo,
    pack_padded,
    pack_padded_with_info,
    unpack_packed,
    validate_cu_seqlens,
)
from .tuning import KernelConfig, KernelTuningOptions

__version__ = "0.2.0"

__all__ = [
    "DebertaV2InferenceEncoder",
    "DisentangledFlashAttention",
    "KernelConfig",
    "KernelTuningOptions",
    "PackedSequenceInfo",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "optimize_deberta",
    "pack_padded",
    "pack_padded_with_info",
    "unpack_packed",
    "validate_cu_seqlens",
]
