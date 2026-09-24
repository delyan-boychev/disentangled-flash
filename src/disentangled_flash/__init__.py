"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton."""

from .deberta import (
    DebertaV2OptimizedEncoder,
    compile_deberta_buckets,
    enable_deberta_inference,
    enable_deberta_training,
    optimize_deberta,
    optimize_deberta_training,
)
from .kernel import DisentangledFlashAttention
from .packed import (
    PackedSequenceInfo,
    pack_padded,
    pack_padded_with_info,
    unpack_packed,
    validate_cu_seqlens,
)
from .training import TritonTrainingDisentangledSelfAttention
from .tuning import KernelConfig, KernelTuningOptions

__version__ = "0.2.0"

__all__ = [
    "DebertaV2OptimizedEncoder",
    "DisentangledFlashAttention",
    "KernelConfig",
    "KernelTuningOptions",
    "PackedSequenceInfo",
    "TritonTrainingDisentangledSelfAttention",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "enable_deberta_training",
    "optimize_deberta",
    "optimize_deberta_training",
    "pack_padded",
    "pack_padded_with_info",
    "unpack_packed",
    "validate_cu_seqlens",
]
