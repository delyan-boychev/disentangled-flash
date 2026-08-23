"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton."""

from .deberta import (
    DebertaV2InferenceEncoder,
    DebertaV2OptimizedEncoder,
    DebertaV2TrainingEncoder,
    compile_deberta_buckets,
    enable_deberta_inference,
    enable_deberta_training,
    optimize_deberta,
    optimize_deberta_training,
)
from .kernel import DisentangledFlashAttention
from .training import TritonTrainingDisentangledSelfAttention
from .tuning import KernelConfig, KernelTuningOptions

__version__ = "0.1.2"

__all__ = [
    "DebertaV2InferenceEncoder",
    "DebertaV2OptimizedEncoder",
    "DebertaV2TrainingEncoder",
    "DisentangledFlashAttention",
    "KernelConfig",
    "KernelTuningOptions",
    "TritonTrainingDisentangledSelfAttention",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "enable_deberta_training",
    "optimize_deberta",
    "optimize_deberta_training",
]
