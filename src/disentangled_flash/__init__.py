"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton."""

from .deberta import (
    DebertaV2InferenceEncoder,
    compile_deberta_buckets,
    enable_deberta_inference,
    optimize_deberta,
)
from .kernel import DisentangledFlashAttention
from .tuning import KernelConfig, KernelTuningOptions
from .training import (
    DebertaV2TrainingEncoder,
    TritonTrainingDisentangledSelfAttention,
    enable_deberta_training,
    optimize_deberta_training,
)

__version__ = "0.1.2"

__all__ = [
    "DebertaV2InferenceEncoder",
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
