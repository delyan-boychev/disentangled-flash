# DisentangledFlash

[![PyPI Version](https://img.shields.io/pypi/v/disentangled-flash.svg)](https://pypi.org/project/disentangled-flash/)
[![CI](https://github.com/delyan-boychev/disentangled-flash/actions/workflows/ci.yml/badge.svg)](https://github.com/delyan-boychev/disentangled-flash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

***Fast exact DeBERTa-style disentangled attention in Triton.***

DisentangledFlash is an inference-oriented implementation of bidirectional
DeBERTa-v2/v3 disentangled self-attention. 

The Triton kernel is inspired by [FlashAttention](https://github.com/Dao-AILab/flash-attention)'s tiling, IO-aware computations, and online softmax without memory materialization, while its relative position encoding implementation is inspired by [FlexAttention](https://pytorch.org/blog/flexattention/). It fuses QK, relative-score lookup, factorized padding-mask application, online softmax, and PV without materializing a `[B, H, L, L]` attention tensor. C2P/P2C projection score GEMMs remain regular PyTorch GEMMs over the pruned active relative-position slots.

The PyTorch-optimized (`torch`) backend is constructed by optimizing operations, employing smart caching techniques, and leveraging fused QKV projection.


## Status

- NVIDIA CUDA + Triton
- inference only
- DeBERTa-v2/v3-style C2P/P2C disentangled attention
- FP16, BF16, strict FP32, and optional fast FP32/TF32 mode
- head dimensions 32, 64, and 128
- factorized 2-D padding masks
- runtime sequence lengths through six bounded kernel families up to 1024
- FlashAttention-style packed, unpadded inference with `cu_seqlens`
- **fused QKV projection is always enabled** for the PyTorch/Triton backends

Training/backward is not implemented. CPU/MPS use the PyTorch backend
for development/benchmarking, not the Triton kernel.

While currently tailored to DeBERTa-v2/v3, the kernel and caching abstractions are designed to be extensible to other architectures requiring factorized relative-position or disentangled attention schemes in the future.

> [!IMPORTANT]
> **GPU Compatibility**: The Triton kernel has been validated and benchmarked primarily on the **NVIDIA RTX 6000 Ada** (Compute Capability 8.9). Further testing, benchmarking, and autotuning calibration are required to ensure optimal performance on other GPU models and hardware architectures. **Pull requests, benchmark results, and configurations for other GPUs are highly welcome!**

## Install

Use a PyTorch build appropriate for your CUDA environment, then install this
repository editable while developing:

```bash
pip install -e ".[dev]"
```

For the pretrained Hugging Face MNLI parity benchmark:

```bash
pip install -e ".[dev,hf]"
```

`sentencepiece` and `protobuf` are included in the `hf` extra because the
DeBERTa-v2 tokenizer uses a SentencePiece `spm.model`.

## Hugging Face usage

```python
from transformers import AutoModelForSequenceClassification
from disentangled_flash import optimize_deberta

model = (
    AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    .cuda()
    .eval()
)

optimize_deberta(
    model.deberta,
    sequence_lengths=[64, 128, 384, 512, 768, 1024],
)
```

`optimize_deberta()` replaces only the DeBERTa encoder attention path. The
existing layer output, residual, FFN, convolution, classifier, and checkpoint
parameter names are preserved.

For lower-level experiments, `enable_deberta_inference(..., backend="torch")`
selects the PyTorch backend instead of Triton.

## CUDA benchmark

Attention-only:

```bash
python -m benchmarks.benchmark_cuda --scope attention
```

Full encoder:

```bash
python -m benchmarks.benchmark_cuda \
  --scope encoder \
  --implementations original,torch,triton \
  --dtypes fp16,fp32 \
  --output encoder_attention_cuda_results.json
```

### Kernel tuning profiles

The Triton backend uses a validated saved configuration when one exactly
matches the GPU and workload. Otherwise it safely falls back to Triton
autotuning on first use. You can override that policy explicitly:

```python
from disentangled_flash import KernelConfig, KernelTuningOptions, optimize_deberta

# Always retune, even when a bundled or personal profile matches.
optimize_deberta(model, tuning=KernelTuningOptions(mode="autotune"))

# Use a personal profile before the bundled profiles.
optimize_deberta(
    model,
    tuning=KernelTuningOptions(profile_paths=("my-gpu-profile.json",)),
)

# Advanced: bypass profiles and autotuning with one explicit configuration.
optimize_deberta(
    model,
    tuning=KernelTuningOptions(
        mode="fixed",
        fixed_config=KernelConfig(64, 64, 4),
    ),
)
```

Personal profiles placed in
`$XDG_CACHE_HOME/disentangled_flash/profiles` (or
`~/.cache/disentangled_flash/profiles`) are discovered automatically. Set
`DISENTANGLED_FLASH_PROFILE_DIR` to use another directory. Explicit profile
paths take precedence, followed by the user directory and bundled profiles.
Installed package files are never modified.

The Triton autotuner specializes only on the finite length regime, head
dimension, dtype/FP32 policy, relative-attention mode, and padding-mask mode.
Exact sequence length, batch size, head count, and active-slot count are runtime
scalars and are excluded from the autotune key. Consequently, changing 384 to
383 does not trigger another benchmark sweep. Use `fixed` to bypass autotuning
entirely; `profile_only` reports a missing finite-family profile.

Generate a resumable profile on a CUDA machine with:

```bash
python -m disentangled_flash.tune \
  --preset standard \
  --output rtx-6000-ada.json
```

Saved profiles use six bounded sequence-length families: `64`, `128`, `384`,
`512`, `768`, and `1024`, plus occupancy families `8` and `32` batch-head
programs. Runtime lengths select the next family (for example, 383 uses the 384
profile). The exact length remains runtime data used for the launch grid, loop
bounds, and partial-tile masks. Relative-position LUTs are prepared at the
family size and indexed with a family offset, so they are reusable by every
shorter exact length in that family. Triton inputs and custom tuning sweeps are
currently capped at 1024.

This runtime-length contract uses profile format 2. Profiles generated by the
former exact-shape kernel are intentionally rejected and should be regenerated.

`quick` checks one representative workload, `standard` covers common production
shapes, and `exhaustive` adds tile boundaries, occupancy levels, and all relative
attention modes. Use `--help` for custom dimensions and candidate files. Every
accepted result is checked against a PyTorch reference with multiple padding
patterns, and the output is saved after each workload so an interrupted run can
resume.

The original backend remains unfused and acts as the reference baseline. The
PyTorch and Triton backends always use one packed QKV projection.

### Packed unpadded inference

Attention modules and optimized encoders accept a FlashAttention-style packed
token layout through `forward_packed(hidden_states, cu_seqlens, max_seqlen)`.
`hidden_states` has shape `[total_tokens, hidden_size]`; `cu_seqlens` is a
contiguous int32/int64 tensor containing cumulative sequence boundaries. No
dense padded batch or cross-sequence attention matrix is constructed. The
Triton projects QKV once for the complete token buffer and dispatches one
attention grid across every sequence/head tile. Each program reads its runtime
boundaries from `cu_seqlens`, so tokens cannot attend across sequences and no
dense padded batch is created. The PyTorch backend retains a segmented reference
implementation for portability and validation.

Use `pack_padded` and `unpack_packed` to convert right-padded tensors at an API
boundary. Empty sequences and non-right-padded masks are rejected explicitly.

## Pretrained task parity + speed

The MNLI script loads `microsoft/deberta-v2-xlarge-mnli`, compares the untouched
Hugging Face model with the same checkpoint using DisentangledFlash, verifies
logits/probabilities/hidden-state parity, and benchmarks the full classification
forward. Tokenization and model loading are excluded from timing.

Defaults are batch size 8 and 500 measured iterations:

```bash
python -m benchmarks.parity_pretrained_mnli
```

The Triton candidate uses packed `cu_seqlens` inference by default while the
untouched Hugging Face reference remains padded. Pass `--layout padded` to
benchmark the regular padded candidate path instead.

## Hostile CUDA validation

```bash
python -m validation.validate_cuda
```

The validation matrix covers FP16/BF16/FP32, boundary sequence lengths, several
padding patterns, and C2P/P2C position modes while reporting raw max/mean errors.

> [!NOTE]
> **Test Coverage**: CPU tests cover the public inference API, packed conversion,
> bounded tuning/profile dispatch, and launch contracts. CUDA correctness and
> performance should still be validated on every supported GPU family before
> publishing a bundled tuning profile.

## GPU calibration

The bounded families prevent retuning for every exact sequence length, but a
missing hardware/workload family still evaluates the configured candidate set
once. Offline calibration per GPU family can pre-select those configurations and
remove that first-use benchmarking cost.

Do not treat the current candidate table as a universal final table for every GPU.


## Results

DisentangledFlash was benchmarked against:

- the original Hugging Face DeBERTa encoder,
- the PyTorch implementation,
- and the Triton DisentangledFlash implementation.

The benchmark covers the full 12-layer encoder, not only the isolated attention operator.

### Benchmark configuration

| Parameter | Value |
|---|---:|
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Encoder layers | 12 |
| FFN intermediate size | 3072 |
| Convolution kernel | 3 |
| Batch sizes | 1, 8, 16, 32 |
| Sequence lengths | 16, 32, 64, 128, 256, 384, 512 |
| Precisions | FP16, strict FP32 |
| Execution mode | Eager |
| GPU | NVIDIA RTX 6000 Ada Generation |
| Compute capability | 8.9 |

> The benchmark snapshot below predates the current always-fused-QKV API and was run with QKV fusion disabled for both the PyTorch and Triton backends. The current DisentangledFlash implementation always uses fused QKV projection.

### Overall encoder speedup

Across all 28 tested `(batch size, sequence length)` configurations per precision:

| Precision | Geomean speedup vs. Hugging Face | Geomean speedup vs. PyTorch impl. | Best speedup vs. PyTorch impl. |
|---|---:|---:|---:|
| FP16 | **1.75×** | **1.32×** | **1.97×** |
| FP32 | **1.56×** | **1.24×** | **1.50×** |

The advantage over the PyTorch implementation increases substantially for longer sequences, where the quadratic attention matrix becomes increasingly expensive.

### FP16 encoder latency

Median end-to-end encoder latency:

| Batch | Seq. length | Hugging Face | PyTorch impl. | DisentangledFlash | vs. HF | vs. PyTorch impl. |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 6.25 ms | 3.79 ms | **3.27 ms** | **1.91×** | **1.16×** |
| 8 | 256 | 7.56 ms | 7.09 ms | **5.27 ms** | **1.43×** | **1.34×** |
| 8 | 384 | 14.16 ms | 14.73 ms | **9.73 ms** | **1.45×** | **1.51×** |
| 8 | 512 | 21.94 ms | 22.70 ms | **12.03 ms** | **1.82×** | **1.89×** |
| 16 | 128 | 6.32 ms | 6.18 ms | **4.84 ms** | **1.30×** | **1.27×** |
| 16 | 256 | 15.40 ms | 16.58 ms | **11.36 ms** | **1.36×** | **1.46×** |
| 16 | 384 | 30.61 ms | 32.86 ms | **20.61 ms** | **1.48×** | **1.59×** |
| 16 | 512 | 52.37 ms | 53.44 ms | **27.19 ms** | **1.93×** | **1.97×** |
| 32 | 128 | 12.50 ms | 13.09 ms | **9.48 ms** | **1.32×** | **1.38×** |
| 32 | 256 | 33.81 ms | 36.35 ms | **24.61 ms** | **1.37×** | **1.48×** |
| 32 | 384 | 67.93 ms | 71.55 ms | **41.08 ms** | **1.65×** | **1.74×** |
| 32 | 512 | 108.11 ms | 110.28 ms | **57.84 ms** | **1.87×** | **1.91×** |

At `B=16, L=512`, DisentangledFlash reduces encoder latency from **53.44 ms to 27.19 ms** relative to the PyTorch path, corresponding to approximately a **49% latency reduction**.

### FP32 encoder latency

Strict FP32 also benefits significantly:

| Batch | Seq. length | Hugging Face | PyTorch impl. | DisentangledFlash | vs. HF | vs. PyTorch impl. |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 12.86 ms | 11.93 ms | **10.22 ms** | **1.26×** | **1.17×** |
| 8 | 256 | 27.24 ms | 27.50 ms | **22.73 ms** | **1.20×** | **1.21×** |
| 8 | 384 | 45.98 ms | 45.91 ms | **34.40 ms** | **1.34×** | **1.33×** |
| 8 | 512 | 70.78 ms | 69.20 ms | **46.09 ms** | **1.54×** | **1.50×** |
| 16 | 128 | 25.50 ms | 24.47 ms | **21.11 ms** | **1.21×** | **1.16×** |
| 16 | 256 | 52.04 ms | 53.26 ms | **41.73 ms** | **1.25×** | **1.28×** |
| 16 | 384 | 93.33 ms | 93.72 ms | **65.53 ms** | **1.42×** | **1.43×** |
| 16 | 512 | 137.84 ms | 137.07 ms | **91.96 ms** | **1.50×** | **1.49×** |
| 32 | 128 | 46.55 ms | 45.94 ms | **38.61 ms** | **1.21×** | **1.19×** |
| 32 | 256 | 109.15 ms | 110.59 ms | **84.57 ms** | **1.29×** | **1.31×** |
| 32 | 384 | 183.83 ms | 185.52 ms | **130.02 ms** | **1.41×** | **1.43×** |
| 32 | 512 | 280.22 ms | 278.36 ms | **187.99 ms** | **1.49×** | **1.48×** |

### Scaling with sequence length

Geometric-mean speedup across all tested batch sizes:

| Sequence length | FP16 vs. HF | FP16 vs. PyTorch impl. | FP32 vs. HF | FP32 vs. PyTorch impl. |
|---:|---:|---:|---:|---:|
| 16 | **1.96×** | **1.17×** | **2.10×** | **1.18×** |
| 32 | **1.89×** | **1.18×** | **1.75×** | **1.17×** |
| 64 | **1.81×** | **1.15×** | **1.52×** | **1.13×** |
| 128 | **1.60×** | **1.23×** | **1.38×** | **1.17×** |
| 256 | **1.52×** | **1.39×** | **1.37×** | **1.27×** |
| 384 | **1.63×** | **1.48×** | **1.40×** | **1.34×** |
| 512 | **1.91×** | **1.69×** | **1.55×** | **1.47×** |

The comparison against the PyTorch implementation is particularly useful: both implementations already avoid several pieces of Hugging Face encoder overhead, so the increasing gap at long sequence lengths isolates the benefit of the streaming Triton attention path more clearly.

### Peak GPU memory

At batch size 32, the memory advantage grows with sequence length:

#### FP16

| Sequence length | Hugging Face | PyTorch impl. | DisentangledFlash | Reduction vs. PyTorch impl. |
|---:|---:|---:|---:|---:|
| 128 | 0.64 GB | 0.73 GB | **0.70 GB** | 4.2% |
| 256 | 0.96 GB | 1.10 GB | **0.97 GB** | 12.3% |
| 384 | 1.45 GB | 1.58 GB | **1.26 GB** | 20.3% |
| 512 | 2.09 GB | 2.14 GB | **1.55 GB** | **27.3%** |

#### FP32

| Sequence length | Hugging Face | PyTorch impl. | DisentangledFlash | Reduction vs. PyTorch impl. |
|---:|---:|---:|---:|---:|
| 128 | 1.25 GB | 1.44 GB | **1.38 GB** | 3.8% |
| 256 | 1.87 GB | 2.19 GB | **1.92 GB** | 12.2% |
| 384 | 2.85 GB | 3.13 GB | **2.50 GB** | 20.3% |
| 512 | 4.12 GB | 4.25 GB | **3.09 GB** | **27.3%** |

This behavior is expected because DisentangledFlash performs tiled streaming softmax and does not materialize the full `[B, H, L, L]` attention score/probability tensor.

### Numerical accuracy

DisentangledFlash was compared directly against the original Hugging Face implementation at both the isolated attention level and across the full 12-layer DeBERTa encoder.

| Precision | Level | Max absolute error | Mean absolute error |
|---|---|---:|---:|
| FP16 | Attention | **7.63e-6** | **2.48e-7** |
| FP16 | Full encoder | **1.56e-2** | **1.12e-3** |
| FP32 | Attention | **4.89e-9** | **1.79e-10** |
| FP32 | Full encoder | **7.57e-6** | **6.26e-7** |

The maximum absolute error is the worst observed value across all tested batch-size and sequence-length configurations. The mean absolute error is averaged across the 28 tested configurations for each precision and level.

### Pretrained-model parity

Task-level parity was additionally tested with the pretrained `microsoft/deberta-v2-xlarge-mnli` checkpoint.

The original Hugging Face model and the same checkpoint with its DeBERTa encoder replaced by DisentangledFlash achieved **full task-level parity** on the parity test.

The test verifies:

- final MNLI predictions,
- classification logits and probabilities,
- final encoder hidden states,
- and the complete sequence-classification inference path.

This test exercising a real pretrained DeBERTa model rather than only synthetic attention tensors.

### Test environment

All current CUDA benchmarks and pretrained-model parity tests were run on:

| Component | Configuration |
|---|---|
| OS | Ubuntu 24.04.3 LTS |
| GPU | NVIDIA RTX 6000 Ada Generation |
| Compute capability | 8.9 |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu13 |
| Triton | 3.7.1 |
| Transformers | 5.15.1 |
| CPU | AMD Ryzen Threadripper PRO 7975WX, 32 cores |
| System RAM | 512 GB |

Latency numbers above are steady-state measurements. Triton compilation and autotuning startup cost are excluded from the reported p50 latency.

Further validation and benchmarking are needed on other GPU architectures and configurations to guarantee optimal tuning and performance across different hardware.


## Attribution

The auditable reference implementation is derived from Hugging Face Transformers
4.57.6 DeBERTa-v2/v3 modeling code and retains its original Apache-2.0 header.
See `THIRD_PARTY_NOTICES.md`.

## Citation

If you use DisentangledFlash in your research or project, please cite it as follows:

```bibtex
@software{boychev2026disentangledflash,
  author = {Boychev, Delyan},
  title = {DisentangledFlash: Fast exact DeBERTa-style disentangled attention in Triton},
  url = {https://github.com/delyan-boychev/disentangled-flash},
  version = {0.1.3},
  year = {2026}
}
```
