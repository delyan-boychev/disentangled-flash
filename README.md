# DisentangledFlash

[![PyPI Version](https://img.shields.io/pypi/v/disentangled-flash.svg?cacheSeconds=300)](https://pypi.org/project/disentangled-flash/)
[![CI](https://github.com/delyan-boychev/disentangled-flash/actions/workflows/ci.yml/badge.svg)](https://github.com/delyan-boychev/disentangled-flash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

***Fast exact DeBERTa-style disentangled attention in Triton.***

DisentangledFlash provides fused Triton and optimized PyTorch backends
for bidirectional DeBERTa-v2/v3 attention without materializing the
`[B, H, L, L]` attention matrix, including inference and experimental
differentiable training paths.


## Status

- NVIDIA CUDA + Triton
- inference plus experimental differentiable training/backward
- DeBERTa-v2/v3-style C2P/P2C disentangled attention
- FP16, BF16, strict FP32, and optional fast FP32/TF32 mode
- head dimensions 32, 64, and 128
- factorized 2-D padding masks
- runtime sequence lengths through nine bounded kernel families up to 8192
- FlashAttention-style packed, unpadded inference and training with `cu_seqlens`
- **fused QKV projection is always enabled** for the PyTorch/Triton backends

The training Triton path requires self-attention and head dimensions 32, 64, or
128. Attention-probability dropout (`attention_probs_dropout_prob`) runs inside
the fused kernels: like FlashAttention, the Philox mask is regenerated in the
backward pass instead of being stored. It does not yet support
`output_attentions=True` or custom pairwise relative-position tensors. CPU/MPS
use the differentiable PyTorch backend rather than Triton.

> [!IMPORTANT]
> The Triton kernel has been validated on NVIDIA RTX
> 6000 Ada (SM 8.9) and NVIDIA H200 (SM 9.0). A reviewed H200 profile for the
> documented compiler stack ships in the package. Other GPUs and stacks use
> bounded autotuning and should be validated locally.

## Install

```bash
pip install -e ".[dev]"
```

For the pretrained GLUE/MNLI evaluation:

```bash
pip install -e ".[dev,benchmark,hf]"
```

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
    sequence_lengths=[64, 128, 384, 512, 768, 1024, 2048, 4096, 8192],
)
```

Only the encoder attention path is replaced; checkpoint names and the remaining
model layers are preserved. Use `enable_deberta_inference(..., backend="torch")`
to select the PyTorch backend.

For training, enable the differentiable path before constructing the optimizer:

```python
from disentangled_flash import optimize_deberta_training

optimize_deberta_training(model.deberta, backend="triton")
```

This preserves the original parameter identities and keeps Q/K/V and relative
position projections in the autograd graph. The same tuning policy applies to
training, with independently selected entries for the LSE-producing forward,
dQ, and dK/dV kernels.

## CUDA benchmark

```bash
pip install -e ".[benchmark]"
```

```bash
python -m benchmarks.benchmark_cuda \
  --tuning-mode profile_only \
  --profile src/disentangled_flash/profiles/h200-sm90-deberta-v2-v3-torch-2.14-cu130-triton-3.8.json \
  --output attention_kernel_results.json
```

The release matrix compares one DeBERTa-v3-base-shaped attention layer across
Hugging Face eager, DF PyTorch, DF Triton, and FlashDeBERTa. It measures
inference forward and training forward+backward in BF16 at
lengths 128, 512, 1024, 2048, 4096, and 8192. Every point contains approximately
16k active tokens (`batch = max(1, 16384 // length)`), so there is no batch or
packed-layout axis. Training is measured with dropout 0 and 0.1. Each case runs
in an isolated process and records latency, effective attention throughput,
peak memory, OOMs, and unavailable optional backends.

Generate latency and memory plots with:

```bash
python -m benchmarks.plot_cuda_results \
  attention_kernel_results.json \
  --output-dir benchmarks/results/kernel
```

### Kernel tuning profiles

Matching saved profiles are used automatically; otherwise Triton runs bounded
autotuning on first use. Override this with:

```python
from disentangled_flash import (
    KernelConfig,
    KernelTuningOptions,
    optimize_deberta,
    optimize_deberta_training,
)

optimize_deberta(model, tuning=KernelTuningOptions(mode="autotune"))
optimize_deberta(
    model,
    tuning=KernelTuningOptions(profile_paths=("my-gpu-profile.json",)),
)
optimize_deberta_training(
    model,
    tuning=KernelTuningOptions(profile_paths=("my-gpu-profile.json",)),
)
optimize_deberta(
    model,
    tuning=KernelTuningOptions(
        mode="fixed",
        fixed_config=KernelConfig(64, 64, 4),
    ),
)
```

Profiles are discovered in the user cache, `DISENTANGLED_FLASH_PROFILE_DIR`, and
the package. Compatibility is keyed by GPU/compiler stack and workload; the
driver is diagnostic only. Exact length, batch size, head count, and active-slot
count are runtime values rather than autotune keys.

Generate a resumable profile on a CUDA machine with:

```bash
python -m disentangled_flash.tune \
  --output rtx-6000-ada.json
```

```bash
python -m disentangled_flash.tune inspect rtx-6000-ada.json
```

The default `standard` preset covers all supported DeBERTa-v2/v3 variants: length
families `64`, `128`, `384`, `512`, `768`, `1024`, `2048`, `4096`, and `8192`,
head dimension 64, both launch-occupancy regimes, C2P+P2C relative attention,
and padded/packed layouts. Like FlashAttention and FlexAttention, FP16 and BF16
share one half-precision schedule family (measured with BF16), while strict FP32
and fast FP32/TF32 are tuned separately over a conservative search: one pipeline
stage, forward tiles up to 64x64, and backward tiles up to 2048 elements. Training phases are tuned with and
without attention dropout, because the in-kernel mask changes register pressure.
That is 162 workload shapes, giving 162 `inference` workloads plus 972 training
workloads (`training_forward`, `backward_dq`, and `backward_dkv` for both dropout
variants), 1134 phase workloads in total. Results are parity-checked and saved after each workload so tuning can
resume. `--passes inference`, `--passes training`, or `--dropout off|on` can
restrict a run; the shape and attention options remain available for custom
architectures.

### Packed unpadded inference and training

`forward_packed(hidden_states, cu_seqlens, max_seqlen)` accepts
FlashAttention-style `[total_tokens, hidden_size]` input. Convert right-padded
input with `pack_padded_with_info` and reuse its metadata across encoder layers:

```python
from disentangled_flash import pack_padded_with_info, unpack_packed

tokens, cu_seqlens, info = pack_padded_with_info(hidden_states, attention_mask)
packed_output = encoder.forward_packed(
    tokens,
    cu_seqlens,
    info.max_seqlen,
    packed_info=info,
).last_hidden_state
output, _ = unpack_packed(
    packed_output,
    cu_seqlens,
    hidden_states.size(1),
    packed_info=info,
)
```

The same call is differentiable when the encoder was installed with
`optimize_deberta_training`; no padded `[B, L, ...]` attention workspace is
created by the packed Triton path.

## Pretrained GLUE/MNLI evaluation

```bash
python -m benchmarks.evaluate_mnli
```

This runs one cold, full GLUE/MNLI matched-validation pass at batch size 8 across
Hugging Face, DF PyTorch, DF Triton, and FlashDeBERTa. It reports performance,
accuracy, numerical error, and classification-decision parity.

To require the bundled H200 configuration and prohibit Triton autotuning:

```bash
python -m benchmarks.evaluate_mnli \
  --require-full-parity \
  --tuning-mode profile_only \
  --profile src/disentangled_flash/profiles/h200-sm90-deberta-v2-v3-torch-2.14-cu130-triton-3.8.json \
  --output mnli_parity.json
```

## CUDA validation

```bash
python -m validation.validate_cuda
```

Run this before publishing a profile for a new GPU or compiler stack.


## Bundled H200 tuning profile

The package automatically discovers the reviewed
[`h200-sm90-deberta-v2-v3-torch-2.14-cu130-triton-3.8.json`](src/disentangled_flash/profiles/h200-sm90-deberta-v2-v3-torch-2.14-cu130-triton-3.8.json)
profile. Its 1,134 validated winners cover inference, training forward, and both
backward phases across the standard DeBERTa-v2/v3 workload families through
length 8192. They require H200 SM 9.0, PyTorch 2.14.0+cu130, CUDA 13.0, and
the recorded Triton 3.8.0 compiler fingerprint; otherwise `auto` mode uses
bounded autotuning.

## Release parity checks

Release validation consists of the full MNLI matched-validation decision-parity
run above and a short BF16 fine-tuning comparison with attention dropout 0.1:

```bash
python -m validation.validate_multistep_training \
  --dropout 0.1 \
  --require-parity \
  --tuning-mode profile_only \
  --profile src/disentangled_flash/profiles/h200-sm90-deberta-v2-v3-torch-2.14-cu130-triton-3.8.json
```

The multi-step report records the loss curve plus output, gradient, and
post-update parameter divergence from Hugging Face. End-to-end performance is
not part of the release benchmark matrix.

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
  version = {0.2.0},
  year = {2026}
}
```
