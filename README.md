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
> The Triton kernel has been validated on NVIDIA RTX 6000 Ada (SM89), H200
> (SM90), and RTX PRO 6000 Blackwell (SM120). A reviewed H200 profile ships in
> the package; other GPUs use the portable heuristic and should be validated
> locally.

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

## Performance

These are BF16 attention-layer results with 12 heads, head dimension 64,
16,384 active tokens per batch, and no dropout. All backends compute exact
DeBERTa C2P/P2C attention. DF Triton uses the built-in heuristic, not a tuned
profile.

Both machines used PyTorch 2.14.0+cu130, Triton 3.8.0, CUDA 13.0, and
FlashDeBERTa 0.0.7; Transformers was 5.17.0 on H200 and 5.18.0 on RTX.

TFLOP/s is FLOPs divided by time. For batch `B`, heads `H`, length `L`, head
dimension `D`, and `R` active relative-position rows, we count
`BH(4L²D + 4LRD)` for forward, `BH(10L²D + 12LRD)` for backward, and
`BH(14L²D + 16LRD)` combined. As in FlashAttention, this excludes softmax,
gathers, and projection layers.

### NVIDIA H200

![Attention layer throughput on H200](docs/figures/kernel_throughput_h200.png)

![Attention kernel incremental peak memory on H200](docs/figures/kernel_memory_h200.png)

At length 8192, DF Triton is 5.19× faster than Hugging Face forward and 2.13×
faster for forward+backward. Incremental peak memory is 0.47 GiB forward and
1.30 GiB combined, versus 19.07 GiB and 19.47 GiB for Hugging Face.

### NVIDIA RTX PRO 6000 Blackwell

![Attention layer throughput on RTX PRO 6000 Blackwell](docs/figures/kernel_throughput_rtx6000.png)

![Attention kernel incremental peak memory on RTX PRO 6000 Blackwell](docs/figures/kernel_memory_rtx6000.png)

At length 8192, DF Triton is 14.4× faster than Hugging Face forward and 5.04×
faster combined; against FlashDeBERTa it is 1.57× and 17.6× faster. Peak memory
matches the H200 run. All 72 cases completed without allocator retries or
`cudaMalloc` calls during timing.

### Benchmark methodology

Install the benchmark dependencies and reproduce the matrix and plots with:

```bash
pip install -e ".[benchmark]"
python -m benchmarks.benchmark_cuda --tuning-mode heuristic --output attention_kernel_results.json
python -m benchmarks.plot_cuda_results attention_kernel_results.json --output-dir docs/figures
```

Pass `--tuning-mode profile_only --profile PROFILE.json` to benchmark a tuned
profile instead; it fails on any workload the profile does not cover.

Each configuration runs in its own process. After one untimed call for
compilation and setup, Triton's `do_bench` uses 3 ms of warmup and 30 ms of
timing, flushing L2 before each iteration; processes are separated by one
second. Backward is timed independently by running forward once and replaying
`out.backward(grad, retain_graph=True)`.

Timing is eager and therefore includes Python, autograd, and launch overhead.
Hollow markers identify points where CPU issue time is at least 80% of measured
latency. The Hugging Face mask and relative-position matrix are constructed
before timing. The report also records min, p10, median, and p90 latency,
incremental peak allocated memory, allocator retries, and `cudaMalloc` calls.

### Kernel tuning profiles

A matching profile is used automatically; otherwise the kernel uses a
hardware-aware heuristic. `heuristic` always ignores profiles, `profile_only`
fails on a miss, and `autotune` benchmarks candidates at runtime.

```python
from disentangled_flash import KernelTuningOptions, optimize_deberta

optimize_deberta(model, tuning=KernelTuningOptions(mode="heuristic"))
```

Generate and inspect a profile with:

```bash
python -m disentangled_flash.tune --output my-gpu.json
python -m disentangled_flash.tune inspect my-gpu.json
```

The default preset tests the heuristic and nearby configs. Release profiles use
`--preset exhaustive`. Runs are parity-checked, resumable, and saved after each
workload.

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

This runs one cold matched-validation pass and reports speed, accuracy, and
decision parity. The release command is:

```bash
python -m benchmarks.evaluate_mnli \
  --dtype fp16 \
  --batch-size 16 \
  --bucket 512 \
  --require-full-parity \
  --tuning-mode heuristic \
  --output mnli_parity_fp16_heuristic.json
```

### H200

On the full 9,815-example split in FP16 at batch size 16, every backend reached
91.74% accuracy with zero decision mismatches against Hugging Face.

| Variant | Full-split time | Speedup vs. HF eager | Decision mismatches | Maximum absolute logit difference |
| --- | ---: | ---: | ---: | ---: |
| Hugging Face eager, padded | 53.27 s | 1.00× | Reference | 0 |
| DF Triton, padded | 23.25 s | 2.29× | 0 / 9,815 | 0.0752 |
| DF Triton, packed | 6.94 s | 7.68× | 0 / 9,815 | 0.0752 |
| FlashDeBERTa, packed | 22.00 s | 2.42× | 0 / 9,815 | 0.0801 |

This is decision parity, not bitwise logit equality. The recorded BF16 run did
not meet full decision parity, so the release claim uses FP16.

### RTX PRO 6000 Blackwell

At batch size 8, the first run includes JIT compilation; the second reuses the
persistent Triton cache. Both runs kept 91.74% accuracy and zero mismatches.

| Variant | Cold full-split time | Compiled-cache time | Cold speedup vs HF | Compiled speedup vs HF | Decision mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hugging Face eager, padded | 88.67 s | 88.74 s | 1.00× | 1.00× | Reference |
| DF Triton, padded | 40.85 s | 38.15 s | 2.17× | 2.33× | 0 / 9,815 |
| DF Triton, packed | 17.27 s | **8.46 s** | 5.14× | **10.49×** | 0 / 9,815 |
| FlashDeBERTa, packed | 43.44 s | 39.15 s | 2.04× | 2.27× | 0 / 9,815 |

`heuristic` skips configuration search, not JIT compilation. Set
`TRITON_CACHE_DIR` to persistent storage to reuse the compiled binaries. The
8.81-second cold-to-cached difference for packed DF Triton also includes normal
run-to-run variation.

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
the heuristic.

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

The 12-layer BF16 check runs 20 optimizer steps at length 1024, batch 2, and
dropout 0.1. Final fixed loss differed by 0.0796%; final parameter relative L2
difference was 0.42% with cosine similarity 0.99999. It validates the training
path, including backward, but does not claim elementwise-identical gradients.

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
  version = {1.1.0},
  year = {2026}
}
```
