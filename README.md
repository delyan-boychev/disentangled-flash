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

## Performance

The benchmark compares exact DeBERTa disentangled attention across Hugging Face
eager, DF PyTorch, DF Triton, and FlashDeBERTa.

### H200

- **Hardware:** one NVIDIA H200 (SM90).
- **Software:** BF16, PyTorch 2.14.0+cu130, Triton 3.8.0, CUDA 13.0,
  Transformers 5.17.0, and FlashDeBERTa 0.0.7.
- **Shape:** 12 heads, head dimension 64, sequence lengths 128 to 8192, and
  `batch = 16384 / length`.
- **Passes:** forward, backward, and forward + backward, with dropout disabled.
- **DF Triton policy:** built-in heuristic, without a tuning profile.

![Attention layer throughput on H200](docs/figures/kernel_throughput_h200.png)

Throughput is algorithmic work divided by time. For batch `B`, heads `H`,
length `L`, head dimension `D`, and `R` active relative-position rows, the
forward, backward, and combined counts are respectively
`BH(4L²D + 4LRD)`, `BH(10L²D + 12LRD)`, and
`BH(14L²D + 16LRD)`. Backward includes score recomputation plus both operand
gradients for QK, C2P, and P2C. As in FlashAttention, softmax, gathers, and
projection layers are excluded.

Median latency in milliseconds, and DF Triton's speedup over Hugging Face eager
and FlashDeBERTa (`*` marks a host-bound DF Triton point):

| Length | Batch | Forward | vs HF | vs FlashDeBERTa | Backward | vs HF | vs FlashDeBERTa | Fwd + bwd | vs HF | vs FlashDeBERTa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 0.42 | 2.99× | 3.21× | 4.16\* | 0.64× | 0.80× | 4.36\* | 1.00× | 1.10× |
| 512 | 32 | 0.81 | 3.60× | 1.86× | 4.09\* | 1.09× | 3.20× | 4.73\* | 1.56× | 3.11× |
| 1024 | 16 | 1.32 | 3.97× | 1.65× | 6.17\* | 1.33× | 9.55× | 7.48\* | 1.80× | 8.23× |
| 2048 | 8 | 2.35 | 4.45× | 1.54× | 10.91 | 1.56× | 17.1× | 13.29 | 2.07× | 14.4× |
| 4096 | 4 | 4.42 | 5.00× | 1.49× | 20.67 | 1.70× | 22.1× | 25.15 | 2.27× | 18.4× |
| 8192 | 2 | 8.51 | 5.19× | 1.48× | 45.67 | 1.57× | 21.7× | 54.39 | 2.13× | 18.5× |

Forward is fastest at every length. For training, DF Triton leads from length
512 up. At 128 every implementation's training step is host-bound, so those
points compare Python and launch overhead rather than kernels. These are
attention-layer numbers, not full encoder training, and not a comparison with
plain FlashAttention without relative bias.

![Attention kernel incremental peak memory on H200](docs/figures/kernel_memory_h200.png)

Incremental peak allocated memory in GiB, above the process's allocation before
the measured call:

| Length | HF forward | DF Triton forward | HF fwd + bwd | DF Triton fwd + bwd | FlashDeBERTa fwd + bwd |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.63 | 0.28 | 0.96 | 0.55 | 1.13 |
| 512 | 1.25 | 0.47 | 1.64 | 1.29 | 0.99 |
| 2048 | 4.64 | 0.47 | 5.04 | 1.30 | 0.95 |
| 8192 | 19.07 | 0.47 | 19.47 | 1.30 | 0.94 |

DF Triton's memory stays flat with length: at 8192 it allocates 98% less than
Hugging Face eager for forward and 93% less for forward+backward. FlashDeBERTa
uses about 0.35 GiB less for training, most of which is DF Triton's per-distance
position-gradient buffers.

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

A matching saved profile is used automatically. Otherwise a deterministic
heuristic picks the config from the workload (length, head dim, phase,
precision) and the GPU's own limits: it reads the shared memory per block and SM
count from the device, so large tiles are never tried on GPUs they don't fit.
If a config still fails to launch, that workload falls back to Triton
autotuning. Under `torch.compile` the config is picked once while tracing, so a
graph recompiles only when the length family changes. Choose a different policy
with:

```python
from disentangled_flash import (
    KernelConfig,
    KernelTuningOptions,
    optimize_deberta,
    optimize_deberta_training,
)

optimize_deberta(model, tuning=KernelTuningOptions(mode="heuristic"))  # ignore profiles
optimize_deberta(model, tuning=KernelTuningOptions(mode="autotune"))  # benchmark at runtime
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

Generate a profile for your GPU with:

```bash
python -m disentangled_flash.tune \
  --output rtx-6000-ada.json
```

```bash
python -m disentangled_flash.tune inspect rtx-6000-ada.json
```

The default `standard` preset is cheap: lengths 128, 512, 2048, and 8192 in BF16
over padded and packed layouts, 84 phase workloads in total. Each one measures
only the heuristic config and its three closest candidates. Shapes it doesn't
cover use the heuristic at runtime.

`--preset exhaustive` is what the bundled profiles use: every length family
(`64` through `8192`), both launch-occupancy regimes, half precision plus strict
and fast FP32, and the full candidate lists, for 1134 phase workloads. Like
FlashAttention and FlexAttention, FP16 and BF16 share one half-precision family,
and FP32 searches only single-stage schedules with small tiles. Training is
tuned with and without attention dropout.

Either way, configs that don't fit the GPU's shared memory are skipped before
compiling, results are parity-checked, and each workload is saved as soon as it
finishes, so tuning can resume. `--passes`, `--dropout off|on`, and the shape
options restrict a run; `--verbose` prints every measured candidate, which is
handy for comparing configs at one exact shape, e.g.
`--preset exhaustive --lengths 128 --batch-heads 1536 --verbose`.

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

To reproduce the release parity run with the portable heuristic and no runtime
autotuning:

```bash
python -m benchmarks.evaluate_mnli \
  --dtype fp16 \
  --batch-size 16 \
  --bucket 512 \
  --require-full-parity \
  --tuning-mode heuristic \
  --output mnli_parity_fp16_heuristic.json
```

### H200 MNLI decision parity

The full `validation_matched` split (9,815 examples; DeBERTa-v2-xlarge-MNLI)
was evaluated in FP16 at batch size 16 using the portable heuristic. All
implementations and layouts had the same accuracy (91.74%) and zero
classification-decision mismatches against Hugging Face:

| Variant | Full-split time | Speedup vs. HF eager | Decision mismatches | Maximum absolute logit difference |
| --- | ---: | ---: | ---: | ---: |
| Hugging Face eager, padded | 53.27 s | 1.00× | Reference | 0 |
| DF PyTorch, padded | 55.58 s | 0.96× | 0 / 9,815 | 0.1094 |
| DF PyTorch, packed | 49.93 s | 1.07× | 0 / 9,815 | 0.1074 |
| DF Triton, padded | 23.25 s | 2.29× | 0 / 9,815 | 0.0752 |
| DF Triton, packed | 6.94 s | 7.68× | 0 / 9,815 | 0.0752 |
| FlashDeBERTa, packed | 22.00 s | 2.42× | 0 / 9,815 | 0.0801 |

This is decision parity, not bitwise or elementwise logit parity: the maximum
logit differences are nonzero. The recorded BF16 run did not meet full decision
parity, so the FP16 result above is the release parity claim. Full-split times
are from one evaluation run on the H200 at batch size 16, and include the
benchmark's end-to-end MNLI evaluation path; they are not kernel-only timings.

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

The BF16 check used a 12-layer DeBERTa-v2-shaped model, length 1024, batch 2,
dropout 0.1, and 20 optimizer steps. It passed the configured final fixed-loss
gate: the candidate/reference fixed losses after step 20 were 0.683213 and
0.683757, respectively (0.0796% relative difference, below the 5% limit).
Reference and candidate fixed losses fell by 0.4446 and 0.4452 over the run.
The check also records backward gradients and post-update parameters: at the
last step, aggregate gradient relative L2 difference was 14.75% (cosine
similarity 0.9891), and parameter relative L2 difference was 0.42% (cosine
similarity 0.99999). Thus this is a multi-step training-path parity check
including backward, not a claim that individual gradients are numerically
identical. End-to-end performance is not part of the release benchmark matrix.

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
  version = {1.0.0},
  year = {2026}
}
```
