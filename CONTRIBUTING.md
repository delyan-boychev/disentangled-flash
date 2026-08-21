# Contributing to DisentangledFlash

Thank you for contributing to DisentangledFlash! This document provides guidelines and commands to help you develop, test, and benchmark the codebase.

---

## Development Install

We recommend using a virtual environment (via `micromamba`, `conda`, or `venv`) with Python 3.10 or newer.

To install the library in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

If you plan to run Hugging Face integration tests or the MNLI parity benchmarks, install with the `hf` extra:

```bash
pip install -e ".[dev,hf]"
```

---

## CPU Tests

Unit tests that run on CPU (or Apple Silicon MPS) check the PyTorch-optimized backend and public API. You can execute these without a GPU using `pytest`:

```bash
pytest
```

---

## CUDA Tests

If you are developing on a CUDA-enabled system, you can run the full testing and validation suite.

1. **PyTest (including CUDA tests)**:
   ```bash
   pytest -m cuda
   ```

2. **Validation Suite**:
   Runs a validation matrix covering various sequence lengths, data types (FP16/BF16/FP32), masking patterns, and relative position modes:
   ```bash
   python -m validation.validate_cuda
   ```

---

## Formatting and Linting

We enforce strict formatting and import ordering rules using `ruff`. Before submitting your changes, please run:

1. **Lint checks**:
   ```bash
   ruff check .
   ```

2. **Formatting checks**:
   ```bash
   ruff format --check .
   ```

To automatically fix lints (such as import ordering) and format files:
```bash
ruff check --fix .
ruff format .
```

---

## How to Add a GPU Configuration

Triton kernel autotuning options are defined in [kernel.py](file:///Users/delyan-boychev/disentangled-flash/src/disentangled_flash/kernel.py).

To tune kernel performance or add a new GPU configuration:
1. **Define Tuning Candidates**: Add or modify block size and warp combinations in `_AUTOTUNE_CONFIGS` at the top of [kernel.py](file:///Users/delyan-boychev/disentangled-flash/src/disentangled_flash/kernel.py):
   ```python
   triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1)
   ```
2. **Adjust Configuration Pruning**: Modify the `_prune_autotune_configs` helper. This function filters configurations to ensure they are resource-safe (e.g. preventing shared-memory allocation failures or heavy register pressure) based on head dimension, sequence length, and data precision (such as FP32).

---

## Benchmark Methodology

We measure the latency and accuracy of the reference, PyTorch-optimized, and Triton-fused implementations.

### 1. Running Benchmarks
* **CUDA Attention & Encoder Benchmark**:
  ```bash
  python -m benchmarks.benchmark_cuda --scope encoder --dtypes fp16,fp32
  ```
* **MPS Benchmark (macOS)**:
  ```bash
  python -m benchmarks.benchmark_mps
  ```
* **MNLI Pretrained Parity Benchmark**:
  Verifies prediction parity against a Hugging Face pre-trained DeBERTa model:
  ```bash
  python -m benchmarks.parity_pretrained_mnli
  ```

### 2. Performance Metrics
* **Warmup**: All benchmarks execute several warm-up runs to ensure kernels/JIT compiles are cached before timing.
* **Latency (p50/p90)**: Reported values are steady-state statistics of execution times. Triton autotuning and JIT compilation times are excluded from reported figures.
* **Precision/Accuracy**: Maximum and mean absolute errors are checked against the reference implementation to guarantee numerical parity.
