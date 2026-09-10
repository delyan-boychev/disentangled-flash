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

## How to Add a GPU Tuning Profile

Run the resumable tuner on the target GPU:

```bash
python -m disentangled_flash.tune \
  --preset standard \
  --output my-gpu-profile.json
```

Use `quick` for a smoke test and `exhaustive` for broad calibration. The tuner
rejects configurations that fail to compile, produce non-finite values, or do
not match the reference implementation across several padding patterns.

To test additional schedules, pass a JSON file containing a list of configuration
objects:

```json
[
  {"block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 1}
]
```

```bash
python -m disentangled_flash.tune \
  --preset standard \
  --candidates candidates.json \
  --output my-gpu-profile.json
```

Before contributing a profile, run the CUDA tests and validation suite, retain
the generated environment metadata, and place the reviewed JSON file in
`src/disentangled_flash/profiles/`.

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
