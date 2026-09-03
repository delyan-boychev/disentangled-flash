import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_cuda.py"
PARITY_PATH = ROOT / "benchmarks" / "parity_pretrained_mnli.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_cuda",
        BENCHMARK_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_parity_module():
    spec = importlib.util.spec_from_file_location(
        "parity_pretrained_mnli",
        PARITY_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_cuda_benchmark_exposes_unpadded_mode():
    benchmark = load_benchmark_module()

    args = benchmark.build_parser().parse_args(
        [
            "--assume-unpadded",
            "--minimum-length-fraction",
            "1.0",
        ]
    )

    assert args.assume_unpadded is True
    assert args.minimum_length_fraction == 1.0


def test_make_models_accepts_unpadded_mode():
    import inspect

    benchmark = load_benchmark_module()
    signature = inspect.signature(benchmark.make_models)
    assert "assume_unpadded" in signature.parameters


def test_pretrained_parity_accepts_torch_backend(monkeypatch):
    parity = load_parity_module()
    monkeypatch.setattr("sys.argv", [str(PARITY_PATH), "--backend", "torch"])

    args = parity.parse_args()

    assert args.backend == "torch"
