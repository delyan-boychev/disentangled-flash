import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_cuda.py"
MNLI_BENCHMARK_PATH = ROOT / "benchmarks" / "parity_pretrained_mnli.py"


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


def test_mnli_benchmark_defaults_to_packed_layout():
    spec = importlib.util.spec_from_file_location(
        "parity_pretrained_mnli",
        MNLI_BENCHMARK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    assert benchmark.parse_args([]).layout == "packed"
    assert benchmark.parse_args(["--layout", "padded"]).layout == "padded"


def test_mnli_packed_path_runs_classifier_without_padding_attention():
    spec = importlib.util.spec_from_file_location(
        "parity_pretrained_mnli_packed",
        MNLI_BENCHMARK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class Embeddings(nn.Module):
        def forward(self, *, input_ids, token_type_ids, mask):
            del token_type_ids, mask
            values = input_ids.float()
            return torch.stack((values, values + 10), dim=-1)

    class Encoder(nn.Module):
        def forward_packed(
            self,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            *,
            output_hidden_states,
            return_dict,
        ):
            assert cu_seqlens.tolist() == [0, 2, 3]
            assert max_seqlen == 2
            assert output_hidden_states is True
            assert return_dict is True
            return SimpleNamespace(last_hidden_state=hidden_states + 1)

    class Model(nn.Module):
        base_model_prefix = "deberta"

        def __init__(self):
            super().__init__()
            self.deberta = SimpleNamespace(embeddings=Embeddings(), encoder=Encoder())
            self.pooler = lambda hidden: hidden[:, 0]
            self.dropout = nn.Identity()
            self.classifier = nn.Identity()

    inputs = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
    }
    logits, hidden = benchmark.forward_model(
        Model(),
        inputs,
        layout="packed",
        output_hidden_states=True,
    )

    torch.testing.assert_close(logits, torch.tensor([[2.0, 12.0], [4.0, 14.0]]))
    assert hidden is not None
    assert torch.equal(hidden[:, -1], torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
