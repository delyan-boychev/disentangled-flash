import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_cuda_benchmark_defaults_to_deberta_v3_base_matrix():
    benchmark = load_benchmark_module()

    args = benchmark.build_parser().parse_args([])

    assert args.scope == "encoder"
    assert args.implementations == ["base", "torch", "triton", "flashdeberta"]
    assert args.layouts == ["padded", "packed"]
    assert args.lengths == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    assert args.hidden_size == 768
    assert args.num_attention_heads == 12
    assert args.num_hidden_layers == 12
    assert args.intermediate_size == 3072
    assert args.vocab_size == 128100
    assert args.parity_samples == 1
    assert args.parity_batch_size == 1


def test_cuda_benchmark_reports_only_real_layouts():
    benchmark = load_benchmark_module()

    assert benchmark.unsupported_layout_reason("base", "packed") is not None
    assert benchmark.unsupported_layout_reason("flashdeberta", "padded") is not None
    assert benchmark.unsupported_layout_reason("torch", "padded") is None
    assert benchmark.unsupported_layout_reason("torch", "packed") is None
    assert benchmark.unsupported_layout_reason("triton", "padded") is None
    assert benchmark.unsupported_layout_reason("triton", "packed") is None


def test_cuda_benchmark_uses_distinct_reproducible_input_batches():
    benchmark = load_benchmark_module()
    embedding = torch.arange(256, dtype=torch.float32).view(64, 4)

    first = benchmark.make_inputs(embedding, 3, 8, 3, 100, 0.5)
    repeated = benchmark.make_inputs(embedding, 3, 8, 3, 100, 0.5)
    measured = benchmark.make_inputs(embedding, 3, 8, 3, 10_100, 0.5)

    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, repeated))
    assert any(not torch.equal(first[index][0], first[index + 1][0]) for index in range(2))
    assert all(not torch.equal(a[0], b[0]) for a, b in zip(first, measured))
    assert all(mask.dtype == torch.long for _, mask in first)


def test_system_details_are_json_serializable_and_safe():
    benchmark = load_benchmark_module()

    details = benchmark.collect_system_details()

    assert {"os", "cpu", "system_memory", "cuda", "nvidia_smi", "packages", "git"} <= set(details)
    assert set(details["environment"]) <= set(benchmark.RELEVANT_ENVIRONMENT_VARIABLES)
    json.dumps(details)


def test_cuda_summary_handles_oom_and_unavailable_parity(capsys):
    benchmark = load_benchmark_module()
    metadata = {
        "implementation": "triton",
        "scope": "encoder",
        "dtype": "fp16",
        "execution": "eager",
        "layout": "packed",
    }
    successful = {
        "status": "ok",
        "batch_size": 1,
        "sequence_length": 64,
        "p50_ms": 1.0,
        "p90_ms": 1.1,
        "docs_per_second": 1000.0,
        "valid_tokens_per_second": 64000.0,
        "max_abs_error": None,
    }
    oom = {
        "status": "oom",
        "batch_size": 32,
        "sequence_length": 8192,
    }

    benchmark.print_summary([{"metadata": metadata, "results": [successful, oom]}])

    output = capsys.readouterr().out
    assert "N/A" in output
    assert "OOM" in output


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


def test_mnli_benchmark_supports_flashdeberta_internal_packed_path():
    spec = importlib.util.spec_from_file_location(
        "parity_pretrained_mnli_flashdeberta",
        MNLI_BENCHMARK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    args = benchmark.parse_args(["--backend", "flashdeberta"])

    assert args.layout == "packed"
    assert benchmark.candidate_execution_layout(args.backend, args.layout) == "padded"
    with pytest.raises(ValueError, match="mask-driven internal varlen"):
        benchmark.candidate_execution_layout("flashdeberta", "padded")


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
            packed_info,
        ):
            assert cu_seqlens.tolist() == [0, 2, 3]
            assert max_seqlen == 2
            assert output_hidden_states is True
            assert return_dict is True
            assert packed_info.offsets == (0, 2, 3)
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
