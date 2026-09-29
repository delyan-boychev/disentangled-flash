import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_cuda.py"
ENCODER_BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_encoder.py"
TRAINING_BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_training.py"
MNLI_EVALUATION_PATH = ROOT / "benchmarks" / "evaluate_mnli.py"
MULTISTEP_VALIDATION_PATH = ROOT / "validation" / "validate_multistep_training.py"


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


def load_encoder_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_encoder",
        ENCODER_BENCHMARK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_training",
        TRAINING_BENCHMARK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_benchmark_defaults_to_constant_token_kernel_matrix():
    benchmark = load_benchmark_module()
    args = benchmark.build_parser().parse_args([])

    assert args.implementations == ["base", "torch", "triton", "flashdeberta"]
    assert "flex" not in args.implementations
    assert args.passes == ["forward", "forward_backward"]
    assert args.lengths == [128, 512, 1024, 2048, 4096, 8192]
    assert args.total_tokens == 16_384
    assert args.head_dim == 64
    assert args.dtype == "bf16"
    assert args.training_dropouts == [0.0]


def test_flashdeberta_config_uses_transformers_compatible_position_types():
    benchmark = load_benchmark_module()
    config = benchmark.make_config(64, 0.0)

    values = benchmark._flashdeberta_config_kwargs(config)

    assert values["pos_att_type"] == ["p2c", "c2p"]


def test_cuda_benchmark_uses_constant_token_batches_and_reports_effective_flops():
    benchmark = load_benchmark_module()

    assert benchmark.batch_size_for_length(128, 16_384) == 128
    assert benchmark.batch_size_for_length(512, 16_384) == 32
    assert benchmark.batch_size_for_length(8192, 16_384) == 2
    forward = benchmark.effective_attention_flops(2, 12, 512, 64, "forward")
    combined = benchmark.effective_attention_flops(2, 12, 512, 64, "forward_backward")
    assert combined == int(forward * 3.5)


def test_torch_kernel_backend_runs_active_tokens_without_layout_axis():
    benchmark = load_benchmark_module()
    config = benchmark.make_config(64, 0.0)
    reference = benchmark.OriginalDisentangledSelfAttention(config)
    module = benchmark.make_module(
        "torch",
        config,
        reference.state_dict(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        pass_mode="forward_backward",
        tuning_mode="auto",
        profile_paths=(),
    ).eval()
    hidden = torch.randn(2, 7, config.hidden_size)
    mask = torch.ones(2, 7, dtype=torch.bool)
    relative = torch.randn(config.position_buckets * 2, config.hidden_size)

    output = module(hidden, mask, relative)

    assert output.shape == hidden.shape


def test_kernel_benchmark_selects_inference_and_training_triton_paths():
    benchmark = load_benchmark_module()
    config = benchmark.make_config(64, 0.0)
    reference = benchmark.OriginalDisentangledSelfAttention(config)
    options = {
        "implementation": "triton",
        "config": config,
        "reference_state": reference.state_dict(),
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "tuning_mode": "auto",
        "profile_paths": (),
    }

    inference = benchmark.make_module(pass_mode="forward", **options)
    training = benchmark.make_module(pass_mode="forward_backward", **options)

    assert isinstance(inference.module, benchmark.InferenceDisentangledSelfAttention)
    assert isinstance(training.module, benchmark.TritonTrainingDisentangledSelfAttention)


def test_cuda_benchmark_reports_only_real_layouts():
    benchmark = load_encoder_benchmark_module()

    assert benchmark.unsupported_layout_reason("base", "packed") is not None
    assert benchmark.unsupported_layout_reason("flashdeberta", "padded") is not None
    assert benchmark.unsupported_layout_reason("torch", "padded") is None
    assert benchmark.unsupported_layout_reason("torch", "packed") is None
    assert benchmark.unsupported_layout_reason("triton", "padded") is None
    assert benchmark.unsupported_layout_reason("triton", "packed") is None


def test_cuda_benchmark_uses_distinct_reproducible_input_batches():
    benchmark = load_encoder_benchmark_module()
    embedding = torch.arange(256, dtype=torch.float32).view(64, 4)

    first = benchmark.make_inputs(embedding, 3, 8, 3, 100, 0.5)
    repeated = benchmark.make_inputs(embedding, 3, 8, 3, 100, 0.5)
    measured = benchmark.make_inputs(embedding, 3, 8, 3, 10_100, 0.5)

    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, repeated))
    assert any(not torch.equal(first[index][0], first[index + 1][0]) for index in range(2))
    assert all(not torch.equal(a[0], b[0]) for a, b in zip(first, measured))
    assert all(mask.dtype == torch.long for _, mask in first)


def test_system_details_are_json_serializable_and_safe():
    benchmark = load_encoder_benchmark_module()

    details = benchmark.collect_system_details()

    assert {"os", "cpu", "system_memory", "cuda", "nvidia_smi", "packages", "git"} <= set(details)
    assert set(details["environment"]) <= set(benchmark.RELEVANT_ENVIRONMENT_VARIABLES)
    json.dumps(details)


def test_cuda_summary_handles_oom_and_unavailable_parity(capsys):
    benchmark = load_encoder_benchmark_module()
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

    benchmark = load_encoder_benchmark_module()
    signature = inspect.signature(benchmark.make_models)
    assert "assume_unpadded" in signature.parameters


def test_training_benchmark_exposes_strict_profile_only_mode():
    benchmark = load_training_benchmark_module()

    args = benchmark.build_parser().parse_args(
        ["--tuning-mode", "profile_only", "--profile", "h200.json"]
    )

    assert args.tuning_mode == "profile_only"
    assert args.profile == ["h200.json"]


def test_training_benchmark_defaults_to_deberta_base_constant_tokens():
    benchmark = load_training_benchmark_module()
    args = benchmark.build_parser().parse_args([])

    assert args.lengths == [128, 512, 2048, 8192]
    assert args.total_tokens == 16_384
    assert args.batch_size is None
    assert args.dtype == "bf16"
    assert args.dropout == 0.1
    assert benchmark.batch_size_for_length(8192, args.total_tokens, args.batch_size) == 2

    config = benchmark.make_config(8192, args.dropout)
    assert config.num_hidden_layers == 12
    assert config.intermediate_size == 3072


def load_mnli_evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_mnli",
        MNLI_EVALUATION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    return evaluation


def test_mnli_evaluation_defaults_to_full_single_pass_matrix():
    evaluation = load_mnli_evaluation_module()
    args = evaluation.parse_args([])

    assert args.implementations == ["base", "torch", "triton", "flashdeberta"]
    assert args.layouts == ["padded", "packed"]
    assert args.split == "validation_matched"
    assert args.batch_size == 8
    assert args.limit == 0
    assert args.runs == 1
    assert args.dtype == "bf16"
    assert args.output == "mnli_parity.json"
    assert evaluation.requested_variants(args.implementations, args.layouts) == (
        evaluation.Variant("base", "padded"),
        evaluation.Variant("torch", "padded"),
        evaluation.Variant("torch", "packed"),
        evaluation.Variant("triton", "padded"),
        evaluation.Variant("triton", "packed"),
        evaluation.Variant("flashdeberta", "packed"),
    )


def test_mnli_evaluation_rows_are_never_repeated_to_fill_a_batch():
    evaluation = load_mnli_evaluation_module()

    ranges = evaluation.batch_ranges(10, 4)

    assert ranges == ((0, 4), (4, 8), (8, 10))
    assert [index for start, end in ranges for index in range(start, end)] == list(range(10))


def test_mnli_full_parity_means_no_decision_mismatches():
    evaluation = load_mnli_evaluation_module()

    assert evaluation.has_full_parity(0)
    assert not evaluation.has_full_parity(1)


def test_mnli_evaluation_exposes_strict_profile_only_mode():
    evaluation = load_mnli_evaluation_module()

    args = evaluation.parse_args(
        ["--tuning-mode", "profile_only", "--profile", "h200.json", "--runs", "3"]
    )

    assert args.tuning_mode == "profile_only"
    assert args.profile == ["h200.json"]
    assert args.runs == 3


def test_multistep_parity_uses_base_shape_and_dropout():
    spec = importlib.util.spec_from_file_location(
        "validate_multistep_training",
        MULTISTEP_VALIDATION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)

    config = validation.make_config(8192, 0.1)
    assert config.vocab_size == 128_100
    assert config.num_hidden_layers == 12
    assert config.intermediate_size == 3072
    assert config.attention_probs_dropout_prob == 0.1


def test_mnli_packed_path_runs_classifier_without_padding_attention():
    evaluation = load_mnli_evaluation_module()

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
            assert output_hidden_states is False
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
    logits = evaluation.forward_model(
        Model(),
        inputs,
        layout="packed",
    )

    torch.testing.assert_close(logits, torch.tensor([[2.0, 12.0], [4.0, 14.0]]))
