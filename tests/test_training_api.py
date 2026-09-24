from __future__ import annotations

import copy

import pytest
import torch

import disentangled_flash
from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
    OriginalDisentangledSelfAttention,
)
from disentangled_flash._torch import TorchTrainingDisentangledSelfAttention
from disentangled_flash.deberta import DebertaV2OptimizedEncoder
from disentangled_flash.training import TritonTrainingDisentangledSelfAttention
from disentangled_flash.tuning import KernelConfig, KernelTuningOptions


def make_config() -> DebertaAttentionConfig:
    return DebertaAttentionConfig(
        hidden_size=256,
        num_attention_heads=4,
        attention_head_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_relative_positions=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        norm_rel_ebd="none",
    )


def test_training_attention_preserves_parameter_layout() -> None:
    config = make_config()
    reference = OriginalDisentangledSelfAttention(config)
    target = TritonTrainingDisentangledSelfAttention(config)
    assert tuple(reference.state_dict()) == tuple(target.state_dict())
    target.load_state_dict(reference.state_dict(), strict=True)


def test_training_attention_accepts_shared_tuning_options() -> None:
    tuning = KernelTuningOptions(
        mode="fixed",
        fixed_config=KernelConfig(32, 64, 4),
    )
    target = TritonTrainingDisentangledSelfAttention(make_config(), tuning=tuning)

    assert target.tuning is tuning


def test_training_attention_falls_back_to_autotune_when_saved_schedule_fails() -> None:
    config = KernelConfig(32, 64, 4)
    target = TritonTrainingDisentangledSelfAttention(
        make_config(),
        tuning=KernelTuningOptions(mode="fixed", fixed_config=config),
    )
    launches = []

    def launch(forward_config, dq_config, dkv_config):
        launches.append((forward_config, dq_config, dkv_config))
        if forward_config is not None:
            raise RuntimeError("out of resources")
        return "autotuned"

    options = {
        "hidden_states": torch.empty(1, 1, 256),
        "sequence_length": 128,
        "batch_heads": 4,
        "active_slots": 128,
        "has_c2p": True,
        "has_p2c": True,
        "layout": "padded",
        "uses_padding_mask": True,
    }
    with torch.enable_grad():
        assert target._launch_with_profile_fallback(launch, options) == "autotuned"
    assert launches == [(config, config, config), (None, None, None)]


def test_profile_only_training_attention_raises_when_saved_schedule_fails(monkeypatch) -> None:
    target = TritonTrainingDisentangledSelfAttention(
        make_config(),
        tuning=KernelTuningOptions(mode="profile_only", use_bundled_profiles=False),
    )
    monkeypatch.setattr(target, "_resolve_kernel_config", lambda **_kwargs: KernelConfig(32, 64, 4))

    def launch(*_configs):
        raise RuntimeError("out of resources")

    with pytest.raises(RuntimeError, match="saved training kernel configuration failed"):
        target._launch_with_profile_fallback(launch, {"hidden_states": torch.empty(1)})


def test_training_public_api_is_exported() -> None:
    assert disentangled_flash.TritonTrainingDisentangledSelfAttention is not None
    assert disentangled_flash.DebertaV2OptimizedEncoder is not None
    assert callable(disentangled_flash.enable_deberta_training)
    assert callable(disentangled_flash.optimize_deberta_training)


def test_torch_training_fallback_matches_reference_and_preserves_grad_in_eval() -> None:
    torch.manual_seed(0)
    config = make_config()
    reference = OriginalDisentangledSelfAttention(config).eval()
    target = TorchTrainingDisentangledSelfAttention(config).eval()
    target.load_state_dict(reference.state_dict(), strict=True)

    hidden = torch.randn(2, 17, config.hidden_size, requires_grad=True)
    target_hidden = hidden.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [[1] * 13 + [0] * 4, [1] * 9 + [0] * 8],
        dtype=torch.long,
    )
    relative = torch.randn(
        config.position_buckets * 2,
        config.hidden_size,
        requires_grad=True,
    )
    target_relative = relative.detach().clone().requires_grad_(True)

    expanded_mask = mask[:, None, None, :] * mask[:, None, :, None]
    expected, _ = reference(
        hidden,
        expanded_mask,
        rel_embeddings=relative,
    )
    actual, _ = target(
        target_hidden,
        mask,
        rel_embeddings=target_relative,
    )

    torch.testing.assert_close(actual, expected)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(target_hidden.grad, hidden.grad)
    torch.testing.assert_close(target_relative.grad, relative.grad)


def test_unified_training_encoder_auto_falls_back_to_torch_on_cpu() -> None:
    config = make_config()
    config = DebertaAttentionConfig(
        **{
            **config.__dict__,
            "num_hidden_layers": 1,
            "intermediate_size": 512,
        }
    )
    source = DebertaV2Encoder(config)
    reference_keys = tuple(source.state_dict())
    target = DebertaV2OptimizedEncoder(
        copy.deepcopy(source),
        config,
        backend="auto",
        inference=False,
    )

    assert target.backend == "torch"
    assert target.inference is False
    assert tuple(target.state_dict()) == reference_keys

    target.eval()
    hidden = torch.randn(1, 8, config.hidden_size, requires_grad=True)
    mask = torch.ones(1, 8, dtype=torch.long)
    output = target(hidden, mask, output_hidden_states=False).last_hidden_state
    output.float().square().mean().backward()
    assert hidden.grad is not None

    with torch.no_grad():
        no_grad_output = target(
            hidden.detach(),
            mask,
            output_hidden_states=False,
        ).last_hidden_state
    assert not no_grad_output.requires_grad
