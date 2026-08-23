from __future__ import annotations

import disentangled_flash

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
)
from disentangled_flash.training import TritonTrainingDisentangledSelfAttention


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


def test_training_public_api_is_exported() -> None:
    assert disentangled_flash.TritonTrainingDisentangledSelfAttention is not None
    assert disentangled_flash.DebertaV2TrainingEncoder is not None
    assert callable(disentangled_flash.enable_deberta_training)
    assert callable(disentangled_flash.optimize_deberta_training)
