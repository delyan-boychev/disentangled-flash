import torch

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
)
from disentangled_flash.deberta import DebertaV2OptimizedEncoder


def _torch_encoder_with_mask_spy(monkeypatch) -> tuple[DebertaV2OptimizedEncoder, list[bool]]:
    from disentangled_flash import _torch

    config = DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_size=16,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=64,
        max_position_embeddings=64,
        position_buckets=32,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        conv_kernel_size=0,
    )
    torch.manual_seed(0)
    encoder = DebertaV2OptimizedEncoder(
        DebertaV2Encoder(config),
        config,
        backend="torch",
        inference=True,
    ).eval()
    encoder.prepare_for_inference(6)

    skipped: list[bool] = []
    attention = _torch._torch_disentangled_attention

    def spy(*args, **kwargs):
        skipped.append(kwargs["assume_unpadded"])
        return attention(*args, **kwargs)

    monkeypatch.setattr(_torch, "_torch_disentangled_attention", spy)
    return encoder, skipped


def test_torch_encoder_skips_mask_for_unpadded_batch(monkeypatch):
    encoder, skipped = _torch_encoder_with_mask_spy(monkeypatch)
    hidden = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.long)

    with torch.inference_mode():
        plans = encoder._active_plans
        expected = encoder.forward_prepared(hidden, mask, plans).last_hidden_state
        assert skipped == [False, False]
        actual = encoder(hidden, mask).last_hidden_state

    assert skipped == [False, False, True, True]
    assert torch.equal(actual, expected)


def test_torch_encoder_keeps_mask_for_padded_batch(monkeypatch):
    encoder, skipped = _torch_encoder_with_mask_spy(monkeypatch)
    hidden = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[1, 4:] = 0

    with torch.inference_mode():
        expected = encoder.forward_prepared(hidden, mask, encoder._active_plans).last_hidden_state
        actual = encoder(hidden, mask).last_hidden_state
        unmasked = encoder.forward_prepared(
            hidden, mask, encoder._active_plans, assume_unpadded=True
        ).last_hidden_state

    assert skipped == [False, False, False, False, True, True]
    assert torch.equal(actual, expected)
    assert not torch.equal(unmasked, expected)
