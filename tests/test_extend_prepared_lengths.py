import torch

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
)
from disentangled_flash.deberta import DebertaV2OptimizedEncoder


def _torch_inference_encoder() -> tuple[DebertaV2OptimizedEncoder, DebertaAttentionConfig]:
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
    return encoder, config


def _encode(encoder: DebertaV2OptimizedEncoder, hidden: torch.Tensor) -> torch.Tensor:
    encoder.activate_shape(hidden.size(1))
    mask = torch.ones(hidden.shape[:2], dtype=torch.long)
    with torch.inference_mode():
        return encoder(hidden, mask, output_hidden_states=False).last_hidden_state


def test_extend_prepared_lengths_keeps_existing_plans():
    encoder, _ = _torch_inference_encoder()
    encoder.prepare_for_inference([5, 9])
    existing = dict(encoder._prepared_plans)
    fused = [layer.attention.self._cached_qkv_weight for layer in encoder.layer]

    encoder.extend_prepared_lengths([9, 40, 3])

    assert list(encoder._prepared_plans) == [5, 9, 40, 3]
    assert encoder._active_sequence_length == 5
    for length, plans in existing.items():
        assert all(new is old for new, old in zip(encoder._prepared_plans[length], plans))
    for layer, weight in zip(encoder.layer, fused):
        attention = layer.attention.self
        assert attention._cached_qkv_weight is weight
        assert attention._cached_pos_key is None and attention._cached_pos_query is None


def test_extend_prepared_lengths_matches_full_preparation():
    extended, _ = _torch_inference_encoder()
    reference, _ = _torch_inference_encoder()
    reference.load_state_dict(extended.state_dict())
    lengths = [5, 40, 3, 100]
    extended.prepare_for_inference(lengths[0])
    for length in lengths[1:]:
        extended.extend_prepared_lengths(length)
    reference.prepare_for_inference(lengths)

    for length in lengths:
        hidden = torch.randn(2, length, 32)
        assert torch.equal(_encode(extended, hidden), _encode(reference, hidden))


def test_extend_prepared_lengths_prepares_when_nothing_is_prepared():
    encoder, _ = _torch_inference_encoder()

    encoder.extend_prepared_lengths([7, 4])

    assert list(encoder._prepared_plans) == [7, 4]
    assert encoder._active_sequence_length == 7
    assert _encode(encoder, torch.randn(1, 4, 32)).shape == (1, 4, 32)


def test_extend_prepared_lengths_starts_over_after_cache_invalidation():
    encoder, _ = _torch_inference_encoder()
    encoder.prepare_for_inference(5)
    encoder.clear_inference_cache()

    encoder.extend_prepared_lengths(9)

    assert list(encoder._prepared_plans) == [9]
    assert _encode(encoder, torch.randn(1, 9, 32)).shape == (1, 9, 32)
