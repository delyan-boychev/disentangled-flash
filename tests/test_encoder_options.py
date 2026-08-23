from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
)
from disentangled_flash.deberta import DebertaV2InferenceEncoder


def test_encoder_propagates_assume_unpadded():
    config = DebertaAttentionConfig(
        hidden_size=64,
        num_attention_heads=1,
        attention_head_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
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

    source = DebertaV2Encoder(config)

    encoder = DebertaV2InferenceEncoder(
        source,
        config,
        backend="triton",
        assume_unpadded=True,
    )

    assert encoder.assume_unpadded is True

    for layer in encoder.layer:
        assert layer.attention.self.assume_unpadded is True
