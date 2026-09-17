import torch

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


def test_packed_encoder_preserves_convolution_mask_dtype():
    config = DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=1,
        attention_head_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=16,
        max_position_embeddings=16,
        position_buckets=-1,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        conv_kernel_size=3,
    )
    encoder = DebertaV2InferenceEncoder(
        DebertaV2Encoder(config),
        config,
        backend="torch",
    ).eval()
    hidden_states = torch.randn(5, 32)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    with torch.inference_mode():
        output = encoder.forward_packed(
            hidden_states,
            cu_seqlens,
            max_seqlen=3,
            output_hidden_states=False,
        )

    assert output.last_hidden_state.shape == hidden_states.shape


def test_packed_encoder_batched_convolution_matches_individual_sequences():
    config = DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=1,
        attention_head_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=16,
        max_position_embeddings=16,
        position_buckets=-1,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        conv_kernel_size=3,
    )
    encoder = DebertaV2InferenceEncoder(
        DebertaV2Encoder(config),
        config,
        backend="torch",
    ).eval()
    sequences = (torch.randn(3, 32), torch.randn(5, 32), torch.randn(2, 32))
    packed = torch.cat(sequences)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32)

    with torch.inference_mode():
        encoder.prepare_for_inference([2, 3, 5])
        expected = []
        for sequence in sequences:
            encoder.activate_shape(sequence.size(0))
            output = encoder(
                sequence.unsqueeze(0),
                torch.ones(1, sequence.size(0), dtype=torch.long),
                output_hidden_states=False,
            )
            expected.append(output.last_hidden_state.squeeze(0))
        actual = encoder.forward_packed(
            packed,
            cu_seqlens,
            max_seqlen=5,
            output_hidden_states=False,
        ).last_hidden_state

    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-5)
