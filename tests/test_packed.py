import pytest
import torch

from disentangled_flash._reference import DebertaAttentionConfig
from disentangled_flash._torch import TorchInferenceDisentangledSelfAttention
from disentangled_flash.packed import pack_padded, unpack_packed, validate_cu_seqlens


def test_pack_and_unpack_right_padded_batch():
    hidden = torch.arange(2 * 4 * 3).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    packed, cu_seqlens, max_seqlen = pack_padded(hidden, mask)
    restored, restored_mask = unpack_packed(packed, cu_seqlens, 4)

    assert cu_seqlens.tolist() == [0, 3, 4]
    assert max_seqlen == 3
    assert torch.equal(restored_mask, mask)
    assert torch.equal(restored[mask], hidden[mask])


def test_packed_attention_matches_independent_unpadded_sequences():
    config = DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=1,
        attention_head_size=32,
        relative_attention=True,
        max_relative_positions=16,
        max_position_embeddings=16,
        position_buckets=-1,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        norm_rel_ebd="none",
    )
    module = TorchInferenceDisentangledSelfAttention(config).eval()
    relative = torch.randn(32, 32)
    sequences = (torch.randn(3, 32), torch.randn(5, 32), torch.randn(2, 32))
    packed = torch.cat(sequences)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32)

    with torch.inference_mode():
        module.prepare_for_inference(relative)
        expected = torch.cat(
            [
                module(
                    sequence.unsqueeze(0),
                    torch.ones(1, sequence.size(0), dtype=torch.bool),
                )[0].squeeze(0)
                for sequence in sequences
            ]
        )
        actual, _ = module.forward_packed(
            packed,
            cu_seqlens,
            max_seqlen=5,
            rel_embeddings=relative,
        )

    torch.testing.assert_close(actual, expected)


def test_cu_seqlens_validation_rejects_bad_boundaries():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_cu_seqlens(torch.tensor([0, 2, 2, 4], dtype=torch.int32), 4)
