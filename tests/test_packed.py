import pytest
import torch

from disentangled_flash import kernel
from disentangled_flash._reference import DebertaAttentionConfig
from disentangled_flash._torch import (
    TorchInferenceDisentangledSelfAttention,
    TorchTrainingDisentangledSelfAttention,
)
from disentangled_flash.kernel import InferenceDisentangledSelfAttention
from disentangled_flash.packed import (
    pack_padded,
    pack_padded_with_info,
    unpack_packed,
    validate_cu_seqlens,
)
from disentangled_flash.tuning import KernelConfig, KernelTuningOptions


def test_pack_and_unpack_right_padded_batch():
    hidden = torch.arange(2 * 4 * 3).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    packed, cu_seqlens, max_seqlen = pack_padded(hidden, mask)
    restored, restored_mask = unpack_packed(packed, cu_seqlens, 4)

    assert cu_seqlens.tolist() == [0, 3, 4]
    assert max_seqlen == 3
    assert torch.equal(restored_mask, mask)
    assert torch.equal(restored[mask], hidden[mask])


def test_packed_info_can_be_reused_for_vectorized_unpack():
    hidden = torch.arange(3 * 5 * 2).view(3, 5, 2)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    packed, cu_seqlens, info = pack_padded_with_info(hidden, mask)
    restored, restored_mask = unpack_packed(
        packed,
        cu_seqlens,
        hidden.size(1),
        packed_info=info,
    )

    assert info.offsets == (0, 5, 8, 9)
    assert info.lengths == (5, 3, 1)
    assert info.max_seqlen == 5
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


def test_packed_training_fallback_preserves_autograd():
    config = DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=1,
        attention_head_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=16,
        max_position_embeddings=16,
        position_buckets=-1,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        norm_rel_ebd="none",
    )
    module = TorchTrainingDisentangledSelfAttention(config)
    relative = torch.randn(32, 32)
    lengths = (3, 5, 2)
    expected_input = torch.randn(sum(lengths), 32, requires_grad=True)
    packed_input = expected_input.detach().clone().requires_grad_(True)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32)

    expected_parts = []
    start = 0
    for length in lengths:
        output, _ = module(
            expected_input[start : start + length].unsqueeze(0),
            torch.ones(1, length, dtype=torch.bool),
            rel_embeddings=relative,
        )
        expected_parts.append(output.squeeze(0))
        start += length
    expected = torch.cat(expected_parts)
    actual, _ = module.forward_packed(
        packed_input,
        cu_seqlens,
        max_seqlen=5,
        rel_embeddings=relative,
    )

    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(packed_input.grad, expected_input.grad)


def test_cu_seqlens_validation_rejects_bad_boundaries():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_cu_seqlens(torch.tensor([0, 2, 2, 4], dtype=torch.int32), 4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(kernel.triton is None, reason="Triton is not installed")
def test_single_launch_packed_triton_matches_torch_segments():
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
    reference = TorchInferenceDisentangledSelfAttention(config).cuda().half().eval()
    candidate = (
        InferenceDisentangledSelfAttention(
            config,
            backend="triton",
            tuning=KernelTuningOptions(
                mode="fixed",
                fixed_config=KernelConfig(32, 32, 4),
            ),
        )
        .cuda()
        .half()
        .eval()
    )
    candidate.load_state_dict(reference.state_dict())
    relative = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    sequences = tuple(
        torch.randn(length, 32, device="cuda", dtype=torch.float16) for length in (3, 5, 2)
    )
    packed = torch.cat(sequences)
    cu_seqlens = torch.tensor([0, 3, 8, 10], device="cuda", dtype=torch.int32)

    with torch.inference_mode():
        reference.prepare_for_inference(relative)
        expected, _ = reference.forward_packed(packed, cu_seqlens, 5)
        candidate.prepare_for_inference(relative)
        actual, _ = candidate.forward_packed(packed, cu_seqlens, 5)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
