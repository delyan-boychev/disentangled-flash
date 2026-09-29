import pytest
import torch

from disentangled_flash import kernel


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(kernel.triton is None, reason="Triton is not installed")
def test_online_softmax_handles_completely_masked_first_tile():
    # Only the last token is kept, so every schedule hits a fully masked tile first.
    sequence_length = 65
    head_dim = 32
    query = torch.randn(1, 1, sequence_length, head_dim, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_mask = torch.zeros(1, sequence_length, dtype=torch.bool, device="cuda")
    attention_mask[:, -1] = True
    delta_to_local = torch.zeros(
        2 * sequence_length - 1,
        dtype=torch.int32,
        device="cuda",
    )

    output = kernel._deberta_attention_op(
        query,
        key,
        value,
        query,
        key,
        delta_to_local,
        attention_mask,
        1,
        sequence_length,
        128,
        1,
        sequence_length - 1,
        1.0,
        True,
        False,
        False,
        False,
        True,
        True,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, -1], value[:, 0, -1], rtol=0.0, atol=1e-6)


def test_head_major_scores_match_matmul_without_copying_the_input():
    from disentangled_flash.kernel import _head_major_scores, _relative_strides

    batch, heads, length, head_dim, slots = 3, 4, 5, 8, 16
    fused = torch.randn(batch, length, 3 * heads * head_dim, dtype=torch.float64)
    query = fused[..., : heads * head_dim].view(batch, length, heads, head_dim).permute(0, 2, 1, 3)
    table = torch.randn(heads, slots, head_dim, dtype=torch.float64)

    scores = _head_major_scores(query, table)

    torch.testing.assert_close(scores, torch.matmul(query, table.transpose(-1, -2)))
    assert scores.permute(1, 0, 2, 3).is_contiguous()
    strides = _relative_strides(scores, scores, True, True)
    assert strides == {
        "stride_rb": length * slots,
        "stride_rh": batch * length * slots,
        "stride_rl": slots,
    }
    assert _relative_strides(query, query, False, False)["stride_rb"] == 0
