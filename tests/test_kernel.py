import pytest
import torch

from disentangled_flash import kernel


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(kernel.triton is None, reason="Triton is not installed")
def test_online_softmax_handles_completely_masked_first_tile():
    # FP32 pruning limits BLOCK_N to 64 here.  With only the final token kept,
    # every candidate must process at least one all-masked tile before it sees
    # the valid key.
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
        1,
        1.0,
        False,
        False,
        False,
        True,
        True,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, -1], value[:, 0, -1], rtol=0.0, atol=1e-6)
