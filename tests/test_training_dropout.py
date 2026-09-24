"""CUDA checks for fused attention-probability dropout."""

from __future__ import annotations

import math

import pytest
import torch

from disentangled_flash import kernel

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    pytest.mark.skipif(kernel.triton is None, reason="Triton is not installed"),
]


def _inputs(batch: int, heads: int, length: int, head_dim: int, dtype: torch.dtype):
    torch.manual_seed(0)
    shape = (batch, heads, length, head_dim)
    tensors = [torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    delta_to_local = torch.zeros(2 * length - 1, device="cuda", dtype=torch.int32)
    mask = torch.ones(batch, length, device="cuda", dtype=torch.bool)
    mask[-1, length // 2 :] = False
    return tensors, delta_to_local, mask


def _reference(query, key, value, mask, keep, dropout_p, score_scale):
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * score_scale
    pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
    scores = scores.masked_fill(~pair_mask, float("-inf"))
    scores = torch.where(~mask[:, None, :, None], torch.zeros_like(scores), scores)
    probabilities = torch.softmax(scores, dim=-1) * keep / (1.0 - dropout_p)
    context = torch.matmul(probabilities, value.float()).to(query.dtype)
    batch, heads, length, head_dim = query.shape
    return context.transpose(1, 2).reshape(batch, length, heads * head_dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_dropout_matches_reference_forward_and_backward(dtype):
    from disentangled_flash.training._kernels import dropout_keep_mask, training_attention

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")
    batch, heads, length, head_dim, dropout_p = 2, 3, 96, 64, 0.25
    (query, key, value), delta_to_local, mask = _inputs(batch, heads, length, head_dim, dtype)
    seed = torch.tensor([1234], device="cuda", dtype=torch.int64)
    score_scale = 1.0 / math.sqrt(head_dim)

    actual = training_attention(
        query,
        key,
        value,
        None,
        None,
        delta_to_local,
        mask,
        score_scale=score_scale,
        has_padding=True,
        strict_fp32=True,
        dropout_p=dropout_p,
        dropout_seed=seed,
    )
    keep = dropout_keep_mask(
        seed, stream_start=0, streams=batch * heads, length=length, dropout_p=dropout_p
    ).view(batch, heads, length, length)
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
    expected = _reference(*reference_inputs, mask, keep, dropout_p, score_scale)
    tolerance = {"rtol": 2e-4, "atol": 2e-4} if dtype == torch.float32 else {}
    torch.testing.assert_close(actual, expected, **(tolerance or {"rtol": 2e-2, "atol": 2e-2}))

    grad_output = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), grad_output)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, grad_output)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, **(tolerance or {"rtol": 3e-2, "atol": 3e-2})
        )


def test_dropout_mask_rate_and_seed_independence():
    from disentangled_flash.training._kernels import dropout_keep_mask

    seed = torch.tensor([7], device="cuda", dtype=torch.int64)
    keep = dropout_keep_mask(seed, stream_start=0, streams=4, length=256, dropout_p=0.1)
    other = dropout_keep_mask(seed + 1, stream_start=0, streams=4, length=256, dropout_p=0.1)
    shifted = dropout_keep_mask(seed, stream_start=1, streams=3, length=256, dropout_p=0.1)

    assert abs(keep.float().mean().item() - 0.9) < 0.01
    assert (keep != other).float().mean().item() > 0.1
    assert not torch.equal(keep[0], keep[1])
    assert torch.equal(keep[1:], shifted)


def test_each_call_draws_a_fresh_dropout_mask():
    from disentangled_flash.training._kernels import training_attention

    (query, key, value), delta_to_local, mask = _inputs(1, 2, 64, 64, torch.float16)
    options = {
        "score_scale": 0.125,
        "has_padding": False,
        "strict_fp32": True,
    }
    first = training_attention(
        query, key, value, None, None, delta_to_local, mask, dropout_p=0.5, **options
    )
    second = training_attention(
        query, key, value, None, None, delta_to_local, mask, dropout_p=0.5, **options
    )
    plain = training_attention(query, key, value, None, None, delta_to_local, mask, **options)

    assert not torch.equal(first, second)
    assert not torch.equal(first, plain)
