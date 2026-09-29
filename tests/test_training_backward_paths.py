"""CUDA check that the atomic-free backward path matches the atomic one."""

from __future__ import annotations

import pytest
import torch

from disentangled_flash import kernel

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    pytest.mark.skipif(kernel.triton is None, reason="Triton is not installed"),
]


def _run(dtype, length, **options):
    from disentangled_flash.position import SharedPositionPlanCache, pad_position_table
    from disentangled_flash.training._kernels import training_attention

    torch.manual_seed(0)
    batch, heads, head_dim = 2, 3, 64
    fused = torch.randn(batch, length, 3 * heads * head_dim, device="cuda", dtype=dtype)
    fused.requires_grad_(True)
    views = [
        fused[..., i * heads * head_dim : (i + 1) * heads * head_dim]
        .view(batch, length, heads, head_dim)
        .permute(0, 2, 1, 3)
        for i in range(3)
    ]
    plan = SharedPositionPlanCache(
        position_buckets=256, max_relative_positions=512, position_embedding_size=256
    ).compact(length, "cuda")
    slots = plan.active_slots.numel()
    pos_key = torch.randn(heads, slots, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    pos_query = torch.randn(heads, slots, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.ones(batch, length, device="cuda", dtype=torch.bool)
    mask[1, length // 2 :] = False
    output = training_attention(
        *views,
        pad_position_table(pos_key),
        pad_position_table(pos_query),
        plan.delta_to_local,
        mask,
        score_scale=(head_dim * 3) ** -0.5,
        has_padding=True,
        strict_fp32=True,
        **options,
    )
    grad = torch.randn_like(output)
    gradients = torch.autograd.grad(output, (fused, pos_key, pos_query), grad)
    return output, gradients, slots == 2 * length - 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_unique_slot_stores_match_atomic_accumulation(dtype):
    output, gradients, unique = _run(dtype, 128)
    assert unique
    unique_output, unique_gradients, _ = _run(dtype, 128, unique_slots=True)

    # The two paths autotune separately, so tiles (and summation order) may differ.
    tolerance = (
        {"rtol": 2e-2, "atol": 2e-2} if dtype == torch.bfloat16 else {"rtol": 1e-4, "atol": 1e-4}
    )
    torch.testing.assert_close(unique_output, output, **tolerance)
    for actual, expected in zip(unique_gradients, gradients):
        torch.testing.assert_close(actual, expected, **tolerance)
