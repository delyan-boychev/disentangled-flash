import torch

from disentangled_flash.position import SharedPositionPlanCache


def test_compact_position_plan_is_linear_size():
    cache = SharedPositionPlanCache(
        position_buckets=256,
        max_relative_positions=512,
        position_embedding_size=256,
        uses_position_bias=True,
    )
    plan = cache.compact(65, torch.device("cpu"))
    assert plan.delta_to_local.shape == (129,)
    assert plan.active_slots.ndim == 1
    assert plan.active_slots.numel() <= 129

def test_active_position_slots_are_contiguous_ranges():
    cache = SharedPositionPlanCache(
        position_buckets=256,
        max_relative_positions=512,
        position_embedding_size=256,
        uses_position_bias=True,
    )

    for sequence_length in (
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ):
        plan = cache.compact(
            sequence_length,
            torch.device("cpu"),
        )

        active = plan.active_slots

        if active.numel() == 0:
            continue

        expected = torch.arange(
            active[0].item(),
            active[-1].item() + 1,
            dtype=active.dtype,
        )

        assert torch.equal(active, expected)
