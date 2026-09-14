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


def test_family_position_plan_covers_shorter_runtime_length():
    cache = SharedPositionPlanCache(
        position_buckets=256,
        max_relative_positions=512,
        position_embedding_size=256,
        uses_position_bias=True,
    )
    runtime = cache.compact(383, torch.device("cpu"))
    family = cache.compact(384, torch.device("cpu"))
    positions = torch.arange(383)
    runtime_indices = positions[:, None] - positions[None, :] + 382
    family_indices = positions[:, None] - positions[None, :] + 383

    runtime_slots = runtime.active_slots[runtime.delta_to_local[runtime_indices].long()]
    family_slots = family.active_slots[family.delta_to_local[family_indices].long()]

    assert torch.equal(family_slots, runtime_slots)
