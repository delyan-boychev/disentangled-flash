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
