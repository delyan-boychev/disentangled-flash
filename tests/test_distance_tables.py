import pytest
import torch

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash._torch import (
    TorchInferenceDisentangledSelfAttention,
    TorchPositionPlan,
    _scores_by_distance,
)


def make_config(
    pos_att_type: tuple[str, ...] = ("c2p", "p2c"),
    position_buckets: int = 32,
) -> DebertaAttentionConfig:
    return DebertaAttentionConfig(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=64,
        max_position_embeddings=64,
        position_buckets=position_buckets,
        share_att_key=True,
        pos_att_type=pos_att_type,
        norm_rel_ebd="none",
    )


def make_module(
    config: DebertaAttentionConfig,
) -> tuple[TorchInferenceDisentangledSelfAttention, torch.Tensor]:
    torch.manual_seed(0)
    module = TorchInferenceDisentangledSelfAttention(config).eval()
    relative = torch.randn(module.pos_ebd_size * 2, config.hidden_size)
    module.prepare_for_inference(relative)
    return module, relative


def gather_plan(
    module: TorchInferenceDisentangledSelfAttention,
    sequence_length: int,
) -> TorchPositionPlan:
    """The plan prepare_shape() built before distance-ordered tables."""

    indices = module.position_plan_cache.dense(sequence_length, "cpu")
    pos_key, pos_query = module._project_active_positions(indices.active_slots)
    return TorchPositionPlan(
        sequence_length=sequence_length,
        active_slots=indices.active_slots,
        c2p_local=indices.pair_to_local,
        p2c_local=indices.pair_to_local,
        pos_key=pos_key,
        pos_query=pos_query,
    )


@pytest.mark.parametrize("length", [1, 2, 5, 9])
def test_scores_by_distance_reads_the_diagonal_band(length):
    scores = torch.randn(2, 3, length, 2 * length - 1)

    band = _scores_by_distance(scores)

    assert band.shape == (2, 3, length, length)
    assert band.untyped_storage().data_ptr() == scores.untyped_storage().data_ptr()
    for i in range(length):
        for j in range(length):
            assert torch.equal(band[..., i, j], scores[..., i, length - 1 - i + j])


@pytest.mark.parametrize("pos_att_type", [("c2p", "p2c"), ("c2p",), ("p2c",)])
@pytest.mark.parametrize("position_buckets", [32, 0])
@pytest.mark.parametrize("length", [1, 2, 7, 33, 100])
def test_prepared_plan_matches_gather_plan(pos_att_type, position_buckets, length):
    module, _ = make_module(make_config(pos_att_type, position_buckets))
    hidden = torch.randn(3, length, 32)
    mask = torch.ones(3, length, dtype=torch.bool)
    mask[1, length // 2 + 1 :] = False

    plan = module.prepare_shape(length)
    with torch.inference_mode():
        expected, _ = module.forward_prepared(hidden, mask, gather_plan(module, length))
        actual, _ = module.forward_prepared(hidden, mask, plan)

    assert plan.by_distance
    assert plan.c2p_local is None and plan.p2c_local is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("length", [1, 7, 100])
def test_prepared_plan_matches_reference_attention(length):
    config = make_config()
    module, relative = make_module(config)
    reference = OriginalDisentangledSelfAttention(config).eval()
    reference.load_state_dict(module.state_dict())
    hidden = torch.randn(2, length, 32)
    mask = torch.ones(2, length, dtype=torch.bool)

    with torch.inference_mode():
        dense_mask = _prepare_attention_mask(mask, length, length)
        expected = reference(hidden, dense_mask, rel_embeddings=relative)[0]
        actual, _ = module.forward_prepared(hidden, mask, module.prepare_shape(length))

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("lengths", [(40, 7, 23), (5, 6, 7, 8, 9), (3, 100, 4)])
def test_lengths_share_distance_tables_in_any_order(lengths):
    module, _ = make_module(make_config())
    inputs = {length: torch.randn(1, length, 32) for length in lengths}
    masks = {length: torch.ones(1, length, dtype=torch.bool) for length in lengths}
    with torch.inference_mode():
        expected = {
            length: module.forward_prepared(hidden, masks[length], gather_plan(module, length))[0]
            for length, hidden in inputs.items()
        }

    plans = {length: module.prepare_shape(length) for length in lengths}
    module.release_position_projection_workspace()

    storages = {plan.pos_key.untyped_storage().data_ptr() for plan in plans.values()}
    assert len(storages) < len(lengths)
    for length, plan in plans.items():
        assert plan.pos_key.shape == (2, 2 * length - 1, 16)
        assert plan.pos_query.shape == (2, 2 * length - 1, 16)
        with torch.inference_mode():
            actual, _ = module.forward_prepared(inputs[length], masks[length], plan)
        torch.testing.assert_close(actual, expected[length], rtol=0, atol=1e-6)


def test_distance_tables_are_cleared_with_the_inference_cache():
    module, _ = make_module(make_config())
    module.prepare_shape(12)
    assert module._distance_position_cache

    module.clear_inference_cache()

    assert not module._distance_position_cache
