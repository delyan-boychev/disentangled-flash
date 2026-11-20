"""Inference caches stay bounded and re-preparation is incremental.

Prepared lengths share one projected position table per layer, the PyTorch backend
keeps no [L, L] gather map per length, and prepare_for_inference() only builds the
lengths that are missing.
"""

import pytest
import torch

from disentangled_flash._reference import DebertaAttentionConfig, DebertaV2Encoder
from disentangled_flash._torch import share_pair_map
from disentangled_flash.deberta import DebertaV2OptimizedEncoder
from disentangled_flash.kernel import InferenceDisentangledSelfAttention
from disentangled_flash.position import (
    SharedPositionPlanCache,
    aligned_slot_count,
    dense_pair_map,
    pad_position_table,
    position_window,
)
from disentangled_flash.tuning import TUNING_SEQUENCE_LENGTHS


def make_config(**overrides) -> DebertaAttentionConfig:
    values = {
        "hidden_size": 32,
        "num_attention_heads": 2,
        "attention_head_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "relative_attention": True,
        "max_relative_positions": 64,
        "max_position_embeddings": 64,
        "position_buckets": 32,
        "share_att_key": True,
        "pos_att_type": ("c2p", "p2c"),
        "conv_kernel_size": 0,
    }
    values.update(overrides)
    return DebertaAttentionConfig(**values)


def make_encoder(config: DebertaAttentionConfig | None = None) -> DebertaV2OptimizedEncoder:
    config = config or make_config()
    torch.manual_seed(0)
    return DebertaV2OptimizedEncoder(
        DebertaV2Encoder(config), config, backend="torch", inference=True
    ).eval()


def encode(encoder: DebertaV2OptimizedEncoder, length: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, length, 32, generator=generator)
    mask = torch.ones(2, length, dtype=torch.long)
    mask[1, length // 2 + 1 :] = 0
    encoder.activate_shape(length)
    with torch.inference_mode():
        return encoder(hidden, mask).last_hidden_state


def test_reprepare_keeps_existing_plans_weights_and_parameters():
    encoder = make_encoder()
    encoder.prepare_for_inference([5, 9])
    plans = dict(encoder._prepared_plans)
    parameters = list(encoder.parameters())
    fused = [layer.attention.self._cached_qkv_weight for layer in encoder.layer]

    encoder.prepare_for_inference([5, 9, 40])

    assert list(encoder._prepared_plans) == [5, 9, 40]
    assert encoder._active_sequence_length == 5
    for length, old in plans.items():
        assert all(new is plan for new, plan in zip(encoder._prepared_plans[length], old))
    assert all(new is old for new, old in zip(encoder.parameters(), parameters))
    for layer, weight in zip(encoder.layer, fused):
        assert layer.attention.self._cached_qkv_weight is weight


def test_reprepare_keeps_exactly_the_requested_lengths_in_order():
    encoder = make_encoder()
    encoder.prepare_for_inference([5, 9, 40])
    kept = encoder._prepared_plans[40]

    encoder.prepare_for_inference([40, 3])

    assert list(encoder._prepared_plans) == [40, 3]
    assert encoder._prepared_plans[40] is kept
    assert encoder._active_sequence_length == 40
    with pytest.raises(ValueError, match="was not prepared"):
        encoder.activate_shape(9)


def test_lengths_added_one_at_a_time_match_one_preparation():
    # The pattern of a caller that discovers lengths at run time.
    incremental, reference = make_encoder(), make_encoder()
    lengths = [37, 5, 100, 3, 64, 65]
    seen: list[int] = []
    for length in lengths:
        seen.append(length)
        incremental.prepare_for_inference(sorted(seen))
    reference.prepare_for_inference(lengths)

    for length in lengths:
        assert torch.equal(encode(incremental, length), encode(reference, length))


def test_prepared_lengths_share_one_position_table_per_layer():
    encoder = make_encoder()
    encoder.prepare_for_inference(range(1, 130))

    for index, layer in enumerate(encoder.layer):
        attention = layer.attention.self
        full = {
            attention._cached_pos_key.untyped_storage().data_ptr(),
            attention._cached_pos_query.untyped_storage().data_ptr(),
        }
        for plans in encoder._prepared_plans.values():
            plan = plans[index]
            assert plan.c2p_local is None and plan.p2c_local is None
            assert plan.pos_key.untyped_storage().data_ptr() in full
            assert plan.pos_query.untyped_storage().data_ptr() in full


def test_layers_share_one_gather_map_per_forward():
    encoder = make_encoder()
    encoder.prepare_for_inference([33])

    shared = share_pair_map(encoder._prepared_plans[33])

    assert len({id(plan.c2p_local) for plan in shared}) == 1
    assert all(plan.p2c_local is plan.c2p_local for plan in shared)
    assert all(plan.c2p_local is None for plan in encoder._prepared_plans[33])
    expected = dense_pair_map(shared[0].delta_to_local, 33)
    assert torch.equal(shared[0].c2p_local, expected)


def test_in_place_parameter_edit_rebuilds_on_the_next_prepare():
    edited, reference = make_encoder(), make_encoder()
    edited.prepare_for_inference([7])
    before = encode(edited, 7)

    for encoder in (edited, reference):
        with torch.no_grad():
            encoder.rel_embeddings.weight.mul_(1.5)
            encoder.layer[0].attention.self.key_proj.weight.mul_(0.5)
    edited.prepare_for_inference([7])
    reference.prepare_for_inference([7])

    after = encode(edited, 7)
    assert not torch.equal(after, before)
    assert torch.equal(after, encode(reference, 7))


def test_cleared_caches_are_rebuilt():
    encoder, reference = make_encoder(), make_encoder()
    encoder.prepare_for_inference([7, 8])
    encoder.train()
    encoder.eval()
    assert not encoder._prepared_plans

    encoder.prepare_for_inference([8, 7])
    reference.prepare_for_inference([8, 7])
    assert torch.equal(encode(encoder, 7), encode(reference, 7))


def test_dense_pair_map_expands_the_lookup():
    cache = SharedPositionPlanCache(
        position_buckets=32, max_relative_positions=64, position_embedding_size=32
    )
    for length in (1, 2, 7, 33, 100):
        compact = cache.compact(length, "cpu")
        expected = torch.empty(length, length, dtype=torch.long)
        for i in range(length):
            for j in range(length):
                expected[i, j] = int(compact.delta_to_local[i - j + length - 1])
        assert torch.equal(dense_pair_map(compact.delta_to_local, length)[0, 0], expected)
        assert torch.equal(cache.dense(length, "cpu").pair_to_local[0, 0], expected)


@pytest.mark.parametrize("buckets,max_relative", [(256, 512), (32, 64), (0, 64)])
def test_position_window_reads_the_same_slots_from_one_table(buckets, max_relative):
    span = buckets if buckets > 0 else max_relative
    cache = SharedPositionPlanCache(
        position_buckets=buckets,
        max_relative_positions=max_relative,
        position_embedding_size=span,
    )
    rows = 2 * span
    lengths = sorted({*range(1, 140), *TUNING_SEQUENCE_LENGTHS[:6]})
    for length in lengths:
        compact = cache.compact(length, "cpu")
        lookup = compact.delta_to_local.to(torch.int32)
        window = position_window(compact.active_slots, lookup, rows)
        slot_count = compact.active_slots.numel()
        assert window is not None
        assert window.width == aligned_slot_count(slot_count)
        assert 0 <= window.start and window.start + window.width <= rows
        assert window.delta_to_local.dtype == torch.int32
        assert int(window.delta_to_local.min()) >= 0
        assert int(window.delta_to_local.max()) < window.width
        global_slots = compact.active_slots[lookup.long()]
        assert torch.equal(window.start + window.delta_to_local.long(), global_slots)


def test_position_window_clamps_at_the_table_end_and_declines_when_too_wide():
    lookup = torch.arange(11, dtype=torch.int32)
    window = position_window(torch.arange(500, 511), lookup, 512)
    assert (window.start, window.width) == (496, 16)
    assert torch.equal(window.delta_to_local, lookup + 4)

    assert position_window(torch.arange(0, 9), torch.arange(9), 10) is None
    assert position_window(torch.empty(0, dtype=torch.long), torch.zeros(3), 10) is None


def test_triton_family_tables_are_views_that_read_what_padded_copies_read():
    # Simulates the kernel's reads (score row offset by the lookup) on CPU.
    config = make_config(position_buckets=256, max_relative_positions=512, hidden_size=32)
    attention = InferenceDisentangledSelfAttention(config, backend="triton").eval()
    torch.manual_seed(0)
    attention.prepare_for_inference(torch.randn(512, 32))
    full_key = attention._cached_pos_key.untyped_storage().data_ptr()
    full_query = attention._cached_pos_query.untyped_storage().data_ptr()
    query = torch.randn(2, 5, 16)

    for representative in TUNING_SEQUENCE_LENGTHS[:6]:
        indices = attention.position_plan_cache.compact(representative, "cpu")
        lookup, pos_key, pos_query = attention._family_position_tables(
            indices.active_slots, indices.delta_to_local
        )
        assert pos_key.untyped_storage().data_ptr() == full_key
        assert pos_query.untyped_storage().data_ptr() == full_query
        assert pos_key.size(-2) % 8 == 0

        start, count = int(indices.active_slots[0]), indices.active_slots.numel()
        padded = pad_position_table(attention._cached_pos_key.narrow(1, start, count))
        old_scores = torch.bmm(query, padded.transpose(1, 2))
        new_scores = torch.bmm(query, pos_key.transpose(1, 2))
        old_lookup = indices.delta_to_local.long()
        assert torch.equal(new_scores[..., lookup.long()], old_scores[..., old_lookup])
