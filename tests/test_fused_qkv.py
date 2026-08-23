import torch

from disentangled_flash._reference import DebertaAttentionConfig
from disentangled_flash._torch import TorchInferenceDisentangledSelfAttention


def make_config(
    *,
    position_buckets: int = 32,
    max_relative_positions: int = 64,
) -> DebertaAttentionConfig:
    return DebertaAttentionConfig(
        hidden_size=64,
        num_attention_heads=1,
        attention_head_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=max_relative_positions,
        max_position_embeddings=max_relative_positions,
        position_buckets=position_buckets,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        norm_rel_ebd="none",
    )


def make_relative(config: DebertaAttentionConfig) -> torch.Tensor:
    return torch.randn(
        config.position_buckets * 2,
        config.hidden_size,
    )


def test_qkv_is_fused_and_shares_parameter_storage():
    config = make_config()
    module = TorchInferenceDisentangledSelfAttention(config).eval()
    relative = make_relative(config)

    module.prepare_for_inference(relative)

    packed_weight = module._cached_qkv_weight
    assert packed_weight is not None
    assert packed_weight.shape == (192, 64)

    packed_storage = packed_weight.untyped_storage().data_ptr()

    assert (
        module.query_proj.weight.untyped_storage().data_ptr()
        == packed_storage
    )
    assert (
        module.key_proj.weight.untyped_storage().data_ptr()
        == packed_storage
    )
    assert (
        module.value_proj.weight.untyped_storage().data_ptr()
        == packed_storage
    )

    elements_per_projection = config.hidden_size * config.hidden_size

    assert module.query_proj.weight.storage_offset() == 0
    assert module.key_proj.weight.storage_offset() == elements_per_projection
    assert (
        module.value_proj.weight.storage_offset()
        == 2 * elements_per_projection
    )

    hidden = torch.randn(2, 8, 64)

    with torch.inference_mode():
        q, k, v = module._project_qkv(hidden)

    assert q.shape == k.shape == v.shape == (2, 8, 64)


def test_prepare_for_inference_can_be_repeated():
    config = make_config()
    module = TorchInferenceDisentangledSelfAttention(config).eval()

    relative = make_relative(config)
    hidden = torch.randn(2, 8, config.hidden_size)
    mask = torch.ones(2, 8, dtype=torch.bool)

    module.prepare_for_inference(relative)
    first_plan = module.prepare_shape(8, "cpu")

    with torch.inference_mode():
        first = module.forward_prepared(
            hidden,
            mask,
            first_plan,
        )[0]

    module.prepare_for_inference(relative)
    second_plan = module.prepare_shape(8, "cpu")

    with torch.inference_mode():
        second = module.forward_prepared(
            hidden,
            mask,
            second_plan,
        )[0]

    torch.testing.assert_close(first, second)


def test_state_dict_round_trip_after_qkv_packing():
    config = make_config()

    source = TorchInferenceDisentangledSelfAttention(config).eval()
    relative = make_relative(config)

    source.prepare_for_inference(relative)

    state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
    }

    restored = TorchInferenceDisentangledSelfAttention(config).eval()
    restored.load_state_dict(state, strict=True)

    # Loading a state dict deliberately invalidates derived inference caches.
    assert restored._cached_qkv_weight is None

    restored.prepare_for_inference(relative)

    hidden = torch.randn(2, 8, config.hidden_size)

    with torch.inference_mode():
        source_qkv = source._project_qkv(hidden)
        restored_qkv = restored._project_qkv(hidden)

    for expected, actual in zip(source_qkv, restored_qkv):
        torch.testing.assert_close(expected, actual)


def test_released_position_workspace_keeps_prepared_plan_usable():
    config = make_config()
    module = TorchInferenceDisentangledSelfAttention(config).eval()

    relative = make_relative(config)
    hidden = torch.randn(2, 8, config.hidden_size)
    mask = torch.ones(2, 8, dtype=torch.bool)

    module.prepare_for_inference(relative)
    plan = module.prepare_shape(8, "cpu")

    module.release_position_projection_workspace()

    assert module._cached_pos_key is None
    assert module._cached_pos_query is None

    with torch.inference_mode():
        prepared_output = module.forward_prepared(
            hidden,
            mask,
            plan,
        )[0]

        # Must reuse the cached plan rather than trying to regenerate projected
        # positions from the released workspace.
        lazy_output = module(
            hidden,
            mask,
        )[0]

    torch.testing.assert_close(
        prepared_output,
        lazy_output,
    )


def test_equivalent_active_slot_ranges_share_projected_storage():
    config = make_config(
        position_buckets=256,
        max_relative_positions=512,
    )
    module = TorchInferenceDisentangledSelfAttention(config).eval()

    relative = make_relative(config)
    module.prepare_for_inference(relative)

    slots_1k = module.position_plan_cache.compact(
        1024,
        torch.device("cpu"),
    ).active_slots

    slots_2k = module.position_plan_cache.compact(
        2048,
        torch.device("cpu"),
    ).active_slots

    assert torch.equal(slots_1k, slots_2k)

    key_1k, query_1k = module._project_active_positions(slots_1k)
    key_2k, query_2k = module._project_active_positions(slots_2k)

    assert key_1k is key_2k
    assert query_1k is query_2k

    assert key_1k is not None
    assert query_1k is not None

    assert key_1k.data_ptr() == key_2k.data_ptr()
    assert query_1k.data_ptr() == query_2k.data_ptr()