import torch

from disentangled_flash._reference import DebertaAttentionConfig
from disentangled_flash._torch import TorchInferenceDisentangledSelfAttention


def test_qkv_is_always_fused_after_prepare():
    config = DebertaAttentionConfig(
        hidden_size=64,
        num_attention_heads=1,
        attention_head_size=64,
        relative_attention=True,
        max_relative_positions=64,
        max_position_embeddings=64,
        position_buckets=32,
        share_att_key=True,
        pos_att_type=("c2p", "p2c"),
        norm_rel_ebd="none",
    )
    module = TorchInferenceDisentangledSelfAttention(config).eval()
    relative = torch.randn(config.position_buckets * 2, config.hidden_size)
    module.prepare_for_inference(relative)

    assert module._cached_qkv_weight is not None
    assert module._cached_qkv_weight.shape == (192, 64)

    hidden = torch.randn(2, 8, 64)
    with torch.inference_mode():
        q, k, v = module._project_qkv(hidden)
    assert q.shape == k.shape == v.shape == (2, 8, 64)
