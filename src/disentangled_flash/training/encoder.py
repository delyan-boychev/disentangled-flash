"""Encoder-level training integration for DisentangledFlash."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .._reference import BaseModelOutput
from ..position import SharedPositionPlanCache
from .attention import TritonTrainingDisentangledSelfAttention


class DebertaV2TrainingEncoder(nn.Module):
    """Drop-in DeBERTa-v2/v3 encoder using differentiable Triton attention.

    The wrapper preserves the original layer/FFN/residual/convolution modules
    and state-dict names while avoiding Hugging Face's dense [B,1,L,L] mask and
    [L,L] relative-position construction. Only each layer's self-attention
    module is replaced.
    """

    def __init__(
        self,
        source_encoder: nn.Module,
        config: Any,
        *,
        fp32_precision: str = "strict",
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(source_encoder, "layer"):
            raise TypeError("source_encoder does not look like a DebertaV2Encoder")
        if float(getattr(config, "attention_probs_dropout_prob", 0.0)) != 0.0:
            raise ValueError(
                "the first DisentangledFlash training backend requires "
                "config.attention_probs_dropout_prob == 0.0"
            )

        self.fp32_precision = fp32_precision
        self.assume_unpadded = assume_unpadded
        self.relative_attention = getattr(source_encoder, "relative_attention", False)
        self.max_relative_positions = getattr(
            source_encoder,
            "max_relative_positions",
            getattr(config, "max_position_embeddings", 512),
        )
        self.position_buckets = getattr(source_encoder, "position_buckets", -1)
        self.norm_rel_ebd = list(getattr(source_encoder, "norm_rel_ebd", ["none"]))
        self.gradient_checkpointing = bool(getattr(source_encoder, "gradient_checkpointing", False))

        # Preserve HF module names and parameter ownership.
        self.layer = source_encoder.layer
        if self.relative_attention:
            self.rel_embeddings = source_encoder.rel_embeddings
        if hasattr(source_encoder, "LayerNorm"):
            self.LayerNorm = source_encoder.LayerNorm
        self.conv = getattr(source_encoder, "conv", None)

        position_embedding_size = self.max_relative_positions
        if self.position_buckets > 0:
            position_embedding_size = self.position_buckets
        pos_att_type = getattr(config, "pos_att_type", ()) or ()
        if isinstance(pos_att_type, str):
            pos_att_type = tuple(part.strip() for part in pos_att_type.split("|") if part)
        uses_position_bias = self.relative_attention and bool(
            {"c2p", "p2c"}.intersection(pos_att_type)
        )
        self.position_plan_cache = SharedPositionPlanCache(
            position_buckets=self.position_buckets,
            max_relative_positions=self.max_relative_positions,
            position_embedding_size=position_embedding_size,
            uses_position_bias=uses_position_bias,
        )

        for layer in self.layer:
            original_attention = layer.attention.self
            replacement = TritonTrainingDisentangledSelfAttention(
                config,
                position_plan_cache=self.position_plan_cache,
                fp32_precision=fp32_precision,
                assume_unpadded=assume_unpadded,
            )
            replacement.load_state_dict(original_attention.state_dict(), strict=True)
            replacement.to(
                device=original_attention.query_proj.weight.device,
                dtype=original_attention.query_proj.weight.dtype,
            )
            replacement.train(original_attention.training)
            layer.attention.self = replacement

    def get_rel_embedding(self) -> torch.Tensor | None:
        relative = self.rel_embeddings.weight if self.relative_attention else None
        if relative is not None and "layer_norm" in self.norm_rel_ebd:
            relative = self.LayerNorm(relative)
        return relative

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError("training encoder requires attention_mask with shape [B, L]")
        if self.assume_unpadded:
            return attention_mask
        if attention_mask.dtype == torch.bool and attention_mask.is_contiguous():
            return attention_mask
        return attention_mask.bool().contiguous()

    def get_rel_pos(self, *args: Any, **kwargs: Any) -> None:
        # Compact delta_to_local plans replace the dense HF relative_pos tensor.
        return None

    def _run_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rel_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        self_output, _ = layer.attention.self(
            hidden_states,
            attention_mask,
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )
        attention_output = layer.attention.output(self_output, hidden_states)
        intermediate_output = layer.intermediate(attention_output)
        return layer.output(intermediate_output, attention_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the training path")
        if query_states is not None:
            raise ValueError("query_states/z_steps are not supported by the training path")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the training path")
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("attention_mask and hidden_states must have matching [B, L]")

        layer_attention_mask = self.get_attention_mask(attention_mask)
        rel_embeddings = self.get_rel_embedding()
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        next_kv = hidden_states
        input_mask = attention_mask

        for index, layer in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                if rel_embeddings is None:
                    output_states = checkpoint(
                        lambda x, layer=layer: self._run_layer(
                            layer,
                            x,
                            layer_attention_mask,
                            None,
                        ),
                        next_kv,
                        use_reentrant=False,
                    )
                else:
                    output_states = checkpoint(
                        lambda x, rel, layer=layer: self._run_layer(
                            layer,
                            x,
                            layer_attention_mask,
                            rel,
                        ),
                        next_kv,
                        rel_embeddings,
                        use_reentrant=False,
                    )
            else:
                output_states = self._run_layer(
                    layer,
                    next_kv,
                    layer_attention_mask,
                    rel_embeddings,
                )

            if index == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            next_kv = output_states

        if not return_dict:
            values = (next_kv, all_hidden_states, None)
            return tuple(value for value in values if value is not None)
        return BaseModelOutput(
            last_hidden_state=next_kv,
            hidden_states=all_hidden_states,
            attentions=None,
        )


def enable_deberta_training(
    model: nn.Module,
    *,
    fp32_precision: str = "strict",
    assume_unpadded: bool = False,
) -> nn.Module:
    """Replace a Hugging Face DeBERTa backbone encoder with the training backend.

    ``model`` is the DeBERTa backbone (the object with ``embeddings`` and
    ``encoder``), not an outer task head. The conversion is in-place and keeps
    checkpoint/state-dict parameter names stable.
    """

    if not hasattr(model, "encoder") or not hasattr(model, "embeddings"):
        raise TypeError("model must be a Hugging Face DeBERTa-v2/v3 backbone")
    if getattr(model, "z_steps", 0) > 1:
        raise ValueError("DeBERTa z_steps > 1 is not supported by the training path")
    if isinstance(model.encoder, DebertaV2TrainingEncoder):
        raise TypeError("the model already uses DebertaV2TrainingEncoder")

    # Avoid converting the inference wrapper: it may have inference-only packed
    # parameter storage. Start training conversion from a normal HF encoder.
    from ..deberta import DebertaV2InferenceEncoder

    if isinstance(model.encoder, DebertaV2InferenceEncoder):
        raise TypeError(
            "cannot convert DebertaV2InferenceEncoder in place; load a normal HF model "
            "and call enable_deberta_training() on that backbone"
        )

    was_training = model.training
    model.encoder = DebertaV2TrainingEncoder(
        model.encoder,
        model.config,
        fp32_precision=fp32_precision,
        assume_unpadded=assume_unpadded,
    )
    model.train(was_training)
    return model


optimize_deberta_training = enable_deberta_training


__all__ = [
    "DebertaV2TrainingEncoder",
    "enable_deberta_training",
    "optimize_deberta_training",
]
