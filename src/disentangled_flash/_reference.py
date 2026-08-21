# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-contained Hugging Face DeBERTa-v2/v3 attention and encoder reference.

The model code below is copied from Transformers 4.57.6
``models/deberta_v2/modeling_deberta_v2.py``.  Only external infrastructure was
replaced: the config, activation registry, gradient-checkpointing base class,
and ``BaseModelOutput`` are local lightweight equivalents.  The attention,
layer, convolution, mask expansion, relative-position construction, and
encoder loop are kept as the auditable baseline for this experiment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import functional as F


@dataclass(frozen=True)
class DebertaAttentionConfig:
    """Self-contained subset of ``DebertaV2Config`` used by these modules."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_head_size: int | None = None
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str | Callable[[torch.Tensor], torch.Tensor] = "gelu"
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-7
    relative_attention: bool = True
    max_relative_positions: int = -1
    max_position_embeddings: int = 512
    position_buckets: int = 256
    share_att_key: bool = True
    pos_att_type: Any = ("p2c", "c2p")
    norm_rel_ebd: str = "layer_norm"
    conv_kernel_size: int = 0
    conv_groups: int = 1
    conv_act: str = "tanh"


@dataclass
class BaseModelOutput:
    """Minimal local equivalent of Transformers' ``BaseModelOutput``."""

    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def __getitem__(self, index: int) -> Any:
        values = tuple(
            value
            for value in (self.last_hidden_state, self.hidden_states, self.attentions)
            if value is not None
        )
        return values[index]


def _gelu_new(tensor: torch.Tensor) -> torch.Tensor:
    return 0.5 * tensor * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi) * (tensor + 0.044715 * torch.pow(tensor, 3.0))
        )
    )


ACT2FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": F.gelu,
    "gelu_new": _gelu_new,
    "gelu_fast": _gelu_new,
    "mish": F.mish,
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "tanh": torch.tanh,
}


# Copied from DebertaSelfOutput with DebertaLayerNorm -> LayerNorm.
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def make_log_bucket_position(
    relative_pos: torch.Tensor,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid)
            / torch.log(torch.tensor((max_position - 1) / mid))
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    bucket_size: int = -1,
    max_position: int = -1,
) -> torch.Tensor:
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)
    query_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    key_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    relative_pos_ids = query_ids[:, None] - key_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        relative_pos_ids = make_log_bucket_position(
            relative_pos_ids,
            bucket_size,
            max_position,
        )
    relative_pos_ids = relative_pos_ids.to(torch.long)
    relative_pos_ids = relative_pos_ids[:query_size, :]
    return relative_pos_ids.unsqueeze(0)


def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


def build_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    if key_layer.size(-2) != query_layer.size(-2):
        return build_relative_position(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    return relative_pos


def _prepare_attention_mask(
    attention_mask: torch.Tensor,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    """Standalone equivalent of ``DebertaV2Encoder.get_attention_mask``."""

    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        return extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    if attention_mask.dim() == 3:
        return attention_mask.unsqueeze(1)
    if attention_mask.dim() == 4:
        return attention_mask
    raise ValueError("attention_mask must have 2, 3, or 4 dimensions")


class DisentangledSelfAttention(nn.Module):
    """Hugging Face DeBERTa-v2/v3 disentangled self-attention baseline."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        default_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", None) or default_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        config.hidden_size,
                        self.all_head_size,
                        bias=True,
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor: torch.Tensor, attention_heads: int) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (attention_heads, -1)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3).contiguous().view(
            -1,
            tensor.size(1),
            tensor.size(-1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states),
            self.num_attention_heads,
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states),
            self.num_attention_heads,
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states),
            self.num_attention_heads,
        )

        relative_attention = None
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(query_layer, scale_factor)
        attention_scores = torch.bmm(
            query_layer,
            key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype),
        )
        if self.relative_attention:
            if rel_embeddings is None:
                raise ValueError("rel_embeddings is required when relative_attention=True")
            rel_embeddings = self.pos_dropout(rel_embeddings)
            relative_attention = self.disentangled_attention_bias(
                query_layer,
                key_layer,
                relative_pos,
                rel_embeddings,
                scale_factor,
            )

        if relative_attention is not None:
            attention_scores = attention_scores + relative_attention
        attention_scores = attention_scores.view(
            -1,
            self.num_attention_heads,
            attention_scores.size(-2),
            attention_scores.size(-1),
        )
        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(
            ~attention_mask,
            torch.finfo(query_layer.dtype).min,
        )
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(
                -1,
                attention_probs.size(-2),
                attention_probs.size(-1),
            ),
            value_layer,
        )
        context_layer = (
            context_layer.view(
                -1,
                self.num_attention_heads,
                context_layer.size(-2),
                context_layer.size(-1),
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        context_layer = context_layer.view(context_layer.size()[:-2] + (-1,))
        if not output_attentions:
            return context_layer, None
        return context_layer, attention_probs

    def disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: torch.Tensor | None,
        rel_embeddings: torch.Tensor,
        scale_factor: int,
    ) -> torch.Tensor:
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer,
                key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}"
            )

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(device=query_layer.device, dtype=torch.long)
        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        batch_size = query_layer.size(0) // self.num_attention_heads

        pos_query_layer = None
        pos_key_layer = None
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings),
                self.num_attention_heads,
            ).repeat(batch_size, 1, 1)
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings),
                self.num_attention_heads,
            ).repeat(batch_size, 1, 1)
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings),
                    self.num_attention_heads,
                ).repeat(batch_size, 1, 1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings),
                    self.num_attention_heads,
                ).repeat(batch_size, 1, 1)

        score: torch.Tensor | int = 0
        if "c2p" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_key_layer, scale_factor)
            c2p_attention = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_position = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_attention = torch.gather(
                c2p_attention,
                dim=-1,
                index=c2p_position.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
                ),
            )
            score = score + c2p_attention / scale.to(dtype=c2p_attention.dtype)

        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_query_layer, scale_factor)
            relative_position = build_rpos(
                query_layer,
                key_layer,
                relative_pos,
                self.max_relative_positions,
                self.position_buckets,
            )
            p2c_position = torch.clamp(-relative_position + att_span, 0, att_span * 2 - 1)
            p2c_attention = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_attention = torch.gather(
                p2c_attention,
                dim=-1,
                index=p2c_position.squeeze(0).expand(
                    [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
                ),
            ).transpose(-1, -2)
            score = score + p2c_attention / scale.to(dtype=p2c_attention.dtype)
        return score


# Backward-compatible experiment name.  This is an alias, not a rewrite.
OriginalDisentangledSelfAttention = DisentangledSelfAttention


class DebertaV2Attention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_output, attention_matrix = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if output_attentions:
            return attention_output, attention_matrix
        return attention_output, None


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class DebertaV2Output(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class DebertaV2Layer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_output, attention_matrix = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return layer_output, attention_matrix
        return layer_output, None


class ConvLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_states: torch.Tensor,
        input_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        output = self.conv(hidden_states.permute(0, 2, 1).contiguous())
        output = output.permute(0, 2, 1).contiguous()
        if input_mask is not None:
            # The upstream implementation uses ``1 - input_mask``, which is
            # valid for tokenizer-produced integer masks but raises for bool
            # masks.  Encoder benchmarks and callers may legitimately provide
            # bool padding masks, so use the equivalent logical operation for
            # that dtype while retaining upstream behavior for numeric masks.
            remove_mask = (
                ~input_mask
                if input_mask.dtype == torch.bool
                else (1 - input_mask).bool()
            )
            output.masked_fill_(remove_mask.unsqueeze(-1).expand(output.size()), 0)
        output = ACT2FN[self.conv_act](self.dropout(output))

        layer_norm_input = residual_states + output
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)
        if input_mask is None:
            return output
        if input_mask.dim() != layer_norm_input.dim():
            if input_mask.dim() == 4:
                input_mask = input_mask.squeeze(1).squeeze(1)
            input_mask = input_mask.unsqueeze(2)
        return output * input_mask.to(output.dtype)


class DebertaV2Encoder(nn.Module):
    """Copied Hugging Face DeBERTa-v2/v3 encoder baseline."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [DebertaV2Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, "position_buckets", -1)
            position_embedding_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                position_embedding_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(position_embedding_size, config.hidden_size)

        self.norm_rel_ebd = [
            item.strip()
            for item in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(
                config.hidden_size,
                config.layer_norm_eps,
                elementwise_affine=True,
            )
        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self) -> torch.Tensor | None:
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (
                extended_attention_mask
                * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            )
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.relative_attention and relative_pos is None:
            if query_states is not None:
                relative_pos = build_relative_position(
                    query_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                relative_pos = build_relative_position(
                    hidden_states,
                    hidden_states,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
        return relative_pos

    def forward(
        self,
        hidden_states: torch.Tensor | Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> BaseModelOutput | tuple[Any, ...]:
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_key_value = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for index, layer_module in enumerate(self.layer):
            output_states, attention_weights = layer_module(
                next_key_value,
                attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions = all_attentions + (attention_weights,)
            if index == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_key_value = (
                        hidden_states[index + 1]
                        if index + 1 < len(self.layer)
                        else None
                    )
            else:
                next_key_value = output_states

        if not return_dict:
            return tuple(
                value
                for value in (output_states, all_hidden_states, all_attentions)
                if value is not None
            )
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


__all__ = [
    "BaseModelOutput",
    "ConvLayer",
    "DebertaAttentionConfig",
    "DebertaV2Attention",
    "DebertaV2Encoder",
    "DebertaV2Intermediate",
    "DebertaV2Layer",
    "DebertaV2Output",
    "DebertaV2SelfOutput",
    "DisentangledSelfAttention",
    "OriginalDisentangledSelfAttention",
    "_prepare_attention_mask",
    "build_relative_position",
    "build_rpos",
    "make_log_bucket_position",
    "scaled_size_sqrt",
]
