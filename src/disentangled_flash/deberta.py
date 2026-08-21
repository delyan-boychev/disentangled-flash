"""Encoder-level DeBERTa-v2/v3 inference fast path.

Hugging Face's regular encoder expands a 2-D padding mask to ``[B, 1, L, L]``
and builds an ``[L, L]`` relative-position tensor before entering its layer
loop.  This wrapper preserves the original layer output, FFN, convolution, and
state-dict layout while passing the original 2-D mask and prepared position
plans directly to the standalone attention implementations.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Iterable
from typing import Any, Callable, Mapping, Union

import torch
from torch import nn

from ._prepared import InferenceDisentangledSelfAttention, PreparedPositionPlan
from ._reference import BaseModelOutput
from .position import SharedPositionPlanCache
from .kernel import (
    TritonInferenceDisentangledSelfAttention,
    TritonPreparedPositionPlan,
)

PositionPlan = Union[PreparedPositionPlan, TritonPreparedPositionPlan]


def _invalidate_encoder_cache_after_load(
    module: torch.nn.Module,
    _incompatible_keys: Any,
) -> None:
    module.clear_inference_cache()


class DebertaV2InferenceEncoder(nn.Module):
    """Inference-only replacement for a Hugging Face ``DebertaV2Encoder``.

    Call :meth:`prepare_for_inference` for all production bucket lengths, then
    :meth:`activate_shape` before executing or compiling a bucket.  The hot
    ``forward`` uses only the active immutable tuple of plans: no dictionary
    lookup, relative-position construction, or cache mutation occurs there.
    """

    def __init__(
        self,
        source_encoder: nn.Module,
        config: Any,
        *,
        backend: str = "triton",
        fp32_precision: str = "strict",
    ) -> None:
        super().__init__()
        if backend not in {"optimized", "triton"}:
            raise ValueError("backend must be 'optimized' or 'triton'")
        if not hasattr(source_encoder, "layer"):
            raise TypeError("source_encoder does not look like a DebertaV2Encoder")

        self.backend = backend
        self.fp32_precision = fp32_precision
        self.relative_attention = getattr(source_encoder, "relative_attention", False)
        self.max_relative_positions = getattr(
            source_encoder,
            "max_relative_positions",
            getattr(config, "max_position_embeddings", 512),
        )
        self.position_buckets = getattr(source_encoder, "position_buckets", -1)
        self.norm_rel_ebd = list(getattr(source_encoder, "norm_rel_ebd", ["none"]))
        self.gradient_checkpointing = False

        # Preserve the exact HF module names so checkpoint keys remain stable.
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

        attention_class: type[InferenceDisentangledSelfAttention]
        if backend == "triton":
            attention_class = TritonInferenceDisentangledSelfAttention
        else:
            attention_class = InferenceDisentangledSelfAttention

        for layer in self.layer:
            original_attention = layer.attention.self
            kwargs: dict[str, Any] = {
                "position_plan_cache": self.position_plan_cache,
            }
            if backend == "triton":
                kwargs["fp32_precision"] = fp32_precision
            replacement = attention_class(config, **kwargs)
            replacement.load_state_dict(original_attention.state_dict(), strict=True)
            replacement.to(
                device=original_attention.query_proj.weight.device,
                dtype=original_attention.query_proj.weight.dtype,
            )
            replacement.train(original_attention.training)
            layer.attention.self = replacement

        self._prepared_plans: dict[int, tuple[PositionPlan, ...]] = {}
        self._active_sequence_length: int | None = None
        self._active_plans: tuple[PositionPlan, ...] | None = None
        self.register_load_state_dict_post_hook(_invalidate_encoder_cache_after_load)

    def get_rel_embedding(self) -> torch.Tensor | None:
        relative = self.rel_embeddings.weight if self.relative_attention else None
        if relative is not None and "layer_norm" in self.norm_rel_ebd:
            relative = self.LayerNorm(relative)
        return relative

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Retain only the supported factorized mask; never expand it."""

        if attention_mask.dim() != 2:
            raise ValueError("the inference encoder requires attention_mask with shape [B, L]")
        return attention_mask

    def get_rel_pos(self, *args: Any, **kwargs: Any) -> None:
        """The compact delta plan replaces Hugging Face's dense relative_pos."""

        return None

    def _attention_modules(self) -> tuple[InferenceDisentangledSelfAttention, ...]:
        return tuple(layer.attention.self for layer in self.layer)

    def clear_inference_cache(self) -> None:
        self._prepared_plans.clear()
        self._active_sequence_length = None
        self._active_plans = None
        self.position_plan_cache.clear()
        for attention in self._attention_modules():
            attention.clear_inference_cache()

    def train(self, mode: bool = True) -> "DebertaV2InferenceEncoder":
        if mode and hasattr(self, "_prepared_plans"):
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn: Any, recurse: bool = True) -> "DebertaV2InferenceEncoder":
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_prepared_plans"):
            self.clear_inference_cache()
        return result

    @torch.no_grad()
    def prepare_for_inference(
        self,
        sequence_lengths: int | Iterable[int],
    ) -> "DebertaV2InferenceEncoder":
        """Prepare every layer and every selected production bucket."""

        if self.training:
            raise RuntimeError("prepare_for_inference() requires encoder.eval()")
        if isinstance(sequence_lengths, int):
            lengths = (sequence_lengths,)
        else:
            lengths = tuple(dict.fromkeys(int(length) for length in sequence_lengths))
        if not lengths or any(length < 1 for length in lengths):
            raise ValueError("sequence_lengths must contain positive integers")

        # Full projected position tables are layer-dependent.  The length plans
        # below share their index tensors through one encoder-owned cache.
        self.clear_inference_cache()
        rel_embeddings = self.get_rel_embedding()
        attentions = self._attention_modules()
        for attention in attentions:
            attention.eval().prepare_for_inference(rel_embeddings)

        device = attentions[0].query_proj.weight.device
        for length in lengths:
            self._prepared_plans[length] = tuple(
                attention.prepare_shape(length, device) for attention in attentions
            )
        self.activate_shape(lengths[0])
        return self

    def activate_shape(self, sequence_length: int) -> "DebertaV2InferenceEncoder":
        """Select a prebuilt bucket outside the compiled hot graph."""

        try:
            plans = self._prepared_plans[sequence_length]
        except KeyError as error:
            raise ValueError(
                f"sequence length {sequence_length} was not prepared; available buckets: "
                f"{sorted(self._prepared_plans)}"
            ) from error
        self._active_sequence_length = sequence_length
        self._active_plans = plans
        return self

    def _validate_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        query_states: torch.Tensor | None,
        relative_pos: torch.Tensor | None,
    ) -> tuple[PositionPlan, ...]:
        if self.training:
            raise RuntimeError("DebertaV2InferenceEncoder requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "DebertaV2InferenceEncoder requires torch.no_grad() or inference_mode()"
            )
        if output_attentions:
            raise ValueError("output_attentions=True is not supported")
        if query_states is not None:
            raise ValueError("query_states/z_steps are not supported by the inference path")
        if relative_pos is not None:
            raise ValueError("custom relative_pos is not supported by the inference path")
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [B, L]")
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("attention_mask and hidden_states must have matching [B, L]")
        if self._active_plans is None or self._active_sequence_length is None:
            raise RuntimeError("call prepare_for_inference() before forward()")
        if hidden_states.size(1) != self._active_sequence_length:
            raise ValueError(
                f"active bucket length {self._active_sequence_length} does not match "
                f"input length {hidden_states.size(1)}; call activate_shape() first"
            )
        return self._active_plans

    def forward_prepared(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plans: tuple[PositionPlan, ...],
        *,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        """Execute the encoder without dense mask/relative-position construction."""

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        next_kv = hidden_states
        input_mask = attention_mask

        for index, (layer, plan) in enumerate(zip(self.layer, plans)):
            self_output, _ = layer.attention.self.forward_prepared(
                next_kv,
                attention_mask,
                plan,
            )
            attention_output = layer.attention.output(self_output, next_kv)
            intermediate_output = layer.intermediate(attention_output)
            output_states = layer.output(intermediate_output, attention_output)

            if index == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            next_kv = output_states

        if not return_dict:
            values = (output_states, all_hidden_states, None)
            return tuple(value for value in values if value is not None)
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        plans = self._validate_forward(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states,
            relative_pos,
        )
        return self.forward_prepared(
            hidden_states,
            attention_mask,
            plans,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def enable_deberta_inference(
    model: nn.Module,
    *,
    backend: str = "triton",
    sequence_lengths: Iterable[int] | int | None = None,
    fp32_precision: str = "strict",
) -> nn.Module:
    """Replace a HF DeBERTa-v2/v3 encoder without changing checkpoint keys.

    ``model`` must be the Hugging Face backbone (the object with ``embeddings``
    and ``encoder``), not the outer GLiNER2 model.  The operation is in-place
    and inference-only.
    """

    if not hasattr(model, "encoder") or not hasattr(model, "embeddings"):
        raise TypeError("model must be a Hugging Face DeBERTa-v2/v3 backbone")
    if getattr(model, "z_steps", 0) > 1:
        raise ValueError("DeBERTa z_steps > 1 is not supported by the inference path")
    if isinstance(model.encoder, DebertaV2InferenceEncoder):
        raise ValueError("the model already uses DebertaV2InferenceEncoder")

    was_training = model.training
    if sequence_lengths is not None and was_training:
        raise RuntimeError("call model.eval() before preparing inference buckets")
    model.encoder = DebertaV2InferenceEncoder(
        model.encoder,
        model.config,
        backend=backend,
        fp32_precision=fp32_precision,
    )
    model.train(was_training)
    if sequence_lengths is not None:
        model.encoder.prepare_for_inference(sequence_lengths)
    return model


def compile_deberta_buckets(
    encoder: DebertaV2InferenceEncoder,
    sequence_lengths: Iterable[int] | None = None,
    *,
    mode: str = "max-autotune-no-cudagraphs",
    fullgraph: bool = True,
    dynamic_batch: bool = True,
    examples: Mapping[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Create isolated compiled encoder callables for prepared length buckets.

    PyTorch 2.13+'s ``isolate_recompiles`` prevents bucket factories from
    sharing one code object's recompile budget.  On older PyTorch releases, a
    distinct cloned code object provides the documented compatibility
    workaround.  Supplying ``examples`` executes one example per bucket so
    Dynamo, Inductor, and Triton autotuning finish during startup.
    """

    if sequence_lengths is None:
        lengths = tuple(sorted(encoder._prepared_plans))
    else:
        lengths = tuple(dict.fromkeys(int(length) for length in sequence_lengths))
    missing = [length for length in lengths if length not in encoder._prepared_plans]
    if missing:
        raise ValueError(f"unprepared bucket lengths: {missing}")

    supports_isolation = "isolate_recompiles" in inspect.signature(
        torch.compile
    ).parameters
    compiled: dict[int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}

    def make_forward(
        plans: tuple[PositionPlan, ...],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def bucket_forward(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            return encoder.forward_prepared(
                hidden_states,
                attention_mask,
                plans,
                output_hidden_states=False,
            ).last_hidden_state

        return bucket_forward

    for length in lengths:
        function = make_forward(encoder._prepared_plans[length])
        compile_kwargs: dict[str, Any] = {
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic_batch,
        }
        if supports_isolation:
            compile_kwargs["isolate_recompiles"] = True
        else:
            clone = types.FunctionType(
                function.__code__.replace(),
                function.__globals__,
                name=f"deberta_bucket_{length}",
                argdefs=function.__defaults__,
                closure=function.__closure__,
            )
            clone.__kwdefaults__ = function.__kwdefaults__
            function = clone
        compiled[length] = torch.compile(function, **compile_kwargs)

    if examples is not None:
        with torch.inference_mode():
            for length, function in compiled.items():
                try:
                    hidden_states, attention_mask = examples[length]
                except KeyError as error:
                    raise ValueError(f"missing compilation example for bucket {length}") from error
                if hidden_states.size(1) != length:
                    raise ValueError(
                        f"bucket {length} example has sequence length {hidden_states.size(1)}"
                    )
                function(hidden_states, attention_mask)
    return compiled




def optimize_deberta(
    model: nn.Module,
    *,
    sequence_lengths: Iterable[int] | int | None = None,
    fp32_precision: str = "strict",
) -> nn.Module:
    """Enable the DisentangledFlash Triton backend on a Hugging Face DeBERTa backbone."""

    return enable_deberta_inference(
        model,
        backend="triton",
        sequence_lengths=sequence_lengths,
        fp32_precision=fp32_precision,
    )


# Compatibility alias for code from the standalone experiment.
enable_deberta_v2_inference = enable_deberta_inference


__all__ = [
    "DebertaV2InferenceEncoder",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "enable_deberta_v2_inference",
    "optimize_deberta",
]
