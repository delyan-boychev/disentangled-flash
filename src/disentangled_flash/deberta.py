"""Unified encoder-level DeBERTa-v2/v3 optimization integration.

Hugging Face's regular encoder expands a 2-D padding mask to ``[B, 1, L, L]``
and builds an ``[L, L]`` relative-position tensor before entering its layer
loop.  This wrapper preserves the original layer output, FFN, convolution, and
state-dict layout while selecting either the Triton or optimized PyTorch
attention backend for inference or differentiable training.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._reference import BaseModelOutput
from ._torch import (
    TorchInferenceDisentangledSelfAttention,
    TorchPositionPlan,
    TorchTrainingDisentangledSelfAttention,
)
from .kernel import (
    TritonInferenceDisentangledSelfAttention,
    TritonPreparedPositionPlan,
)
from .kernel import (
    triton as _triton,
)
from .position import SharedPositionPlanCache
from .tuning import KernelTuningOptions, ProfileRegistry

PositionPlan = TorchPositionPlan | TritonPreparedPositionPlan


def _resolve_backend(
    backend: str,
    *,
    source_attention: nn.Module,
    config: Any,
    inference: bool,
) -> str:
    """Resolve backend selection without falling back to Hugging Face attention."""

    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be 'auto', 'torch', or 'triton'")
    if backend == "torch":
        return "torch"

    weight = source_attention.query_proj.weight
    head_dim = getattr(config, "attention_head_size", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    triton_supported = _triton is not None and int(head_dim) in {32, 64, 128}
    if not inference:
        triton_supported = (
            triton_supported
            and hasattr(torch.library, "triton_op")
            and hasattr(
                torch.library,
                "wrap_triton",
            )
        )
        if float(getattr(config, "attention_probs_dropout_prob", 0.0)) != 0.0:
            triton_supported = False
    if backend == "auto":
        triton_supported = (
            triton_supported
            and weight.device.type == "cuda"
            and weight.dtype in {torch.float16, torch.bfloat16, torch.float32}
        )
    return "triton" if triton_supported else "torch"


def _invalidate_encoder_cache_after_load(
    module: torch.nn.Module,
    _incompatible_keys: Any,
) -> None:
    module.clear_inference_cache()


class DebertaV2OptimizedEncoder(nn.Module):
    """Single optimized replacement for a Hugging Face ``DebertaV2Encoder``.

    This module is a drop-in replacement for the Hugging Face encoder in DeBERTa-v2
    and DeBERTa-v3 models. It uses the fast, exact, Triton-fused or PyTorch-optimized
    disentangled self-attention mechanisms under the hood.

    ``inference=True`` enables the parameter-derived packed/cached path and is
    therefore intended for inference-only models.  ``inference=False`` keeps
    parameter identities stable and projections differentiable.  In the latter
    mode, ``model.eval()`` still permits backward; only ``no_grad()`` or
    ``inference_mode()`` disables the Triton LSE/autograd state.
    """

    def __init__(
        self,
        source_encoder: nn.Module,
        config: Any,
        *,
        backend: str = "triton",
        inference: bool = True,
        fp32_precision: str = "strict",
        tuning: KernelTuningOptions | None = None,
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(source_encoder, "layer"):
            raise TypeError("source_encoder does not look like a DebertaV2Encoder")

        first_attention = source_encoder.layer[0].attention.self
        self.requested_backend = backend
        self.inference = inference
        self.backend = _resolve_backend(
            backend,
            source_attention=first_attention,
            config=config,
            inference=inference,
        )
        self.fp32_precision = fp32_precision
        self.tuning = tuning or KernelTuningOptions()
        self.assume_unpadded = assume_unpadded
        self.relative_attention = getattr(source_encoder, "relative_attention", False)
        self.max_relative_positions = getattr(
            source_encoder,
            "max_relative_positions",
            getattr(config, "max_position_embeddings", 512),
        )
        self.position_buckets = getattr(source_encoder, "position_buckets", -1)
        self.norm_rel_ebd = list(getattr(source_encoder, "norm_rel_ebd", ["none"]))
        self.gradient_checkpointing = (
            False if inference else bool(getattr(source_encoder, "gradient_checkpointing", False))
        )

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

        attention_class: type[nn.Module]
        if self.inference and self.backend == "triton":
            attention_class = TritonInferenceDisentangledSelfAttention
        elif self.inference:
            attention_class = TorchInferenceDisentangledSelfAttention
        elif self.backend == "triton":
            # Lazy import avoids a package-level cycle: training._kernels imports
            # the canonical forward kernel from kernel.py.
            from .training.attention import TritonTrainingDisentangledSelfAttention

            attention_class = TritonTrainingDisentangledSelfAttention
        else:
            attention_class = TorchTrainingDisentangledSelfAttention

        profile_registry = (
            ProfileRegistry.from_options(self.tuning)
            if backend == "triton" and self.tuning.mode in {"auto", "profile_only"}
            else None
        )
        for layer in self.layer:
            original_attention = layer.attention.self
            kwargs: dict[str, Any] = {
                "position_plan_cache": self.position_plan_cache,
            }
            if self.backend == "triton":
                kwargs["fp32_precision"] = fp32_precision
            if self.inference and self.backend == "triton":
                kwargs["tuning"] = self.tuning
                kwargs["profile_registry"] = profile_registry
            kwargs["assume_unpadded"] = assume_unpadded
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
        if self.inference:
            self.register_load_state_dict_post_hook(_invalidate_encoder_cache_after_load)

    def get_rel_embedding(self) -> torch.Tensor | None:
        relative = self.rel_embeddings.weight if self.relative_attention else None
        if relative is not None and "layer_norm" in self.norm_rel_ebd:
            relative = self.LayerNorm(relative)
        return relative

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Retain only the supported factorized mask; never expand it."""

        if attention_mask.dim() != 2:
            raise ValueError("the optimized encoder requires attention_mask with shape [B, L]")
        if (
            not self.inference
            and self.backend == "triton"
            and not self.assume_unpadded
            and (attention_mask.dtype != torch.bool or not attention_mask.is_contiguous())
        ):
            return attention_mask.bool().contiguous()
        return attention_mask

    def get_rel_pos(self, *args: Any, **kwargs: Any) -> None:
        """The compact delta plan replaces Hugging Face's dense relative_pos."""

        return

    def _attention_modules(self) -> tuple[nn.Module, ...]:
        return tuple(layer.attention.self for layer in self.layer)

    def clear_inference_cache(self) -> None:
        self._prepared_plans.clear()
        self._active_sequence_length = None
        self._active_plans = None
        self.position_plan_cache.clear()
        for attention in self._attention_modules():
            if hasattr(attention, "clear_inference_cache"):
                attention.clear_inference_cache()

    def train(self, mode: bool = True) -> DebertaV2OptimizedEncoder:
        if self.inference and mode and hasattr(self, "_prepared_plans"):
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn: Any, recurse: bool = True) -> DebertaV2OptimizedEncoder:
        result = super()._apply(fn, recurse=recurse)
        if self.inference and hasattr(self, "_prepared_plans"):
            self.clear_inference_cache()
        return result

    @torch.no_grad()
    def prepare_for_inference(
        self,
        sequence_lengths: int | Iterable[int],
    ) -> DebertaV2OptimizedEncoder:
        """Prepare every layer and every selected production bucket."""

        if not self.inference:
            raise RuntimeError(
                "prepare_for_inference() requires optimize_deberta(..., inference=True)"
            )
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

        # Every prepared shape now owns its compact projected position tensors.
        # The full per-layer projected position tables were only preparation
        # workspace and need not remain resident during inference.
        for attention in attentions:
            attention.release_position_projection_workspace()

        self.activate_shape(lengths[0])
        return self

    def activate_shape(self, sequence_length: int) -> DebertaV2OptimizedEncoder:
        """Select a prebuilt bucket outside the compiled hot graph."""

        if not self.inference:
            raise RuntimeError("activate_shape() is available only when inference=True")
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
        if not self.inference:
            raise RuntimeError("prepared inference forward requires inference=True")
        if self.training:
            raise RuntimeError("the cached inference path requires model.eval()")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "the cached inference path requires torch.no_grad() or inference_mode()"
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

        if not self.inference:
            raise RuntimeError("forward_prepared() is available only when inference=True")

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        next_kv = hidden_states
        input_mask = attention_mask

        # Normalize a real padding mask only once for the complete encoder.
        # With assume_unpadded=True the Triton kernel never reads the mask, so
        # preserve the original tensor without allocating a bool copy.
        layer_attention_mask = attention_mask

        if (
            self.backend == "triton"
            and not self.assume_unpadded
            and (attention_mask.dtype != torch.bool or not attention_mask.is_contiguous())
        ):
            layer_attention_mask = attention_mask.bool().contiguous()

        for index, (layer, plan) in enumerate(zip(self.layer, plans)):
            self_output, _ = layer.attention.self.forward_prepared(
                next_kv,
                layer_attention_mask,
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

    def _run_differentiable_layer(
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

    def _forward_differentiable(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
        output_attentions: bool,
        query_states: torch.Tensor | None,
        relative_pos: torch.Tensor | None,
        return_dict: bool,
    ) -> Any:
        if output_attentions:
            raise ValueError("output_attentions=True is not supported by the optimized path")
        if query_states is not None:
            raise ValueError("query_states/z_steps are not supported by the optimized path")
        if relative_pos is not None:
            raise ValueError("custom relative_pos tensors are not supported by the optimized path")
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("attention_mask and hidden_states must have matching [B, L]")

        layer_attention_mask = self.get_attention_mask(attention_mask)
        rel_embeddings = self.get_rel_embedding()
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        next_kv = hidden_states
        input_mask = attention_mask

        for index, layer in enumerate(self.layer):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                if rel_embeddings is None:
                    output_states = checkpoint(
                        lambda x, layer=layer: self._run_differentiable_layer(
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
                        lambda x, rel, layer=layer: self._run_differentiable_layer(
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
                output_states = self._run_differentiable_layer(
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
        if not self.inference:
            return self._forward_differentiable(
                hidden_states,
                attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                return_dict=return_dict,
            )

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


class DebertaV2InferenceEncoder(DebertaV2OptimizedEncoder):
    """Compatibility wrapper for the unified encoder with ``inference=True``."""

    def __init__(
        self,
        source_encoder: nn.Module,
        config: Any,
        *,
        backend: str = "triton",
        fp32_precision: str = "strict",
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__(
            source_encoder,
            config,
            backend=backend,
            inference=True,
            fp32_precision=fp32_precision,
            assume_unpadded=assume_unpadded,
        )


class DebertaV2TrainingEncoder(DebertaV2OptimizedEncoder):
    """Compatibility wrapper for the unified encoder with ``inference=False``."""

    def __init__(
        self,
        source_encoder: nn.Module,
        config: Any,
        *,
        backend: str = "triton",
        fp32_precision: str = "strict",
        assume_unpadded: bool = False,
    ) -> None:
        super().__init__(
            source_encoder,
            config,
            backend=backend,
            inference=False,
            fp32_precision=fp32_precision,
            assume_unpadded=assume_unpadded,
        )


def _enable_deberta(
    model: nn.Module,
    *,
    backend: str,
    inference: bool,
    sequence_lengths: Iterable[int] | int | None,
    fp32_precision: str,
    assume_unpadded: bool,
) -> nn.Module:
    if not hasattr(model, "encoder") or not hasattr(model, "embeddings"):
        raise TypeError("model must be a Hugging Face DeBERTa-v2/v3 backbone")
    if getattr(model, "z_steps", 0) > 1:
        raise ValueError("DeBERTa z_steps > 1 is not supported by the optimized path")
    if isinstance(model.encoder, DebertaV2OptimizedEncoder):
        raise TypeError("the model already uses an optimized DeBERTa encoder")
    if not inference and sequence_lengths is not None:
        raise ValueError("sequence_lengths is only valid when inference=True")

    was_training = model.training
    if inference and sequence_lengths is not None and was_training:
        raise RuntimeError("call model.eval() before preparing inference buckets")

    model.encoder = DebertaV2OptimizedEncoder(
        model.encoder,
        model.config,
        backend=backend,
        inference=inference,
        fp32_precision=fp32_precision,
        assume_unpadded=assume_unpadded,
    )
    model.train(was_training)
    if inference and sequence_lengths is not None:
        model.encoder.prepare_for_inference(sequence_lengths)
    return model


def enable_deberta_inference(
    model: nn.Module,
    *,
    backend: str = "triton",
    sequence_lengths: Iterable[int] | int | None = None,
    fp32_precision: str = "strict",
    tuning: KernelTuningOptions | None = None,
    assume_unpadded: bool = False,
) -> nn.Module:
    """Replace a HF DeBERTa-v2/v3 encoder without changing checkpoint keys.

    ``model`` must be the Hugging Face backbone (the object with ``embeddings``
    and ``encoder``), not the outer GLiNER2 model.  The operation is in-place
    and inference-only.
    """

    return _enable_deberta(
        model,
        backend=backend,
        inference=True,
        sequence_lengths=sequence_lengths,
        fp32_precision=fp32_precision,
        tuning=tuning,
        assume_unpadded=assume_unpadded,
    )


def compile_deberta_buckets(
    encoder: DebertaV2OptimizedEncoder,
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
    if not encoder.inference:
        raise ValueError(
            "compile_deberta_buckets() requires an encoder configured "
            "with inference=True"
        )

    if sequence_lengths is None:
        lengths = tuple(sorted(encoder._prepared_plans))
    else:
        lengths = tuple(dict.fromkeys(int(length) for length in sequence_lengths))
    missing = [length for length in lengths if length not in encoder._prepared_plans]
    if missing:
        raise ValueError(f"unprepared bucket lengths: {missing}")

    supports_isolation = "isolate_recompiles" in inspect.signature(torch.compile).parameters
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
    backend: str = "triton",
    inference: bool = True,
    sequence_lengths: Iterable[int] | int | None = None,
    fp32_precision: str = "strict",
    tuning: KernelTuningOptions | None = None,
    assume_unpadded: bool = False,
) -> nn.Module:
    """Enable the unified optimized DeBERTa backend.

    ``inference=True`` is the default production mode and may install packed,
    parameter-derived caches during inference preparation.  Set
    ``inference=False`` before constructing an optimizer or training so
    parameter identities remain stable and all projections stay differentiable.
    ``backend='triton'`` falls back when the Triton runtime/model configuration
    is unsupported. ``backend='auto'`` additionally considers the model's
    current device and dtype before choosing Triton.
    """

    return _enable_deberta(
        model,
        backend=backend,
        inference=inference,
        sequence_lengths=sequence_lengths,
        fp32_precision=fp32_precision,
        tuning=tuning,
        assume_unpadded=assume_unpadded,
    )


def enable_deberta_training(
    model: nn.Module,
    *,
    backend: str = "triton",
    fp32_precision: str = "strict",
    assume_unpadded: bool = False,
) -> nn.Module:
    """Enable the same optimized encoder with differentiable parameter handling."""

    return optimize_deberta(
        model,
        backend=backend,
        inference=False,
        fp32_precision=fp32_precision,
        assume_unpadded=assume_unpadded,
    )


optimize_deberta_training = enable_deberta_training


# Compatibility alias for code from the standalone experiment.
enable_deberta_v2_inference = enable_deberta_inference


__all__ = [
    "DebertaV2InferenceEncoder",
    "DebertaV2OptimizedEncoder",
    "DebertaV2TrainingEncoder",
    "compile_deberta_buckets",
    "enable_deberta_inference",
    "enable_deberta_training",
    "enable_deberta_v2_inference",
    "optimize_deberta",
    "optimize_deberta_training",
]
