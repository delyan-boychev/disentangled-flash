"""Command-line tuner for the fused DeBERTa Triton attention kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from . import kernel
from .position import GradientBand, SharedPositionPlanCache, aligned_slot_count
from .tuning import (
    DEFAULT_DKV_KERNEL_CONFIGS,
    DEFAULT_DQ_KERNEL_CONFIGS,
    DEFAULT_KERNEL_CONFIGS,
    KERNEL_SOURCE_DIGEST,
    TUNING_BATCH_HEADS,
    TUNING_SEQUENCE_LENGTHS,
    CompilerSpec,
    DeviceResources,
    HardwareSpec,
    KernelConfig,
    KernelProfile,
    ProfileEntry,
    WorkloadKey,
    current_provenance,
    hardware_safe_candidates,
    heuristic_config,
    load_profile,
    merge_profile_entry,
    prepare_profile_retarget,
    save_profile,
    search_neighborhood,
    tuning_dtype,
)

# DeBERTa's default attention dropout. Any non-zero rate uses the same schedule.
TUNING_DROPOUT_P = 0.1


@dataclass(frozen=True)
class TuningCase:
    sequence_length: int
    head_dim: int
    batch_heads: int
    dtype: str
    has_c2p: bool
    has_p2c: bool
    fp32_precision: str
    layout: str = "padded"
    uses_padding_mask: bool = True

    def __post_init__(self) -> None:
        if self.layout not in {"padded", "packed"}:
            raise ValueError("layout must be padded or packed")
        if self.layout == "packed" and self.uses_padding_mask:
            raise ValueError("packed tuning cases cannot use a padding mask")


PRESETS = {
    "quick": {
        "lengths": (64,),
        "head_dims": (64,),
        "batch_heads": (8,),
        "dtypes": ("float16",),
        "relative_modes": ("both",),
        "layouts": ("padded", "packed"),
        "passes": ("inference", "training"),
        "dropout": ("off",),
        "search": "full",
    },
    "standard": {
        "lengths": (128, 512, 2048, 8192),
        "head_dims": (64,),
        "batch_heads": (32,),
        "dtypes": ("bfloat16",),
        "relative_modes": ("both",),
        "layouts": ("padded", "packed"),
        "passes": ("inference", "training"),
        "dropout": ("off", "on"),
        "search": "neighborhood",
    },
    "exhaustive": {
        "lengths": TUNING_SEQUENCE_LENGTHS,
        "head_dims": (64,),
        "batch_heads": TUNING_BATCH_HEADS,
        "dtypes": ("bfloat16", "float32"),
        "relative_modes": ("both",),
        "layouts": ("padded", "packed"),
        "passes": ("inference", "training"),
        "dropout": ("off", "on"),
        "search": "full",
    },
}


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _relative_flags(mode: str) -> tuple[bool, bool]:
    if mode not in {"none", "c2p", "p2c", "both"}:
        raise ValueError(f"unknown relative mode: {mode}")
    return mode in {"c2p", "both"}, mode in {"p2c", "both"}


def _dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def _cases(args: argparse.Namespace) -> Iterable[TuningCase]:
    preset = PRESETS[args.preset]
    lengths = args.lengths or preset["lengths"]
    head_dims = args.head_dims or preset["head_dims"]
    batch_heads = args.batch_heads or preset["batch_heads"]
    dtypes = args.dtypes or preset["dtypes"]
    relative_modes = args.relative_modes or preset["relative_modes"]
    layouts = args.layouts or preset["layouts"]
    unknown_dtypes = set(dtypes) - {"float16", "bfloat16", "float32"}
    unknown_modes = set(relative_modes) - {"none", "c2p", "p2c", "both"}
    unknown_layouts = set(layouts) - {"padded", "packed"}
    if unknown_dtypes:
        raise ValueError(f"unsupported dtypes: {sorted(unknown_dtypes)}")
    if unknown_modes:
        raise ValueError(f"unsupported relative modes: {sorted(unknown_modes)}")
    if unknown_layouts:
        raise ValueError(f"unsupported layouts: {sorted(unknown_layouts)}")
    unsupported_head_dims = set(head_dims) - {32, 64, 128}
    if unsupported_head_dims:
        raise ValueError(f"unsupported head dimensions: {sorted(unsupported_head_dims)}")
    # Measure one dtype per family.
    families: dict[str, str] = {}
    for dtype_name in dtypes:
        families.setdefault(tuning_dtype(dtype_name), dtype_name)
    dtypes = tuple(families.values())
    if max(lengths) > TUNING_SEQUENCE_LENGTHS[-1]:
        raise ValueError(f"tuning lengths must not exceed {TUNING_SEQUENCE_LENGTHS[-1]}")
    for length in lengths:
        for head_dim in head_dims:
            for occupancy in batch_heads:
                for dtype_name in dtypes:
                    for relative_mode in relative_modes:
                        has_c2p, has_p2c = _relative_flags(relative_mode)
                        precisions = ("strict", "fast") if dtype_name == "float32" else ("strict",)
                        for precision in precisions:
                            for layout in layouts:
                                padding_modes = (True, False) if layout == "padded" else (False,)
                                for uses_padding_mask in padding_modes:
                                    yield TuningCase(
                                        sequence_length=length,
                                        head_dim=head_dim,
                                        batch_heads=occupancy,
                                        dtype=dtype_name,
                                        has_c2p=has_c2p,
                                        has_p2c=has_p2c,
                                        fp32_precision=precision,
                                        layout=layout,
                                        uses_padding_mask=uses_padding_mask,
                                    )


def _dropout_modes(args: argparse.Namespace) -> tuple[bool, ...]:
    modes = args.dropout or PRESETS[args.preset]["dropout"]
    unknown = set(modes) - {"off", "on"}
    if unknown:
        raise ValueError(f"unsupported dropout modes: {sorted(unknown)}")
    return tuple(mode == "on" for mode in modes)


def _parse_candidate_list(payload: object, context: str) -> tuple[KernelConfig, ...]:
    if not isinstance(payload, list):
        raise TypeError(f"{context} must be a JSON array")
    configs = tuple(KernelConfig.from_dict(item) for item in payload)
    if not configs or len(configs) != len(set(configs)):
        raise ValueError(f"{context} must contain unique configurations")
    return configs


def _load_candidates(
    path: Path | None,
) -> tuple[
    tuple[KernelConfig, ...],
    tuple[KernelConfig, ...],
    tuple[KernelConfig, ...],
]:
    """Return forward, dQ, and dK/dV candidates from a list or a per-phase object."""

    if path is None:
        return (
            DEFAULT_KERNEL_CONFIGS,
            DEFAULT_DQ_KERNEL_CONFIGS,
            DEFAULT_DKV_KERNEL_CONFIGS,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read candidate file {path}: {error}") from error
    if isinstance(payload, dict):
        phases = ("forward", "dq", "dkv")
        if set(payload) != set(phases):
            raise ValueError("candidate object must contain exactly forward, dq, and dkv")
        forward, dq, dkv = (
            _parse_candidate_list(payload[phase], f"{phase} candidates") for phase in phases
        )
        return forward, dq, dkv
    if not isinstance(payload, list):
        raise TypeError("candidate file must contain a JSON array or a per-phase object")
    configs = _parse_candidate_list(payload, "candidate file")
    return configs, configs, configs


def _make_padded_inputs(
    case: TuningCase,
    device: torch.device,
    *,
    position_buckets: int = 256,
    max_relative_positions: int = 512,
    position_embedding_size: int = 256,
) -> tuple[tuple[object, ...], WorkloadKey]:
    dtype = _dtype(case.dtype)
    batch_size = 1
    num_heads = case.batch_heads
    length = case.sequence_length
    head_dim = case.head_dim
    query = torch.randn(batch_size, num_heads, length, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    uses_positions = case.has_c2p or case.has_p2c
    position_cache = SharedPositionPlanCache(
        position_buckets=position_buckets,
        max_relative_positions=max_relative_positions,
        position_embedding_size=position_embedding_size,
        uses_position_bias=uses_positions,
    )
    position_plan = position_cache.compact(length, device)
    active_slots = position_plan.active_slots.numel()
    table_width = aligned_slot_count(active_slots)
    relative_shape = (num_heads, batch_size, length, table_width)
    c2p = (
        torch.randn(relative_shape, device=device, dtype=dtype).permute(1, 0, 2, 3)
        if case.has_c2p
        else query
    )
    p2c = (
        torch.randn(relative_shape, device=device, dtype=dtype).permute(1, 0, 2, 3)
        if case.has_p2c
        else key
    )
    attention_mask = torch.ones(batch_size, length, device=device, dtype=torch.bool)
    scale_factor = 1 + int(case.has_c2p) + int(case.has_p2c)
    score_scale = (head_dim * scale_factor) ** -0.5
    args = (
        query,
        key,
        value,
        c2p,
        p2c,
        position_plan.delta_to_local,
        attention_mask,
        num_heads,
        length,
        length,
        table_width,
        length - 1,
        score_scale * 1.4426950408889634,
        case.uses_padding_mask,
        case.has_c2p,
        case.has_p2c,
        dtype == torch.bfloat16,
        dtype == torch.float32,
        case.fp32_precision == "strict",
    )
    workload = WorkloadKey(
        sequence_length=length,
        head_dim=head_dim,
        batch_heads=case.batch_heads,
        active_slots=active_slots,
        dtype=case.dtype,
        has_c2p=case.has_c2p,
        has_p2c=case.has_p2c,
        fp32_precision=case.fp32_precision,
        layout="padded",
        uses_padding_mask=case.uses_padding_mask,
    )
    return args, workload


def _packed_lengths(max_seqlen: int, occupancy: int) -> tuple[int, ...]:
    sequence_count = 4 if occupancy >= 4 else 1
    if sequence_count == 1:
        return (max_seqlen,)
    return (
        max_seqlen,
        max(1, max_seqlen - 1),
        max(1, (2 * max_seqlen) // 3),
        max(1, max_seqlen // 3),
    )


def _make_packed_inputs(
    case: TuningCase,
    device: torch.device,
    *,
    position_buckets: int = 256,
    max_relative_positions: int = 512,
    position_embedding_size: int = 256,
) -> tuple[tuple[object, ...], WorkloadKey]:
    dtype = _dtype(case.dtype)
    lengths = _packed_lengths(case.sequence_length, case.batch_heads)
    if case.batch_heads % len(lengths):
        raise ValueError("packed occupancy must be divisible by the sequence count")
    num_heads = case.batch_heads // len(lengths)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.int32)
    total_tokens = boundaries[-1]
    query = torch.randn(num_heads, total_tokens, case.head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    uses_positions = case.has_c2p or case.has_p2c
    position_cache = SharedPositionPlanCache(
        position_buckets=position_buckets,
        max_relative_positions=max_relative_positions,
        position_embedding_size=position_embedding_size,
        uses_position_bias=uses_positions,
    )
    position_plan = position_cache.compact(case.sequence_length, device)
    active_slots = position_plan.active_slots.numel()
    table_width = aligned_slot_count(active_slots)
    relative_shape = (num_heads, total_tokens, table_width)
    c2p = torch.randn(relative_shape, device=device, dtype=dtype) if case.has_c2p else query
    p2c = torch.randn(relative_shape, device=device, dtype=dtype) if case.has_p2c else key
    scale_factor = 1 + int(case.has_c2p) + int(case.has_p2c)
    score_scale = (case.head_dim * scale_factor) ** -0.5
    args = (
        query,
        key,
        value,
        c2p,
        p2c,
        position_plan.delta_to_local,
        cu_seqlens,
        case.sequence_length,
        case.sequence_length,
        table_width,
        case.sequence_length - 1,
        score_scale * 1.4426950408889634,
        case.has_c2p,
        case.has_p2c,
        dtype == torch.bfloat16,
        dtype == torch.float32,
        case.fp32_precision == "strict",
    )
    workload = WorkloadKey(
        sequence_length=case.sequence_length,
        head_dim=case.head_dim,
        batch_heads=case.batch_heads,
        active_slots=active_slots,
        dtype=case.dtype,
        has_c2p=case.has_c2p,
        has_p2c=case.has_p2c,
        fp32_precision=case.fp32_precision,
        layout="packed",
        uses_padding_mask=False,
    )
    return args, workload


def _make_inputs(
    case: TuningCase,
    device: torch.device,
    **position_options: int,
) -> tuple[tuple[object, ...], WorkloadKey]:
    if case.layout == "packed":
        return _make_packed_inputs(case, device, **position_options)
    return _make_padded_inputs(case, device, **position_options)


@dataclass(frozen=True)
class TrainingInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    pos_key: torch.Tensor | None
    pos_query: torch.Tensor | None
    delta_to_local: torch.Tensor
    sequence_metadata: torch.Tensor
    max_seqlen: int
    score_scale: float
    dropout_p: float = 0.0
    dropout_seed: torch.Tensor | None = None
    gradient_band: GradientBand | None = None

    def grad_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.query, self.key, self.value]
        if self.pos_key is not None:
            tensors.append(self.pos_key)
        if self.pos_query is not None:
            tensors.append(self.pos_query)
        return tuple(tensors)


def _make_training_inputs(
    case: TuningCase,
    device: torch.device,
    *,
    position_buckets: int = 256,
    max_relative_positions: int = 512,
    position_embedding_size: int = 256,
    dropout_p: float = 0.0,
) -> tuple[TrainingInputs, WorkloadKey]:
    dtype = _dtype(case.dtype)
    if case.layout == "packed":
        lengths = _packed_lengths(case.sequence_length, case.batch_heads)
        if case.batch_heads % len(lengths):
            raise ValueError("packed occupancy must be divisible by the sequence count")
        num_heads = case.batch_heads // len(lengths)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        total_tokens = boundaries[-1]
        shape = (num_heads, total_tokens, case.head_dim)
        sequence_metadata = torch.tensor(boundaries, device=device, dtype=torch.int32)
    else:
        num_heads = case.batch_heads
        shape = (1, num_heads, case.sequence_length, case.head_dim)
        sequence_metadata = torch.ones(
            (1, case.sequence_length),
            device=device,
            dtype=torch.bool,
        )
    query = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    uses_positions = case.has_c2p or case.has_p2c
    position_cache = SharedPositionPlanCache(
        position_buckets=position_buckets,
        max_relative_positions=max_relative_positions,
        position_embedding_size=position_embedding_size,
        uses_position_bias=uses_positions,
    )
    position_plan = position_cache.compact(case.sequence_length, device)
    active_slots = position_plan.active_slots.numel()
    position_shape = (num_heads, aligned_slot_count(active_slots), case.head_dim)
    pos_key = (
        torch.randn(position_shape, device=device, dtype=dtype, requires_grad=True)
        if case.has_c2p
        else None
    )
    pos_query = (
        torch.randn(position_shape, device=device, dtype=dtype, requires_grad=True)
        if case.has_p2c
        else None
    )
    scale_factor = 1 + int(case.has_c2p) + int(case.has_p2c)
    inputs = TrainingInputs(
        query=query,
        key=key,
        value=value,
        pos_key=pos_key,
        pos_query=pos_query,
        delta_to_local=position_plan.delta_to_local,
        sequence_metadata=sequence_metadata,
        max_seqlen=case.sequence_length,
        score_scale=(case.head_dim * scale_factor) ** -0.5,
        dropout_p=dropout_p,
        gradient_band=position_plan.gradient_band,
        dropout_seed=(
            torch.tensor([0x1BF52], device=device, dtype=torch.int64) if dropout_p else None
        ),
    )
    workload = WorkloadKey(
        sequence_length=case.sequence_length,
        head_dim=case.head_dim,
        batch_heads=case.batch_heads,
        active_slots=active_slots,
        dtype=case.dtype,
        has_c2p=case.has_c2p,
        has_p2c=case.has_p2c,
        fp32_precision=case.fp32_precision,
        layout=case.layout,
        uses_padding_mask=case.uses_padding_mask,
        phase="training_forward",
        has_dropout=dropout_p > 0.0,
    )
    return inputs, workload


def _run_training_forward(
    inputs: TrainingInputs,
    workload: WorkloadKey,
    forward_config: KernelConfig,
    dq_config: KernelConfig,
    dkv_config: KernelConfig,
) -> torch.Tensor:
    from .training._kernels import training_attention, training_attention_packed

    options = {
        "score_scale": inputs.score_scale,
        "strict_fp32": workload.fp32_precision == "strict",
        "forward_config": forward_config,
        "dq_config": dq_config,
        "dkv_config": dkv_config,
        "dropout_p": inputs.dropout_p,
        "dropout_seed": inputs.dropout_seed,
        "gradient_band": inputs.gradient_band,
    }
    if workload.layout == "packed":
        return training_attention_packed(
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.pos_key,
            inputs.pos_query,
            inputs.delta_to_local,
            inputs.sequence_metadata,
            max_seqlen=inputs.max_seqlen,
            **options,
        )
    return training_attention(
        inputs.query,
        inputs.key,
        inputs.value,
        inputs.pos_key,
        inputs.pos_query,
        inputs.delta_to_local,
        inputs.sequence_metadata,
        has_padding=workload.uses_padding_mask,
        **options,
    )


def _time_cuda(callable_: Any, *, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def _run_training_config(
    inputs: TrainingInputs,
    workload: WorkloadKey,
    grad_output: torch.Tensor,
    *,
    forward_config: KernelConfig,
    dq_config: KernelConfig,
    dkv_config: KernelConfig,
    warmup: int,
    repetitions: int,
) -> tuple[float, float, torch.Tensor, tuple[torch.Tensor, ...]]:
    output = _run_training_forward(
        inputs,
        workload,
        forward_config,
        dq_config,
        dkv_config,
    )
    grad_tensors = inputs.grad_tensors()

    def run_forward() -> torch.Tensor:
        return _run_training_forward(
            inputs,
            workload,
            forward_config,
            dq_config,
            dkv_config,
        )

    forward_ms = _time_cuda(run_forward, warmup=warmup, repetitions=repetitions)

    def run_backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            output,
            grad_tensors,
            grad_output,
            retain_graph=True,
        )

    gradients = run_backward()
    backward_ms = _time_cuda(run_backward, warmup=warmup, repetitions=repetitions)
    return forward_ms, backward_ms, output, gradients


def _reference(arguments: tuple[object, ...], *, query_chunk_size: int = 32) -> torch.Tensor:
    (
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local,
        attention_mask,
        num_heads,
        sequence_length,
        _length_regime,
        _active_slots,
        position_offset,
        score_scale_log2,
        use_padding_mask,
        has_c2p,
        has_p2c,
        _is_bf16,
        _is_fp32,
        _strict_fp32,
    ) = arguments
    score_scale = float(score_scale_log2) / 1.4426950408889634
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    positions = torch.arange(sequence_length, device=query.device)
    key_positions = positions[None, :]
    key_transposed = key.float().transpose(-1, -2)
    value_float = value.float()
    p2c_float = p2c.float() if has_p2c else None
    outputs = []
    for start in range(0, sequence_length, query_chunk_size):
        end = min(start + query_chunk_size, sequence_length)
        query_positions = positions[start:end, None]
        local = delta_to_local[query_positions - key_positions + position_offset].long()
        scores = torch.matmul(query[:, :, start:end].float(), key_transposed)
        gather_indices = local.expand(query.size(0), num_heads, -1, -1)
        if has_c2p:
            scores += torch.gather(c2p[:, :, start:end].float(), -1, gather_indices)
        if has_p2c:
            assert p2c_float is not None
            p2c_by_key = p2c_float.unsqueeze(-3).expand(-1, -1, end - start, -1, -1)
            scores += torch.gather(
                p2c_by_key,
                -1,
                local[None, None, :, :, None].expand(query.size(0), num_heads, -1, -1, -1),
            ).squeeze(-1)
        scores *= score_scale
        if use_padding_mask:
            pair_mask = attention_mask[:, None, start:end, None] & attention_mask[:, None, None, :]
            scores = scores.masked_fill(~pair_mask, float("-inf"))
            padded_queries = ~attention_mask[:, None, start:end, None]
            scores = torch.where(padded_queries, torch.zeros_like(scores), scores)
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), value_float))
    output = torch.cat(outputs, dim=2).to(query.dtype)
    return output.transpose(1, 2).reshape(query.size(0), sequence_length, -1)


def _packed_reference(arguments: tuple[object, ...]) -> torch.Tensor:
    (
        query,
        key,
        value,
        c2p,
        p2c,
        delta_to_local,
        cu_seqlens,
        max_seqlen,
        length_regime,
        active_slots,
        position_offset,
        score_scale_log2,
        has_c2p,
        has_p2c,
        is_bf16,
        is_fp32,
        strict_fp32,
    ) = arguments
    num_heads = query.size(0)
    boundaries = cu_seqlens.detach().cpu().tolist()
    outputs = []
    for start, end in pairwise(boundaries):
        length = end - start
        mask = torch.ones((1, length), device=query.device, dtype=torch.bool)
        padded_arguments = (
            query[:, start:end].unsqueeze(0),
            key[:, start:end].unsqueeze(0),
            value[:, start:end].unsqueeze(0),
            c2p[:, start:end].unsqueeze(0) if has_c2p else query[:, start:end].unsqueeze(0),
            p2c[:, start:end].unsqueeze(0) if has_p2c else key[:, start:end].unsqueeze(0),
            delta_to_local,
            mask,
            num_heads,
            length,
            length_regime,
            active_slots,
            position_offset,
            score_scale_log2,
            False,
            has_c2p,
            has_p2c,
            is_bf16,
            is_fp32,
            strict_fp32,
        )
        outputs.append(_reference(padded_arguments))
    output = torch.cat(outputs, dim=1)
    if output.size(1) != query.size(1) or max_seqlen < max(
        end - start for start, end in pairwise(boundaries)
    ):
        raise ValueError("invalid packed tuning boundaries")
    return output.squeeze(0)


def _dropout_keep(inputs: TrainingInputs, stream_start: int, streams: int, length: int):
    if not inputs.dropout_p:
        return None
    from .training._kernels import dropout_keep_mask

    assert inputs.dropout_seed is not None
    return dropout_keep_mask(
        inputs.dropout_seed,
        stream_start=stream_start,
        streams=streams,
        length=length,
        dropout_p=inputs.dropout_p,
    )


def _training_reference(inputs: TrainingInputs, workload: WorkloadKey) -> torch.Tensor:
    def sequence_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = query.size(-2)
        positions = torch.arange(length, device=query.device)
        local = inputs.delta_to_local[
            positions[:, None] - positions[None, :] + inputs.max_seqlen - 1
        ].long()
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        if inputs.pos_key is not None:
            c2p = torch.matmul(query.float(), inputs.pos_key.float().transpose(-1, -2))
            scores = scores + torch.gather(
                c2p,
                -1,
                local.expand(query.size(0), -1, -1),
            )
        if inputs.pos_query is not None:
            p2c = torch.matmul(key.float(), inputs.pos_query.float().transpose(-1, -2))
            # scores[h, i, j] += p2c[h, j, local[i, j]]. Gathering from an
            # expanded [H, L, L, R] view would allocate all of it in backward.
            scores = scores + torch.gather(
                p2c,
                -1,
                local.transpose(0, 1).expand(query.size(0), -1, -1),
            ).transpose(-1, -2)
        scores = scores * inputs.score_scale
        if mask is not None:
            pair_mask = mask[:, None] & mask[None, :]
            scores = scores.masked_fill(~pair_mask, float("-inf"))
            scores = torch.where(~mask[:, None], torch.zeros_like(scores), scores)
        probabilities = torch.softmax(scores, dim=-1)
        if keep is not None:
            probabilities = probabilities * keep / (1.0 - inputs.dropout_p)
        context = torch.matmul(probabilities, value.float()).to(query.dtype)
        return context.transpose(0, 1).reshape(length, -1)

    if workload.layout == "packed":
        num_heads = inputs.query.size(0)
        boundaries = inputs.sequence_metadata.detach().cpu().tolist()
        return torch.cat(
            [
                sequence_attention(
                    inputs.query[:, start:end],
                    inputs.key[:, start:end],
                    inputs.value[:, start:end],
                    None,
                    _dropout_keep(inputs, sequence * num_heads, num_heads, end - start),
                )
                for sequence, (start, end) in enumerate(pairwise(boundaries))
            ],
            dim=0,
        )
    mask = inputs.sequence_metadata[0] if workload.uses_padding_mask else None
    num_heads, length = inputs.query.size(1), inputs.query.size(2)
    return sequence_attention(
        inputs.query[0],
        inputs.key[0],
        inputs.value[0],
        mask,
        _dropout_keep(inputs, 0, num_heads, length),
    ).unsqueeze(0)


def _reference_for_workload(arguments: tuple[object, ...], workload: WorkloadKey) -> torch.Tensor:
    return _packed_reference(arguments) if workload.layout == "packed" else _reference(arguments)


def _run_config(
    arguments: tuple[object, ...],
    config: KernelConfig,
    workload: WorkloadKey,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, torch.Tensor]:
    configured_arguments = arguments + (
        config.block_m,
        config.block_n,
        config.num_warps,
        config.num_stages,
    )
    operation = (
        kernel._deberta_attention_packed_configured_op
        if workload.layout == "packed"
        else kernel._deberta_attention_configured_op
    )
    output = operation(*configured_arguments)
    torch.cuda.synchronize()
    for _ in range(warmup):
        output = operation(*configured_arguments)
    torch.cuda.synchronize()
    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation(*configured_arguments)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings), output


def _validate_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    strict_fp32: bool,
) -> None:
    if not torch.isfinite(actual).all():
        raise ValueError("kernel produced non-finite output")
    if actual.dtype == torch.float32 and strict_fp32:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    elif actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _validate_gradients(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    strict_fp32: bool,
) -> None:
    if len(actual) != len(expected):
        raise ValueError("training gradient tuple length mismatch")
    for actual_gradient, expected_gradient in zip(actual, expected):
        _validate_output(actual_gradient, expected_gradient, strict_fp32=strict_fp32)


def _validation_metadata(workload: WorkloadKey) -> dict[str, str]:
    if workload.dtype == "float32" and workload.fp32_precision == "strict":
        rtol, atol = "0.0002", "0.0002"
    elif workload.dtype == "float32":
        rtol, atol = "0.005", "0.005"
    else:
        rtol, atol = "0.02", "0.02"
    if workload.layout == "packed":
        patterns = "mixed_lengths,reversed_boundaries"
    elif workload.uses_padding_mask:
        patterns = "full,partial,one_token,fully_masked"
    else:
        patterns = "dense_unmasked"
    metadata = {"rtol": rtol, "atol": atol, "patterns": patterns}
    if workload.has_dropout:
        metadata["dropout_p"] = str(TUNING_DROPOUT_P)
    return metadata


def _validate_mask_patterns(arguments: tuple[object, ...], config: KernelConfig) -> None:
    base_mask = arguments[6]
    length = base_mask.size(1)
    partial = torch.zeros_like(base_mask)
    partial[:, length // 2 :] = True
    only_last = torch.zeros_like(base_mask)
    only_last[:, -1] = True
    config_values = (
        config.block_m,
        config.block_n,
        config.num_warps,
        config.num_stages,
    )
    for mask in (partial, only_last, torch.zeros_like(base_mask)):
        masked_arguments = arguments[:6] + (mask,) + arguments[7:]
        expected = _reference(masked_arguments)
        actual = kernel._deberta_attention_configured_op(*(masked_arguments + config_values))
        torch.cuda.synchronize()
        _validate_output(actual, expected, strict_fp32=bool(arguments[18]))


def _validate_packed_patterns(arguments: tuple[object, ...], config: KernelConfig) -> None:
    boundaries = arguments[6].detach().cpu().tolist()
    lengths = [end - start for start, end in pairwise(boundaries)]
    reversed_boundaries = [0]
    for length in reversed(lengths):
        reversed_boundaries.append(reversed_boundaries[-1] + length)
    alternate_cu_seqlens = torch.tensor(
        reversed_boundaries,
        device=arguments[6].device,
        dtype=arguments[6].dtype,
    )
    alternate_arguments = arguments[:6] + (alternate_cu_seqlens,) + arguments[7:]
    config_values = (
        config.block_m,
        config.block_n,
        config.num_warps,
        config.num_stages,
    )
    expected = _packed_reference(alternate_arguments)
    actual = kernel._deberta_attention_packed_configured_op(*(alternate_arguments + config_values))
    torch.cuda.synchronize()
    _validate_output(actual, expected, strict_fp32=bool(arguments[16]))


def _profile_provenance(
    args: argparse.Namespace, candidates: tuple[KernelConfig, ...]
) -> dict[str, str]:
    candidate_payload = json.dumps(
        [candidate.to_dict() for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **current_provenance(args.seed),
        "preset": args.preset,
        "layouts": ",".join(args.layouts or PRESETS[args.preset]["layouts"]),
        "passes": ",".join(args.passes or PRESETS[args.preset]["passes"]),
        "warmup": str(args.warmup),
        "repetitions": str(args.repetitions),
        "tie_margin": str(args.tie_margin),
        "search": "retarget" if getattr(args, "retarget", False) else "hierarchical",
        "kernel_digest": KERNEL_SOURCE_DIGEST,
        "candidate_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _choose_winner(
    results: list[tuple[float, KernelConfig]],
    candidates: tuple[KernelConfig, ...],
    tie_margin: float,
) -> tuple[float, KernelConfig]:
    if not results:
        raise RuntimeError("no correct kernel configuration survived")
    best_latency = min(latency for latency, _config in results)
    near_ties = [result for result in results if result[0] <= best_latency * (1 + tie_margin)]
    return min(near_ties, key=lambda result: candidates.index(result[1]))


def _retarget_candidates(
    seed: KernelConfig,
    candidates: tuple[KernelConfig, ...],
) -> tuple[KernelConfig, ...]:
    """Use an old winner plus a small conservative neighborhood on a new compiler."""

    selected = [seed]
    selected.extend(
        config
        for config in candidates
        if (
            (config.block_m, config.block_n, config.num_warps)
            == (seed.block_m, seed.block_n, seed.num_warps)
            or config in {KernelConfig(32, 32, 4), KernelConfig(64, 64, 4)}
        )
    )
    return tuple(dict.fromkeys(selected))


@dataclass(frozen=True)
class SearchResult:
    latency_ms: float
    config: KernelConfig
    rejected: tuple[str, ...] = ()


def _format_config(config: KernelConfig) -> str:
    return f"{config.block_m}x{config.block_n}w{config.num_warps}s{config.num_stages}"


def _search_configs(
    candidates: tuple[KernelConfig, ...],
    evaluate: Callable[[KernelConfig], float],
    *,
    tie_margin: float,
    label: str,
    seed: KernelConfig | None = None,
    retarget: bool = False,
    verbose: bool = False,
) -> SearchResult:
    """Tune tile/warps first and pipeline depth around the strongest shapes."""

    ordered = tuple(dict.fromkeys(((seed,) if seed is not None else ()) + candidates))
    if retarget and seed is not None:
        base = _retarget_candidates(seed, ordered)
        refinements: tuple[KernelConfig, ...] = ()
    else:
        base = tuple(config for config in ordered if config.num_stages == 1)
        refinements = tuple(config for config in ordered if config.num_stages > 1)

    results: list[tuple[float, KernelConfig]] = []
    rejected: list[str] = []

    def run_configs(configs: Iterable[KernelConfig]) -> None:
        for config in configs:
            try:
                results.append((evaluate(config), config))
                if verbose:
                    print(f"  {label} {_format_config(config)}: {results[-1][0]:.4f} ms")
            # Compile and resource failures raise different exception types.
            except Exception as error:  # noqa: BLE001
                print(f"  rejected {label} {config}: {type(error).__name__}: {error}")
                rejected.append(f"{_format_config(config)}:{type(error).__name__}")
                torch.cuda.empty_cache()

    run_configs(base)
    if not results:
        raise RuntimeError(f"no correct {label} configuration survived")
    if refinements:
        strongest = {
            (config.block_m, config.block_n, config.num_warps)
            for _latency, config in sorted(results, key=lambda result: result[0])[:2]
        }
        run_configs(
            config
            for config in refinements
            if (config.block_m, config.block_n, config.num_warps) in strongest
        )
    latency, winner = _choose_winner(results, ordered, tie_margin)
    return SearchResult(latency, winner, tuple(rejected))


def run(args: argparse.Namespace) -> None:
    if kernel.triton is None or not torch.cuda.is_available():
        raise RuntimeError("the tuning command requires CUDA and Triton")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hardware = HardwareSpec.current(device)
    compiler = CompilerSpec.current()
    passes = args.passes or PRESETS[args.preset]["passes"]
    unknown_passes = set(passes) - {"inference", "training"}
    if unknown_passes:
        raise ValueError(f"unsupported tuning passes: {sorted(unknown_passes)}")
    forward_candidates, dq_candidates, dkv_candidates = _load_candidates(args.candidates)
    all_candidates = tuple(dict.fromkeys(forward_candidates + dq_candidates + dkv_candidates))
    output_path = args.output.expanduser().resolve()
    profile: KernelProfile | None = load_profile(output_path) if output_path.exists() else None
    retarget = bool(getattr(args, "retarget", False))
    if retarget and profile is None:
        raise ValueError("retarget requires an existing profile")
    if profile is not None:
        compatibility = profile.compatibility(hardware, compiler)
        if compatibility == "incompatible":
            raise ValueError("the output profile has incompatible hardware or launch schema")
        if retarget:
            profile = prepare_profile_retarget(
                profile,
                hardware=hardware,
                compiler=compiler,
                provenance=_profile_provenance(args, all_candidates),
            )
            save_profile(profile, output_path)
            print(f"Retargeting {len(profile.entries)} saved winners for the current compiler")
        elif compatibility != "exact":
            details = "; ".join(profile.compiler.mismatches(compiler))
            suffix = f" ({details})" if details else ""
            raise ValueError(
                "the output profile requires retargeting"
                f"{suffix}; run `python -m disentangled_flash.tune retarget {output_path}`"
            )
    seed_entries = {} if profile is None else {entry.workload: entry for entry in profile.entries}

    resources = DeviceResources.current(device)
    search_mode = PRESETS[args.preset]["search"]

    def is_allowed(workload: WorkloadKey, config: KernelConfig) -> bool:
        return config in hardware_safe_candidates(
            (config,),
            phase=workload.phase,
            head_dim=workload.head_dim,
            dtype=workload.dtype,
            resources=resources,
        )

    def heuristic_for(workload: WorkloadKey, batch_heads: int) -> KernelConfig:
        return heuristic_config(
            phase=workload.phase,
            sequence_length=workload.sequence_length,
            head_dim=workload.head_dim,
            dtype=workload.dtype,
            batch_heads=batch_heads,
            resources=resources,
        )

    def phase_candidates(
        workload: WorkloadKey,
        candidates: tuple[KernelConfig, ...],
        batch_heads: int,
    ) -> tuple[KernelConfig, ...]:
        """Configs this GPU can run, narrowed to the heuristic's neighbors for standard."""

        safe = hardware_safe_candidates(
            candidates,
            phase=workload.phase,
            head_dim=workload.head_dim,
            dtype=workload.dtype,
            resources=resources,
        )
        if search_mode == "neighborhood":
            return search_neighborhood(heuristic_for(workload, batch_heads), safe)
        return safe

    completed = {
        workload
        for workload, entry in seed_entries.items()
        if entry.validated and is_allowed(workload, entry.config)
    }

    def is_cached(workload: WorkloadKey) -> bool:
        return workload in completed and not args.retest

    def search_phase(
        workload: WorkloadKey,
        candidates: tuple[KernelConfig, ...],
        evaluate: Callable[[KernelConfig], float],
        label: str,
        batch_heads: int,
    ) -> tuple[SearchResult, str]:
        """Reuse a validated winner, revalidate a pending seed, or search from scratch."""

        seed = seed_entries.get(workload)
        if is_cached(workload):
            assert seed is not None
            return SearchResult(seed.latency_ms, seed.config), "cached"
        candidates = phase_candidates(workload, candidates, batch_heads)
        if seed is not None and not is_allowed(workload, seed.config):
            seed = None
        pending = seed is not None and not seed.validated
        result = _search_configs(
            candidates,
            evaluate,
            tie_margin=args.tie_margin,
            label=label,
            seed=None if seed is None else seed.config,
            retarget=pending,
            verbose=args.verbose,
        )
        return result, "retarget" if pending else "hierarchical"

    def record(
        workload: WorkloadKey,
        result: SearchResult,
        search: str,
        extra: dict[str, str] | None = None,
    ) -> None:
        nonlocal profile
        validation = {**_validation_metadata(workload), "search": search, **(extra or {})}
        if result.rejected:
            validation["rejected"] = ";".join(result.rejected)
        profile = merge_profile_entry(
            profile,
            hardware=hardware,
            compiler=compiler,
            entry=ProfileEntry(
                workload=workload,
                config=result.config,
                latency_ms=result.latency_ms,
                validation=validation,
            ),
            provenance=_profile_provenance(args, all_candidates),
        )
        save_profile(profile, output_path)
        completed.add(workload)

    cases = list(_cases(args))
    if not torch.cuda.is_bf16_supported() and any(case.dtype == "bfloat16" for case in cases):
        cases = [
            replace(case, dtype="float16") if case.dtype == "bfloat16" else case for case in cases
        ]
        print("Measuring the half-precision family with FP16; this GPU lacks BF16")
    dropout_modes = _dropout_modes(args)
    training_cases = (
        [(case, dropout) for case in cases for dropout in dropout_modes]
        if "training" in passes
        else []
    )
    workload_count = len(cases) * int("inference" in passes) + 3 * len(training_cases)
    print(
        f"Tuning {workload_count} workloads on {hardware.name}; "
        f"forward={len(forward_candidates)}, dQ={len(dq_candidates)}, "
        f"dK/dV={len(dkv_candidates)} candidates"
    )
    inference_cases = cases if "inference" in passes else []
    for index, case in enumerate(inference_cases, start=1):
        arguments, workload = _make_inputs(
            case,
            device,
            position_buckets=args.position_buckets,
            max_relative_positions=args.max_relative_positions,
            position_embedding_size=args.position_embedding_size,
        )
        if retarget and workload not in seed_entries:
            continue
        if is_cached(workload):
            print(f"[{index}/{len(inference_cases)} inference] cached {workload}")
            continue
        expected = _reference_for_workload(arguments, workload)
        strict_index = 16 if workload.layout == "packed" else 18

        def evaluate_inference(
            config: KernelConfig,
            _arguments: tuple[object, ...] = arguments,
            _workload: WorkloadKey = workload,
            _expected: torch.Tensor = expected,
            _strict_index: int = strict_index,
        ) -> float:
            latency, actual = _run_config(
                _arguments,
                config,
                _workload,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            _validate_output(actual, _expected, strict_fp32=bool(_arguments[_strict_index]))
            if _workload.layout == "packed":
                _validate_packed_patterns(_arguments, config)
            elif _workload.uses_padding_mask:
                _validate_mask_patterns(_arguments, config)
            return latency

        result, search = search_phase(
            workload, forward_candidates, evaluate_inference, "inference", case.batch_heads
        )
        record(workload, result, search)
        print(
            f"[{index}/{len(inference_cases)} inference] "
            f"{result.latency_ms:.4f} ms {result.config} {workload}"
        )

    for index, (case, dropout) in enumerate(training_cases, start=1):
        inputs, base_workload = _make_training_inputs(
            case,
            device,
            position_buckets=args.position_buckets,
            max_relative_positions=args.max_relative_positions,
            position_embedding_size=args.position_embedding_size,
            dropout_p=TUNING_DROPOUT_P if dropout else 0.0,
        )
        phase_workloads = {
            phase: replace(base_workload, phase=phase)
            for phase in ("training_forward", "backward_dq", "backward_dkv")
        }
        if retarget and not any(workload in seed_entries for workload in phase_workloads.values()):
            continue
        if all(is_cached(workload) for workload in phase_workloads.values()):
            print(f"[{index}/{len(training_cases)} training] cached {base_workload}")
            continue
        expected = _training_reference(inputs, base_workload)
        grad_output = torch.randn_like(expected)
        # Drop the reference graph now; it holds L x L activations.
        expected_gradients = torch.autograd.grad(expected, inputs.grad_tensors(), grad_output)
        expected = expected.detach()
        strict_fp32 = base_workload.dtype == "float32" and (
            base_workload.fp32_precision == "strict"
        )

        def baseline(workload: WorkloadKey, batch_heads: int = case.batch_heads) -> KernelConfig:
            seed = seed_entries.get(workload)
            if seed is not None and is_allowed(workload, seed.config):
                return seed.config
            return heuristic_for(workload, batch_heads)

        dq_baseline = baseline(phase_workloads["backward_dq"])
        dkv_baseline = baseline(phase_workloads["backward_dkv"])

        def measure(
            forward_config: KernelConfig,
            dq_config: KernelConfig,
            dkv_config: KernelConfig,
            *,
            _inputs: TrainingInputs = inputs,
            _workload: WorkloadKey = base_workload,
            _grad_output: torch.Tensor = grad_output,
            _expected: torch.Tensor = expected,
            _expected_gradients: tuple[torch.Tensor, ...] = expected_gradients,
            _strict_fp32: bool = strict_fp32,
        ) -> tuple[float, float]:
            forward_ms, backward_ms, actual, gradients = _run_training_config(
                _inputs,
                _workload,
                _grad_output,
                forward_config=forward_config,
                dq_config=dq_config,
                dkv_config=dkv_config,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            _validate_output(actual, _expected, strict_fp32=_strict_fp32)
            _validate_gradients(
                gradients,
                _expected_gradients,
                strict_fp32=_strict_fp32,
            )
            return forward_ms, backward_ms

        forward, forward_search = search_phase(
            phase_workloads["training_forward"],
            forward_candidates,
            lambda config, _dq=dq_baseline, _dkv=dkv_baseline: measure(config, _dq, _dkv)[0],
            "training forward",
            case.batch_heads,
        )
        dq, dq_search = search_phase(
            phase_workloads["backward_dq"],
            dq_candidates,
            lambda config, _fwd=forward.config, _dkv=dkv_baseline: measure(_fwd, config, _dkv)[1],
            "dQ",
            case.batch_heads,
        )
        dkv, dkv_search = search_phase(
            phase_workloads["backward_dkv"],
            dkv_candidates,
            lambda config, _fwd=forward.config, _dq=dq.config: measure(_fwd, _dq, config)[1],
            "dK/dV",
            case.batch_heads,
        )

        # Check the three winners together.
        combined_forward_ms, combined_backward_ms = measure(forward.config, dq.config, dkv.config)
        refined = False
        # dQ was timed with the baseline dK/dV. If the final pair is slower,
        # re-tune dQ once against the chosen dK/dV.
        if dq_search != "cached" and combined_backward_ms > dq.latency_ms * (1 + args.tie_margin):
            retuned = _search_configs(
                phase_candidates(phase_workloads["backward_dq"], dq_candidates, case.batch_heads),
                lambda config, _fwd=forward.config, _dkv=dkv.config: measure(_fwd, config, _dkv)[1],
                tie_margin=args.tie_margin,
                label="dQ refinement",
                seed=dq.config,
                verbose=args.verbose,
            )
            dq = replace(retuned, rejected=dq.rejected + retuned.rejected)
            combined_forward_ms, combined_backward_ms = measure(
                forward.config, dq.config, dkv.config
            )
            refined = True

        combined = {
            "combined_forward_ms": f"{combined_forward_ms:.9g}",
            "combined_backward_ms": f"{combined_backward_ms:.9g}",
        }
        for phase, result, search in (
            ("training_forward", forward, forward_search),
            ("backward_dq", dq, dq_search),
            ("backward_dkv", dkv, dkv_search),
        ):
            if search == "cached":
                continue
            extra = dict(combined)
            if refined and phase == "backward_dq":
                extra["refined"] = "dq"
            record(phase_workloads[phase], result, search, extra)
        print(
            f"[{index}/{len(training_cases)} training] "
            f"fwd={forward.latency_ms:.4f} ms {forward.config}; "
            f"dQ-total={dq.latency_ms:.4f} ms {dq.config}; "
            f"dK/dV-total={dkv.latency_ms:.4f} ms {dkv.config}; "
            f"combined={combined_forward_ms:.4f}+{combined_backward_ms:.4f} ms "
            f"{base_workload}"
        )
    print(f"Saved {len(profile.entries) if profile else 0} entries to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="standard",
        help="standard: a few shapes around the heuristic; exhaustive: release profiles",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", type=_csv_ints)
    parser.add_argument("--head-dims", type=_csv_ints)
    parser.add_argument("--batch-heads", type=_csv_ints)
    parser.add_argument("--dtypes", type=_csv_strings)
    parser.add_argument("--relative-modes", type=_csv_strings)
    parser.add_argument(
        "--dropout",
        type=_csv_strings,
        help="comma-separated off and/or on attention-dropout variants for training",
    )
    parser.add_argument(
        "--layouts",
        type=_csv_strings,
        help="comma-separated padded and/or packed layouts; padded tunes masked and unmasked",
    )
    parser.add_argument(
        "--passes",
        type=_csv_strings,
        help="comma-separated inference and/or training passes",
    )
    parser.add_argument("--position-buckets", type=int, default=256)
    parser.add_argument("--max-relative-positions", type=int, default=512)
    parser.add_argument("--position-embedding-size", type=int, default=256)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=0.02,
        help="prefer an earlier conservative candidate when it is within this fraction",
    )
    parser.add_argument("--retest", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every measured candidate, e.g. to compare configs at one shape",
    )
    return parser


def inspect_profile(path: Path, device: int = 0) -> bool:
    """Print whether a saved profile is reusable by the current compiler and GPU."""

    profile = load_profile(path.expanduser().resolve())
    print(f"Profile: {path}")
    print(
        f"Format: {profile.format_version}; kernel: {profile.kernel_version}; "
        f"digest: {profile.kernel_digest[:12]}"
    )
    print(
        f"GPU: {profile.hardware.name} "
        f"sm_{profile.hardware.compute_capability[0]}{profile.hardware.compute_capability[1]}"
    )
    print(f"Entries: {len(profile.entries)}")
    compiler = CompilerSpec.current()
    issues = list(profile.compiler.mismatches(compiler))
    if torch.cuda.is_available():
        hardware = HardwareSpec.current(device)
        compatibility = profile.compatibility(hardware, compiler)
        if compatibility == "exact":
            print("Compatibility: exact")
            print("Compatible: yes")
            return True
        if compatibility == "retargetable":
            if profile.kernel_digest != KERNEL_SOURCE_DIGEST:
                issues.append("kernel source digest differs")
            print("Compatibility: retargetable")
            print(f"Retarget with: python -m disentangled_flash.tune retarget {path}")
        else:
            issues.append(f"hardware: profile={profile.hardware.name!r}, current={hardware.name!r}")
            print("Compatibility: incompatible")
    else:
        issues.append("hardware: CUDA is unavailable, so GPU compatibility was not checked")
        print("Compatibility: unknown")
    if issues:
        print("Compatible: no")
        for issue in issues:
            print(f"  - {issue}")
    return False


def build_inspect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a kernel tuning profile")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments[:1] == ["inspect"]:
        inspect_args = build_inspect_parser().parse_args(arguments[1:])
        inspect_profile(inspect_args.profile, inspect_args.device)
        return
    if arguments[:1] == ["retarget"]:
        if len(arguments) < 2:
            raise SystemExit("retarget requires a profile path")
        args = build_parser().parse_args(["--output", arguments[1], *arguments[2:]])
        args.retarget = True
    else:
        args = build_parser().parse_args(arguments)
        args.retarget = False
    if (
        args.warmup < 0
        or args.repetitions < 1
        or not 0 <= args.tie_margin <= 0.25
        or args.position_buckets < -1
        or (args.position_buckets > 0 and args.max_relative_positions < 1)
        or args.position_embedding_size < 1
    ):
        raise SystemExit("invalid timing, tie-margin, or relative-position arguments; use --help")
    run(args)


if __name__ == "__main__":
    main()
