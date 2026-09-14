"""Command-line tuner for the fused DeBERTa Triton attention kernel."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from . import kernel
from .position import SharedPositionPlanCache
from .tuning import (
    DEFAULT_KERNEL_CONFIGS,
    HardwareSpec,
    KernelConfig,
    KernelProfile,
    ProfileEntry,
    TUNING_SEQUENCE_LENGTHS,
    WorkloadKey,
    load_profile,
    merge_profile_entry,
    save_profile,
)


@dataclass(frozen=True)
class TuningCase:
    sequence_length: int
    head_dim: int
    batch_heads: int
    dtype: str
    has_c2p: bool
    has_p2c: bool
    fp32_precision: str


PRESETS = {
    "quick": {
        "lengths": (64,),
        "head_dims": (64,),
        "batch_heads": (8,),
        "dtypes": ("float16",),
        "relative_modes": ("both",),
    },
    "standard": {
        "lengths": TUNING_SEQUENCE_LENGTHS,
        "head_dims": (32, 64, 128),
        "batch_heads": (1, 8, 32),
        "dtypes": ("float16", "bfloat16", "float32"),
        "relative_modes": ("both",),
    },
    "exhaustive": {
        "lengths": TUNING_SEQUENCE_LENGTHS,
        "head_dims": (32, 64, 128),
        "batch_heads": (1, 4, 8, 16, 32),
        "dtypes": ("float16", "bfloat16", "float32"),
        "relative_modes": ("none", "c2p", "p2c", "both"),
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
    unknown_dtypes = set(dtypes) - {"float16", "bfloat16", "float32"}
    unknown_modes = set(relative_modes) - {"none", "c2p", "p2c", "both"}
    if unknown_dtypes:
        raise ValueError(f"unsupported dtypes: {sorted(unknown_dtypes)}")
    if unknown_modes:
        raise ValueError(f"unsupported relative modes: {sorted(unknown_modes)}")
    unsupported_head_dims = set(head_dims) - {32, 64, 128}
    if unsupported_head_dims:
        raise ValueError(f"unsupported head dimensions: {sorted(unsupported_head_dims)}")
    if max(lengths) > TUNING_SEQUENCE_LENGTHS[-1]:
        raise ValueError(
            f"tuning lengths must not exceed {TUNING_SEQUENCE_LENGTHS[-1]}"
        )
    for length in lengths:
        for head_dim in head_dims:
            for occupancy in batch_heads:
                for dtype_name in dtypes:
                    for relative_mode in relative_modes:
                        has_c2p, has_p2c = _relative_flags(relative_mode)
                        precisions = ("strict", "fast") if dtype_name == "float32" else ("strict",)
                        for precision in precisions:
                            yield TuningCase(
                                sequence_length=length,
                                head_dim=head_dim,
                                batch_heads=occupancy,
                                dtype=dtype_name,
                                has_c2p=has_c2p,
                                has_p2c=has_p2c,
                                fp32_precision=precision,
                            )


def _load_candidates(path: Path | None) -> tuple[KernelConfig, ...]:
    if path is None:
        return DEFAULT_KERNEL_CONFIGS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read candidate file {path}: {error}") from error
    if not isinstance(payload, list):
        raise TypeError("candidate file must contain a JSON array")
    configs = tuple(KernelConfig.from_dict(item) for item in payload)
    if not configs or len(configs) != len(set(configs)):
        raise ValueError("candidate file must contain unique configurations")
    return configs


def _make_inputs(
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
    relative_shape = (batch_size, num_heads, length, active_slots)
    c2p = torch.randn(relative_shape, device=device, dtype=dtype) if case.has_c2p else query
    p2c = torch.randn(relative_shape, device=device, dtype=dtype) if case.has_p2c else key
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
        active_slots,
        score_scale * 1.4426950408889634,
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
    )
    return args, workload


def _reference(arguments: tuple[object, ...]) -> torch.Tensor:
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
        _active_slots,
        score_scale_log2,
        has_c2p,
        has_p2c,
        _is_bf16,
        _is_fp32,
        _strict_fp32,
    ) = arguments
    score_scale = float(score_scale_log2) / 1.4426950408889634
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    positions = torch.arange(sequence_length, device=query.device)
    local = delta_to_local[positions[:, None] - positions[None, :] + sequence_length - 1].long()
    if has_c2p:
        scores += torch.gather(
            c2p.float(),
            -1,
            local.expand(query.size(0), num_heads, -1, -1),
        )
    if has_p2c:
        p2c_by_key = p2c.float().unsqueeze(-3).expand(-1, -1, sequence_length, -1, -1)
        scores += torch.gather(
            p2c_by_key,
            -1,
            local[None, None, :, :, None].expand(query.size(0), num_heads, -1, -1, -1),
        ).squeeze(-1)
    scores *= score_scale
    pair_mask = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
    scores = scores.masked_fill(~pair_mask, float("-inf"))
    padded_queries = ~attention_mask[:, None, :, None]
    scores = torch.where(padded_queries, torch.zeros_like(scores), scores)
    output = torch.matmul(torch.softmax(scores, dim=-1), value.float()).to(query.dtype)
    return output.transpose(1, 2).reshape(query.size(0), sequence_length, -1)


def _run_config(
    arguments: tuple[object, ...],
    config: KernelConfig,
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
    output = kernel._deberta_attention_configured_op(*configured_arguments)
    torch.cuda.synchronize()
    for _ in range(warmup):
        output = kernel._deberta_attention_configured_op(*configured_arguments)
    torch.cuda.synchronize()
    timings = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = kernel._deberta_attention_configured_op(*configured_arguments)
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
        _validate_output(actual, expected, strict_fp32=bool(arguments[15]))


def _environment(seed: int) -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "triton": getattr(kernel.triton, "__version__", "unknown"),
        "cuda": str(torch.version.cuda),
        "driver": str(torch.cuda.driver_version())
        if hasattr(torch.cuda, "driver_version")
        else "unknown",
        "seed": str(seed),
    }


def run(args: argparse.Namespace) -> None:
    if kernel.triton is None or not torch.cuda.is_available():
        raise RuntimeError("the tuning command requires CUDA and Triton")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hardware = HardwareSpec.current(device)
    candidates = _load_candidates(args.candidates)
    output_path = args.output.expanduser().resolve()
    profile: KernelProfile | None = load_profile(output_path) if output_path.exists() else None
    if profile is not None and not profile.hardware.matches(hardware):
        raise ValueError("the output profile belongs to a different GPU model")
    completed = set() if profile is None else {entry.workload for entry in profile.entries}
    cases = list(_cases(args))
    if not torch.cuda.is_bf16_supported():
        skipped = sum(case.dtype == "bfloat16" for case in cases)
        cases = [case for case in cases if case.dtype != "bfloat16"]
        if skipped:
            print(f"Skipping {skipped} BF16 workloads unsupported by this GPU")
    print(f"Tuning {len(cases)} workloads on {hardware.name}; {len(candidates)} candidates")
    for index, case in enumerate(cases, start=1):
        arguments, workload = _make_inputs(
            case,
            device,
            position_buckets=args.position_buckets,
            max_relative_positions=args.max_relative_positions,
            position_embedding_size=args.position_embedding_size,
        )
        if workload in completed and not args.retest:
            print(f"[{index}/{len(cases)}] cached {workload}")
            continue
        expected = _reference(arguments)
        winners: list[tuple[float, KernelConfig]] = []
        for config in candidates:
            try:
                latency, actual = _run_config(
                    arguments,
                    config,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                _validate_output(actual, expected, strict_fp32=bool(arguments[15]))
                _validate_mask_patterns(arguments, config)
                winners.append((latency, config))
            # Backends report compile/resource failures through several exception
            # families, so one bad candidate must not abort the complete run.
            except Exception as error:  # noqa: BLE001
                print(f"  rejected {config}: {type(error).__name__}: {error}")
                torch.cuda.empty_cache()
        if not winners:
            raise RuntimeError(f"no correct configuration survived for {workload}")
        best_latency = min(latency for latency, _config in winners)
        near_ties = [
            result for result in winners if result[0] <= best_latency * (1 + args.tie_margin)
        ]
        latency, winner = min(near_ties, key=lambda result: candidates.index(result[1]))
        entry = ProfileEntry(workload=workload, config=winner, latency_ms=latency)
        profile = merge_profile_entry(
            profile,
            hardware=hardware,
            entry=entry,
            environment=_environment(args.seed),
        )
        save_profile(profile, output_path)
        completed.add(workload)
        print(f"[{index}/{len(cases)}] {latency:.4f} ms {winner} {workload}")
    print(f"Saved {len(profile.entries) if profile else 0} entries to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="standard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", type=_csv_ints)
    parser.add_argument("--head-dims", type=_csv_ints)
    parser.add_argument("--batch-heads", type=_csv_ints)
    parser.add_argument("--dtypes", type=_csv_strings)
    parser.add_argument("--relative-modes", type=_csv_strings)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
