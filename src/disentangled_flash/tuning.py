"""Kernel tuning policy, profile validation, and profile persistence."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import torch

PROFILE_FORMAT_VERSION = 4
KERNEL_PROFILE_VERSION = "deberta-attention-launch-schema-v1"
TuningMode = Literal["auto", "heuristic", "autotune", "profile_only", "fixed"]
KernelLayout = Literal["padded", "packed"]
KernelPhase = Literal["inference", "training_forward", "backward_dq", "backward_dkv"]
ProfileCompatibility = Literal["exact", "retargetable", "incompatible"]
TUNING_SEQUENCE_LENGTHS = (64, 128, 384, 512, 768, 1024, 2048, 4096, 8192)
TUNING_BATCH_HEADS = (8, 32)


def _canonical_code(node: Any) -> str:
    # Unlike ast.dump, skip empty fields so new Python versions hash the same.
    if isinstance(node, ast.AST):
        fields = ",".join(
            f"{name}={_canonical_code(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and value != []
        )
        return f"{type(node).__name__}({fields})"
    if isinstance(node, list):
        return "[" + ",".join(_canonical_code(item) for item in node) + "]"
    return repr(node)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


class _DropDispatchCode(ast.NodeTransformer):
    # Module classes only choose configs, so they don't affect saved winners.
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None


def _launch_code(source: str) -> ast.AST:
    return _DropDispatchCode().visit(_strip_docstrings(ast.parse(source)))


def _kernel_source_digest() -> str:
    """Fingerprint the kernel and launch code, ignoring comments, imports and classes."""

    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in (Path("kernel.py"), Path("training") / "_kernels.py"):
        tree = _launch_code((package / relative).read_text(encoding="utf-8"))
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canonical_code(tree).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


KERNEL_SOURCE_DIGEST = _kernel_source_digest()


def tuning_dtype(dtype: str) -> str:
    """Map a dtype to its tuning family: FP16 and BF16 share "half", like FlashAttention."""

    if dtype in {"float16", "bfloat16", "half"}:
        return "half"
    if dtype == "float32":
        return "float32"
    raise ValueError("dtype must be float16, bfloat16, or float32")


def tuning_sequence_length(sequence_length: int) -> int:
    """Map an exact runtime length to the finite profiled kernel family."""

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    for representative in TUNING_SEQUENCE_LENGTHS:
        if sequence_length <= representative:
            return representative
    # Longer inputs reuse the 8192 schedule.
    return TUNING_SEQUENCE_LENGTHS[-1]


def tuning_batch_heads(batch_heads: int) -> int:
    """Map exact batch/head occupancy to a small launch-occupancy family."""

    if isinstance(batch_heads, bool) or not isinstance(batch_heads, int):
        raise TypeError("batch_heads must be an integer")
    if batch_heads <= 0:
        raise ValueError("batch_heads must be positive")
    for representative in TUNING_BATCH_HEADS:
        if batch_heads <= representative:
            return representative
    return TUNING_BATCH_HEADS[-1]


def _require_exact_keys(
    data: Mapping[str, Any], required: set[str], optional: set[str], context: str
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


@dataclass(frozen=True, order=True)
class KernelConfig:
    """One resource-safe Triton launch configuration."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int = 1

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.block_m, self.block_n, self.num_warps, self.num_stages)
        ):
            raise TypeError("kernel configuration fields must be integers")
        if self.block_m not in {16, 32, 64, 128}:
            raise ValueError("block_m must be one of 16, 32, 64, or 128")
        if self.block_n not in {16, 32, 64, 128}:
            raise ValueError("block_n must be one of 16, 32, 64, or 128")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("num_warps must be one of 1, 2, 4, or 8")
        if not 1 <= self.num_stages <= 5:
            raise ValueError("num_stages must be between 1 and 5")

    def to_dict(self) -> dict[str, int]:
        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelConfig:
        _require_exact_keys(
            data,
            {"block_m", "block_n", "num_warps", "num_stages"},
            set(),
            "kernel config",
        )
        if any(isinstance(data[name], bool) or not isinstance(data[name], int) for name in data):
            raise TypeError("kernel config fields must be integers")
        return cls(**data)


BASE_FORWARD_KERNEL_CONFIGS = (
    KernelConfig(16, 16, 2),
    KernelConfig(16, 32, 2),
    KernelConfig(32, 32, 2),
    KernelConfig(32, 32, 4),
    KernelConfig(32, 64, 4),
    KernelConfig(64, 32, 4),
    KernelConfig(64, 64, 4),
    KernelConfig(64, 128, 4),
    KernelConfig(128, 64, 4),
)

# Deeper pipelines are only tried on the best one-stage shapes.
DEFAULT_KERNEL_CONFIGS = BASE_FORWARD_KERNEL_CONFIGS + (
    KernelConfig(32, 32, 2, 2),
    KernelConfig(32, 32, 4, 2),
    KernelConfig(32, 64, 4, 2),
    KernelConfig(64, 32, 4, 2),
    KernelConfig(64, 64, 4, 2),
    KernelConfig(32, 64, 4, 3),
    KernelConfig(64, 32, 4, 3),
)

# Backward holds more live state, so its search is smaller. 8 warps help at
# head dim 128.
BASE_BACKWARD_KERNEL_CONFIGS = (
    KernelConfig(16, 16, 4),
    KernelConfig(32, 32, 4),
    KernelConfig(32, 64, 4),
    KernelConfig(64, 32, 4),
    KernelConfig(64, 64, 4),
    KernelConfig(32, 64, 8),
    KernelConfig(64, 32, 8),
)

DEFAULT_DQ_KERNEL_CONFIGS = BASE_BACKWARD_KERNEL_CONFIGS + (
    KernelConfig(32, 32, 4, 2),
    KernelConfig(32, 64, 4, 2),
    KernelConfig(64, 32, 4, 2),
    KernelConfig(32, 64, 8, 2),
)

# dK/dV iterates over keys, so N-oriented tiles come first on ties.
DEFAULT_DKV_KERNEL_CONFIGS = (
    KernelConfig(16, 16, 4),
    KernelConfig(32, 32, 4),
    KernelConfig(64, 32, 4),
    KernelConfig(32, 64, 4),
    KernelConfig(64, 64, 4),
    KernelConfig(64, 32, 8),
    KernelConfig(32, 64, 8),
    KernelConfig(32, 32, 4, 2),
    KernelConfig(64, 32, 4, 2),
    KernelConfig(32, 64, 4, 2),
    KernelConfig(64, 32, 8, 2),
)

# Kept for backwards compatibility.
DEFAULT_BACKWARD_KERNEL_CONFIGS = tuple(
    dict.fromkeys(DEFAULT_DQ_KERNEL_CONFIGS + DEFAULT_DKV_KERNEL_CONFIGS)
)


# FP32 tiles use twice the registers, so FP32 only searches one-stage schedules
# with small tiles. Same limits as the runtime autotune pruning.
FP32_MAX_FORWARD_TILE = 64 * 64
FP32_MAX_BACKWARD_TILE = 2048


def conservative_candidates(
    configs: tuple[KernelConfig, ...],
    *,
    dtype: str,
    phase: str,
) -> tuple[KernelConfig, ...]:
    """Restrict FP32 searches to resource-safe schedules; other dtypes pass through."""

    if tuning_dtype(dtype) != "float32":
        return configs
    limit = (
        FP32_MAX_FORWARD_TILE
        if phase in {"inference", "training_forward"}
        else FP32_MAX_BACKWARD_TILE
    )
    kept = tuple(
        config
        for config in configs
        if config.num_stages == 1 and config.block_m * config.block_n <= limit
    )
    return kept or (KernelConfig(32, 32, 4),)


FORWARD_PHASES = frozenset({"inference", "training_forward"})
REGISTER_BUDGET = 160


@dataclass(frozen=True)
class DeviceResources:
    """What the heuristic needs to know about a GPU, read from the device."""

    shared_memory_per_block: int
    multiprocessor_count: int
    compute_capability: tuple[int, int] = (8, 0)

    @property
    def pipelines_loads(self) -> bool:
        return self.compute_capability >= (8, 0)

    @classmethod
    def current(cls, device: torch.device | str | int | None = None) -> DeviceResources:
        return _device_resources(_device_index(device))


def _device_index(device: torch.device | str | int | None) -> int:
    if isinstance(device, int):
        return device
    resolved = torch.device(device or "cuda")
    return resolved.index if resolved.index is not None else torch.cuda.current_device()


@cache
def _device_resources(index: int) -> DeviceResources:
    properties = torch.cuda.get_device_properties(index)
    shared_memory = getattr(properties, "shared_memory_per_block_optin", 0)
    if not shared_memory:
        shared_memory = getattr(properties, "shared_memory_per_block", 48 * 1024)
    return DeviceResources(
        shared_memory_per_block=int(shared_memory),
        multiprocessor_count=int(properties.multi_processor_count),
        compute_capability=(properties.major, properties.minor),
    )


def _element_size(dtype: str) -> int:
    return 4 if tuning_dtype(dtype) == "float32" else 2


def estimate_shared_memory(
    config: KernelConfig,
    *,
    phase: str,
    head_dim: int,
    dtype: str,
) -> int:
    """Estimate shared memory in bytes; streamed tiles are buffered once per stage."""

    m, n, d, stages = config.block_m, config.block_n, head_dim, config.num_stages
    if phase in FORWARD_PHASES:
        elements = m * d + m * n + stages * 2 * n * d
    elif phase == "backward_dq":
        elements = 2 * m * d + m * n + stages * 2 * n * d
    else:
        elements = 2 * n * d + 2 * m * n + stages * 2 * m * d
    return elements * _element_size(dtype)


def estimate_registers(config: KernelConfig, *, phase: str, head_dim: int) -> float:
    """Estimate FP32 values held per thread."""

    m, n, d = config.block_m, config.block_n, head_dim
    if phase in FORWARD_PHASES:
        values = m * d + 2 * m * n
    elif phase == "backward_dq":
        values = m * d + 4 * m * n
    else:
        values = 2 * n * d + 4 * m * n
    return values / (32 * config.num_warps)


def fits_device(
    config: KernelConfig,
    *,
    phase: str,
    head_dim: int,
    dtype: str,
    resources: DeviceResources,
) -> bool:
    budget = int(resources.shared_memory_per_block * 0.9)
    return estimate_shared_memory(config, phase=phase, head_dim=head_dim, dtype=dtype) <= budget


def hardware_safe_candidates(
    configs: tuple[KernelConfig, ...],
    *,
    phase: str,
    head_dim: int,
    dtype: str,
    resources: DeviceResources,
) -> tuple[KernelConfig, ...]:
    """Drop configs that won't fit on this GPU."""

    kept = tuple(
        config
        for config in conservative_candidates(configs, dtype=dtype, phase=phase)
        if fits_device(config, phase=phase, head_dim=head_dim, dtype=dtype, resources=resources)
    )
    return kept or (KernelConfig(16, 16, 4),)


# Always fits: about 35 KB at worst (FP32, head dim 128, dK/dV), under the
# 48 KB every CUDA GPU allows per block.
HEURISTIC_FLOOR = KernelConfig(16, 16, 4)


def _base_heuristic(phase: str, sequence_length: int, dtype: str) -> KernelConfig:
    fp32 = tuning_dtype(dtype) == "float32"
    if phase == "inference" and sequence_length <= 128:
        return KernelConfig(32, 64, 4)
    if phase in FORWARD_PHASES:
        return KernelConfig(64, 64, 4)
    if phase == "backward_dq":
        if fp32:
            return KernelConfig(32, 32, 4)
        if sequence_length <= 128:
            return KernelConfig(64, 32, 4)
        if sequence_length <= 4096:
            return KernelConfig(32, 32, 4)
        return KernelConfig(16, 16, 4)
    return KernelConfig(16, 16, 4)


def _shrink(config: KernelConfig) -> KernelConfig:
    if config.num_stages > 1:
        return replace(config, num_stages=config.num_stages - 1)
    if config.block_m >= config.block_n and config.block_m > 16:
        return replace(config, block_m=config.block_m // 2)
    if config.block_n > 16:
        return replace(config, block_n=config.block_n // 2)
    return config


@lru_cache(maxsize=4096)
def heuristic_config(
    *,
    phase: str,
    sequence_length: int,
    head_dim: int,
    dtype: str,
    batch_heads: int | None,
    resources: DeviceResources,
) -> KernelConfig:
    """Pick a config without benchmarking, from the workload and the GPU's limits.

    batch_heads=None (a symbolic batch under torch.compile) skips the SM split.
    """

    length = tuning_sequence_length(sequence_length)
    config = _base_heuristic(phase, length, dtype)
    if head_dim >= 128 and config.block_m * config.block_n > 256:
        config = _shrink(config)
    if (
        phase in FORWARD_PHASES
        and tuning_dtype(dtype) == "half"
        and resources.pipelines_loads
        and (length > 128 or phase == "training_forward")
    ):
        config = replace(config, num_stages=2)

    while config != _shrink(config) and not (
        fits_device(config, phase=phase, head_dim=head_dim, dtype=dtype, resources=resources)
        and estimate_registers(config, phase=phase, head_dim=head_dim) <= REGISTER_BUDGET
    ):
        config = _shrink(config)

    parallel = "block_n" if phase == "backward_dkv" else "block_m"
    while batch_heads is not None and getattr(config, parallel) > 16:
        programs = -(-sequence_length // getattr(config, parallel)) * batch_heads
        if programs >= resources.multiprocessor_count:
            break
        config = replace(config, **{parallel: getattr(config, parallel) // 2})
    return config


def search_neighborhood(
    center: KernelConfig,
    candidates: tuple[KernelConfig, ...],
    *,
    size: int = 4,
) -> tuple[KernelConfig, ...]:
    """Return the center config plus its closest candidates."""

    def distance(config: KernelConfig) -> float:
        return (
            abs(math.log2(config.block_m / center.block_m))
            + abs(math.log2(config.block_n / center.block_n))
            + abs(math.log2(config.num_warps / center.num_warps))
            + abs(config.num_stages - center.num_stages)
        )

    others = sorted(
        (config for config in candidates if config != center),
        key=lambda config: (distance(config), candidates.index(config)),
    )
    return (center, *others[: size - 1])


@dataclass(frozen=True, order=True)
class WorkloadKey:
    """Finite workload family used to select a saved kernel schedule."""

    sequence_length: int
    head_dim: int
    batch_heads: int
    active_slots: int
    dtype: str
    has_c2p: bool
    has_p2c: bool
    fp32_precision: str
    layout: KernelLayout = "padded"
    uses_padding_mask: bool = True
    phase: KernelPhase = "inference"
    has_dropout: bool = False

    def __post_init__(self) -> None:
        for name in ("sequence_length", "head_dim", "batch_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "sequence_length", tuning_sequence_length(self.sequence_length))
        object.__setattr__(self, "batch_heads", tuning_batch_heads(self.batch_heads))
        if isinstance(self.active_slots, bool) or not isinstance(self.active_slots, int):
            raise TypeError("active_slots must be an integer")
        if self.active_slots < 0:
            raise ValueError("active_slots must be non-negative")
        if self.active_slots:
            object.__setattr__(
                self,
                "active_slots",
                tuning_sequence_length(self.active_slots),
            )
        if not isinstance(self.has_c2p, bool) or not isinstance(self.has_p2c, bool):
            raise TypeError("has_c2p and has_p2c must be booleans")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        object.__setattr__(self, "dtype", tuning_dtype(self.dtype))
        if self.fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be strict or fast")
        if self.layout not in {"padded", "packed"}:
            raise ValueError("layout must be padded or packed")
        if not isinstance(self.uses_padding_mask, bool):
            raise TypeError("uses_padding_mask must be a boolean")
        if self.layout == "packed" and self.uses_padding_mask:
            raise ValueError("packed workloads cannot use a padding mask")
        if self.phase not in {
            "inference",
            "training_forward",
            "backward_dq",
            "backward_dkv",
        }:
            raise ValueError(
                "phase must be inference, training_forward, backward_dq, or backward_dkv"
            )
        if not isinstance(self.has_dropout, bool):
            raise TypeError("has_dropout must be a boolean")
        if self.has_dropout and self.phase == "inference":
            raise ValueError("inference workloads cannot use attention dropout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_regime": self.sequence_length,
            "head_dim": self.head_dim,
            "occupancy_regime": self.batch_heads,
            "slot_regime": self.active_slots,
            "dtype": self.dtype,
            "has_c2p": self.has_c2p,
            "has_p2c": self.has_p2c,
            "fp32_precision": self.fp32_precision,
            "layout": self.layout,
            "uses_padding_mask": self.uses_padding_mask,
            "phase": self.phase,
            "has_dropout": self.has_dropout,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkloadKey:
        fields = {
            "length_regime",
            "head_dim",
            "occupancy_regime",
            "slot_regime",
            "dtype",
            "has_c2p",
            "has_p2c",
            "fp32_precision",
            "layout",
            "uses_padding_mask",
        }
        # Older profiles may lack phase and has_dropout.
        _require_exact_keys(data, fields, {"phase", "has_dropout"}, "workload key")
        integer_fields = {"length_regime", "head_dim", "occupancy_regime", "slot_regime"}
        if any(
            isinstance(data[name], bool) or not isinstance(data[name], int)
            for name in integer_fields
        ):
            raise TypeError("workload dimensions must be integers")
        if not isinstance(data["has_c2p"], bool) or not isinstance(data["has_p2c"], bool):
            raise TypeError("workload attention-mode fields must be booleans")
        if not isinstance(data["dtype"], str) or not isinstance(data["fp32_precision"], str):
            raise TypeError("workload dtype and fp32_precision must be strings")
        if not isinstance(data["layout"], str):
            raise TypeError("workload layout must be a string")
        if not isinstance(data["uses_padding_mask"], bool):
            raise TypeError("workload uses_padding_mask must be a boolean")
        if not isinstance(data.get("has_dropout", False), bool):
            raise TypeError("workload has_dropout must be a boolean")
        return cls(
            sequence_length=data["length_regime"],
            head_dim=data["head_dim"],
            batch_heads=data["occupancy_regime"],
            active_slots=data["slot_regime"],
            dtype=data["dtype"],
            has_c2p=data["has_c2p"],
            has_p2c=data["has_p2c"],
            fp32_precision=data["fp32_precision"],
            layout=data["layout"],
            uses_padding_mask=data["uses_padding_mask"],
            phase=data.get("phase", "inference"),
            has_dropout=data.get("has_dropout", False),
        )


def _current_triton_key() -> str:
    """Return the compiler identity used by Inductor's own cache when available."""

    try:
        from torch._inductor.runtime.triton_compat import triton_key

        key = triton_key()
        if key:
            return str(key)
    except (ImportError, RuntimeError):
        pass
    try:
        import triton

        return f"version:{triton.__version__}"
    except ImportError:
        return "unavailable"


@dataclass(frozen=True)
class CompilerSpec:
    """Compiler identity that makes a measured launch configuration reusable."""

    torch: str
    triton_key: str
    cuda_runtime: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in self.to_dict().values()):
            raise ValueError("compiler identity fields must be non-empty strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "torch": self.torch,
            "triton_key": self.triton_key,
            "cuda_runtime": self.cuda_runtime,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompilerSpec:
        _require_exact_keys(data, {"torch", "triton_key", "cuda_runtime"}, set(), "compiler")
        if not all(isinstance(value, str) for value in data.values()):
            raise TypeError("compiler identity fields must be strings")
        return cls(**data)

    @classmethod
    def current(cls) -> CompilerSpec:
        return cls(
            torch=torch.__version__,
            triton_key=_current_triton_key(),
            cuda_runtime=str(torch.version.cuda),
        )

    def mismatches(self, other: CompilerSpec) -> tuple[str, ...]:
        return tuple(
            f"{name}: profile={getattr(self, name)!r}, current={getattr(other, name)!r}"
            for name in ("torch", "triton_key", "cuda_runtime")
            if getattr(self, name) != getattr(other, name)
        )


def current_provenance(seed: int | None = None) -> dict[str, str]:
    """Return reproducibility metadata that does not control profile matching."""

    provenance = {
        "driver": (
            str(torch.cuda.driver_version())
            if torch.cuda.is_available() and hasattr(torch.cuda, "driver_version")
            else "unknown"
        ),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    if seed is not None:
        provenance["seed"] = str(seed)
    return provenance


@dataclass(frozen=True)
class HardwareSpec:
    """Portable identity for one GPU model and architecture."""

    backend: str
    name: str
    compute_capability: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not isinstance(self.name, str):
            raise TypeError("hardware backend and name must be strings")
        if not isinstance(self.compute_capability, tuple):
            object.__setattr__(self, "compute_capability", tuple(self.compute_capability))
        if self.backend != "cuda":
            raise ValueError("only CUDA tuning profiles are currently supported")
        if not self.name.strip():
            raise ValueError("hardware name must not be empty")
        if (
            len(self.compute_capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.compute_capability
            )
            or min(self.compute_capability) < 0
        ):
            raise ValueError("compute_capability must contain two non-negative integers")

    @property
    def normalized_name(self) -> str:
        return " ".join(self.name.casefold().split())

    def matches(self, other: HardwareSpec) -> bool:
        return (
            self.backend == other.backend
            and self.compute_capability == other.compute_capability
            and self.normalized_name == other.normalized_name
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "name": self.name,
            "compute_capability": list(self.compute_capability),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareSpec:
        _require_exact_keys(
            data,
            {"backend", "name", "compute_capability"},
            set(),
            "hardware",
        )
        capability = data["compute_capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        ):
            raise ValueError("compute_capability must be a two-integer JSON array")
        if not isinstance(data["backend"], str) or not isinstance(data["name"], str):
            raise TypeError("hardware backend and name must be strings")
        return cls(
            backend=data["backend"],
            name=data["name"],
            compute_capability=(capability[0], capability[1]),
        )

    @classmethod
    def current(cls, device: torch.device | str | int | None = None) -> HardwareSpec:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to identify tuning-profile hardware")
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device or "cuda")
        )
        if resolved.type != "cuda":
            raise ValueError("tuning-profile hardware must be a CUDA device")
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        return cls(
            backend="cuda",
            name=torch.cuda.get_device_name(index),
            compute_capability=tuple(torch.cuda.get_device_capability(index)),
        )


@dataclass(frozen=True)
class ProfileEntry:
    workload: WorkloadKey
    config: KernelConfig
    latency_ms: float
    validated: bool = True
    validation: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload.to_dict(),
            "config": self.config.to_dict(),
            "latency_ms": self.latency_ms,
            "validated": self.validated,
            "validation": dict(self.validation),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProfileEntry:
        _require_exact_keys(
            data,
            {"workload", "config", "latency_ms", "validated", "validation"},
            set(),
            "profile entry",
        )
        if not isinstance(data["validated"], bool):
            raise TypeError("profile entry validated must be a boolean")
        if not isinstance(data["validation"], dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in data["validation"].items()
        ):
            raise ValueError("profile validation metadata must map strings to strings")
        latency = data["latency_ms"]
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise TypeError("profile entry latency_ms must be numeric")
        return cls(
            workload=WorkloadKey.from_dict(data["workload"]),
            config=KernelConfig.from_dict(data["config"]),
            latency_ms=float(latency),
            validated=data["validated"],
            validation=data["validation"],
        )


def _load_entries(items: Any) -> tuple[ProfileEntry, ...]:
    """Parse saved entries. Old separate FP16/BF16 entries collapse to the first one."""

    if not isinstance(items, list):
        raise TypeError("profile entries must be a JSON array")
    entries: dict[WorkloadKey, ProfileEntry] = {}
    saved_dtypes: dict[WorkloadKey, set[str]] = {}
    for item in items:
        entry = ProfileEntry.from_dict(item)
        saved_dtype = item["workload"]["dtype"]
        seen = saved_dtypes.setdefault(entry.workload, set())
        if saved_dtype in seen or (seen and not seen | {saved_dtype} <= {"float16", "bfloat16"}):
            raise ValueError("profile contains duplicate workload entries")
        seen.add(saved_dtype)
        entries.setdefault(entry.workload, entry)
    return tuple(entries.values())


@dataclass(frozen=True)
class KernelProfile:
    hardware: HardwareSpec
    compiler: CompilerSpec
    entries: tuple[ProfileEntry, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)
    format_version: int = PROFILE_FORMAT_VERSION
    kernel_version: str = KERNEL_PROFILE_VERSION
    kernel_digest: str = KERNEL_SOURCE_DIGEST

    def __post_init__(self) -> None:
        if self.format_version != PROFILE_FORMAT_VERSION:
            raise ValueError(f"unsupported profile format version: {self.format_version}")
        if not isinstance(self.kernel_version, str) or not self.kernel_version:
            raise ValueError("kernel profile version must be a non-empty string")
        if not isinstance(self.kernel_digest, str) or not self.kernel_digest:
            raise ValueError("kernel digest must be a non-empty string")
        keys = [entry.workload for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("profile contains duplicate workload entries")

    def compatibility(
        self,
        hardware: HardwareSpec,
        compiler: CompilerSpec,
        *,
        kernel_digest: str = KERNEL_SOURCE_DIGEST,
    ) -> ProfileCompatibility:
        """Classify reuse without confusing a measured compiler with a launch ABI."""

        if not self.hardware.matches(hardware) or self.kernel_version != KERNEL_PROFILE_VERSION:
            return "incompatible"
        if self.compiler == compiler and self.kernel_digest == kernel_digest:
            return "exact"
        return "retargetable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "kernel_version": self.kernel_version,
            "kernel_digest": self.kernel_digest,
            "hardware": self.hardware.to_dict(),
            "compiler": self.compiler.to_dict(),
            "provenance": dict(self.provenance),
            "entries": [
                entry.to_dict() for entry in sorted(self.entries, key=lambda item: item.workload)
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelProfile:
        format_version = data.get("format_version") if isinstance(data, Mapping) else None
        if format_version == 2:
            raise ValueError(
                "profile format 2 lacks compiler and kernel-layout compatibility; "
                "regenerate it with the current tuner"
            )
        if format_version == 3:
            _require_exact_keys(
                data,
                {
                    "format_version",
                    "kernel_version",
                    "hardware",
                    "compiler",
                    "provenance",
                    "entries",
                },
                set(),
                "profile",
            )
            provenance = dict(data["provenance"])
            provenance["migrated_from_format"] = "3"
            return cls(
                hardware=HardwareSpec.from_dict(data["hardware"]),
                compiler=CompilerSpec.from_dict(data["compiler"]),
                provenance=provenance,
                entries=_load_entries(data["entries"]),
                kernel_digest=f"legacy-v3:{data['kernel_version']}",
            )
        if format_version != PROFILE_FORMAT_VERSION:
            raise ValueError(f"unsupported profile format version: {format_version}")
        _require_exact_keys(
            data,
            {
                "format_version",
                "kernel_version",
                "kernel_digest",
                "hardware",
                "compiler",
                "provenance",
                "entries",
            },
            set(),
            "profile",
        )
        if not isinstance(data["provenance"], dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in data["provenance"].items()
        ):
            raise ValueError("profile provenance must map strings to strings")
        if not isinstance(data["entries"], list):
            raise TypeError("profile entries must be a JSON array")
        if isinstance(data["format_version"], bool) or not isinstance(data["format_version"], int):
            raise TypeError("profile format_version must be an integer")
        if not isinstance(data["kernel_version"], str):
            raise TypeError("profile kernel_version must be a string")
        if not isinstance(data["kernel_digest"], str):
            raise TypeError("profile kernel_digest must be a string")
        return cls(
            format_version=data["format_version"],
            kernel_version=data["kernel_version"],
            kernel_digest=data["kernel_digest"],
            hardware=HardwareSpec.from_dict(data["hardware"]),
            compiler=CompilerSpec.from_dict(data["compiler"]),
            provenance=data["provenance"],
            entries=_load_entries(data["entries"]),
        )


def load_profile(path: str | os.PathLike[str]) -> KernelProfile:
    profile_path = Path(path)
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load tuning profile {profile_path}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError(f"tuning profile {profile_path} must contain a JSON object")
    return KernelProfile.from_dict(data)


def save_profile(profile: KernelProfile, path: str | os.PathLike[str]) -> Path:
    """Atomically save a validated profile without touching package files."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), indent=2, sort_keys=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def default_user_profile_directory() -> Path:
    override = os.environ.get("DISENTANGLED_FLASH_PROFILE_DIR")
    if override:
        return Path(override).expanduser()
    cache_root = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return root / "disentangled_flash" / "profiles"


def _load_profile_directory(path: Path) -> list[KernelProfile]:
    if not path.is_dir():
        return []
    profiles = []
    for profile_path in sorted(path.glob("*.json")):
        try:
            profiles.append(load_profile(profile_path))
        except (TypeError, ValueError) as error:
            warnings.warn(
                f"ignoring incompatible tuning profile {profile_path}: {error}",
                stacklevel=2,
            )
    return profiles


def load_bundled_profiles() -> tuple[KernelProfile, ...]:
    loaded: list[KernelProfile] = []
    root = resources.files(__package__).joinpath("profiles")
    if not root.is_dir():
        return ()
    for item in sorted(root.iterdir(), key=lambda entry: entry.name):
        if item.name.endswith(".json"):
            with resources.as_file(item) as profile_path:
                loaded.append(load_profile(profile_path))
    return tuple(loaded)


@dataclass(frozen=True)
class KernelTuningOptions:
    """How launch configs are picked.

    auto: saved profile, else heuristic. heuristic: never profiles. autotune:
    benchmark at runtime. profile_only: fail on a miss. fixed: fixed_config.
    """

    mode: TuningMode = "auto"
    profile_paths: tuple[str | os.PathLike[str], ...] = ()
    fixed_config: KernelConfig | None = None
    candidates: tuple[KernelConfig, ...] | None = None
    use_user_profiles: bool = True
    use_bundled_profiles: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_paths", tuple(self.profile_paths))
        if self.candidates is not None:
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.mode not in {"auto", "heuristic", "autotune", "profile_only", "fixed"}:
            raise ValueError(
                "tuning mode must be auto, heuristic, autotune, profile_only, or fixed"
            )
        if self.mode == "fixed" and self.fixed_config is None:
            raise ValueError("fixed tuning mode requires fixed_config")
        if self.mode != "fixed" and self.fixed_config is not None:
            raise ValueError("fixed_config is only valid in fixed tuning mode")
        if self.mode in {"fixed", "profile_only", "heuristic"} and self.candidates is not None:
            raise ValueError("candidates are only valid in auto or autotune mode")
        if self.candidates is not None:
            if not self.candidates:
                raise ValueError("candidates must not be empty")
            if len(self.candidates) != len(set(self.candidates)):
                raise ValueError("candidates must not contain duplicates")


class ProfileRegistry:
    """Ordered collection where earlier profiles have higher priority."""

    def __init__(self, profiles: Iterable[KernelProfile] = ()) -> None:
        self.profiles = tuple(profiles)

    @classmethod
    def from_options(cls, options: KernelTuningOptions) -> ProfileRegistry:
        profiles = [load_profile(path) for path in options.profile_paths]
        if options.use_user_profiles:
            profiles.extend(_load_profile_directory(default_user_profile_directory()))
        if options.use_bundled_profiles:
            profiles.extend(load_bundled_profiles())
        return cls(profiles)

    def resolve(
        self,
        hardware: HardwareSpec,
        compiler: CompilerSpec,
        workload: WorkloadKey,
    ) -> KernelConfig | None:
        for profile in self.profiles:
            if profile.compatibility(hardware, compiler) != "exact":
                continue
            for entry in profile.entries:
                if entry.validated and entry.workload == workload:
                    return entry.config
        return None

    def explain_miss(
        self,
        hardware: HardwareSpec,
        compiler: CompilerSpec,
        workload: WorkloadKey,
    ) -> str:
        if not self.profiles:
            return "no tuning profiles are available"
        hardware_matches = [
            profile for profile in self.profiles if profile.hardware.matches(hardware)
        ]
        if not hardware_matches:
            return f"no profile matches GPU {hardware.name!r} {hardware.compute_capability}"
        exact_matches = [
            profile
            for profile in hardware_matches
            if profile.compatibility(hardware, compiler) == "exact"
        ]
        if not exact_matches:
            retargetable = [
                profile
                for profile in hardware_matches
                if profile.compatibility(hardware, compiler) == "retargetable"
            ]
            if retargetable:
                details = "; ".join(retargetable[0].compiler.mismatches(compiler))
                suffix = f" ({details})" if details else " (kernel source changed)"
                return f"profile requires local retargeting{suffix}"
            return "profile launch schema is incompatible"
        return f"no validated profile entry matches workload {workload}"


def length_family(sequence_length: Any) -> int:
    """Like tuning_sequence_length, but also works on a symbolic length under compile."""

    for representative in TUNING_SEQUENCE_LENGTHS:
        if sequence_length <= representative:
            return representative
    return TUNING_SEQUENCE_LENGTHS[-1]


def occupancy_family(batch_heads: Any) -> int:
    # A symbolic batch would add a guard and recompile per family, so under
    # compile assume the batch fills the GPU.
    if torch.compiler.is_compiling():
        return TUNING_BATCH_HEADS[-1]
    for representative in TUNING_BATCH_HEADS:
        if batch_heads <= representative:
            return representative
    return TUNING_BATCH_HEADS[-1]


_REGISTRIES: dict[int, ProfileRegistry] = {}


def register_profile_registry(registry: ProfileRegistry) -> int:
    _REGISTRIES[id(registry)] = registry
    return id(registry)


def _constant_under_compile(function: Any) -> Any:
    marker = getattr(torch.compiler, "assume_constant_result", None)
    return marker(function) if marker is not None else function


@_constant_under_compile
def resolve_launch_config(
    registry_key: int,
    mode: str,
    device_index: int,
    workload_fields: tuple[Any, ...],
    batch_heads: int | None,
) -> tuple[int, int, int, int] | None:
    """Pick a config from the profiles or the heuristic.

    Takes only constants, so torch.compile runs it once while tracing and bakes
    the result into the graph. workload_fields are WorkloadKey's fields in order.
    """

    return _resolve_launch_config(registry_key, mode, device_index, workload_fields, batch_heads)


@lru_cache(maxsize=4096)
def _resolve_launch_config(
    registry_key: int,
    mode: str,
    device_index: int,
    workload_fields: tuple[Any, ...],
    batch_heads: int | None,
) -> tuple[int, int, int, int] | None:
    workload = WorkloadKey(*workload_fields)
    registry = _REGISTRIES[registry_key]
    config = None
    if mode != "heuristic":
        config = registry.resolve(
            HardwareSpec.current(device_index), CompilerSpec.current(), workload
        )
    if config is None and mode == "profile_only":
        raise RuntimeError(
            registry.explain_miss(
                HardwareSpec.current(device_index), CompilerSpec.current(), workload
            )
        )
    if config is None:
        config = heuristic_config(
            phase=workload.phase,
            sequence_length=workload.sequence_length,
            head_dim=workload.head_dim,
            dtype=workload.dtype,
            batch_heads=batch_heads,
            resources=DeviceResources.current(device_index),
        )
    return config.block_m, config.block_n, config.num_warps, config.num_stages


def merge_profile_entry(
    profile: KernelProfile | None,
    *,
    hardware: HardwareSpec,
    compiler: CompilerSpec,
    entry: ProfileEntry,
    provenance: Mapping[str, str],
) -> KernelProfile:
    """Insert or replace one finite workload family while preserving others."""

    if profile is not None and not profile.hardware.matches(hardware):
        raise ValueError("cannot merge tuning results for different GPU models")
    if profile is not None and profile.compiler != compiler:
        details = "; ".join(profile.compiler.mismatches(compiler))
        raise ValueError(f"cannot merge tuning results from different compilers ({details})")
    existing = {} if profile is None else {item.workload: item for item in profile.entries}
    existing[entry.workload] = entry
    return KernelProfile(
        hardware=hardware,
        compiler=compiler,
        entries=tuple(existing.values()),
        provenance=dict(provenance),
        kernel_digest=KERNEL_SOURCE_DIGEST,
    )


def prepare_profile_retarget(
    profile: KernelProfile,
    *,
    hardware: HardwareSpec,
    compiler: CompilerSpec,
    provenance: Mapping[str, str],
) -> KernelProfile:
    """Preserve old winners as pending seeds for an interruptible local retarget."""

    compatibility = profile.compatibility(hardware, compiler)
    if compatibility == "incompatible":
        raise ValueError("cannot retarget a profile with different hardware or launch schema")
    if compatibility == "exact":
        # Resuming a retarget; keep the progress.
        return profile
    source_compiler = json.dumps(profile.compiler.to_dict(), sort_keys=True, separators=(",", ":"))
    pending = tuple(
        ProfileEntry(
            workload=entry.workload,
            config=entry.config,
            latency_ms=entry.latency_ms,
            validated=False,
            validation={
                **entry.validation,
                "retarget_status": "pending",
                "source_compiler": source_compiler,
                "source_kernel_digest": profile.kernel_digest,
            },
        )
        for entry in profile.entries
    )
    return KernelProfile(
        hardware=hardware,
        compiler=compiler,
        entries=pending,
        provenance=dict(provenance),
        kernel_digest=KERNEL_SOURCE_DIGEST,
    )


__all__ = [
    "BASE_BACKWARD_KERNEL_CONFIGS",
    "BASE_FORWARD_KERNEL_CONFIGS",
    "DEFAULT_BACKWARD_KERNEL_CONFIGS",
    "DEFAULT_DKV_KERNEL_CONFIGS",
    "DEFAULT_DQ_KERNEL_CONFIGS",
    "DEFAULT_KERNEL_CONFIGS",
    "FP32_MAX_BACKWARD_TILE",
    "FP32_MAX_FORWARD_TILE",
    "KERNEL_PROFILE_VERSION",
    "KERNEL_SOURCE_DIGEST",
    "PROFILE_FORMAT_VERSION",
    "TUNING_BATCH_HEADS",
    "TUNING_SEQUENCE_LENGTHS",
    "CompilerSpec",
    "HardwareSpec",
    "KernelConfig",
    "KernelProfile",
    "KernelTuningOptions",
    "ProfileCompatibility",
    "ProfileEntry",
    "ProfileRegistry",
    "WorkloadKey",
    "conservative_candidates",
    "current_provenance",
    "default_user_profile_directory",
    "load_bundled_profiles",
    "load_profile",
    "merge_profile_entry",
    "prepare_profile_retarget",
    "save_profile",
    "tuning_batch_heads",
    "tuning_sequence_length",
]
