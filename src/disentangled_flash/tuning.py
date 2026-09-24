"""Kernel tuning policy, profile validation, and profile persistence."""

from __future__ import annotations

import json
import math
import os
import platform
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import torch

PROFILE_FORMAT_VERSION = 3
KERNEL_PROFILE_VERSION = "deberta-attention-forward-runtime-length-v3"
TuningMode = Literal["auto", "autotune", "profile_only", "fixed"]
KernelLayout = Literal["padded", "packed"]
TUNING_SEQUENCE_LENGTHS = (64, 128, 384, 512, 768, 1024, 2048, 4096, 8192)
TUNING_BATCH_HEADS = (8, 32)


def tuning_sequence_length(sequence_length: int) -> int:
    """Map an exact runtime length to the finite profiled kernel family."""

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    for representative in TUNING_SEQUENCE_LENGTHS:
        if sequence_length <= representative:
            return representative
    # Keep the profile family bounded. The 8192 schedule remains a valid
    # conservative dispatch choice for longer inputs, even though the kernel
    # still receives and masks the exact runtime length.
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


DEFAULT_KERNEL_CONFIGS = (
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
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.fp32_precision not in {"strict", "fast"}:
            raise ValueError("fp32_precision must be strict or fast")
        if self.layout not in {"padded", "packed"}:
            raise ValueError("layout must be padded or packed")
        if not isinstance(self.uses_padding_mask, bool):
            raise TypeError("uses_padding_mask must be a boolean")
        if self.layout == "packed" and self.uses_padding_mask:
            raise ValueError("packed workloads cannot use a padding mask")

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
        _require_exact_keys(data, fields, set(), "workload key")
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


@dataclass(frozen=True)
class KernelProfile:
    hardware: HardwareSpec
    compiler: CompilerSpec
    entries: tuple[ProfileEntry, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)
    format_version: int = PROFILE_FORMAT_VERSION
    kernel_version: str = KERNEL_PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.format_version != PROFILE_FORMAT_VERSION:
            raise ValueError(f"unsupported profile format version: {self.format_version}")
        if self.kernel_version != KERNEL_PROFILE_VERSION:
            raise ValueError(f"incompatible kernel profile version: {self.kernel_version}")
        keys = [entry.workload for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("profile contains duplicate workload entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "kernel_version": self.kernel_version,
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
        if format_version != PROFILE_FORMAT_VERSION:
            if format_version == 2:
                raise ValueError(
                    "profile format 2 lacks compiler and kernel-layout compatibility; "
                    "regenerate it with the current tuner"
                )
            raise ValueError(f"unsupported profile format version: {format_version}")
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
        return cls(
            format_version=data["format_version"],
            kernel_version=data["kernel_version"],
            hardware=HardwareSpec.from_dict(data["hardware"]),
            compiler=CompilerSpec.from_dict(data["compiler"]),
            provenance=data["provenance"],
            entries=tuple(ProfileEntry.from_dict(entry) for entry in data["entries"]),
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
    """Policy controlling saved profiles and Triton autotuning."""

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
        if self.mode not in {"auto", "autotune", "profile_only", "fixed"}:
            raise ValueError("tuning mode must be auto, autotune, profile_only, or fixed")
        if self.mode == "fixed" and self.fixed_config is None:
            raise ValueError("fixed tuning mode requires fixed_config")
        if self.mode != "fixed" and self.fixed_config is not None:
            raise ValueError("fixed_config is only valid in fixed tuning mode")
        if self.mode in {"fixed", "profile_only"} and self.candidates is not None:
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
            if not profile.hardware.matches(hardware) or profile.compiler != compiler:
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
        compiler_matches = [profile for profile in hardware_matches if profile.compiler == compiler]
        if not compiler_matches:
            details = "; ".join(hardware_matches[0].compiler.mismatches(compiler))
            return f"profile compiler is incompatible ({details})"
        return f"no validated profile entry matches workload {workload}"


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
    )


__all__ = [
    "DEFAULT_KERNEL_CONFIGS",
    "KERNEL_PROFILE_VERSION",
    "PROFILE_FORMAT_VERSION",
    "TUNING_BATCH_HEADS",
    "TUNING_SEQUENCE_LENGTHS",
    "CompilerSpec",
    "HardwareSpec",
    "KernelConfig",
    "KernelProfile",
    "KernelTuningOptions",
    "ProfileEntry",
    "ProfileRegistry",
    "WorkloadKey",
    "current_provenance",
    "default_user_profile_directory",
    "load_bundled_profiles",
    "load_profile",
    "merge_profile_entry",
    "save_profile",
    "tuning_batch_heads",
    "tuning_sequence_length",
]
