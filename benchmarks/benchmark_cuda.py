"""Isolated CUDA benchmark for DeBERTa-v3-base encoder backends.

By default this benchmarks the reference, PyTorch, Triton, and FlashDeBERTa
implementations under the four requested execution configurations:

* FP16 eager
* FP16 ``torch.compile``
* FP32 eager
* FP32 ``torch.compile``

Each implementation/dtype/execution combination runs in a fresh process.  All
implementations receive identically seeded embedding-derived inputs, while each
measured iteration receives a different token batch and padding pattern.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from disentangled_flash._reference import (
    DebertaAttentionConfig,
    DebertaV2Encoder,
    OriginalDisentangledSelfAttention,
    _prepare_attention_mask,
)
from disentangled_flash._torch import TorchInferenceDisentangledSelfAttention
from disentangled_flash.deberta import DebertaV2InferenceEncoder
from disentangled_flash.kernel import TritonInferenceDisentangledSelfAttention
from disentangled_flash.packed import pack_padded_with_info, unpack_packed

IMPLEMENTATIONS = {
    "base": OriginalDisentangledSelfAttention,
    "original": OriginalDisentangledSelfAttention,
    "torch": TorchInferenceDisentangledSelfAttention,
    "triton": TritonInferenceDisentangledSelfAttention,
    "flashdeberta": None,
}
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

MODEL_ARCHITECTURE = "microsoft/deberta-v3-base"
RELEVANT_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TRITON_CACHE_DIR",
    "TRITON_PRINT_AUTOTUNING",
    "DISENTANGLED_FLASH_PROFILE_DIR",
    "FLASHDEBERTA_FWD_BLOCK_M",
    "FLASHDEBERTA_FWD_BLOCK_N",
    "FLASHDEBERTA_FWD_NUM_STAGES",
    "FLASHDEBERTA_FWD_NUM_WARPS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
)


def _run_metadata_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("disentangled-flash", "torch", "triton", "transformers", "flashdeberta"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _memory_details() -> dict[str, int | None]:
    total_bytes = None
    available_bytes = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_bytes = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available_bytes = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        mac_total = _run_metadata_command(["sysctl", "-n", "hw.memsize"])
        if mac_total is not None:
            total_bytes = int(mac_total)
    return {"total_bytes": total_bytes, "available_bytes": available_bytes}


def _cpu_details() -> dict[str, Any]:
    model = platform.processor() or None
    physical_cores = None
    if sys.platform.startswith("linux"):
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
            sockets: set[tuple[str, str]] = set()
            for processor in cpuinfo.split("\n\n"):
                fields = {}
                for line in processor.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        fields[key.strip()] = value.strip()
                if "physical id" in fields and "core id" in fields:
                    sockets.add((fields["physical id"], fields["core id"]))
            physical_cores = len(sockets) or None
        except OSError:
            pass
    elif sys.platform == "darwin":
        model = _run_metadata_command(["sysctl", "-n", "machdep.cpu.brand_string"]) or model
        physical = _run_metadata_command(["sysctl", "-n", "hw.physicalcpu"])
        physical_cores = int(physical) if physical is not None else None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        "model": model,
        "physical_cores": physical_cores,
        "host_logical_cores": os.cpu_count(),
        "process_affinity_logical_cores": len(affinity) if affinity is not None else None,
        "process_affinity_cpu_ids": affinity,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "architecture": platform.machine(),
    }


def _read_cgroup_value(paths: tuple[str, ...]) -> str | None:
    for path in paths:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _scheduler_details() -> dict[str, Any]:
    slurm = {
        key: os.environ[key]
        for key in RELEVANT_ENVIRONMENT_VARIABLES
        if key.startswith("SLURM_") and key in os.environ
    }
    return {
        "slurm": slurm or None,
        "cgroup": {
            "cpu_set": _read_cgroup_value(
                ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset/cpuset.cpus")
            ),
            "cpu_quota": _read_cgroup_value(
                ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            ),
            "memory_limit_bytes": _read_cgroup_value(
                ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
            ),
            "memory_current_bytes": _read_cgroup_value(
                ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
            ),
        },
    }


def _git_details() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = _run_metadata_command(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run_metadata_command(["git", "branch", "--show-current"], cwd=root)
    status = _run_metadata_command(["git", "status", "--porcelain"], cwd=root)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _nvidia_smi_details() -> dict[str, Any]:
    query = (
        "index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,"
        "power.limit,clocks.max.sm,clocks.max.memory"
    )
    output = _run_metadata_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    fields = (
        "index",
        "name",
        "uuid",
        "pci_bus_id",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "power_limit_w",
        "max_sm_clock_mhz",
        "max_memory_clock_mhz",
    )
    gpus = []
    if output:
        for line in output.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) == len(fields):
                gpus.append(dict(zip(fields, values)))
    return {
        "raw": output,
        "gpus": gpus,
        "topology": _run_metadata_command(["nvidia-smi", "topo", "-m"]),
    }


def _torch_cuda_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if not torch.cuda.is_available():
        return details
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_vram_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
        for attribute in (
            "l2_cache_size",
            "max_threads_per_multi_processor",
            "shared_memory_per_multiprocessor",
        ):
            if hasattr(properties, attribute):
                device[attribute] = getattr(properties, attribute)
        devices.append(device)
    details["devices"] = devices
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        details["selected_device_memory_at_start"] = {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
        }
    except RuntimeError:
        pass
    return details


def collect_system_details() -> dict[str, Any]:
    """Return a JSON-safe reproducibility snapshot without exposing arbitrary secrets."""

    uname = platform.uname()
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "node": uname.node,
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "platform": platform.platform(),
        },
        "cpu": _cpu_details(),
        "system_memory": _memory_details(),
        "allocation": _scheduler_details(),
        "cuda": _torch_cuda_details(),
        "nvidia_smi": _nvidia_smi_details(),
        "packages": _package_versions(),
        "git": _git_details(),
        "environment": {
            key: os.environ[key] for key in RELEVANT_ENVIRONMENT_VARIABLES if key in os.environ
        },
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def unsupported_layout_reason(implementation: str, layout: str) -> str | None:
    if layout == "packed" and implementation in {"base", "original"}:
        return "the base encoder has no external cu_seqlens packed interface"
    if layout == "padded" and implementation == "flashdeberta":
        return (
            "FlashDeBERTa automatically uses its mask-driven internal varlen path; "
            "it has no switch for a distinct dense padded FlashDeBERTa path"
        )
    return None


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def configure_fp32(precision: str) -> None:
    torch.set_float32_matmul_precision("highest" if precision == "strict" else "high")
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee" if precision == "strict" else "tf32"
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = precision == "fast"


def compile_isolated(
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compile one bucket without sharing Dynamo's per-code-object budget.

    PyTorch 2.13 added ``isolate_recompiles=True`` for this exact factory
    pattern.  Older releases use the officially documented code-object cloning
    workaround so separate length buckets cannot exhaust one another's default
    eight-entry recompile limit.
    """

    compile_kwargs: dict[str, Any] = {
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if "isolate_recompiles" in inspect.signature(torch.compile).parameters:
        compile_kwargs["isolate_recompiles"] = True
    elif isinstance(function, types.FunctionType):
        clone = types.FunctionType(
            function.__code__.replace(),
            function.__globals__,
            name=function.__name__,
            argdefs=function.__defaults__,
            closure=function.__closure__,
        )
        clone.__kwdefaults__ = function.__kwdefaults__
        function = clone
    return torch.compile(function, **compile_kwargs)


def initialize_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, (torch.nn.Linear, torch.nn.Conv1d)):
                child.weight.normal_(mean=0.0, std=0.02, generator=generator)
                if child.bias is not None:
                    child.bias.zero_()
            elif isinstance(child, torch.nn.Embedding):
                child.weight.normal_(mean=0.0, std=0.02, generator=generator)
            elif isinstance(child, torch.nn.LayerNorm):
                if child.weight is not None:
                    child.weight.fill_(1.0)
                if child.bias is not None:
                    child.bias.zero_()


def make_embedding_table(
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    table = torch.empty(vocab_size, hidden_size, dtype=torch.float32, device="cpu")
    table.normal_(mean=0.0, std=0.02, generator=generator)
    table[0].zero_()
    return table.to(device=device, dtype=dtype)


def make_inputs(
    embedding_table: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    minimum_length_fraction: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    minimum_length = max(1, int(sequence_length * minimum_length_fraction))
    for run in range(count):
        generator = torch.Generator(device="cpu").manual_seed(seed + run)
        token_ids = torch.randint(
            1,
            embedding_table.size(0),
            (batch_size, sequence_length),
            generator=generator,
            device="cpu",
        )
        lengths = torch.randint(
            minimum_length,
            sequence_length + 1,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        positions = torch.arange(sequence_length, device="cpu")[None, :]
        mask_cpu = positions < lengths[:, None]
        token_ids.masked_fill_(~mask_cpu, 0)

        token_ids = token_ids.to(device=embedding_table.device)
        # Match tokenizer output. DeBERTa's convolution computes
        # ``1 - input_mask`` and therefore requires an integer mask.
        mask = mask_cpu.to(device=embedding_table.device, dtype=torch.long)
        hidden_states = embedding_table[token_ids].contiguous()
        batches.append((hidden_states, mask))
    return batches


def make_models(
    scope: str,
    implementation: str,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    hidden_size: int,
    num_attention_heads: int,
    attention_head_size: int,
    fp32_precision: str,
    sequence_lengths: list[int],
    num_hidden_layers: int,
    intermediate_size: int,
    conv_kernel_size: int,
    assume_unpadded: bool,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.Tensor | None,
    float,
]:
    config = DebertaAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_head_size=attention_head_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        relative_attention=True,
        max_relative_positions=-1,
        max_position_embeddings=512,
        position_buckets=256,
        share_att_key=True,
        pos_att_type=("p2c", "c2p"),
        conv_kernel_size=conv_kernel_size,
        conv_act="gelu",
    )
    if scope == "encoder":
        reference = DebertaV2Encoder(config)
    else:
        reference = OriginalDisentangledSelfAttention(config)
    initialize_parameters(reference, seed)
    if scope == "encoder":
        if implementation in {"base", "original"}:
            target = reference
        elif implementation == "flashdeberta":
            try:
                from flashdeberta.model import FlashDebertaV2Encoder
                from transformers import DebertaV2Config
            except ImportError as exc:
                raise RuntimeError(
                    "flashdeberta benchmark requires the optional dependencies; install "
                    "them with: pip install -U flashdeberta 'transformers>=4.57'"
                ) from exc

            flash_config = DebertaV2Config(
                vocab_size=128100,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                max_position_embeddings=512,
                relative_attention=True,
                position_buckets=256,
                max_relative_positions=-1,
                pos_att_type=["p2c", "c2p"],
                share_att_key=True,
                conv_kernel_size=conv_kernel_size,
                conv_act="gelu",
                norm_rel_ebd="layer_norm",
                layer_norm_eps=1e-7,
                type_vocab_size=0,
            )
            target = FlashDebertaV2Encoder(flash_config)
            target.load_state_dict(reference.state_dict(), strict=True)
        else:
            source = DebertaV2Encoder(config)
            source.load_state_dict(reference.state_dict(), strict=True)
            target = DebertaV2InferenceEncoder(
                source,
                config,
                backend=implementation,
                fp32_precision=fp32_precision,
                assume_unpadded=(assume_unpadded if implementation == "triton" else False),
            )
    elif implementation == "triton":
        target = TritonInferenceDisentangledSelfAttention(
            config,
            fp32_precision=fp32_precision,
            assume_unpadded=assume_unpadded,
        )
        target.load_state_dict(reference.state_dict(), strict=True)
    elif implementation == "torch":
        target = TorchInferenceDisentangledSelfAttention(config)
        target.load_state_dict(reference.state_dict(), strict=True)
    elif implementation in {"base", "original"}:
        target = OriginalDisentangledSelfAttention(config)
        target.load_state_dict(reference.state_dict(), strict=True)
    else:
        raise ValueError(f"unsupported attention implementation: {implementation}")
    reference = reference.to(device=device, dtype=dtype).eval()
    target = target.to(device=device, dtype=dtype).eval()

    rel_embeddings = None
    if scope == "attention":
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        rel_embeddings = torch.empty(
            config.position_buckets * 2,
            config.hidden_size,
            dtype=torch.float32,
            device="cpu",
        )
        rel_embeddings.normal_(mean=0.0, std=0.02, generator=generator)
        rel_embeddings = rel_embeddings.to(device=device, dtype=dtype)

    preparation_ms = 0.0
    if implementation in {"torch", "triton"}:
        torch.cuda.synchronize()
        started = time.perf_counter()
        if scope == "encoder":
            target.prepare_for_inference(sequence_lengths)
        else:
            target.prepare_for_inference(rel_embeddings)
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - started) * 1000.0
    return reference, target, rel_embeddings, preparation_ms


def make_callable(
    scope: str,
    implementation: str,
    target: torch.nn.Module,
    rel_embeddings: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
    layout: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if scope == "encoder":
        if implementation in {"torch", "triton"}:
            target.activate_shape(sequence_length)

        if layout == "packed" and implementation in {"torch", "triton"}:

            def call_packed(
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                packed, cu_seqlens, packed_info = pack_padded_with_info(
                    hidden_states,
                    attention_mask,
                )
                output = target.forward_packed(
                    packed,
                    cu_seqlens,
                    packed_info.max_seqlen,
                    output_hidden_states=False,
                    return_dict=True,
                    packed_info=packed_info,
                ).last_hidden_state
                padded, _ = unpack_packed(
                    output,
                    cu_seqlens,
                    hidden_states.size(1),
                    packed_info=packed_info,
                )
                return padded

            return call_packed

        def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            output = target(
                hidden_states,
                attention_mask,
                output_hidden_states=False,
            )
            return output.last_hidden_state

        return call

    if implementation in {"base", "original"}:

        def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return target(
                hidden_states,
                _prepare_attention_mask(
                    attention_mask,
                    hidden_states.size(1),
                    hidden_states.size(1),
                ),
                rel_embeddings=rel_embeddings,
            )[0]

        return call

    plan = target.prepare_shape(sequence_length, device)

    def call(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return target.forward_prepared(hidden_states, attention_mask, plan)[0]

    return call


def measure_generated_calls(
    call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embedding_table: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    minimum_length_fraction: float,
    log_prefix: str,
) -> tuple[list[float], list[int]]:
    timings = []
    valid_tokens = []
    for index in range(count):
        hidden_states, attention_mask = make_inputs(
            embedding_table,
            batch_size,
            sequence_length,
            1,
            seed + index,
            minimum_length_fraction,
        )[0]
        print(
            f"{log_prefix} measured run {index + 1:02d}/{count:02d} "
            f"valid_tokens={int(attention_mask.sum().item())}",
            flush=True,
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call(hidden_states, attention_mask)
        end.record()
        end.synchronize()
        milliseconds = start.elapsed_time(end)
        timings.append(milliseconds)
        valid_tokens.append(int(attention_mask.sum().item()))
        print(
            f"{log_prefix} completed run {index + 1:02d}/{count:02d} latency={milliseconds:.4f} ms",
            flush=True,
        )
    return timings, valid_tokens


def compare_outputs(
    scope: str,
    reference: torch.nn.Module,
    target_call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rel_embeddings: torch.Tensor | None,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    layout: str,
) -> tuple[float, float]:
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for hidden_states, attention_mask in batches:
        if scope == "encoder":
            reference_output = reference(
                hidden_states,
                attention_mask,
                output_hidden_states=False,
            ).last_hidden_state
        else:
            reference_output = reference(
                hidden_states,
                _prepare_attention_mask(
                    attention_mask,
                    hidden_states.size(1),
                    hidden_states.size(1),
                ),
                rel_embeddings=rel_embeddings,
            )[0]
        target_output = target_call(hidden_states, attention_mask)
        difference = (reference_output.float() - target_output.float()).abs()
        if layout == "packed":
            difference = difference[attention_mask.bool()]
        maximum = max(maximum, difference.max().item())
        absolute_sum += difference.sum().item()
        element_count += difference.numel()
    return maximum, absolute_sum / element_count


def compare_generated_outputs(
    scope: str,
    reference: torch.nn.Module,
    target_call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rel_embeddings: torch.Tensor | None,
    embedding_table: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    minimum_length_fraction: float,
    layout: str,
) -> tuple[float, float]:
    maximum = 0.0
    mean_sum = 0.0
    for index in range(count):
        batch = make_inputs(
            embedding_table,
            batch_size,
            sequence_length,
            1,
            seed + index,
            minimum_length_fraction,
        )
        batch_maximum, batch_mean = compare_outputs(
            scope,
            reference,
            target_call,
            rel_embeddings,
            batch,
            layout,
        )
        maximum = max(maximum, batch_maximum)
        mean_sum += batch_mean
    return maximum, mean_sum / count


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)


def exception_result(
    *,
    status: str,
    batch_size: int,
    sequence_length: int,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "status": status,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def clear_cuda_after_failure() -> None:
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    torch.cuda.empty_cache()


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.assume_unpadded and not math.isclose(
        args.minimum_length_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("--assume-unpadded requires --minimum-length-fraction 1.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation: {args.implementation}")
    if args.dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {args.dtype}")
    if args.hidden_size != args.num_attention_heads * args.attention_head_size:
        raise ValueError("hidden_size must equal num_attention_heads * attention_head_size")
    unsupported_reason = unsupported_layout_reason(args.implementation, args.layout)
    if unsupported_reason is not None:
        metadata = {
            "status": "completed",
            "implementation": args.implementation,
            "model_architecture": MODEL_ARCHITECTURE,
            "scope": args.scope,
            "layout": args.layout,
            "dtype": args.dtype,
            "execution": args.execution,
            "system": collect_system_details(),
        }
        error = NotImplementedError(unsupported_reason)
        results = [
            exception_result(
                status="unsupported",
                batch_size=batch_size,
                sequence_length=sequence_length,
                stage="layout_selection",
                error=error,
            )
            for sequence_length in args.lengths
            for batch_size in args.batches
        ]
        payload = {"metadata": metadata, "results": results}
        write_json(args.worker_output, payload)
        return payload
    if args.implementation == "triton":
        # Triton documents this switch as the supported way to report tuning
        # time and the winning configuration for every new tuning key.
        os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

    configure_fp32(args.fp32_precision)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    reference, target, rel_embeddings, preparation_ms = make_models(
        args.scope,
        args.implementation,
        dtype,
        device,
        args.seed,
        args.hidden_size,
        args.num_attention_heads,
        args.attention_head_size,
        args.fp32_precision,
        args.lengths,
        args.num_hidden_layers,
        args.intermediate_size,
        args.conv_kernel_size,
        args.assume_unpadded,
    )
    embedding_table = make_embedding_table(
        args.vocab_size,
        args.hidden_size,
        dtype,
        device,
        args.seed + 2,
    )

    metadata = {
        "status": "running",
        "implementation": args.implementation,
        "model_architecture": MODEL_ARCHITECTURE,
        "scope": args.scope,
        "layout": args.layout,
        "dtype": args.dtype,
        "execution": args.execution,
        "compile_mode": args.compile_mode if args.execution == "compile" else None,
        "fullgraph": args.fullgraph if args.execution == "compile" else None,
        "dynamic_batch": args.dynamic if args.execution == "compile" else None,
        "preparation_ms": preparation_ms,
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "attention_head_size": args.attention_head_size,
        "qkv_projection": ("fused" if args.implementation in {"torch", "triton"} else "separate"),
        "layout_execution": (
            "external_cu_seqlens"
            if args.layout == "packed" and args.implementation in {"torch", "triton"}
            else "internal_mask_varlen"
            if args.implementation == "flashdeberta"
            else "dense_masked"
        ),
        "assume_unpadded": (args.assume_unpadded and args.implementation == "triton"),
        "fp32_precision": args.fp32_precision,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "conv_kernel_size": args.conv_kernel_size,
        "vocab_size": args.vocab_size,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "system": collect_system_details(),
        "input_seed_policy": {
            "shared_across_backends_and_layouts": True,
            "different_tensor_each_iteration": True,
            "warmup_seed_offset": 0,
            "measurement_seed_offset": 10000,
            "compile_probe_seed_offset": -1,
        },
    }
    print(
        f"worker scope={args.scope} implementation={args.implementation} dtype={args.dtype} "
        f"execution={args.execution} gpu={metadata['gpu']} "
        f"prepare={preparation_ms:.3f} ms",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    payload = {"metadata": metadata, "results": results}
    write_json(args.worker_output, payload)
    with torch.inference_mode():
        for sequence_length in args.lengths:
            eager_call = make_callable(
                args.scope,
                args.implementation,
                target,
                rel_embeddings,
                sequence_length,
                device,
                args.layout,
            )
            if args.execution == "compile":
                call = compile_isolated(
                    eager_call,
                    mode=args.compile_mode,
                    fullgraph=args.fullgraph,
                    dynamic=args.dynamic,
                )
            else:
                call = eager_call

            compile_ms = 0.0
            if args.execution == "compile":
                probe_batch = min(args.batches)
                try:
                    probe = make_inputs(
                        embedding_table,
                        probe_batch,
                        sequence_length,
                        1,
                        args.seed + sequence_length * 100_000 - 1,
                        args.minimum_length_fraction,
                    )[0]
                    print(
                        f"[{args.implementation}/{args.dtype}/{args.execution} "
                        f"L={sequence_length}] compiling with dynamic batch probe "
                        f"B={probe_batch}",
                        flush=True,
                    )
                    torch.cuda.synchronize()
                    compile_started = time.perf_counter()
                    call(*probe)
                    torch.cuda.synchronize()
                    compile_ms = (time.perf_counter() - compile_started) * 1000.0
                    print(
                        f"[{args.implementation}/{args.dtype}/{args.execution} "
                        f"L={sequence_length}] compile completed in {compile_ms:.3f} ms",
                        flush=True,
                    )
                    del probe
                except torch.OutOfMemoryError as error:
                    for batch_size in args.batches:
                        results.append(
                            exception_result(
                                status="oom",
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                stage="compile_probe",
                                error=error,
                            )
                        )
                    write_json(args.worker_output, payload)
                    print(
                        f"[{args.implementation}/{args.dtype}/{args.execution} "
                        f"L={sequence_length}] OOM during compilation; recorded, continuing",
                        flush=True,
                    )
                    clear_cuda_after_failure()
                    continue
                except Exception as error:  # noqa: BLE001 - benchmark must retain diagnostics
                    for batch_size in args.batches:
                        results.append(
                            exception_result(
                                status="error",
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                stage="compile_probe",
                                error=error,
                            )
                        )
                    write_json(args.worker_output, payload)
                    clear_cuda_after_failure()
                    continue

            for batch_size in args.batches:
                log_prefix = (
                    f"[{args.implementation}/{args.dtype}/{args.execution} "
                    f"B={batch_size} L={sequence_length}]"
                )
                stage = "input_generation"
                try:
                    shape_seed = args.seed + sequence_length * 100_000 + batch_size * 1_000
                    stage = "warmup"
                    torch.cuda.synchronize()
                    setup_started = time.perf_counter()
                    for index in range(args.warmup):
                        hidden_states, attention_mask = make_inputs(
                            embedding_table,
                            batch_size,
                            sequence_length,
                            1,
                            shape_seed + index,
                            args.minimum_length_fraction,
                        )[0]
                        print(
                            f"{log_prefix} warmup {index + 1:02d}/{args.warmup:02d}",
                            flush=True,
                        )
                        call(hidden_states, attention_mask)
                        del hidden_states, attention_mask
                    torch.cuda.synchronize()
                    setup_ms = (time.perf_counter() - setup_started) * 1000.0

                    baseline_allocated_bytes = torch.cuda.memory_allocated()
                    baseline_reserved_bytes = torch.cuda.memory_reserved()
                    torch.cuda.reset_peak_memory_stats()

                    stage = "measurement"
                    timings, valid_tokens = measure_generated_calls(
                        call,
                        embedding_table,
                        batch_size,
                        sequence_length,
                        args.samples,
                        shape_seed + 10_000,
                        args.minimum_length_fraction,
                        log_prefix,
                    )

                    peak_allocated_bytes = torch.cuda.max_memory_allocated()
                    peak_reserved_bytes = torch.cuda.max_memory_reserved()
                    incremental_peak_allocated_bytes = max(
                        0,
                        peak_allocated_bytes - baseline_allocated_bytes,
                    )
                    incremental_peak_reserved_bytes = max(
                        0,
                        peak_reserved_bytes - baseline_reserved_bytes,
                    )

                    stage = "parity"
                    parity_batch_size = min(batch_size, args.parity_batch_size)
                    try:
                        max_error, mean_error = compare_generated_outputs(
                            args.scope,
                            reference,
                            call,
                            rel_embeddings,
                            embedding_table,
                            parity_batch_size,
                            sequence_length,
                            args.parity_samples,
                            shape_seed + 10_000,
                            args.minimum_length_fraction,
                            args.layout,
                        )
                        parity_status = "ok"
                        parity_error = None
                    except torch.OutOfMemoryError as error:
                        max_error = None
                        mean_error = None
                        parity_status = "oom"
                        parity_error = str(error)
                        clear_cuda_after_failure()

                    total_seconds = sum(timings) / 1000.0
                    result = {
                        "status": "ok",
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "samples": args.samples,
                        "p50_ms": statistics.median(timings),
                        "p90_ms": percentile(timings, 0.90),
                        "p95_ms": percentile(timings, 0.95),
                        "mean_ms": statistics.mean(timings),
                        "docs_per_second": batch_size * len(timings) / total_seconds,
                        "valid_tokens_per_second": sum(valid_tokens) / total_seconds,
                        "mean_valid_tokens_per_batch": statistics.mean(valid_tokens),
                        "parity_status": parity_status,
                        "parity_batch_size": parity_batch_size,
                        "parity_samples": args.parity_samples,
                        "parity_error": parity_error,
                        "max_abs_error": max_error,
                        "mean_abs_error": mean_error,
                        "compile_ms_for_length": compile_ms,
                        "warmup_ms": setup_ms,
                        "baseline_allocated_bytes": baseline_allocated_bytes,
                        "baseline_reserved_bytes": baseline_reserved_bytes,
                        "peak_allocated_bytes": peak_allocated_bytes,
                        "peak_reserved_bytes": peak_reserved_bytes,
                        "incremental_peak_allocated_bytes": incremental_peak_allocated_bytes,
                        "incremental_peak_reserved_bytes": incremental_peak_reserved_bytes,
                    }
                    results.append(result)
                    print(
                        f"{log_prefix} summary p50={result['p50_ms']:.4f} ms "
                        f"p90={result['p90_ms']:.4f} ms "
                        f"docs/s={result['docs_per_second']:.2f} "
                        f"valid_tok/s={result['valid_tokens_per_second']:.2f} "
                        f"parity={parity_status}",
                        flush=True,
                    )
                except torch.OutOfMemoryError as error:
                    results.append(
                        exception_result(
                            status="oom",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            stage=stage,
                            error=error,
                        )
                    )
                    print(f"{log_prefix} OOM during {stage}; recorded, continuing", flush=True)
                except Exception as error:  # noqa: BLE001 - retain every failed shape
                    results.append(
                        exception_result(
                            status="error",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            stage=stage,
                            error=error,
                        )
                    )
                    print(
                        f"{log_prefix} ERROR during {stage}: {type(error).__name__}: {error}",
                        flush=True,
                    )
                finally:
                    clear_cuda_after_failure()
                    write_json(args.worker_output, payload)

    metadata["status"] = (
        "completed_with_errors"
        if any(result["status"] == "error" for result in results)
        else "completed"
    )
    write_json(args.worker_output, payload)
    return payload


def print_summary(payloads: list[dict[str, Any]]) -> None:
    baseline = {}
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["implementation"] not in {"base", "original"}:
            continue
        for result in payload["results"]:
            if result["status"] != "ok":
                continue
            key = (
                metadata["dtype"],
                metadata["execution"],
                result["batch_size"],
                result["sequence_length"],
            )
            baseline[key] = result["p50_ms"]

    print()
    scopes = sorted({payload["metadata"]["scope"] for payload in payloads})
    print(f"Final CUDA {'/'.join(scopes)} summary")
    print(
        f"{'dtype':>5} {'execution':>9} {'layout':>7} {'implementation':>14} "
        f"{'B':>3} {'L':>4} "
        f"{'p50 ms':>10} {'p90 ms':>10} {'speedup':>9} "
        f"{'docs/s':>12} {'valid tok/s':>14} {'max error':>12}"
    )
    for payload in payloads:
        metadata = payload["metadata"]
        for result in payload["results"]:
            if result["status"] != "ok":
                print(
                    f"{metadata['dtype']:>5} {metadata['execution']:>9} "
                    f"{metadata['layout']:>7} "
                    f"{metadata['implementation']:>14} "
                    f"{result['batch_size']:>3} {result['sequence_length']:>4} "
                    f"{result['status'].upper():>10} {'-':>10} {'-':>9} "
                    f"{'-':>12} {'-':>14} {'-':>12}"
                )
                continue
            key = (
                metadata["dtype"],
                metadata["execution"],
                result["batch_size"],
                result["sequence_length"],
            )
            reference_ms = baseline.get(key)
            speedup = reference_ms / result["p50_ms"] if reference_ms is not None else 1.0
            max_error = result["max_abs_error"]
            error_text = "N/A" if max_error is None else f"{max_error:.6g}"
            print(
                f"{metadata['dtype']:>5} {metadata['execution']:>9} "
                f"{metadata['layout']:>7} "
                f"{metadata['implementation']:>14} "
                f"{result['batch_size']:>3} {result['sequence_length']:>4} "
                f"{result['p50_ms']:>10.4f} {result['p90_ms']:>10.4f} "
                f"{speedup:>8.2f}x {result['docs_per_second']:>12.2f} "
                f"{result['valid_tokens_per_second']:>14.2f} "
                f"{error_text:>12}"
            )


def run_parent(args: argparse.Namespace) -> None:
    if args.assume_unpadded and not math.isclose(
        args.minimum_length_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("--assume-unpadded requires --minimum-length-fraction 1.0")
    invalid_implementations = set(args.implementations) - set(IMPLEMENTATIONS)
    invalid_dtypes = set(args.dtypes) - set(DTYPES)
    invalid_executions = set(args.executions) - {"eager", "compile"}
    invalid_layouts = set(args.layouts) - {"padded", "packed"}
    if invalid_implementations or invalid_dtypes or invalid_executions or invalid_layouts:
        raise ValueError(
            f"invalid selections: implementations={sorted(invalid_implementations)}, "
            f"dtypes={sorted(invalid_dtypes)}, executions={sorted(invalid_executions)}, "
            f"layouts={sorted(invalid_layouts)}"
        )
    if args.scope == "attention" and "flashdeberta" in args.implementations:
        raise ValueError("flashdeberta is an encoder-only benchmark implementation")

    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_path = Path(args.output).resolve()
    aggregate = {
        "schema_version": 2,
        "system": collect_system_details(),
        "requested": {
            "implementations": args.implementations,
            "model_architecture": MODEL_ARCHITECTURE,
            "scope": args.scope,
            "dtypes": args.dtypes,
            "executions": args.executions,
            "layouts": args.layouts,
            "batches": args.batches,
            "lengths": args.lengths,
            "warmup": args.warmup,
            "samples": args.samples,
            "parity_samples": args.parity_samples,
            "parity_batch_size": args.parity_batch_size,
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "attention_head_size": args.attention_head_size,
            "qkv_projection": {
                "base": "separate",
                "original": "separate",
                "torch": "fused",
                "triton": "fused",
                "flashdeberta": "separate",
            },
            "assume_unpadded": args.assume_unpadded,
            "fp32_precision": args.fp32_precision,
            "num_hidden_layers": args.num_hidden_layers,
            "intermediate_size": args.intermediate_size,
            "conv_kernel_size": args.conv_kernel_size,
            "vocab_size": args.vocab_size,
            "minimum_length_fraction": args.minimum_length_fraction,
        },
        "workers": payloads,
        "failures": failures,
    }
    write_json(output_path, aggregate)
    with tempfile.TemporaryDirectory(prefix="deberta_cuda_benchmark_") as temporary_dir:
        for dtype in args.dtypes:
            for execution in args.executions:
                for layout in args.layouts:
                    for implementation in args.implementations:
                        worker_output = Path(temporary_dir) / (
                            f"{implementation}_{dtype}_{execution}_{layout}.json"
                        )
                        command = [
                            sys.executable,
                            "-m",
                            "benchmarks.benchmark_cuda",
                            "--worker",
                            "--scope",
                            args.scope,
                            "--implementation",
                            implementation,
                            "--dtype",
                            dtype,
                            "--execution",
                            execution,
                            "--layout",
                            layout,
                            "--worker-output",
                            str(worker_output),
                            "--batches",
                            ",".join(str(value) for value in args.batches),
                            "--lengths",
                            ",".join(str(value) for value in args.lengths),
                            "--warmup",
                            str(args.warmup),
                            "--samples",
                            str(args.samples),
                            "--parity-samples",
                            str(args.parity_samples),
                            "--parity-batch-size",
                            str(args.parity_batch_size),
                            "--compile-mode",
                            args.compile_mode,
                            "--vocab-size",
                            str(args.vocab_size),
                            "--minimum-length-fraction",
                            str(args.minimum_length_fraction),
                            "--seed",
                            str(args.seed),
                            "--hidden-size",
                            str(args.hidden_size),
                            "--num-attention-heads",
                            str(args.num_attention_heads),
                            "--attention-head-size",
                            str(args.attention_head_size),
                            "--fp32-precision",
                            args.fp32_precision,
                            "--num-hidden-layers",
                            str(args.num_hidden_layers),
                            "--intermediate-size",
                            str(args.intermediate_size),
                            "--conv-kernel-size",
                            str(args.conv_kernel_size),
                        ]
                        command.append("--fullgraph" if args.fullgraph else "--no-fullgraph")
                        command.append("--dynamic" if args.dynamic else "--static")
                        print()
                        print(
                            f"Starting worker: implementation={implementation} dtype={dtype} "
                            f"execution={execution} layout={layout}",
                            flush=True,
                        )
                        if args.assume_unpadded:
                            command.append("--assume-unpadded")
                        completed = subprocess.run(command, cwd=os.getcwd(), check=False)
                        if completed.returncode != 0:
                            failures.append(
                                {
                                    "implementation": implementation,
                                    "dtype": dtype,
                                    "execution": execution,
                                    "layout": layout,
                                    "returncode": completed.returncode,
                                }
                            )
                            write_json(output_path, aggregate)
                            continue
                        payload = json.loads(worker_output.read_text(encoding="utf-8"))
                        payloads.append(payload)
                        if payload["metadata"]["status"] == "completed_with_errors":
                            failures.append(
                                {
                                    "implementation": implementation,
                                    "dtype": dtype,
                                    "execution": execution,
                                    "layout": layout,
                                    "reason": "one or more shapes produced a non-OOM error",
                                }
                            )
                        write_json(output_path, aggregate)

    write_json(output_path, aggregate)
    print_summary(payloads)
    print()
    print(f"Saved complete results to {output_path}")
    if failures:
        print(f"Failed workers: {json.dumps(failures)}")
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["attention", "encoder"], default="encoder")
    parser.add_argument(
        "--implementations",
        type=parse_csv,
        default=["base", "torch", "triton", "flashdeberta"],
    )
    parser.add_argument("--dtypes", type=parse_csv, default=["fp16", "fp32"])
    parser.add_argument("--executions", type=parse_csv, default=["eager", "compile"])
    parser.add_argument("--layouts", type=parse_csv, default=["padded", "packed"])
    parser.add_argument("--batches", type=parse_csv_ints, default=[1, 8, 16, 32])
    parser.add_argument(
        "--lengths",
        type=parse_csv_ints,
        default=[64, 128, 256, 512, 1024, 2048, 4098, 8192],
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=1,
        help="Fresh samples used for numerical parity outside timed measurements.",
    )
    parser.add_argument(
        "--parity-batch-size",
        type=int,
        default=1,
        help="Maximum parity batch size; performance still uses each requested batch size.",
    )
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-attention-heads", type=int, default=12)
    parser.add_argument("--attention-head-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument(
        "--fp32-precision",
        choices=["strict", "fast"],
        default="strict",
    )
    parser.add_argument("--vocab-size", type=int, default=128100)
    parser.add_argument("--minimum-length-fraction", type=float, default=0.60)
    parser.add_argument(
        "--assume-unpadded",
        action="store_true",
        help=("Use the Triton no-padding specialization. Requires --minimum-length-fraction 1.0."),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="deberta_v3_base_encoder_cuda_results.json")
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=True)
    parser.add_argument("--static", dest="dynamic", action="store_false")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--implementation", choices=sorted(IMPLEMENTATIONS.keys()), help=argparse.SUPPRESS
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), help=argparse.SUPPRESS)
    parser.add_argument("--execution", choices=["eager", "compile"], help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=["padded", "packed"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if not all(
            [args.implementation, args.dtype, args.execution, args.layout, args.worker_output]
        ):
            raise SystemExit(
                "worker mode requires implementation, dtype, execution, layout, and output"
            )
        try:
            run_worker(args)
        except torch.OutOfMemoryError as error:
            results = [
                exception_result(
                    status="oom",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    stage="model_initialization",
                    error=error,
                )
                for sequence_length in args.lengths
                for batch_size in args.batches
            ]
            payload = {
                "metadata": {
                    "status": "completed",
                    "implementation": args.implementation,
                    "model_architecture": MODEL_ARCHITECTURE,
                    "scope": args.scope,
                    "layout": args.layout,
                    "dtype": args.dtype,
                    "execution": args.execution,
                    "system": collect_system_details(),
                },
                "results": results,
            }
            write_json(args.worker_output, payload)
            print(
                "CUDA OOM during model initialization; recorded as an expected capacity "
                "result rather than a benchmark failure",
                flush=True,
            )
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
