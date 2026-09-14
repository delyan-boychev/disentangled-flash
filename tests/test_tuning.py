import json

import pytest
import torch

from disentangled_flash.tune import TuningCase, _make_inputs, _reference
from disentangled_flash.tuning import (
    HardwareSpec,
    KernelConfig,
    KernelProfile,
    KernelTuningOptions,
    ProfileEntry,
    ProfileRegistry,
    WorkloadKey,
    load_profile,
    merge_profile_entry,
    save_profile,
    tuning_sequence_length,
)


def make_hardware(name: str = "NVIDIA RTX 6000 Ada Generation") -> HardwareSpec:
    return HardwareSpec(backend="cuda", name=name, compute_capability=(8, 9))


def make_workload(length: int = 128) -> WorkloadKey:
    return WorkloadKey(
        sequence_length=length,
        head_dim=64,
        batch_heads=12,
        active_slots=128,
        dtype="float16",
        has_c2p=True,
        has_p2c=True,
        fp32_precision="strict",
    )


def make_profile(config: KernelConfig | None = None) -> KernelProfile:
    entry = ProfileEntry(
        workload=make_workload(),
        config=config or KernelConfig(64, 64, 4),
        latency_ms=0.125,
    )
    return KernelProfile(
        hardware=make_hardware(),
        entries=(entry,),
        environment={"torch": "test", "triton": "test", "cuda": "test"},
    )


def test_profile_json_round_trip(tmp_path):
    destination = tmp_path / "profile.json"
    expected = make_profile()

    assert save_profile(expected, destination) == destination.resolve()
    assert load_profile(destination) == expected
    assert json.loads(destination.read_text())["format_version"] == 1


def test_registry_uses_explicit_order_and_normalized_gpu_name():
    preferred = make_profile(KernelConfig(32, 64, 4))
    fallback = make_profile(KernelConfig(64, 64, 4))
    registry = ProfileRegistry((preferred, fallback))
    hardware = make_hardware("  nvidia   RTX 6000 ada GENERATION ")

    assert registry.resolve(hardware, make_workload()) == KernelConfig(32, 64, 4)
    assert registry.resolve(hardware, make_workload(256)) is None


@pytest.mark.parametrize(
    ("length", "representative"),
    [(1, 64), (64, 64), (65, 128), (383, 384), (384, 384), (513, 768), (2048, 1024)],
)
def test_tuning_sequence_lengths_use_bounded_families(length, representative):
    assert tuning_sequence_length(length) == representative


def test_profile_for_384_resolves_runtime_length_383():
    entry = ProfileEntry(
        workload=make_workload(384),
        config=KernelConfig(64, 64, 4),
        latency_ms=0.125,
    )
    profile = KernelProfile(hardware=make_hardware(), entries=(entry,))
    registry = ProfileRegistry((profile,))

    assert registry.resolve(make_hardware(), make_workload(383)) == KernelConfig(64, 64, 4)


def test_registry_ignores_unvalidated_entries():
    profile = KernelProfile(
        hardware=make_hardware(),
        entries=(
            ProfileEntry(
                workload=make_workload(),
                config=KernelConfig(64, 64, 4),
                latency_ms=1.0,
                validated=False,
            ),
        ),
    )

    assert ProfileRegistry((profile,)).resolve(make_hardware(), make_workload()) is None


def test_merge_replaces_only_the_matching_workload():
    original = make_profile(KernelConfig(32, 32, 2))
    replacement = ProfileEntry(
        workload=make_workload(),
        config=KernelConfig(64, 64, 4),
        latency_ms=0.1,
    )

    merged = merge_profile_entry(
        original,
        hardware=make_hardware(),
        entry=replacement,
        environment={"torch": "new"},
    )

    assert merged.entries == (replacement,)
    assert merged.environment == {"torch": "new"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "fixed"}, "requires fixed_config"),
        ({"mode": "auto", "fixed_config": KernelConfig(32, 32, 2)}, "only valid"),
        ({"mode": "invalid"}, "tuning mode"),
        ({"candidates": ()}, "must not be empty"),
        (
            {"mode": "profile_only", "candidates": (KernelConfig(32, 32, 2),)},
            "only valid in auto or autotune",
        ),
    ],
)
def test_tuning_options_reject_invalid_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KernelTuningOptions(**kwargs)


def test_tuning_options_normalize_user_lists():
    options = KernelTuningOptions(
        profile_paths=["profile.json"],
        candidates=[KernelConfig(32, 32, 2)],
    )

    assert options.profile_paths == ("profile.json",)
    assert options.candidates == (KernelConfig(32, 32, 2),)


def test_profile_rejects_unknown_fields(tmp_path):
    payload = make_profile().to_dict()
    payload["run_this"] = "not accepted"
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields"):
        load_profile(path)


def test_profile_rejects_duplicate_workloads():
    entry = make_profile().entries[0]
    with pytest.raises(ValueError, match="duplicate workload"):
        KernelProfile(hardware=make_hardware(), entries=(entry, entry))


def test_profile_rejects_different_hardware_merge():
    with pytest.raises(ValueError, match="different GPU"):
        merge_profile_entry(
            make_profile(),
            hardware=HardwareSpec("cuda", "Other GPU", (8, 9)),
            entry=make_profile().entries[0],
            environment={},
        )


def test_tuning_reference_covers_relative_scores_and_fully_masked_rows():
    case = TuningCase(8, 32, 2, "float32", True, True, "strict")
    arguments, workload = _make_inputs(case, torch.device("cpu"))
    empty_mask = torch.zeros_like(arguments[6])
    masked_arguments = arguments[:6] + (empty_mask,) + arguments[7:]

    output = _reference(masked_arguments)

    assert workload.active_slots == 64
    assert output.shape == (1, 8, 64)
    assert torch.isfinite(output).all()
