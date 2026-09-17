import json

import pytest
import torch

from disentangled_flash.kernel import AUTOTUNE_SPECIALIZATION_KEY
from disentangled_flash.tune import (
    PRESETS,
    TuningCase,
    _make_inputs,
    _packed_reference,
    _reference,
    build_parser,
    inspect_profile,
)
from disentangled_flash.tuning import (
    CompilerSpec,
    HardwareSpec,
    KernelConfig,
    KernelProfile,
    KernelTuningOptions,
    ProfileEntry,
    ProfileRegistry,
    WorkloadKey,
    load_bundled_profiles,
    load_profile,
    merge_profile_entry,
    save_profile,
    tuning_sequence_length,
)


def make_hardware(name: str = "NVIDIA RTX 6000 Ada Generation") -> HardwareSpec:
    return HardwareSpec(backend="cuda", name=name, compute_capability=(8, 9))


def make_compiler(
    torch_version: str = "2.13.0",
    triton_key: str = "triton-test-key",
    cuda_runtime: str = "12.8",
) -> CompilerSpec:
    return CompilerSpec(torch_version, triton_key, cuda_runtime)


def make_workload(
    length: int = 128,
    *,
    layout: str = "padded",
    uses_padding_mask: bool = True,
) -> WorkloadKey:
    return WorkloadKey(
        sequence_length=length,
        head_dim=64,
        batch_heads=12,
        active_slots=128,
        dtype="float16",
        has_c2p=True,
        has_p2c=True,
        fp32_precision="strict",
        layout=layout,
        uses_padding_mask=uses_padding_mask,
    )


def make_profile(config: KernelConfig | None = None) -> KernelProfile:
    entry = ProfileEntry(
        workload=make_workload(),
        config=config or KernelConfig(64, 64, 4),
        latency_ms=0.125,
    )
    return KernelProfile(
        hardware=make_hardware(),
        compiler=make_compiler(),
        entries=(entry,),
        provenance={"driver": "test"},
    )


def test_profile_json_round_trip(tmp_path):
    destination = tmp_path / "profile.json"
    expected = make_profile()

    assert save_profile(expected, destination) == destination.resolve()
    assert load_profile(destination) == expected
    payload = json.loads(destination.read_text())
    assert payload["format_version"] == 3
    assert payload["compiler"]["triton_key"] == "triton-test-key"
    assert payload["entries"][0]["workload"]["length_regime"] == 128
    assert payload["entries"][0]["workload"]["layout"] == "padded"
    assert "sequence_length" not in payload["entries"][0]["workload"]


def test_bundled_profiles_are_parseable_and_fully_validated():
    profiles = load_bundled_profiles()

    assert profiles
    assert all(profile.entries for profile in profiles)
    assert all(entry.validated for profile in profiles for entry in profile.entries)


def test_registry_uses_explicit_order_and_normalized_gpu_name():
    preferred = make_profile(KernelConfig(32, 64, 4))
    fallback = make_profile(KernelConfig(64, 64, 4))
    registry = ProfileRegistry((preferred, fallback))
    hardware = make_hardware("  nvidia   RTX 6000 ada GENERATION ")

    assert registry.resolve(hardware, make_compiler(), make_workload()) == KernelConfig(32, 64, 4)
    assert registry.resolve(hardware, make_compiler(), make_workload(256)) is None


@pytest.mark.parametrize(
    ("length", "representative"),
    [
        (1, 64),
        (64, 64),
        (65, 128),
        (383, 384),
        (384, 384),
        (513, 768),
        (1025, 2048),
        (2049, 4096),
        (4097, 8192),
        (16384, 8192),
    ],
)
def test_tuning_sequence_lengths_use_bounded_families(length, representative):
    assert tuning_sequence_length(length) == representative


def test_standard_is_the_broadest_supported_preset():
    assert set(PRESETS) == {"quick", "standard"}
    assert PRESETS["standard"]["relative_modes"] == ("none", "c2p", "p2c", "both")
    assert PRESETS["standard"]["layouts"] == ("padded", "packed")
    assert PRESETS["standard"]["lengths"][-1] == 8192
    assert build_parser().parse_args(["--output", "profile.json"]).preset == "standard"


def test_packed_tuning_case_uses_mixed_boundaries_and_separate_workload():
    case = TuningCase(
        8,
        32,
        8,
        "float32",
        True,
        True,
        "strict",
        layout="packed",
        uses_padding_mask=False,
    )
    arguments, workload = _make_inputs(case, torch.device("cpu"))

    assert arguments[6].tolist() == [0, 8, 15, 20, 22]
    assert workload.layout == "packed"
    assert workload.uses_padding_mask is False
    assert _packed_reference(arguments).shape == (22, 64)


def test_profile_for_384_resolves_runtime_length_383():
    entry = ProfileEntry(
        workload=make_workload(384),
        config=KernelConfig(64, 64, 4),
        latency_ms=0.125,
    )
    profile = KernelProfile(
        hardware=make_hardware(),
        compiler=make_compiler(),
        entries=(entry,),
    )
    registry = ProfileRegistry((profile,))

    assert registry.resolve(make_hardware(), make_compiler(), make_workload(383)) == KernelConfig(
        64, 64, 4
    )


def test_registry_isolates_padded_and_packed_winners():
    profile = KernelProfile(
        hardware=make_hardware(),
        compiler=make_compiler(),
        entries=(
            ProfileEntry(
                workload=make_workload(),
                config=KernelConfig(64, 64, 4),
                latency_ms=0.1,
            ),
        ),
    )
    registry = ProfileRegistry((profile,))

    assert registry.resolve(make_hardware(), make_compiler(), make_workload()) is not None
    assert (
        registry.resolve(
            make_hardware(),
            make_compiler(),
            make_workload(layout="packed", uses_padding_mask=False),
        )
        is None
    )


@pytest.mark.parametrize(
    "compiler",
    [
        make_compiler(torch_version="2.14.0"),
        make_compiler(triton_key="new-triton-key"),
        make_compiler(cuda_runtime="13.0"),
    ],
)
def test_registry_rejects_incompatible_compilers(compiler):
    registry = ProfileRegistry((make_profile(),))

    assert registry.resolve(make_hardware(), compiler, make_workload()) is None
    assert "incompatible" in registry.explain_miss(make_hardware(), compiler, make_workload())


def test_autotune_key_excludes_exact_runtime_dimensions():
    assert "LENGTH_REGIME" in AUTOTUNE_SPECIALIZATION_KEY
    assert not {
        "SEQUENCE_LENGTH",
        "BATCH_SIZE",
        "NUM_HEADS",
        "ACTIVE_SLOTS",
    }.intersection(AUTOTUNE_SPECIALIZATION_KEY)


def test_registry_ignores_unvalidated_entries():
    profile = KernelProfile(
        hardware=make_hardware(),
        compiler=make_compiler(),
        entries=(
            ProfileEntry(
                workload=make_workload(),
                config=KernelConfig(64, 64, 4),
                latency_ms=1.0,
                validated=False,
            ),
        ),
    )

    assert (
        ProfileRegistry((profile,)).resolve(make_hardware(), make_compiler(), make_workload())
        is None
    )


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
        compiler=make_compiler(),
        entry=replacement,
        provenance={"driver": "new"},
    )

    assert merged.entries == (replacement,)
    assert merged.provenance == {"driver": "new"}


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


def test_profile_rejects_format_two_with_migration_message(tmp_path):
    payload = make_profile().to_dict()
    payload["format_version"] = 2
    path = tmp_path / "old-profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks compiler and kernel-layout"):
        load_profile(path)


def test_profile_inspection_reports_compiler_mismatch(tmp_path, capsys):
    path = tmp_path / "profile.json"
    save_profile(make_profile(), path)

    assert inspect_profile(path) is False
    output = capsys.readouterr().out
    assert "Compatible: no" in output
    assert "profile=" in output


def test_profile_rejects_duplicate_workloads():
    entry = make_profile().entries[0]
    with pytest.raises(ValueError, match="duplicate workload"):
        KernelProfile(
            hardware=make_hardware(),
            compiler=make_compiler(),
            entries=(entry, entry),
        )


def test_profile_rejects_different_hardware_merge():
    with pytest.raises(ValueError, match="different GPU"):
        merge_profile_entry(
            make_profile(),
            hardware=HardwareSpec("cuda", "Other GPU", (8, 9)),
            compiler=make_compiler(),
            entry=make_profile().entries[0],
            provenance={},
        )


def test_driver_provenance_does_not_control_compatibility():
    profile = make_profile()
    changed_driver = KernelProfile(
        hardware=profile.hardware,
        compiler=profile.compiler,
        entries=profile.entries,
        provenance={"driver": "different"},
    )

    assert ProfileRegistry((changed_driver,)).resolve(
        make_hardware(), make_compiler(), make_workload()
    ) == KernelConfig(64, 64, 4)


def test_tuning_reference_covers_relative_scores_and_fully_masked_rows():
    case = TuningCase(8, 32, 2, "float32", True, True, "strict")
    arguments, workload = _make_inputs(case, torch.device("cpu"))
    empty_mask = torch.zeros_like(arguments[6])
    masked_arguments = arguments[:6] + (empty_mask,) + arguments[7:]

    output = _reference(masked_arguments)

    assert workload.active_slots == 64
    assert output.shape == (1, 8, 64)
    assert torch.isfinite(output).all()


def test_tuning_reference_query_chunking_preserves_results():
    case = TuningCase(8, 32, 2, "float32", True, True, "strict")
    arguments, _workload = _make_inputs(case, torch.device("cpu"))

    one_chunk = _reference(arguments, query_chunk_size=8)
    small_chunks = _reference(arguments, query_chunk_size=3)

    torch.testing.assert_close(small_chunks, one_chunk)
