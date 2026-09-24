import json

import pytest
import torch

from disentangled_flash.kernel import AUTOTUNE_SPECIALIZATION_KEY
from disentangled_flash.tune import (
    PRESETS,
    TuningCase,
    _cases,
    _load_candidates,
    _make_inputs,
    _make_training_inputs,
    _packed_reference,
    _reference,
    _search_configs,
    _training_reference,
    build_parser,
    inspect_profile,
    main,
)
from disentangled_flash.tuning import (
    DEFAULT_DKV_KERNEL_CONFIGS,
    DEFAULT_DQ_KERNEL_CONFIGS,
    DEFAULT_KERNEL_CONFIGS,
    KERNEL_SOURCE_DIGEST,
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
    prepare_profile_retarget,
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
    phase: str = "inference",
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
        phase=phase,
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
    assert payload["format_version"] == 4
    assert payload["kernel_digest"] == KERNEL_SOURCE_DIGEST
    assert payload["compiler"]["triton_key"] == "triton-test-key"
    assert payload["entries"][0]["workload"]["length_regime"] == 128
    assert payload["entries"][0]["workload"]["layout"] == "padded"
    assert payload["entries"][0]["workload"]["phase"] == "inference"
    assert "sequence_length" not in payload["entries"][0]["workload"]


def test_profile_phase_separates_training_forward_and_backward_winners():
    entries = (
        ProfileEntry(make_workload(phase="training_forward"), KernelConfig(64, 32, 4), 0.1),
        ProfileEntry(make_workload(phase="backward_dq"), KernelConfig(32, 64, 4), 0.2),
        ProfileEntry(make_workload(phase="backward_dkv"), KernelConfig(64, 64, 8), 0.3),
    )
    registry = ProfileRegistry((KernelProfile(make_hardware(), make_compiler(), entries=entries),))

    assert registry.resolve(
        make_hardware(), make_compiler(), make_workload(phase="training_forward")
    ) == KernelConfig(64, 32, 4)
    assert registry.resolve(
        make_hardware(), make_compiler(), make_workload(phase="backward_dq")
    ) == KernelConfig(32, 64, 4)
    assert registry.resolve(
        make_hardware(), make_compiler(), make_workload(phase="backward_dkv")
    ) == KernelConfig(64, 64, 8)


def test_legacy_format_three_workload_defaults_to_inference_phase():
    payload = make_workload().to_dict()
    payload.pop("phase")

    assert WorkloadKey.from_dict(payload).phase == "inference"


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


def test_standard_covers_supported_deberta_variants_and_all_kernel_phases():
    assert set(PRESETS) == {"quick", "standard"}
    assert PRESETS["standard"]["head_dims"] == (64,)
    assert PRESETS["standard"]["batch_heads"] == (8, 32)
    assert PRESETS["standard"]["relative_modes"] == ("both",)
    assert PRESETS["standard"]["layouts"] == ("padded", "packed")
    assert PRESETS["standard"]["passes"] == ("inference", "training")
    assert PRESETS["standard"]["lengths"][-1] == 8192
    args = build_parser().parse_args(["--output", "profile.json"])
    assert args.preset == "standard"
    assert args.passes is None
    cases = list(_cases(args))
    assert len(cases) == 216
    assert len(cases) * (1 + 3) == 864


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


def test_packed_training_tuning_case_builds_differentiable_reference():
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
    inputs, workload = _make_training_inputs(case, torch.device("cpu"))

    output = _training_reference(inputs, workload)
    gradients = torch.autograd.grad(output.square().mean(), inputs.grad_tensors())

    assert output.shape == (22, 64)
    assert workload.phase == "training_forward"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


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
def test_registry_requires_retargeting_for_different_compilers(compiler):
    registry = ProfileRegistry((make_profile(),))

    assert registry.resolve(make_hardware(), compiler, make_workload()) is None
    assert "retargeting" in registry.explain_miss(make_hardware(), compiler, make_workload())


def test_profile_compatibility_distinguishes_exact_retargetable_and_incompatible():
    profile = make_profile()

    assert profile.compatibility(make_hardware(), make_compiler()) == "exact"
    assert (
        profile.compatibility(make_hardware(), make_compiler(triton_key="different"))
        == "retargetable"
    )
    incompatible = KernelProfile(
        hardware=profile.hardware,
        compiler=profile.compiler,
        entries=profile.entries,
        kernel_version="different-launch-schema",
    )
    assert incompatible.compatibility(make_hardware(), make_compiler()) == "incompatible"


def test_prepare_retarget_preserves_winners_as_pending_seeds():
    source = make_profile()
    current = make_compiler(triton_key="different")

    pending = prepare_profile_retarget(
        source,
        hardware=make_hardware(),
        compiler=current,
        provenance={"kind": "retarget"},
    )

    assert pending.compiler == current
    assert pending.kernel_digest == KERNEL_SOURCE_DIGEST
    assert pending.entries[0].config == source.entries[0].config
    assert pending.entries[0].validated is False
    assert pending.entries[0].validation["retarget_status"] == "pending"


def test_default_candidates_search_pipeline_depth_by_phase():
    assert {config.num_stages for config in DEFAULT_KERNEL_CONFIGS} == {1, 2, 3}
    assert {config.num_stages for config in DEFAULT_DQ_KERNEL_CONFIGS} == {1, 2}
    assert {config.num_stages for config in DEFAULT_DKV_KERNEL_CONFIGS} == {1, 2}


def test_hierarchical_search_refines_stages_only_for_strong_shapes():
    candidates = (
        KernelConfig(32, 32, 4, 1),
        KernelConfig(64, 64, 4, 1),
        KernelConfig(32, 32, 4, 2),
        KernelConfig(64, 64, 4, 2),
    )
    measured = []
    latencies = {
        candidates[0]: 2.0,
        candidates[1]: 1.0,
        candidates[2]: 1.8,
        candidates[3]: 0.8,
    }

    result = _search_configs(
        candidates,
        lambda config: measured.append(config) or latencies[config],
        tie_margin=0.0,
        label="test",
    )

    assert measured == list(candidates)
    assert result.latency_ms == 0.8
    assert result.config == KernelConfig(64, 64, 4, 2)
    assert result.rejected == ()


def test_hierarchical_search_skips_stages_for_weak_shapes_and_records_rejections():
    weak = KernelConfig(16, 16, 4)
    strong = KernelConfig(64, 64, 4)
    broken = KernelConfig(32, 32, 4)
    candidates = (
        weak,
        strong,
        broken,
        KernelConfig(16, 16, 4, 2),
        KernelConfig(64, 64, 4, 2),
        KernelConfig(64, 64, 4, 3),
    )
    latencies = {
        weak: 3.0,
        strong: 1.0,
        KernelConfig(16, 16, 4, 2): 2.5,
        KernelConfig(64, 64, 4, 2): 0.9,
    }
    measured = []

    def evaluate(config):
        measured.append(config)
        if config == broken or config.num_stages == 3:
            raise RuntimeError("out of resources")
        return latencies[config]

    result = _search_configs(candidates, evaluate, tie_margin=0.0, label="test")

    # Only two stage-one shapes survive, so both remain eligible for refinement.
    assert KernelConfig(16, 16, 4, 2) in measured
    assert result.config == KernelConfig(64, 64, 4, 2)
    assert result.rejected == ("32x32w4s1:RuntimeError", "64x64w4s3:RuntimeError")

    latencies[broken] = 2.0
    measured.clear()
    result = _search_configs(
        candidates,
        lambda config: measured.append(config) or latencies.get(config, 0.5),
        tie_margin=0.0,
        label="test",
    )
    assert KernelConfig(16, 16, 4, 2) not in measured
    assert result.config == KernelConfig(64, 64, 4, 3)


def test_search_raises_when_every_candidate_fails():
    def evaluate(_config):
        raise RuntimeError("compile failure")

    with pytest.raises(RuntimeError, match="no correct test configuration"):
        _search_configs((KernelConfig(32, 32, 4),), evaluate, tie_margin=0.0, label="test")


def test_retarget_search_measures_only_seed_neighborhood():
    seed = KernelConfig(32, 64, 8, 2)
    candidates = (
        KernelConfig(16, 16, 4),
        KernelConfig(32, 32, 4),
        KernelConfig(32, 64, 8),
        KernelConfig(64, 64, 4),
        KernelConfig(64, 32, 4, 2),
    )
    measured = []

    result = _search_configs(
        candidates,
        lambda config: measured.append(config) or 1.0,
        tie_margin=0.0,
        label="test",
        seed=seed,
        retarget=True,
    )

    assert measured == [
        seed,
        KernelConfig(32, 32, 4),
        KernelConfig(32, 64, 8),
        KernelConfig(64, 64, 4),
    ]
    assert result.config == seed


def test_candidate_file_accepts_shared_list_and_per_phase_object(tmp_path):
    shared = [KernelConfig(32, 32, 4).to_dict()]
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(shared))
    assert _load_candidates(path) == ((KernelConfig(32, 32, 4),),) * 3

    phased = {
        "forward": [KernelConfig(64, 64, 4, 2).to_dict()],
        "dq": [KernelConfig(32, 64, 4).to_dict()],
        "dkv": [KernelConfig(64, 32, 8).to_dict()],
    }
    path = tmp_path / "phased.json"
    path.write_text(json.dumps(phased))
    assert _load_candidates(path) == (
        (KernelConfig(64, 64, 4, 2),),
        (KernelConfig(32, 64, 4),),
        (KernelConfig(64, 32, 8),),
    )

    path.write_text(json.dumps({"forward": phased["forward"]}))
    with pytest.raises(ValueError, match="forward, dq, and dkv"):
        _load_candidates(path)


def test_retarget_is_resumable_on_an_exact_profile():
    current = make_compiler(triton_key="different")
    pending = prepare_profile_retarget(
        make_profile(),
        hardware=make_hardware(),
        compiler=current,
        provenance={},
    )
    revalidated = merge_profile_entry(
        pending,
        hardware=make_hardware(),
        compiler=current,
        entry=ProfileEntry(make_workload(), KernelConfig(32, 64, 4, 2), 0.1),
        provenance={},
    )

    resumed = prepare_profile_retarget(
        revalidated,
        hardware=make_hardware(),
        compiler=current,
        provenance={},
    )

    assert resumed == revalidated
    assert resumed.entries[0].validated


def test_format_three_profile_migrates_as_retargetable(tmp_path):
    payload = make_profile().to_dict()
    payload.pop("kernel_digest")
    payload["format_version"] = 3
    payload["kernel_version"] = "deberta-attention-forward-runtime-length-v3"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    profile = load_profile(path)
    registry = ProfileRegistry((profile,))

    assert profile.provenance["migrated_from_format"] == "3"
    assert profile.compatibility(make_hardware(), make_compiler()) == "retargetable"
    assert registry.resolve(make_hardware(), make_compiler(), make_workload()) is None
    assert "retargeting" in registry.explain_miss(make_hardware(), make_compiler(), make_workload())


def test_retarget_command_targets_the_profile(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("disentangled_flash.tune.run", captured.append)
    profile = tmp_path / "profile.json"

    main(["retarget", str(profile), "--passes", "training"])
    main(["--output", str(profile)])

    assert captured[0].retarget is True
    assert captured[0].output == profile
    assert captured[0].passes == ("training",)
    assert captured[1].retarget is False
    with pytest.raises(SystemExit, match="profile path"):
        main(["retarget"])


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
