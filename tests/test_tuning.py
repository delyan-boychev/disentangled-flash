import json

import pytest
import torch

from disentangled_flash.kernel import AUTOTUNE_SPECIALIZATION_KEY
from disentangled_flash.tune import (
    PRESETS,
    TuningCase,
    _cases,
    _dropout_modes,
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
    DeviceResources,
    HardwareSpec,
    KernelConfig,
    KernelProfile,
    KernelTuningOptions,
    ProfileEntry,
    ProfileRegistry,
    WorkloadKey,
    conservative_candidates,
    fits_device,
    hardware_safe_candidates,
    heuristic_config,
    load_bundled_profiles,
    load_profile,
    merge_profile_entry,
    prepare_profile_retarget,
    save_profile,
    search_neighborhood,
    tuning_dtype,
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


def test_exhaustive_covers_supported_deberta_variants_and_all_kernel_phases():
    assert set(PRESETS) == {"quick", "standard", "exhaustive"}
    exhaustive = PRESETS["exhaustive"]
    assert exhaustive["head_dims"] == (64,)
    assert exhaustive["batch_heads"] == (8, 32)
    assert exhaustive["layouts"] == ("padded", "packed")
    assert exhaustive["lengths"][-1] == 8192
    assert exhaustive["search"] == "full"
    args = build_parser().parse_args(["--output", "profile.json", "--preset", "exhaustive"])
    cases = list(_cases(args))
    # 9 lengths x 2 occupancies x (half, FP32 strict, FP32 fast) x 3 layouts.
    assert len(cases) == 162
    assert len(cases) + 3 * len(cases) * len(_dropout_modes(args)) == 1134


def test_standard_is_the_default_and_stays_small():
    args = build_parser().parse_args(["--output", "profile.json"])
    assert args.preset == "standard"
    assert PRESETS["standard"]["search"] == "neighborhood"
    cases = list(_cases(args))
    # 4 lengths x 3 layouts, BF16 only.
    assert len(cases) == 12
    assert len(cases) + 3 * len(cases) * len(_dropout_modes(args)) == 84


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


def test_half_precision_family_shares_fp16_and_bf16_workloads():
    fp16 = make_workload()
    bf16 = WorkloadKey(**{**_workload_kwargs(fp16), "dtype": "bfloat16"})
    fp32 = WorkloadKey(**{**_workload_kwargs(fp16), "dtype": "float32"})

    assert tuning_dtype("float16") == tuning_dtype("bfloat16") == "half"
    assert fp16 == bf16
    assert fp16.to_dict()["dtype"] == "half"
    assert fp32 != fp16
    with pytest.raises(ValueError, match="dtype"):
        tuning_dtype("float64")


def _workload_kwargs(workload):
    return {
        "sequence_length": workload.sequence_length,
        "head_dim": workload.head_dim,
        "batch_heads": workload.batch_heads,
        "active_slots": workload.active_slots,
        "dtype": workload.dtype,
        "has_c2p": workload.has_c2p,
        "has_p2c": workload.has_p2c,
        "fp32_precision": workload.fp32_precision,
        "layout": workload.layout,
        "uses_padding_mask": workload.uses_padding_mask,
        "phase": workload.phase,
        "has_dropout": workload.has_dropout,
    }


def test_dropout_is_a_separate_training_workload():
    plain = make_workload(phase="backward_dq")
    dropped = WorkloadKey(**{**_workload_kwargs(plain), "has_dropout": True})

    assert plain != dropped
    assert WorkloadKey.from_dict(dropped.to_dict()) == dropped
    legacy = plain.to_dict()
    legacy.pop("has_dropout")
    assert WorkloadKey.from_dict(legacy).has_dropout is False
    with pytest.raises(ValueError, match="inference workloads cannot use attention dropout"):
        WorkloadKey(**{**_workload_kwargs(make_workload()), "has_dropout": True})


def test_legacy_fp16_and_bf16_twins_collapse_on_load(tmp_path):
    payload = make_profile().to_dict()
    fp16_entry = payload["entries"][0]
    fp16_entry["workload"]["dtype"] = "float16"
    bf16_entry = json.loads(json.dumps(fp16_entry))
    bf16_entry["workload"]["dtype"] = "bfloat16"
    bf16_entry["config"] = KernelConfig(32, 32, 4).to_dict()
    payload["entries"] = [fp16_entry, bf16_entry]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    profile = load_profile(path)

    assert len(profile.entries) == 1
    assert profile.entries[0].workload.dtype == "half"
    assert profile.entries[0].config == KernelConfig.from_dict(fp16_entry["config"])

    payload["entries"] = [fp16_entry, json.loads(json.dumps(fp16_entry))]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_profile(path)


def test_cases_measure_one_dtype_per_precision_family():
    args = build_parser().parse_args(
        ["--output", "p.json", "--dtypes", "float16,bfloat16,float32", "--lengths", "64"]
    )

    assert {case.dtype for case in _cases(args)} == {"float16", "float32"}


def test_dropout_modes_follow_cli_and_reject_unknown_values():
    args = build_parser().parse_args(["--output", "p.json", "--dropout", "on"])
    assert _dropout_modes(args) == (True,)
    assert _dropout_modes(
        build_parser().parse_args(["--output", "p.json", "--preset", "quick"])
    ) == (False,)
    args = build_parser().parse_args(["--output", "p.json", "--dropout", "sometimes"])
    with pytest.raises(ValueError, match="dropout modes"):
        _dropout_modes(args)


def test_fp32_searches_only_conservative_single_stage_schedules():
    for phase, candidates, limit in (
        ("inference", DEFAULT_KERNEL_CONFIGS, 64 * 64),
        ("training_forward", DEFAULT_KERNEL_CONFIGS, 64 * 64),
        ("backward_dq", DEFAULT_DQ_KERNEL_CONFIGS, 2048),
        ("backward_dkv", DEFAULT_DKV_KERNEL_CONFIGS, 2048),
    ):
        fp32 = conservative_candidates(candidates, dtype="float32", phase=phase)
        assert fp32
        assert all(config.num_stages == 1 for config in fp32)
        assert all(config.block_m * config.block_n <= limit for config in fp32)
        assert conservative_candidates(candidates, dtype="bfloat16", phase=phase) == candidates

    assert conservative_candidates(
        (KernelConfig(128, 64, 4),), dtype="float32", phase="inference"
    ) == (KernelConfig(32, 32, 4),)


H200 = DeviceResources(
    shared_memory_per_block=232448, multiprocessor_count=132, compute_capability=(9, 0)
)
SMALL_GPU = DeviceResources(
    shared_memory_per_block=65536, multiprocessor_count=40, compute_capability=(7, 5)
)
MINIMUM_GPU = DeviceResources(
    shared_memory_per_block=48 * 1024, multiprocessor_count=16, compute_capability=(7, 0)
)


def test_shared_memory_filter_drops_large_tiles_on_small_gpus():
    large = KernelConfig(128, 64, 4)
    options = {"phase": "inference", "head_dim": 128, "dtype": "bfloat16"}

    assert fits_device(large, resources=H200, **options)
    assert not fits_device(large, resources=SMALL_GPU, **options)
    safe = hardware_safe_candidates(DEFAULT_KERNEL_CONFIGS, resources=SMALL_GPU, **options)
    assert large not in safe
    assert all(fits_device(config, resources=SMALL_GPU, **options) for config in safe)


def test_bundled_profile_winners_all_fit_the_h200_estimate():
    profile = load_bundled_profiles()[0]

    assert all(
        fits_device(
            entry.config,
            phase=entry.workload.phase,
            head_dim=entry.workload.head_dim,
            dtype=entry.workload.dtype,
            resources=H200,
        )
        for entry in profile.entries
    )


@pytest.mark.parametrize("resources", [H200, SMALL_GPU])
@pytest.mark.parametrize("phase", ["inference", "training_forward", "backward_dq", "backward_dkv"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("length", [64, 128, 512, 2048, 8192])
def test_heuristic_always_returns_a_safe_config(resources, phase, dtype, length):
    for head_dim in (32, 64, 128):
        config = heuristic_config(
            phase=phase,
            sequence_length=length,
            head_dim=head_dim,
            dtype=dtype,
            batch_heads=384,
            resources=resources,
        )
        assert config in hardware_safe_candidates(
            (config,), phase=phase, head_dim=head_dim, dtype=dtype, resources=resources
        )
        assert config.num_stages <= 2


def test_heuristic_splits_tiles_when_the_launch_underfills_the_gpu():
    options = {
        "phase": "inference",
        "sequence_length": 8192,
        "head_dim": 64,
        "dtype": "bfloat16",
        "resources": H200,
    }
    saturated = heuristic_config(batch_heads=24, **options)
    underfilled = heuristic_config(batch_heads=1, **options)

    assert saturated == KernelConfig(64, 64, 4, 2)
    assert underfilled.block_m < saturated.block_m


def test_search_neighborhood_starts_at_the_center():
    center = KernelConfig(32, 64, 4)
    neighborhood = search_neighborhood(center, DEFAULT_KERNEL_CONFIGS)

    assert neighborhood[0] == center
    assert len(neighborhood) == 4
    assert KernelConfig(32, 64, 4, 2) in neighborhood
    assert KernelConfig(128, 64, 4) not in neighborhood


def test_heuristic_mode_rejects_candidate_lists():
    with pytest.raises(ValueError, match="candidates are only valid"):
        KernelTuningOptions(mode="heuristic", candidates=(KernelConfig(32, 32, 4),))
    assert KernelTuningOptions(mode="heuristic").mode == "heuristic"


def test_fingerprint_ignores_module_classes_and_imports():
    from disentangled_flash.tuning import _canonical_code, _launch_code

    base = "import torch\n\ndef launch(x):\n    return x + 1\n"
    edited = (
        "import torch\nimport math\n\nclass Wrapper:\n    pass\n\n"
        "def launch(x):\n    # tweak\n    return x + 1\n"
    )
    changed = "def launch(x):\n    return x + 2\n"

    assert _canonical_code(_launch_code(base)) == _canonical_code(_launch_code(edited))
    assert _canonical_code(_launch_code(base)) != _canonical_code(_launch_code(changed))


def _fake_device(monkeypatch, resources=H200):
    from disentangled_flash import tuning

    monkeypatch.setattr(
        tuning.DeviceResources, "current", classmethod(lambda cls, d=None: resources)
    )
    monkeypatch.setattr(
        tuning.HardwareSpec, "current", classmethod(lambda cls, d=None: make_hardware())
    )
    monkeypatch.setattr(tuning.CompilerSpec, "current", classmethod(lambda cls: make_compiler()))
    tuning._resolve_launch_config.cache_clear()


def _fields(workload):
    return (
        workload.sequence_length,
        workload.head_dim,
        workload.batch_heads,
        workload.active_slots,
        workload.dtype,
        workload.has_c2p,
        workload.has_p2c,
        workload.fp32_precision,
        workload.layout,
        workload.uses_padding_mask,
        workload.phase,
        workload.has_dropout,
    )


def test_resolver_prefers_profiles_and_falls_back_to_the_heuristic(monkeypatch):
    from disentangled_flash.tuning import register_profile_registry, resolve_launch_config

    _fake_device(monkeypatch)
    saved = KernelConfig(64, 64, 4)
    key = register_profile_registry(ProfileRegistry((make_profile(saved),)))
    hit = _fields(make_workload())
    miss = _fields(make_workload(512))

    assert resolve_launch_config(key, "auto", 0, hit, 12) == (64, 64, 4, 1)
    heuristic = heuristic_config(
        phase="inference",
        sequence_length=512,
        head_dim=64,
        dtype="float16",
        batch_heads=12,
        resources=H200,
    )
    expected = (heuristic.block_m, heuristic.block_n, heuristic.num_warps, heuristic.num_stages)
    assert resolve_launch_config(key, "auto", 0, miss, 12) == expected
    assert resolve_launch_config(key, "heuristic", 0, hit, 12) != (64, 64, 4, 1)
    with pytest.raises(RuntimeError, match="no validated profile entry"):
        resolve_launch_config(key, "profile_only", 0, miss, 12)


def test_length_family_matches_the_profile_families():
    from disentangled_flash.tuning import length_family, occupancy_family

    for length in (1, 64, 65, 383, 384, 8192, 20000):
        assert length_family(length) == tuning_sequence_length(length)
    assert occupancy_family(8) == 8
    assert occupancy_family(1536) == 32


def test_config_resolution_compiles_with_fullgraph_and_one_graph_per_family(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    from disentangled_flash.tuning import (
        length_family,
        occupancy_family,
        register_profile_registry,
        resolve_launch_config,
    )

    _fake_device(monkeypatch)

    class Launcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.registry_key = register_profile_registry(ProfileRegistry())

        def forward(self, hidden):
            fields = (
                length_family(hidden.size(1)),
                64,
                occupancy_family(hidden.size(0) * 12),
                128,
                "float16",
                True,
                True,
                "strict",
                "padded",
                True,
                "inference",
                False,
            )
            block_m, block_n, _, _ = resolve_launch_config(
                self.registry_key, "auto", 0, fields, None
            )
            return hidden * block_m + block_n

    launcher = Launcher()
    counter = CompileCounter()
    torch._dynamo.reset()
    compiled = torch.compile(launcher, backend=counter, fullgraph=True, dynamic=True)
    for batch, length in ((2, 100), (3, 120), (4, 700)):
        expected = launcher(torch.ones(batch, length))
        torch.testing.assert_close(compiled(torch.ones(batch, length)), expected)

    # 100 and 120 share the 128 family and the batch never recompiles; 700 is new.
    assert counter.frame_count == 2


@pytest.mark.parametrize("phase", ["inference", "training_forward", "backward_dq", "backward_dkv"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_heuristic_floor_fits_the_smallest_cuda_gpu(phase, dtype):
    from disentangled_flash.tuning import HEURISTIC_FLOOR

    assert fits_device(
        HEURISTIC_FLOOR, phase=phase, head_dim=128, dtype=dtype, resources=MINIMUM_GPU
    )
    for length in (64, 512, 8192):
        config = heuristic_config(
            phase=phase,
            sequence_length=length,
            head_dim=128,
            dtype=dtype,
            batch_heads=384,
            resources=MINIMUM_GPU,
        )
        assert fits_device(config, phase=phase, head_dim=128, dtype=dtype, resources=MINIMUM_GPU)


def test_heuristic_pipelines_only_where_the_hardware_supports_it():
    def forward(dtype, resources, length=2048):
        return heuristic_config(
            phase="inference",
            sequence_length=length,
            head_dim=64,
            dtype=dtype,
            batch_heads=96,
            resources=resources,
        )

    assert forward("bfloat16", H200).num_stages == 2
    assert forward("bfloat16", SMALL_GPU).num_stages == 1  # no async copies before sm80
    assert forward("float32", H200).num_stages == 1
    assert forward("bfloat16", H200, length=128).num_stages == 1
    backward = heuristic_config(
        phase="backward_dq",
        sequence_length=2048,
        head_dim=64,
        dtype="bfloat16",
        batch_heads=96,
        resources=H200,
    )
    assert backward.num_stages == 1


def test_heuristic_matches_the_h200_profile_for_saturated_inference():
    profile = load_bundled_profiles()[0]
    for length in (512, 1024, 2048, 4096, 8192):
        entry = next(
            entry
            for entry in profile.entries
            if entry.workload.phase == "inference"
            and entry.workload.sequence_length == length
            and entry.workload.dtype == "half"
            and entry.workload.batch_heads == 32
            and entry.workload.layout == "padded"
            and not entry.workload.uses_padding_mask
        )
        config = heuristic_config(
            phase="inference",
            sequence_length=length,
            head_dim=64,
            dtype="bfloat16",
            batch_heads=16384 // length * 12,
            resources=H200,
        )
        assert config == entry.config


@pytest.mark.parametrize(
    ("phase", "length", "expected"),
    [
        # Best or near-best in H200 sweeps at the benchmark shapes.
        ("training_forward", 128, KernelConfig(64, 64, 4, 2)),
        ("training_forward", 512, KernelConfig(64, 64, 4, 2)),
        ("backward_dq", 128, KernelConfig(64, 32, 4)),
        ("backward_dq", 512, KernelConfig(32, 32, 4)),
        ("backward_dkv", 128, KernelConfig(16, 16, 4)),
        ("backward_dkv", 512, KernelConfig(16, 16, 4)),
    ],
)
def test_heuristic_training_configs_at_saturated_shapes(phase, length, expected):
    config = heuristic_config(
        phase=phase,
        sequence_length=length,
        head_dim=64,
        dtype="bfloat16",
        batch_heads=16384 // length * 12,
        resources=H200,
    )
    assert config == expected


def test_padded_position_columns_are_never_read():
    case = TuningCase(128, 64, 2, "float32", has_c2p=True, has_p2c=True, fp32_precision="strict")
    arguments, _ = _make_inputs(case, torch.device("cpu"))
    c2p, p2c = arguments[3], arguments[4]
    real = 2 * 128 - 1

    assert c2p.size(-1) == 256 and arguments[10] == 256
    expected = _reference(arguments)
    noisy = list(arguments)
    noisy[3] = c2p.clone()
    noisy[4] = p2c.clone()
    noisy[3][..., real:] = 1e6
    noisy[4][..., real:] = -1e6
    torch.testing.assert_close(_reference(tuple(noisy)), expected)
