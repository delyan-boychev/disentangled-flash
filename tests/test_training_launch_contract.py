"""Check the shared forward launch ABI even on hosts without CUDA/Triton."""

import ast
from pathlib import Path


def test_training_forward_matches_shared_kernel_signature():
    package = Path(__file__).parents[1] / "src" / "disentangled_flash"
    kernel = ast.parse((package / "kernel.py").read_text())
    training = ast.parse((package / "training" / "_kernels.py").read_text())
    forward = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.FunctionDef) and node.name == "_deberta_attention_forward_kernel"
    )
    launcher = next(
        node
        for node in ast.walk(training)
        if isinstance(node, ast.FunctionDef) and node.name == "_training_attention_forward_op"
    )
    calls = [
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "SCORE_SCALE_LOG2" for keyword in node.keywords)
    ]
    assert len(calls) == 1
    signature = {arg.arg for arg in forward.args.args}
    keywords = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    assert "USE_PADDING_MASK" in signature
    assert "USE_PADDING_MASK" in keywords
    assert {"POSITION_OFFSET", "LENGTH_REGIME"} <= signature
    assert {"POSITION_OFFSET", "LENGTH_REGIME"} <= keywords
    assert "HAS_PADDING" not in keywords
    assert keywords <= signature


def test_training_forward_specializations_remain_in_autotune_family_key():
    path = Path(__file__).parents[1] / "src" / "disentangled_flash" / "kernel.py"
    module = ast.parse(path.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "AUTOTUNE_SPECIALIZATION_KEY"
            for target in node.targets
        )
    )
    values = {element.value for element in assignment.value.elts}

    assert {"STORE_LSE", "PHYSICAL_PAIRS", "USE_PADDING_MASK"} <= values
    assert not {"SEQUENCE_LENGTH", "BATCH_SIZE", "NUM_HEADS", "ACTIVE_SLOTS"} & values


def test_backward_kernels_keep_their_own_padding_argument():
    path = Path(__file__).parents[1] / "src/disentangled_flash/training/_kernels.py"
    module = ast.parse(path.read_text())
    for name in ("_backward_dq_dc_kernel", "_backward_dkv_dt_kernel"):
        kernel = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert "HAS_PADDING" in {arg.arg for arg in kernel.args.args}


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_dropout_launch_options_match_every_dropout_kernel_signature():
    package = Path(__file__).parents[1] / "src" / "disentangled_flash"
    kernel = ast.parse((package / "kernel.py").read_text())
    training = ast.parse((package / "training" / "_kernels.py").read_text())
    options = _function(training, "_dropout_launch_options")
    returned = next(node for node in ast.walk(options) if isinstance(node, ast.Dict))
    option_names = {key.value for key in returned.keys}

    assert option_names == {"dropout_seed", "DROPOUT_P", "DROPOUT_SCALE", "HAS_DROPOUT"}
    kernels = [
        _function(kernel, "_deberta_attention_forward_kernel"),
        _function(kernel, "_deberta_attention_packed_forward_kernel"),
        _function(training, "_backward_dq_dc_kernel"),
        _function(training, "_backward_dkv_dt_kernel"),
        _function(training, "_packed_backward_dq_dc_kernel"),
        _function(training, "_packed_backward_dkv_dt_kernel"),
    ]
    for function in kernels:
        assert option_names <= {arg.arg for arg in function.args.args}, function.name


def test_every_training_kernel_launch_forwards_dropout_options():
    path = Path(__file__).parents[1] / "src/disentangled_flash/training/_kernels.py"
    module = ast.parse(path.read_text())
    for name in (
        "_training_attention_forward_op",
        "_training_attention_backward_op",
        "_training_attention_packed_forward_op",
        "_training_attention_packed_backward_op",
    ):
        launches = [
            node
            for node in ast.walk(_function(module, name))
            if isinstance(node, ast.Call)
            and any(keyword.arg == "STRICT_FP32" for keyword in node.keywords)
        ]
        expected = 1 if name.endswith("forward_op") else 2
        assert len(launches) == expected, name
        for launch in launches:
            assert any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Call)
                and getattr(keyword.value.func, "id", "") == "_dropout_launch_options"
                for keyword in launch.keywords
            ), name


def test_training_autotune_keys_separate_dropout_schedules():
    path = Path(__file__).parents[1] / "src/disentangled_flash/training/_kernels.py"
    source = path.read_text()

    assert '_TRAINING_FORWARD_EXTRA_KEY = ("HAS_DROPOUT",)' in source
    module = ast.parse(source)
    backward_key = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_BACKWARD_AUTOTUNE_KEY"
            for target in node.targets
        )
    )
    assert "HAS_DROPOUT" in {element.value for element in backward_key.value.elts}
