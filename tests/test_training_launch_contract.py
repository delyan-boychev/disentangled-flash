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
