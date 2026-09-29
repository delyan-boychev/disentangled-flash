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


def test_padded_kernels_read_relative_tables_through_strides():
    package = Path(__file__).parents[1] / "src" / "disentangled_flash"
    kernel = ast.parse((package / "kernel.py").read_text())
    training = ast.parse((package / "training" / "_kernels.py").read_text())
    strides = {"stride_rb", "stride_rh", "stride_rl"}
    for function in (
        _function(kernel, "_deberta_attention_forward_kernel"),
        _function(training, "_backward_dq_dc_kernel"),
        _function(training, "_backward_dkv_dt_kernel"),
    ):
        assert strides <= {arg.arg for arg in function.args.args}, function.name
        source = ast.unparse(function)
        assert "* ACTIVE_SLOTS" not in source, function.name


def test_head_major_gradient_gemms_match_the_batched_reduction():
    import torch

    batch, heads, length, head_dim, slots = 2, 3, 4, 8, 16
    query = torch.randn(batch, heads, length, head_dim, dtype=torch.float64)
    table = torch.randn(heads, slots, head_dim, dtype=torch.float64)
    grad = torch.randn(heads, batch, length, slots, dtype=torch.float64)
    grad_bhlr = grad.permute(1, 0, 2, 3)

    rows = grad.flatten(1, 2)
    head_rows = query.permute(1, 0, 2, 3).reshape(heads, batch * length, head_dim)
    grad_query = torch.bmm(rows, table).view(heads, batch, length, head_dim).permute(1, 0, 2, 3)
    grad_table = torch.bmm(rows.transpose(1, 2), head_rows)

    torch.testing.assert_close(grad_query, torch.matmul(grad_bhlr, table))
    torch.testing.assert_close(
        grad_table, torch.matmul(grad_bhlr.transpose(-1, -2), query).sum(dim=0)
    )


def _unpacked_names(function: ast.FunctionDef, source_name: str) -> list[str]:
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, (ast.Name, ast.Attribute))
            and ast.unparse(node.value) == source_name
            and isinstance(node.targets[0], ast.Tuple)
        ):
            return [element.id for element in node.targets[0].elts]
    raise AssertionError(f"{function.name} does not unpack {source_name}")


def test_autograd_contexts_match_op_signatures():
    path = Path(__file__).parents[1] / "src/disentangled_flash/training/_kernels.py"
    module = ast.parse(path.read_text())
    for op, setup, backward, backward_op in (
        (
            "_training_attention_forward_op",
            "_setup_training_attention_context",
            "_training_attention_autograd_backward",
            "_training_attention_backward_op",
        ),
        (
            "_training_attention_packed_forward_op",
            "_setup_training_attention_packed_context",
            "_training_attention_packed_autograd_backward",
            "_training_attention_packed_backward_op",
        ),
    ):
        signature = [arg.arg for arg in _function(module, op).args.args]
        assert _unpacked_names(_function(module, setup), "inputs") == signature, op
        saved = next(
            node
            for node in ast.walk(_function(module, setup))
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "save_for_backward"
        )
        saved_names = [ast.unparse(arg) for arg in saved.args]
        unpacked = _unpacked_names(_function(module, backward), "ctx.saved_tensors")
        assert [name.replace("attention_output", "output") for name in saved_names] == unpacked, op
        call = next(
            node
            for node in ast.walk(_function(module, backward))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == backward_op
        )
        # *ctx.dq_config and *ctx.dkv_config each expand to four launch values.
        passed = sum(4 if isinstance(arg, ast.Starred) else 1 for arg in call.args)
        assert passed == len(_function(module, backward_op).args.args), backward_op


def test_backward_kernels_accept_and_receive_unique_slots():
    path = Path(__file__).parents[1] / "src/disentangled_flash/training/_kernels.py"
    module = ast.parse(path.read_text())
    for name in (
        "_backward_dq_dc_kernel",
        "_backward_dkv_dt_kernel",
        "_packed_backward_dq_dc_kernel",
        "_packed_backward_dkv_dt_kernel",
    ):
        assert "UNIQUE_SLOTS" in {arg.arg for arg in _function(module, name).args.args}, name
    for name in ("_training_attention_backward_op", "_training_attention_packed_backward_op"):
        launches = [
            node
            for node in ast.walk(_function(module, name))
            if isinstance(node, ast.Call)
            and any(keyword.arg == "STRICT_FP32" for keyword in node.keywords)
        ]
        assert len(launches) == 2
        for launch in launches:
            assert any(keyword.arg == "UNIQUE_SLOTS" for keyword in launch.keywords), name
