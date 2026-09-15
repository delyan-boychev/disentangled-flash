"""Check Triton launch ABIs on hosts without CUDA or Triton."""

import ast
from pathlib import Path


def test_configured_launcher_passes_all_runtime_strides():
    path = Path(__file__).parents[1] / "src" / "disentangled_flash" / "kernel.py"
    module = ast.parse(path.read_text())
    kernel = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_deberta_attention_forward_kernel"
    )
    launcher = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_deberta_attention_configured"
    )
    call = next(
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "ACTIVE_SLOTS" for keyword in node.keywords)
    )

    runtime_argument_count = next(
        index for index, argument in enumerate(kernel.args.args) if argument.arg == "ACTIVE_SLOTS"
    )
    assert runtime_argument_count == 20
    assert len(call.args) == runtime_argument_count


def test_packed_configured_launcher_matches_kernel_runtime_arguments():
    path = Path(__file__).parents[1] / "src" / "disentangled_flash" / "kernel.py"
    module = ast.parse(path.read_text())
    kernel = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_deberta_attention_packed_forward_kernel"
    )
    launcher = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_launch_deberta_attention_packed_configured"
    )
    call = next(
        node
        for node in ast.walk(launcher)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "ACTIVE_SLOTS" for keyword in node.keywords)
    )

    runtime_argument_count = next(
        index for index, argument in enumerate(kernel.args.args) if argument.arg == "ACTIVE_SLOTS"
    )
    assert runtime_argument_count == 17
    assert len(call.args) == runtime_argument_count
