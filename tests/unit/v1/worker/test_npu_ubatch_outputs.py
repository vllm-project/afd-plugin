# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tensor/alias contracts without importing the native NPU worker stack."""

from __future__ import annotations
import __future__

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def ubatch_outputs():
    source = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/compat/patches/npu/mrv2_ubatching.py"
    )
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"cat_ubatch_outputs", "use_npu_ubatch_output_merge"}
    ]
    # Only function bindings are simulated; tensor operations use real torch.
    bindings = SimpleNamespace(
        slice_model_inputs=object(), merge_ubatch_outputs=object()
    )
    namespace = {
        "torch": torch,
        "contextmanager": contextmanager,
        "ubatch_utils": bindings,
    }
    exec(
        compile(tree, str(source), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return SimpleNamespace(
        cat=namespace["cat_ubatch_outputs"],
        scope=namespace["use_npu_ubatch_output_merge"],
        bindings=bindings,
    )


@pytest.mark.parametrize("auxiliary", [False, True])
def test_merge_preserves_ascend_nested_auxiliary_outputs(ubatch_outputs, auxiliary):
    first, second = torch.arange(5), torch.arange(5, 12)
    outputs = (
        [(first, [first + 10, first + 20]), (second, [second + 10, second + 20])]
        if auxiliary
        else [first, second]
    )
    merged = ubatch_outputs.cat(outputs)
    if auxiliary:
        hidden, aux = merged
        torch.testing.assert_close(hidden, torch.arange(12))
        assert isinstance(aux, list)
        torch.testing.assert_close(aux[0], torch.arange(12) + 10)
        torch.testing.assert_close(aux[1], torch.arange(12) + 20)
    else:
        torch.testing.assert_close(merged, torch.arange(12))


@pytest.mark.parametrize("fail", [False, True])
def test_output_merge_scope_restores_alias_without_touching_slicing(
    ubatch_outputs, fail
):
    bindings = ubatch_outputs.bindings
    original_slice = bindings.slice_model_inputs
    original_merge = bindings.merge_ubatch_outputs

    def exercise():
        with ubatch_outputs.scope():
            assert bindings.slice_model_inputs is original_slice
            assert bindings.merge_ubatch_outputs is not original_merge
            torch.testing.assert_close(
                bindings.merge_ubatch_outputs([torch.arange(5), torch.arange(5, 12)]),
                torch.arange(12),
            )
            if fail:
                raise RuntimeError("model failed")

    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            exercise()
    else:
        exercise()
    assert bindings.slice_model_inputs is original_slice
    assert bindings.merge_ubatch_outputs is original_merge
