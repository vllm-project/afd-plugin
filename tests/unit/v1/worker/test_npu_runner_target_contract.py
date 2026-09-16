# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only behavioral checks for native runner call boundaries."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_dummy_boundary():
    """Load the boundary methods without importing NPU-only extension modules."""
    source_path = Path("afd_plugin/v1/worker/npu/attention_model_runner.py")
    source = ast.parse(source_path.read_text())
    runner = next(node for node in source.body if isinstance(node, ast.ClassDef))
    methods = [
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_dummy_run", "_dummy_run_inference_mode"}
    ]

    class NativeRunner:
        def _dummy_run(self, *args, **kwargs):
            return self.record("native", args, kwargs)

    runner.bases = [ast.Name(id="NativeRunner", ctx=ast.Load())]
    runner.body = methods
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            runner,
        ],
        type_ignores=[],
    )
    namespace = {
        "NativeRunner": NativeRunner,
        "torch": SimpleNamespace(inference_mode=nullcontext),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace
    )
    return namespace[runner.name]()


@pytest.mark.parametrize("use_ubatching", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_dummy_forwards_native_noop_flag_and_restores_state(use_ubatching, fail):
    runner = _load_dummy_boundary()
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=use_ubatching),
    )
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = "stale"
    runner._afd_async_moe_ubatch_metadata = None
    calls = []

    def record(route, args, kwargs):
        calls.append((route, args, kwargs))
        assert runner._afd_is_graph_capturing
        if fail:
            raise RuntimeError("native failure")
        return "hidden", "sample"

    runner.record = record
    runner._dummy_run_with_ubatches = lambda *args, **kwargs: record(
        "ubatch", args, kwargs
    )
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            runner._dummy_run(8, is_graph_capturing=True, skip_gdn_state_update=True)
    else:
        assert runner._dummy_run(
            8, is_graph_capturing=True, skip_gdn_state_update=True
        ) == ("hidden", "sample")
    assert calls[0][0] == ("ubatch" if use_ubatching else "native")
    assert calls[0][1] == (8,)
    assert calls[0][2]["skip_gdn_state_update"] is True
    assert runner._afd_is_graph_capturing is False
    assert runner._afd_pending_metadata is None
