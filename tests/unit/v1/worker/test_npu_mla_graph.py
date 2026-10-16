# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

# Fake dependency modules in this file intentionally receive attributes at
# runtime before the plugin module under test imports them.
# mypy: disable-error-code="attr-defined"
import importlib
import sys
from importlib.util import find_spec
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
NPU_RUNTIME_MODULES = ("vllm", "vllm_ascend", "torch_npu")


def _reload_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> ModuleType:
    """Reload a module and restore both import caches after the test."""
    package_name, module_attribute = module_name.rsplit(".", 1)
    package = importlib.import_module(package_name)
    monkeypatch.setattr(package, module_attribute, None, raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def _preimport_real_module(module_name: str) -> None:
    """Import the real module before the fake sys.modules swap.

    ``_reload_module`` only restores state monkeypatch recorded: when the real
    module was never imported, teardown has no original sys.modules entry or
    parent-package attribute to restore, and the fresh fake-bound module leaks
    into later tests through the package attribute. Pre-importing guarantees a
    genuine original exists to restore. CPU-only environments use the fake
    path when the NPU runtime is unavailable; failures from an installed
    runtime remain visible.
    """
    if not all(find_spec(dependency) is not None for dependency in NPU_RUNTIME_MODULES):
        return
    importlib.import_module(module_name)


def test_preimport_real_module_surfaces_installed_runtime_failures(monkeypatch):
    def fail_import(_module_name):
        raise RuntimeError("ABI mismatch")

    monkeypatch.setattr(
        f"{__name__}.find_spec",
        lambda _dependency: SimpleNamespace(),
    )
    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _preimport_real_module("afd_plugin.v1.worker.npu.mla_graph")


def _load_mla_graph_module(monkeypatch):
    _preimport_real_module("afd_plugin.v1.worker.npu.mla_graph")
    fake_ascend = ModuleType("vllm_ascend")
    fake_ascend.__path__ = []
    fake_compilation = ModuleType("vllm_ascend.compilation")
    fake_compilation.__path__ = []
    fake_acl_graph = ModuleType("vllm_ascend.compilation.acl_graph")

    class GraphParams:
        def __init__(self, events, workspaces, handles, attn_params):
            self.events = events
            self.workspaces = workspaces
            self.handles = handles
            self.attn_params = attn_params

    fake_acl_graph.GraphParams = GraphParams
    monkeypatch.setitem(sys.modules, "vllm_ascend", fake_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.compilation",
        fake_compilation,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.compilation.acl_graph",
        fake_acl_graph,
    )

    module_name = "afd_plugin.v1.worker.npu.mla_graph"
    return _reload_module(monkeypatch, module_name)


def _graph_params(
    num_tokens,
    workspace,
    events,
    handles,
    attn_params,
):
    return SimpleNamespace(
        events={num_tokens: list(events)},
        workspaces={num_tokens: workspace},
        handles={num_tokens: list(handles)},
        attn_params={num_tokens: list(attn_params)},
    )


def test_merge_mla_graph_params_is_layer_major_ubatch_minor(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    stage_params = (
        _graph_params(
            4,
            workspace,
            ["e00", "e10"],
            ["h00", "h10"],
            ["p00", "p10"],
        ),
        _graph_params(
            4,
            workspace,
            ["e01", "e11"],
            ["h01", "h11"],
            ["p01", "p11"],
        ),
    )
    metadata = [
        {"layer0": "m00", "layer1": "m10"},
        {"layer0": "m01", "layer1": "m11"},
    ]

    merged_metadata, merged = mla_graph.merge_mla_graph_params(
        metadata,
        stage_params,
        4,
    )

    assert list(merged_metadata) == [
        ("layer0", 0),
        ("layer0", 1),
        ("layer1", 0),
        ("layer1", 1),
    ]
    assert list(merged_metadata.values()) == ["m00", "m01", "m10", "m11"]
    assert merged.events[4] == ["e00", "e01", "e10", "e11"]
    assert merged.handles[4] == ["h00", "h01", "h10", "h11"]
    assert merged.attn_params[4] == ["p00", "p01", "p10", "p11"]
    assert merged.workspaces[4] is workspace


def test_merge_mla_graph_params_requires_two_metadata_stages(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
        _graph_params(4, workspace, ["e1"], ["h1"], ["p1"]),
    )

    with pytest.raises(RuntimeError, match="exactly two metadata stages"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_requires_matching_layer_order(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(
            4,
            workspace,
            ["e00", "e10"],
            ["h00", "h10"],
            ["p00", "p10"],
        ),
        _graph_params(
            4,
            workspace,
            ["e01", "e11"],
            ["h01", "h11"],
            ["p01", "p11"],
        ),
    )

    with pytest.raises(RuntimeError, match="layer order differs"):
        mla_graph.merge_mla_graph_params(
            [
                {"layer0": "m00", "layer1": "m10"},
                {"layer1": "m11", "layer0": "m01"},
            ],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_requires_shared_workspace(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    graph_params = (
        _graph_params(4, object(), ["e0"], ["h0"], ["p0"]),
        _graph_params(4, object(), ["e1"], ["h1"], ["p1"]),
    )

    with pytest.raises(RuntimeError, match="shared FIA workspace"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}, {"layer0": "m1"}],
            graph_params,
            4,
        )


def test_merge_mla_graph_params_rejects_record_count_mismatch(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    workspace = object()
    graph_params = (
        _graph_params(4, workspace, ["e0"], ["h0"], ["p0"]),
        _graph_params(4, workspace, [], [], []),
    )

    with pytest.raises(RuntimeError, match="record count mismatch"):
        mla_graph.merge_mla_graph_params(
            [{"layer0": "m0"}, {"layer0": "m1"}],
            graph_params,
            4,
        )


def test_override_mla_graph_params_restores_context_after_error(monkeypatch):
    mla_graph = _load_mla_graph_module(monkeypatch)
    original_metadata = [{"layer0": "old"}]
    original_kwargs = {"afd_metadata": object()}
    context = SimpleNamespace(
        attn_metadata=original_metadata,
        additional_kwargs=original_kwargs,
    )
    merged_metadata = {("layer0", 0): "new"}
    merged_params = object()

    with (
        pytest.raises(RuntimeError, match="update failed"),
        mla_graph.override_mla_graph_params(
            context,
            merged_metadata,
            merged_params,
        ),
    ):
        assert context.attn_metadata is merged_metadata
        assert context.additional_kwargs is not original_kwargs
        assert (
            context.additional_kwargs["afd_metadata"]
            is (original_kwargs["afd_metadata"])
        )
        assert context.additional_kwargs["afd_mla_graph_params"] is merged_params
        raise RuntimeError("update failed")

    assert context.attn_metadata is original_metadata
    assert context.additional_kwargs is original_kwargs
