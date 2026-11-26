# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host contracts for native graph tasks captured by two AFD microbatches."""

import sys
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType

import pytest

from tests.unit.fakes.async_ffn import load_host_module


@pytest.fixture
def graph_runtime(monkeypatch):
    stage: ContextVar[int] = ContextVar("test_ubatch")
    ubatching = ModuleType("vllm.v1.worker.ubatching")
    vars(ubatching)["dbo_current_ubatch_id"] = stage.get
    updatable = ModuleType("vllm_ascend.compilation.updatable_graph")
    vars(updatable)["_ACTIVE_GRAPH"] = ContextVar("test_capture_graph", default=None)
    monkeypatch.setitem(sys.modules, ubatching.__name__, ubatching)
    monkeypatch.setitem(sys.modules, updatable.__name__, updatable)
    return load_host_module(monkeypatch, "afd_plugin.v1.worker.npu.mla_graph"), stage


@dataclass(frozen=True)
class _NativeProvider:
    layer_name: str

    def resolve(self, metadata):
        return metadata[self.layer_name]


class _NativeGraph:
    def __init__(self):
        self.providers = []

    def register_task(self, operation, kwargs, provider):
        self.providers.append(provider)

    def resolve(self, metadata):
        # UpdatableGraph groups captured records by provider identity.
        values = {provider: provider.resolve(metadata) for provider in self.providers}
        return [values[provider] for provider in self.providers]


class _NativeManager:
    def __init__(self, graph):
        self.graph = graph

    def run_fullgraph(self, desc):
        return self._updatable_graph_replay(desc, {"layer": "parent"})

    def _updatable_graph_replay(self, desc, attn_metadata):
        if desc == "failed":
            raise RuntimeError("update failed")
        return self.graph.resolve(attn_metadata)


def test_equal_shape_stages_refresh_each_captured_task(graph_runtime):
    runtime, stage = graph_runtime
    graph = _NativeGraph()
    manager = _NativeManager(graph)
    # Interleave capture order independently of the order of live stage metadata.
    with runtime.use_ubatch_graph_registration(graph):
        for stage_id in (1, 0):
            token = stage.set(stage_id)
            try:
                graph.register_task(None, {}, _NativeProvider("layer"))
            finally:
                stage.reset(token)

    for live_lengths in ((17, 41), (30, 52)):
        metadata = [{"layer": length} for length in live_lengths]
        with runtime.use_ubatch_replay_metadata(manager, metadata):
            assert manager.run_fullgraph("same-shape") == list(reversed(live_lengths))


@pytest.mark.parametrize("instance_override", [False, True])
def test_replay_failure_restores_the_native_binding(
    graph_runtime, instance_override, monkeypatch
):
    runtime, _ = graph_runtime
    graph = _NativeGraph()
    graph.register_task(None, {}, _NativeProvider("layer"))
    manager = _NativeManager(graph)
    if instance_override:
        original_replay = manager._updatable_graph_replay

        def prior_override(desc, attn_metadata):
            return (*original_replay(desc, attn_metadata), "instance")

        monkeypatch.setattr(manager, "_updatable_graph_replay", prior_override)

    expected = manager.run_fullgraph("one-stage")
    with (
        pytest.raises(RuntimeError, match="update failed"),
        runtime.use_ubatch_replay_metadata(manager, [{"layer": 17}, {"layer": 41}]),
    ):
        manager.run_fullgraph("failed")
    assert manager.run_fullgraph("one-stage") == expected
