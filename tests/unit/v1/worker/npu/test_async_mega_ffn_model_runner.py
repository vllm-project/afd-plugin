# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model weight handoff and execution for the dedicated Mega FFN runner."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import load_host_module


@pytest.fixture
def runner_type(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    class ModelOwner:
        model: SimpleNamespace

        def get_model(self):
            return self.model

    # Model ownership is the only base-runner dependency in these host tests.
    base = ModuleType("afd_plugin.v1.worker.npu.ffn_model_runner")
    monkeypatch.setattr(base, "AFDNPUFFNModelRunner", ModelOwner, raising=False)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    return load_host_module(
        monkeypatch, "afd_plugin.v1.worker.npu.async_mega_ffn_model_runner"
    ).MegaNPUFFNModelRunner


def test_mega_ffn_runner_prepares_layers_before_connector_init(runner_type):
    from afd_plugin.connectors.npu.async_mega import CAMAsyncMegaAFDConnector

    events = []

    class Connector(CAMAsyncMegaAFDConnector):
        is_initialized = False
        role_rank = 1

        def prepare_layers(self, layers):
            events.append(("layers", layers))

        def init_afd_connector(self, *, preparation_error=None):
            assert preparation_error is None
            events.append(("init", None))

    runner = object.__new__(runner_type)
    runner.connector = object.__new__(Connector)
    layers = {3: (["w1"], ["w2"], ["s1"], ["s2"])}

    def weights(rank):
        assert rank == 1
        return layers

    runner.model = SimpleNamespace(get_mega_expert_weights=weights)
    runner.initialize_afd_connector()
    runner.connector.is_initialized = True
    runner.initialize_afd_connector()
    assert events == [("layers", layers), ("init", None)]


@pytest.mark.parametrize("invalid_interface", [False, True])
def test_preparation_failure_still_joins_peer_startup(runner_type, invalid_interface):
    from afd_plugin.connectors.npu.async_mega import CAMAsyncMegaAFDConnector

    class Connector(CAMAsyncMegaAFDConnector):
        is_initialized = False
        role_rank = 1

        def init_afd_connector(self, *, preparation_error=None):
            # This is the peer-startup seam, reached even after model failure.
            assert preparation_error is not None
            raise RuntimeError(f"peer preparation failed: {preparation_error}")

    def fail_conversion(rank):
        raise ValueError("invalid FP4 storage")

    runner = object.__new__(runner_type)
    runner.connector = object.__new__(Connector)
    runner.model = (
        SimpleNamespace()
        if invalid_interface
        else SimpleNamespace(get_mega_expert_weights=fail_conversion)
    )
    reason = (
        "must provide get_mega_expert_weights"
        if invalid_interface
        else "invalid FP4 storage"
    )
    with pytest.raises(RuntimeError, match=f"peer preparation failed: .*{reason}"):
        runner.initialize_afd_connector()


def test_mega_ffn_runner_checks_connector_before_setup(runner_type):
    runner = object.__new__(runner_type)
    runner.connector = SimpleNamespace()
    with pytest.raises(TypeError, match="requires CAMAsyncMegaAFDConnector"):
        runner.initialize_afd_connector()


def test_mega_ffn_runner_executes_one_connector_step(runner_type):
    from afd_plugin.connectors.npu.async_mega import CAMAsyncMegaAFDConnector

    steps = []

    class Connector(CAMAsyncMegaAFDConnector):
        is_initialized = True

        def execute_next_experts(self):
            steps.append("experts")

    runner = object.__new__(runner_type)
    runner.connector = object.__new__(Connector)
    runner.initialize_afd_connector()
    runner.execute_connector_driven_step()
    assert steps == ["experts"]
