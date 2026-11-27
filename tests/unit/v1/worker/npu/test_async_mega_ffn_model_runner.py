# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model weight handoff and execution for the dedicated Mega FFN runner."""

from types import SimpleNamespace

from tests.unit.fakes.npu_runtime import require_npu_runtime as _require_npu_runtime


def test_mega_ffn_runner_prepares_layers_before_connector_init():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.async_mega_ffn_model_runner import (
        MegaNPUFFNModelRunner,
    )

    events = []

    class _Connector:
        is_initialized = False
        role_rank = 1

        def prepare_layers(self, layers):
            events.append(("layers", layers))

        def init_afd_connector(self):
            events.append(("init", None))

    runner = object.__new__(MegaNPUFFNModelRunner)
    runner.connector = _Connector()
    layers = {3: (["w1"], ["w2"], ["s1"], ["s2"])}

    def weights(rank):
        assert rank == 1
        return layers

    runner.model = SimpleNamespace(get_mega_expert_weights=weights)

    runner.initialize_afd_connector()
    runner.connector.is_initialized = True
    runner.initialize_afd_connector()

    assert events == [("layers", layers), ("init", None)]


def test_mega_ffn_runner_executes_one_connector_step():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.async_mega_ffn_model_runner import (
        MegaNPUFFNModelRunner,
    )

    steps = []

    class _Connector:
        def execute_next_experts(self):
            steps.append("experts")

    runner = object.__new__(MegaNPUFFNModelRunner)
    runner.connector = _Connector()

    runner.execute_connector_driven_step()

    assert steps == ["experts"]
