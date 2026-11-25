# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU lifecycle and preflight tests for ``AsyncGPUFFNBackend``."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config import CUDAGraphMode  # noqa: E402

from afd_plugin.connectors.gpu.async_nvshmem.connector import (  # noqa: E402
    GpuAsyncAFDConnector,
)
from afd_plugin.ffn_faas.platforms.gpu import async_backend as backend_mod  # noqa: E402
from afd_plugin.ffn_faas.platforms.gpu.environment import (  # noqa: E402
    GPUAsyncFFNEnvironment,
)
from afd_plugin.ffn_faas.platforms.gpu.settings import (  # noqa: E402
    GPUAsyncFFNConfig,
)
from afd_plugin.ffn_faas.runner import FFNModelRunner  # noqa: E402
from afd_plugin.ffn_faas.service import FFNService  # noqa: E402
from tests.unit.fakes.vllm_config import mrv2_vllm_config  # noqa: E402


class _FakeAsyncModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((4, 4), dtype=torch.bfloat16))

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return (1, 2)


class _FakeAsyncSession:
    def __init__(self, connector: GpuAsyncAFDConnector, events: list[str]) -> None:
        self.connector = connector
        self.events = events

    def finish_receiver(self) -> None:
        self.events.append("finish_receiver")
        self.connector.receiver_finished = True


class _FakeAsyncEnvironment:
    def __init__(self, events: list[str], config: SimpleNamespace) -> None:
        self.events = events
        self._config = config
        self.device = torch.device("cpu")
        self.ep_group = SimpleNamespace(world_size=2)

    def build_config(self, args: Namespace) -> SimpleNamespace:
        self.events.append("build_config")
        return self._config

    def initialize_device(self, config: SimpleNamespace) -> None:
        self.events.append("initialize_device")

    def initialize_compute(self, config: SimpleNamespace) -> None:
        self.events.append("initialize_compute")

    def activate_thread(self) -> None:
        self.events.append("activate_thread")

    def close(self) -> None:
        self.events.append("environment_close")


class _FakeAsyncRunner:
    def __init__(
        self,
        events: list[str],
        *,
        step_outcomes: list[bool | BaseException] | None = None,
        drain_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.step_outcomes = list(step_outcomes or [False])
        self.drain_error = drain_error
        connector = object.__new__(GpuAsyncAFDConnector)
        connector.expert_per_rank = 4
        connector.receiver_finished = False
        connector.window = SimpleNamespace(total_bytes=8192, remote_peers={0, 1})
        self.connector = connector
        self.session = _FakeAsyncSession(connector, events)
        self.model = _FakeAsyncModule()

    def load_model(self) -> None:
        self.events.append("load_model")

    def get_model(self) -> nn.Module:
        return self.model

    def capture_padded_ffn_graphs(self) -> int:
        self.events.append("capture_padded_ffn_graphs")
        return 4096

    def initialize_afd_connector(self) -> None:
        self.events.append("initialize_afd_connector")

    def execute_connector_driven_step(self, *, cancel_requested) -> bool:
        self.events.append("execute_step")
        if not self.step_outcomes:
            return not cancel_requested()
        outcome = self.step_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def drain(self) -> None:
        self.events.append("drain")
        if self.drain_error is not None:
            raise self.drain_error

    def shutdown(self) -> None:
        self.events.append("runner_shutdown")


def _async_vllm_config() -> SimpleNamespace:
    config = mrv2_vllm_config(
        dtype=torch.bfloat16,
        role="ffn",
        architecture="AFDDeepseekV2ForCausalLM",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        data_parallel_size=2,
        enforce_eager=False,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    config.additional_config["afd"] = {
        "role": "ffn",
        "connector": "GpuAsyncAFDConnector",
        "async": True,
        "compute_gate_on_attention": True,
        "num_attention_ranks": 2,
        "num_ffn_ranks": 2,
    }
    return config


@pytest.fixture
def backend_harness(monkeypatch, tmp_path):
    def build(*, step_outcomes=None, drain_error=None):
        events: list[str] = []
        vllm_config = _async_vllm_config()
        env = _FakeAsyncEnvironment(events, vllm_config)
        runner = _FakeAsyncRunner(
            events,
            step_outcomes=step_outcomes,
            drain_error=drain_error,
        )
        monkeypatch.setattr(backend_mod, "GPUAsyncFFNEnvironment", lambda _s: env)
        monkeypatch.setattr(
            backend_mod,
            "GPUAsyncFFNModelRunner",
            lambda *args, **kwargs: runner,
        )
        settings = GPUAsyncFFNConfig(model=tmp_path, ffn_ranks=2, rank=1, local_rank=1)
        backend = backend_mod.AsyncGPUFFNBackend(settings, Namespace())
        service = FFNService(rank=1, model_path=str(tmp_path), backend_name="gpu-async")
        host = FFNModelRunner(service, lambda: backend, shutdown_timeout=1.0)
        return backend, service, host, events, runner

    return build


def test_prepare_captures_graphs_before_rendezvous_and_stops_cleanly(
    backend_harness,
):
    _, service, host, events, _ = backend_harness(step_outcomes=[True, False])

    host.run()

    assert events.index("load_model") < events.index("capture_padded_ffn_graphs")
    assert events.index("capture_padded_ffn_graphs") < events.index(
        "initialize_afd_connector"
    )
    assert events.index("drain") < events.index("finish_receiver")
    assert events.index("finish_receiver") < events.index("runner_shutdown")
    assert events.index("runner_shutdown") < events.index("environment_close")

    snapshot = service.snapshot()
    assert snapshot["phase"] == "stopped"
    assert snapshot["resources_released"] is True
    assert snapshot["completed_batches"] == 1
    assert snapshot["model_loaded"] is True
    assert snapshot["loaded_experts"] == 8
    assert snapshot["total_experts"] == 8
    assert snapshot["local_expert_first"] == 4
    assert snapshot["local_expert_last"] == 7
    assert snapshot["parameter_bytes"] == 32
    assert snapshot["graph_bytes"] == 4096
    assert snapshot["async_paired"] is True
    assert snapshot["window_bytes"] == 8192
    assert snapshot["rma_peer_count"] == 2
    assert snapshot["compute_ep_world_size"] == 2


def test_host_stop_drains_and_releases_resources(backend_harness):
    _, service, host, events, runner = backend_harness(step_outcomes=[True, True])

    def stop_on_second_step(*, cancel_requested):
        events.append("execute_step")
        if cancel_requested():
            return False
        service.request_stop()
        return True

    runner.execute_connector_driven_step = stop_on_second_step
    host.run()

    assert "drain" in events
    assert "finish_receiver" in events
    assert service.snapshot()["phase"] == "stopped"
    assert service.snapshot()["resources_released"] is True


@pytest.mark.parametrize(
    "step_outcomes, drain_error, cause_type, cause_match",
    [
        ([RuntimeError("step boom")], None, RuntimeError, "step boom"),
        ([False], TimeoutError("drain timeout"), TimeoutError, "drain timeout"),
    ],
    ids=["step-failure", "drain-timeout"],
)
def test_failure_retains_resources_without_closing(
    backend_harness, step_outcomes, drain_error, cause_type, cause_match
):
    _, service, host, events, _ = backend_harness(
        step_outcomes=step_outcomes,
        drain_error=drain_error,
    )

    with pytest.raises(RuntimeError, match="FFN receiver failed") as exc_info:
        host.run()

    assert isinstance(exc_info.value.__cause__, cause_type)
    assert cause_match in str(exc_info.value.__cause__)
    assert "runner_shutdown" not in events
    assert "environment_close" not in events
    snapshot = service.snapshot()
    assert snapshot["phase"] == "failed"
    assert snapshot["resources_released"] is False


def test_close_rejects_unfinished_receiver(backend_harness):
    backend, _, _, events, runner = backend_harness()
    backend.runner = runner

    with pytest.raises(RuntimeError, match="has not drained"):
        backend.close()

    assert "runner_shutdown" not in events
    assert "environment_close" not in events


def test_async_environment_preflights_nvshmem_before_cuda(monkeypatch, tmp_path):
    from afd_plugin.connectors.gpu.async_nvshmem import library as nvshmem_lib

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        nvshmem_lib,
        "find_host_library",
        lambda: (_ for _ in ()).throw(RuntimeError("missing nvshmem")),
    )
    cuda_called = False

    def fail_if_cuda(_device):
        nonlocal cuda_called
        cuda_called = True

    monkeypatch.setattr(torch.accelerator, "set_device_index", fail_if_cuda)
    env = GPUAsyncFFNEnvironment(GPUAsyncFFNConfig(model=tmp_path, ffn_ranks=2, rank=0))

    with pytest.raises(RuntimeError, match="missing nvshmem"):
        env.initialize_device(_async_vllm_config())

    assert cuda_called is False
