# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the GPU EPLB backend with a fake vLLM controller (no vLLM)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.eplb.gpu import GpuVllmEplbBackend

MODEL_CONFIG = object()


class FakeModelState:
    def __init__(self) -> None:
        self.logical_to_physical_map = object()
        self.logical_replica_count = object()


class FakeState:
    def __init__(self, interval: int = 3) -> None:
        self.expert_rearrangement_step = 0
        self.expert_rearrangement_step_interval = interval
        self.model_states = {"model": FakeModelState()}


class FakeController:
    def __init__(self, state: FakeState | None = None) -> None:
        self.state = state
        self.calls: list[tuple] = []
        self.models_added = True

    def prepare_load(self) -> None:
        self.calls.append(("prepare_load",))

    def maybe_register_model(self, model, model_config) -> bool:
        self.calls.append(("maybe_register_model", model, model_config))
        return self.models_added

    def maybe_start_async_loop(self, models_added, load_dummy_weights=False) -> None:
        self.calls.append(("maybe_start_async_loop", models_added, load_dummy_weights))

    def prepare_forward(self, model_config, num_unpadded_tokens) -> None:
        self.calls.append(("prepare_forward", model_config, num_unpadded_tokens))

    def step(self, is_dummy: bool = False) -> None:
        self.calls.append(("step", is_dummy))
        if self.state is None:
            return
        self.state.expert_rearrangement_step += 1
        if (
            self.state.expert_rearrangement_step
            >= self.state.expert_rearrangement_step_interval
        ):
            self.state.expert_rearrangement_step = 0


def _backend(controller: FakeController) -> GpuVllmEplbBackend:
    backend = object.__new__(GpuVllmEplbBackend)
    backend.vllm_config = SimpleNamespace()
    backend.model_config = MODEL_CONFIG
    backend.controller = controller
    return backend


def test_initialize_model_creates_state_registers_and_starts_async_loop():
    controller = FakeController()
    model = object()

    _backend(controller).initialize_model(model, load_dummy_weights=True)

    assert controller.calls == [
        ("prepare_load",),
        ("maybe_register_model", model, MODEL_CONFIG),
        ("maybe_start_async_loop", True, True),
    ]


def test_prepare_forward_publishes_the_unpadded_count():
    controller = FakeController()

    _backend(controller).prepare_forward(17, is_dummy=True)

    assert controller.calls == [("prepare_forward", MODEL_CONFIG, 17)]


def test_step_reports_a_rebalance_when_the_interval_is_reached():
    state = FakeState(interval=3)
    controller = FakeController(state)
    backend = _backend(controller)

    assert [backend.step() for _ in range(3)] == [False, False, True]


def test_step_advances_the_counter_on_dummy_steps_without_recording():
    state = FakeState(interval=2)
    controller = FakeController(state)
    backend = _backend(controller)

    backend.step(is_dummy=True)

    assert controller.calls == [("step", True)]
    assert state.expert_rearrangement_step == 1


def test_step_without_state_is_a_noop_and_returns_false():
    backend = _backend(FakeController(state=None))

    assert backend.step() is False
    assert backend.get_logical_to_physical_map() is None
    assert backend.get_replica_count() is None
    assert backend.get_eplb_state() is None


def test_maps_are_the_persistent_upstream_tensors():
    state = FakeState()
    backend = _backend(FakeController(state))
    model_state = state.model_states["model"]

    assert backend.get_logical_to_physical_map() is (
        model_state.logical_to_physical_map
    )
    assert backend.get_replica_count() is model_state.logical_replica_count
    assert backend.get_eplb_state() is state


def test_backend_works_without_a_python_side_state_before_load():
    backend = _backend(FakeController(state=None))

    backend.prepare_forward(0)
    backend.step()

    assert backend.get_eplb_state() is None


@pytest.mark.parametrize("models_added", [True, False])
def test_async_loop_start_forwards_the_registration_result(models_added):
    controller = FakeController()
    controller.models_added = models_added

    _backend(controller).initialize_model(object())

    assert controller.calls[-1] == (
        "maybe_start_async_loop",
        models_added,
        False,
    )
