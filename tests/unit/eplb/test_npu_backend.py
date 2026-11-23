# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for the Ascend EPLB backend with a fake controller (no NPU)."""

from __future__ import annotations

from types import SimpleNamespace

from afd_plugin.eplb.npu import AscendEplbBackend

MODEL_CONFIG = object()


class FakeModelState:
    def __init__(self) -> None:
        self.logical_to_physical_map = object()
        self.logical_replica_count = object()


class FakeState:
    def __init__(self, interval: int = 3, step: int = 0) -> None:
        self.expert_rearrangement_step = step
        self.expert_rearrangement_step_interval = interval
        self.model_states = {"model": FakeModelState()}


class FakeController:
    def __init__(self, state: FakeState | None = None) -> None:
        self.state = state
        self.calls: list[tuple] = []

    def prepare_load(self) -> None:
        self.calls.append(("prepare_load",))

    def maybe_register_model(self, model, model_config) -> bool:
        self.calls.append(("maybe_register_model", model, model_config))
        return True

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


def _backend(controller: FakeController) -> AscendEplbBackend:
    backend = object.__new__(AscendEplbBackend)
    backend.vllm_config = SimpleNamespace()
    backend.model_config = MODEL_CONFIG
    backend.controller = controller
    return backend


def test_initialize_model_registers_the_ascend_state():
    controller = FakeController()
    model = object()

    _backend(controller).initialize_model(model)

    assert controller.calls == [
        ("prepare_load",),
        ("maybe_register_model", model, MODEL_CONFIG),
        ("maybe_start_async_loop", True, False),
    ]


def test_prepare_forward_publishes_the_unpadded_count():
    controller = FakeController()

    _backend(controller).prepare_forward(9, is_dummy=True)

    assert controller.calls == [("prepare_forward", MODEL_CONFIG, 9)]


def test_step_reports_the_synchronous_rearrangement_boundary():
    controller = FakeController(FakeState(interval=3))
    backend = _backend(controller)

    assert [backend.step() for _ in range(3)] == [False, False, True]
    assert controller.calls == [("step", False)] * 3


def test_dummy_steps_still_enter_the_collective():
    controller = FakeController(FakeState(interval=2))
    backend = _backend(controller)

    backend.step(is_dummy=True)
    backend.step(is_dummy=True)

    assert controller.calls == [("step", True), ("step", True)]


def test_state_accessors_return_the_persistent_upstream_tensors():
    state = FakeState()
    backend = _backend(FakeController(state))
    model_state = state.model_states["model"]

    assert backend.get_logical_to_physical_map() is (
        model_state.logical_to_physical_map
    )
    assert backend.get_replica_count() is model_state.logical_replica_count
    assert backend.get_eplb_state() is state


def test_accessors_are_empty_before_state_is_created():
    backend = _backend(FakeController(state=None))

    assert backend.get_logical_to_physical_map() is None
    assert backend.get_replica_count() is None
    assert backend.get_eplb_state() is None
    assert backend.step() is False
