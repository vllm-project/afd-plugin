# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit tests for ``FFNEplbLifecycle`` using a fake backend (no devices)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.eplb.backend import EplbBackend
from afd_plugin.eplb.lifecycle import FFNEplbLifecycle

STEP_INTERVAL = 3


class FakeEplbBackend:
    """Records calls and mimics step-interval rebalancing plus load windows."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.initialized_model = None
        self.recorded_load = 0
        self.recording_enabled = False
        self.step_index = 0
        # Persistent map updated in place, as graph replay requires.
        self.logical_to_physical_map = [0, 1]
        self.replica_count = [1, 1]
        self.state = object()

    def initialize_model(self, model, *, load_dummy_weights: bool = False) -> None:
        self.initialized_model = model
        self.load_dummy_weights = load_dummy_weights
        self.events.append(("initialize_model", model, load_dummy_weights))

    def prepare_forward(
        self, num_unpadded_tokens: int, *, is_dummy: bool = False
    ) -> None:
        self.events.append(("prepare_forward", num_unpadded_tokens, is_dummy))
        self.recording_enabled = not is_dummy
        if self.recording_enabled:
            self.recorded_load += num_unpadded_tokens

    def step(self, *, is_dummy: bool = False) -> bool:
        self.events.append(("step", is_dummy))
        self.step_index += 1
        rebalanced = self.step_index % STEP_INTERVAL == 0
        if rebalanced:
            self.logical_to_physical_map[:] = [1, 0]
        return rebalanced

    def get_logical_to_physical_map(self):
        return self.logical_to_physical_map

    def get_replica_count(self):
        return self.replica_count

    def get_eplb_state(self):
        return self.state


def _lifecycle(backend) -> FFNEplbLifecycle:
    return FFNEplbLifecycle(SimpleNamespace(), SimpleNamespace(), backend)


def test_fake_backend_satisfies_protocol():
    assert isinstance(FakeEplbBackend(), EplbBackend)


def test_disabled_lifecycle_is_a_noop():
    lifecycle = _lifecycle(None)

    lifecycle.initialize_model(object())
    lifecycle.prepare_forward(8)
    lifecycle.prepare_forward(0, is_dummy=True)

    assert lifecycle.is_enabled is False
    assert lifecycle.step() is False
    assert lifecycle.step(is_dummy=True) is False
    assert lifecycle.step_count == 0
    assert lifecycle.get_eplb_state() is None


def test_initialize_model_delegates_to_backend():
    backend = FakeEplbBackend()
    model = object()

    lifecycle = _lifecycle(backend)
    lifecycle.initialize_model(model)

    assert lifecycle.is_enabled is True
    assert backend.initialized_model is model
    assert lifecycle.get_eplb_state() is backend.state


def test_prepare_forward_passes_unpadded_count_and_dummy_flag():
    backend = FakeEplbBackend()
    lifecycle = _lifecycle(backend)

    lifecycle.prepare_forward(5)
    lifecycle.prepare_forward(0, is_dummy=True)

    assert backend.events == [
        ("prepare_forward", 5, False),
        ("prepare_forward", 0, True),
    ]


def test_padding_rows_are_excluded_from_recorded_load():
    backend = FakeEplbBackend()
    lifecycle = _lifecycle(backend)
    real_tokens, padded_tokens = 5, 8

    # The runner reports only the real attention tokens, never the padded batch.
    lifecycle.prepare_forward(real_tokens)
    lifecycle.step()

    assert backend.recorded_load == real_tokens
    assert backend.recorded_load != padded_tokens


def test_dummy_steps_record_no_load_but_advance_the_step_counter():
    backend = FakeEplbBackend()
    lifecycle = _lifecycle(backend)

    # Warmup, graph capture and idle ranks all run dummy steps.
    for _ in range(STEP_INTERVAL - 1):
        lifecycle.prepare_forward(0, is_dummy=True)
        lifecycle.step(is_dummy=True)

    assert backend.recorded_load == 0
    assert lifecycle.step_count == STEP_INTERVAL - 1
    assert backend.step_index == STEP_INTERVAL - 1


def test_lockstep_ranks_rebalance_on_identical_step_indices():
    # One rank has real work every step; the other is idle and runs dummy steps.
    busy_backend, idle_backend = FakeEplbBackend(), FakeEplbBackend()
    busy, idle = _lifecycle(busy_backend), _lifecycle(idle_backend)
    busy_rebalances: list[int] = []
    idle_rebalances: list[int] = []

    for index in range(1, 3 * STEP_INTERVAL + 1):
        busy.prepare_forward(4)
        idle.prepare_forward(0, is_dummy=True)
        if busy.step():
            busy_rebalances.append(index)
        if idle.step(is_dummy=True):
            idle_rebalances.append(index)

    assert busy_rebalances == idle_rebalances == [3, 6, 9]
    assert busy.step_count == idle.step_count == 3 * STEP_INTERVAL
    assert busy_backend.recorded_load == 4 * 3 * STEP_INTERVAL
    assert idle_backend.recorded_load == 0


def test_step_returns_true_only_on_rebalance():
    lifecycle = _lifecycle(FakeEplbBackend())

    results = [lifecycle.step() for _ in range(STEP_INTERVAL)]

    assert results == [False] * (STEP_INTERVAL - 1) + [True]


def test_map_is_updated_in_place_across_a_rebalance():
    backend = FakeEplbBackend()
    lifecycle = _lifecycle(backend)
    persistent_map = backend.get_logical_to_physical_map()
    persistent_count = backend.get_replica_count()

    for _ in range(STEP_INTERVAL):
        lifecycle.step()

    # Same objects (as captured graphs hold them) with updated contents.
    assert backend.get_logical_to_physical_map() is persistent_map
    assert backend.get_replica_count() is persistent_count
    assert persistent_map == [1, 0]


@pytest.mark.parametrize("is_dummy", [False, True])
def test_step_forwards_dummy_flag(is_dummy):
    backend = FakeEplbBackend()

    _lifecycle(backend).step(is_dummy=is_dummy)

    assert backend.events == [("step", is_dummy)]
