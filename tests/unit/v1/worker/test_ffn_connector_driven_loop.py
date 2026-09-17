# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The connector-driven FFN drain loop, without a GPU.

The worker-loop tests stub ``execute_connector_driven_step`` wholesale, so the
loop underneath it -- idle-poll return, state checking, per-item metadata
installation, output send-back -- ran only on a device. The NPU twin of this
loop is unit-tested; this is the same shape with a fake work-item connector.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.connectors.gpu.async_gpu import (  # noqa: E402
    ConnectorShutdown,
    GpuAsyncTransferState,
)
from afd_plugin.v1.worker import ffn_model_runner as module  # noqa: E402

NUM_LAYERS = 3


class _FakeWorkItem:
    def __init__(self, layer_idx, states, metadata):
        self.layer_idx = layer_idx
        self.hidden_states = torch.zeros(2, 4)
        self.context = SimpleNamespace(states=states, metadata=metadata)


class _FakeConnector:
    """Hands out a scripted sequence of work items, then whatever ends it."""

    def __init__(self, script):
        self.script = list(script)
        self.received = []
        self.sent = []

    def recv_ffn_work_item(self, *, stage_idx, max_num_tokens, **_):
        self.received.append((stage_idx, max_num_tokens))
        if not self.script:
            raise TimeoutError
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    def send_ffn_work_item_output(self, work_item, output):
        self.sent.append((work_item.layer_idx, output))
        return output


def _states():
    state = object.__new__(GpuAsyncTransferState)
    state.group_list = None
    state.expand_x_shared = None
    return state


@pytest.fixture
def runner(monkeypatch):
    forward_context = SimpleNamespace(
        dp_metadata="stale",
        additional_kwargs={},
        all_moe_layers=[f"model.layers.{i}.mlp" for i in range(NUM_LAYERS)],
        moe_layer_index=None,
    )

    @contextmanager
    def fake_context(_vllm_config):
        yield forward_context

    monkeypatch.setattr(module, "_ffn_forward_context", fake_context)

    runner = object.__new__(module.GPUFFNModelRunner)
    runner.num_layers = NUM_LAYERS
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
    )
    runner._compute_work_item = lambda item, states: torch.full((1,), item.layer_idx)
    runner.forward_context = forward_context
    return runner


def test_an_idle_poll_returns_instead_of_blocking(runner):
    # This is what lets the worker loop observe its shutdown event.
    runner.connector = _FakeConnector([])

    assert runner._ffn_forward_connector_driven() is None
    assert len(runner.connector.received) == 1


def test_every_drained_item_is_computed_and_sent_back(runner):
    items = [_FakeWorkItem(i, _states(), f"meta-{i}") for i in range(2)]
    runner.connector = _FakeConnector(items)

    runner._ffn_forward_connector_driven()

    assert [layer for layer, _ in runner.connector.sent] == [0, 1]
    assert [int(out) for _, out in runner.connector.sent] == [0, 1]


def test_each_item_installs_its_own_metadata_and_layer(runner):
    # Successive work items may belong to different layers of different
    # replicas, so per-item context installation is the whole contract.
    runner.connector = _FakeConnector(
        [_FakeWorkItem(2, _states(), "meta-2")],
    )

    runner._ffn_forward_connector_driven()

    assert runner.forward_context.additional_kwargs["afd_metadata"] == "meta-2"
    assert runner.forward_context.moe_layer_index == 2
    # A control-plane run leaves dp_metadata behind; this path must clear it.
    assert runner.forward_context.dp_metadata is None


def test_the_drain_is_bounded_by_the_layer_count(runner):
    endless = [_FakeWorkItem(0, _states(), "m") for _ in range(NUM_LAYERS + 5)]
    runner.connector = _FakeConnector(endless)

    runner._ffn_forward_connector_driven()

    # Returning after a bounded drain is what gives the worker loop its turn.
    assert len(runner.connector.sent) == NUM_LAYERS


def test_a_foreign_transfer_state_is_refused(runner):
    runner.connector = _FakeConnector(
        [_FakeWorkItem(0, SimpleNamespace(), "meta")],
    )

    with pytest.raises(RuntimeError, match="GpuAsyncTransferState"):
        runner._ffn_forward_connector_driven()


def test_a_peer_shutdown_propagates(runner):
    # The worker loop treats this as an ordinary exit; the loop must not
    # swallow it into an idle-poll return.
    runner.connector = _FakeConnector([ConnectorShutdown("peer left")])

    with pytest.raises(ConnectorShutdown):
        runner._ffn_forward_connector_driven()
