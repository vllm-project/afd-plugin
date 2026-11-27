# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async Mega connector step/close contracts on the host."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.npu.async_mega import (  # noqa: E402
    connector as connector_module,
)


def _host_connector():
    # The Worker may close the connector while its FFN loop is still running;
    # build the connector without process groups or device buffers.
    connector = object.__new__(connector_module.CAMAsyncMegaAFDConnector)
    connector.stages = [
        connector_module.MegaStage(group=object(), buffer=SimpleNamespace())
    ]
    connector.next_stage = 0
    connector.layer_ids = [0]
    connector.layer_slots = {0: 0}
    connector.layered_weights = (object(),)
    connector._step_lock = threading.Lock()
    connector._closed = False
    return connector


class _CompletedEvent:
    def record(self, *_args, **_kwargs):
        pass

    def query(self):
        return True


@pytest.fixture
def host_close(monkeypatch):
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    monkeypatch.setattr(torch.npu, "Event", _CompletedEvent)
    monkeypatch.setattr(connector_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(
        connector_module.dist, "destroy_process_group", lambda *args, **kwargs: None
    )


def test_no_step_submits_after_close(host_close):
    connector = _host_connector()
    connector.stages[0].buffer.destroy = lambda: None
    submitted = []
    connector.ops = SimpleNamespace(mega_ffn=lambda *args: submitted.append("op"))

    connector.execute_next_experts()
    connector.close()
    connector.execute_next_experts()

    assert submitted == ["op"]


def test_close_waits_for_the_in_flight_step(host_close):
    connector = _host_connector()
    op_started = threading.Event()
    op_release = threading.Event()
    events = []
    connector.stages[0].buffer.destroy = lambda: events.append("destroy")

    def mega_ffn(*_args):
        op_started.set()
        assert op_release.wait(5)
        events.append("op")

    connector.ops = SimpleNamespace(mega_ffn=mega_ffn)
    step = threading.Thread(target=connector.execute_next_experts)
    step.start()
    assert op_started.wait(5)

    closer = threading.Thread(target=connector.close)
    closer.start()
    op_release.set()
    step.join(5)
    closer.join(5)

    # The window may only be destroyed after the operator using it returned.
    assert not step.is_alive()
    assert not closer.is_alive()
    assert events == ["op", "destroy"]


def test_close_reports_a_step_that_does_not_return(host_close, monkeypatch):
    connector = _host_connector()
    monkeypatch.setattr(connector_module, "CLOSE_LOCK_TIMEOUT_SECONDS", 0.05)
    step_started = threading.Event()
    step_release = threading.Event()
    destroyed = []
    connector.stages[0].buffer.destroy = lambda: destroyed.append("destroy")

    def mega_ffn(*_args):
        step_started.set()
        assert step_release.wait(5)

    connector.ops = SimpleNamespace(mega_ffn=mega_ffn)
    step = threading.Thread(target=connector.execute_next_experts)
    step.start()
    assert step_started.wait(5)

    with pytest.raises(RuntimeError, match="retain resources"):
        connector.close()

    step_release.set()
    step.join(5)
    assert destroyed == []
    assert connector.stages


def test_close_reports_device_work_that_never_completes(host_close, monkeypatch):
    connector = _host_connector()
    monkeypatch.setattr(connector_module, "CLOSE_DEVICE_TIMEOUT_SECONDS", 0.05)
    destroyed = []
    connector.stages[0].buffer.destroy = lambda: destroyed.append("destroy")

    class _PendingEvent:
        def query(self):
            return False

    connector.stages[0].completion = _PendingEvent()

    with pytest.raises(RuntimeError, match="retain resources"):
        connector.close()

    assert destroyed == []
    assert connector.stages


class _FakeOps:
    def __init__(self):
        self.sent = []
        self.received = []
        self.sequence_calls = []
        self._next_ticket = 0

    @staticmethod
    def split_sequence(iteration):
        return (iteration - 1) % 32 + 1

    def atten_send(self, _hidden, _ids, _weights, buffer, _members, sequence, **_kw):
        self.sent.append((buffer, sequence))
        self.sequence_calls.append(sequence)
        self._next_ticket += 1
        return SimpleNamespace(ticket=self._next_ticket)

    def atten_recv(self, ticket, buffer):
        self.received.append(ticket)
        return SimpleNamespace(tensor=ticket.ticket, buffer=buffer)


def _attention_connector(num_ubatches=2):
    connector = _host_connector()
    connector.num_ubatches = num_ubatches
    connector.ubatch_tickets = [None] * num_ubatches
    connector.ubatch_outputs = [None] * num_ubatches
    connector._recv_locks = [threading.Lock() for _ in range(num_ubatches)]
    connector._send_lock = threading.Lock()
    connector.attn_tp = 1
    connector.role_rank = 0
    connector.ops = _FakeOps()
    return connector


def _send(connector, stage):
    context = SimpleNamespace(metadata=SimpleNamespace(stage_idx=stage, layer_idx=0))
    connector.send_attn_output(
        torch.zeros(2, 4),
        context,
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        topk_weights=torch.zeros(2, 1),
    )
    return context


def test_both_ubatches_share_one_window_and_one_sequence_ring(monkeypatch):
    connector = _attention_connector()
    monkeypatch.setattr(connector_module, "dbo_enabled", lambda: True)
    monkeypatch.setattr(connector_module, "dbo_maybe_run_recv_hook", lambda: None)
    hooks = []
    monkeypatch.setattr(connector_module, "dbo_register_recv_hook", hooks.append)

    _send(connector, 0)
    _send(connector, 1)

    assert connector.ops.sent == [(connector.stages[0].buffer, 1),
                                  (connector.stages[0].buffer, 2)]
    assert len(hooks) == 2


def test_receive_hook_and_owner_return_the_same_result_once(monkeypatch):
    connector = _attention_connector()
    hooks = []
    monkeypatch.setattr(connector_module, "dbo_enabled", lambda: True)
    monkeypatch.setattr(connector_module, "dbo_maybe_run_recv_hook", lambda: None)
    monkeypatch.setattr(connector_module, "dbo_register_recv_hook", hooks.append)

    _send(connector, 0)
    hooks[0]()

    first = connector.recv_ffn_output(ubatch_idx=0)
    second = connector.recv_ffn_output(ubatch_idx=0)

    assert first is second
    assert len(connector.ops.received) == 1


def test_peer_result_is_taken_back_before_the_next_send(monkeypatch):
    connector = _attention_connector()
    order = []

    class _Hooks:
        def __init__(self):
            self.registered = []

    hooks = _Hooks()

    def register(hook):
        hooks.registered.append(hook)

    def run_hook():
        if hooks.registered:
            order.append("take_back")
            hooks.registered.pop(0)()

    def send(*args, **kwargs):
        order.append("send")
        return _attention_connector_ops_send(connector, *args, **kwargs)

    monkeypatch.setattr(connector_module, "dbo_enabled", lambda: True)
    monkeypatch.setattr(connector_module, "dbo_maybe_run_recv_hook", run_hook)
    monkeypatch.setattr(connector_module, "dbo_register_recv_hook", register)
    connector.ops.atten_send = send

    _send(connector, 0)
    _send(connector, 1)

    assert order == ["send", "take_back", "send"]


def _attention_connector_ops_send(connector, _hidden, _ids, _weights, buffer,
                                  _members, sequence, **_kw):
    connector.ops.sent.append((buffer, sequence))
    return SimpleNamespace(ticket=len(connector.ops.sent))
