# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU stream protocol checks; device qualification is recorded separately."""

import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from afd_plugin.async_cam_startup import AsyncCAMGraphStartup
from tests.unit.fakes.async_ffn import load_async_modules


@pytest.fixture
def pipeline(monkeypatch):
    _, connector, runtime = load_async_modules(monkeypatch)
    module = sys.modules["afd_plugin.v1.worker.npu.async_ffn_two_stream"]
    npu, history, produced = module.torch.npu, MagicMock(), []
    for name in ("compute_stream", "send_stream", "ready0", "done0", "ready1", "done1"):
        history.attach_mock(MagicMock(), name)
    npu.NPUGraph.side_effect = lambda: MagicMock()
    npu.graph_pool_handle.side_effect = object
    npu.Stream.side_effect = [history.compute_stream, history.send_stream]
    npu.Event.side_effect = [
        history.ready0,
        history.done0,
        history.ready1,
        history.done1,
    ]

    class Tensor:
        pass

    def receive():
        hidden = Tensor()
        produced.append(weakref.ref(hidden))
        state = SimpleNamespace(
            dynamic_scales=1, group_list=2, token_nums_rankid_layeridx=hidden
        )
        return SimpleNamespace(
            hidden_states=hidden, context=SimpleNamespace(states=state)
        )

    def send(output, context):
        assert output is context.states.token_nums_rankid_layeridx

    history.receive.side_effect = receive
    history.compute.side_effect = lambda hidden, *a, **kw: hidden
    history.send.side_effect = send
    config = connector.AFDAsyncExtraInfo(ffn_two_stream=True)
    owner = module.AsyncFFNTwoStream(
        "npu:0",
        config,
        SimpleNamespace(recv_attn_output=history.receive, send_ffn_output=history.send),
        history.compute,
    )
    history.reset_mock()
    return SimpleNamespace(
        owner=owner, npu=npu, history=history, produced=produced, runtime=runtime
    )


@pytest.fixture
def warmup_runner(pipeline):
    p = pipeline
    runtime = p.runtime
    runner = runtime.AsyncNPUFFNModelRunner.__new__(runtime.AsyncNPUFFNModelRunner)
    runner.model = object()
    runner._two_stream = p.owner
    startup = AsyncCAMGraphStartup(1, 1, 2)
    store = MagicMock()
    startup._store = store
    runner.connector = SimpleNamespace(graph_startup=startup)
    p.history.done0.query.return_value = True
    runner._ffn_forward_connector_driven(single=True)
    p.history.reset_mock()
    p.produced.clear()
    return runner, store


@pytest.mark.parametrize("outcome", ["ready", "peer_failure", "timeout"])
def test_warmup_observes_completion_or_failure_without_blocking_fence(
    pipeline, warmup_runner, monkeypatch, outcome
):
    p = pipeline
    runner, store = warmup_runner
    p.history.done1.query.return_value = outcome == "ready"
    store.check.return_value = outcome == "peer_failure"
    store.get.return_value = b"Attention rank failed"
    clock = iter([0, p.runtime.GRAPH_STARTUP_TIMEOUT_SECONDS])
    monkeypatch.setattr(p.runtime.time, "monotonic", lambda: next(clock))
    p.npu.synchronize.side_effect = AssertionError("warmup must poll completion")
    if outcome == "ready":
        runner._ffn_forward_connector_driven(single=True)
        p.owner.close(receiver_exited=True)
        p.history.reset_mock()
        assert all(ref() is None for ref in p.produced)
    else:
        error, message = (
            (RuntimeError, "Attention rank failed")
            if outcome == "peer_failure"
            else (TimeoutError, "warmup timed out; retain resources")
        )
        with pytest.raises(error, match=message):
            runner._ffn_forward_connector_driven(single=True)
        with pytest.raises(RuntimeError, match="failed; retain resources"):
            p.owner.close(receiver_exited=True)
        p.history.reset_mock()
        assert all(ref() is not None for ref in p.produced)


def test_eager_orders_events_and_retires_before_allocating_third_item(pipeline):
    p = pipeline
    p.owner.submit(3)
    expected = []
    for index, slot in enumerate((0, 1, 0)):
        if index == 2:
            expected += ["done0.synchronize", "compute_stream.wait_event"]
        expected += [
            "receive",
            "compute",
            f"ready{slot}.record",
            "send_stream.wait_event",
            "send",
            f"done{slot}.record",
        ]
    assert [c[0] for c in p.history.mock_calls] == expected
    p.history.send_stream.wait_event.assert_has_calls(
        [call(p.history.ready0), call(p.history.ready1), call(p.history.ready0)]
    )
    p.history.compute_stream.wait_event.assert_called_once_with(p.history.done0)
    p.history.ready0.record.assert_has_calls([call(p.owner.compute)] * 2)
    p.history.done0.record.assert_has_calls([call(p.owner.send)] * 2)
    p.history.reset_mock()  # Drop mock call arguments before checking tensor lifetime.
    assert [ref() is not None for ref in p.produced] == [False, True, True]
    p.owner.complete()
    assert all(ref() is None for ref in p.produced)


def test_full_keeps_slot_addresses_and_reuses_with_device_edges(pipeline):
    p = pipeline
    p.owner.capture()
    graphs = [
        graph
        for slot in p.owner.slots
        for graph in (slot.compute_graph, slot.send_graph)
    ]
    assert len({id(graph) for graph in graphs}) == 4
    assert len({pool for slot in p.owner.slots for pool in slot.pools}) == 4
    assert len(p.produced) == 2 and p.produced[0]() is not p.produced[1]()
    p.history.reset_mock()  # Call recording must not keep captured tensors alive.
    p.owner.submit(5)
    p.history.done0.synchronize.assert_not_called()
    p.history.compute_stream.wait_event.assert_any_call(p.history.done0)
    assert len(p.produced) == 2
    p.owner.complete()
    assert all(ref() is not None for ref in p.produced)
    with pytest.raises(RuntimeError, match="receiver exit"):
        p.owner.close(receiver_exited=False)
    p.owner.close(receiver_exited=True)
    p.history.reset_mock()
    assert all(ref() is None for ref in p.produced)
    assert all(graph.reset.call_count == 1 for graph in graphs)


@pytest.mark.parametrize(
    "failure", ["capture", "receive", "compute", "send", "retire", "replay", "fence"]
)
def test_partial_failure_retains_resources_and_disallows_retry(pipeline, failure):
    p = pipeline
    if failure == "replay":
        p.owner.capture()
    failing_call = {
        "capture": p.history.send,
        "receive": p.history.receive,
        "compute": p.history.compute,
        "send": p.history.send,
        "retire": p.history.done0.synchronize,
        "replay": p.owner.slots[0].compute_graph.replay
        if failure == "replay"
        else None,
        "fence": p.npu.synchronize,
    }[failure]
    failing_call.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="native failure"):
        if failure == "capture":
            p.owner.capture()
        else:
            p.owner.submit(3 if failure == "retire" else 1)
            p.owner.complete()
    for operation in (
        p.owner.capture,
        lambda: p.owner.submit(1),
        lambda: p.owner.close(receiver_exited=True),
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert all(
        slot.compute_graph is None or not slot.compute_graph.reset.called
        for slot in p.owner.slots
    )
    p.history.reset_mock()
    assert all(ref() is not None for ref in p.produced)


@pytest.mark.parametrize("budgets", [(25, 24, 24), (12, 23, 24), (12, 24, 25)])
def test_invalid_budget_rejected_before_communication(monkeypatch, budgets):
    _, connector, runtime = load_async_modules(monkeypatch)
    config = connector.AFDAsyncExtraInfo(
        ffn_two_stream=True,
        ffn_compute_aic_num=budgets[0],
        ffn_compute_aiv_num=budgets[1],
        ffn_combine_aiv_num=budgets[2],
    )
    communication = MagicMock()
    with pytest.raises(ValueError, match="core limits"):
        runtime.AsyncFFNTwoStream("npu:0", config, communication, MagicMock())
    assert not communication.mock_calls


def test_pending_work_cannot_be_resubmitted_or_closed(pipeline):
    owner = pipeline.owner
    owner.submit(1)
    with pytest.raises(RuntimeError, match="not ready"):
        owner.submit(1)
    with pytest.raises(RuntimeError, match="device completion"):
        owner.close(receiver_exited=True)
    owner.complete()
    owner.close(receiver_exited=True)
    owner.close(receiver_exited=True)
