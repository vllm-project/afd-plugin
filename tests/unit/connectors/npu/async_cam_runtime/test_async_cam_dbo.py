# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CAM's physical window and owner result survive native DBO handoffs."""

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.v1.worker import ubatching  # noqa: E402

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata  # noqa: E402
from afd_plugin.connectors.npu.async_cam_runtime import async_cam_dbo  # noqa: E402


def _context(stage):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=3, stage_idx=stage, seq_len=2
        )
    )


def test_peer_receives_once_before_sending_and_owner_consumes_its_result(monkeypatch):
    identity = threading.get_ident()
    registry = {identity: 0}
    contexts = [SimpleNamespace(recv_hook=None) for _ in range(2)]
    monkeypatch.setattr(ubatching, "_THREAD_ID_TO_CONTEXT", registry)
    monkeypatch.setattr(ubatching, "_CURRENT_CONTEXTS", contexts)
    monkeypatch.setattr(ubatching, "_NUM_UBATCHES", 2)
    events = []
    outstanding: list[AFDTransferContext] = []

    def send(hidden, context, **kwargs):
        assert not outstanding, "a second dispatch overwrote CAM's physical window"
        outstanding.append(context)
        events.append(("send", context.metadata.stage_idx, registry[identity]))

    def receive(*, ref_tensor, ubatch_idx, context):
        assert outstanding.pop() is context
        assert ubatch_idx == context.metadata.stage_idx
        events.append(("recv", ubatch_idx, registry[identity]))
        return ref_tensor + 10

    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    hidden = [torch.full((2, 1), stage) for stage in range(2)]
    first = async_cam_dbo.send_async_cam_remote_call(connector, hidden[0], _context(0))
    registry[identity] = 1
    second = async_cam_dbo.send_async_cam_remote_call(connector, hidden[1], _context(1))
    assert events == [("send", 0, 0), ("recv", 0, 1), ("send", 1, 1)]

    registry[identity] = 0
    torch.testing.assert_close(first(), hidden[0] + 10)
    torch.testing.assert_close(first(), hidden[0] + 10)
    ubatching.dbo_maybe_run_recv_hook()  # Last-layer tail drains the peer's result.
    registry[identity] = 1
    torch.testing.assert_close(second(), hidden[1] + 10)
    assert events[-1] == ("recv", 1, 0)
    assert len([event for event in events if event[0] == "recv"]) == 2
    assert not outstanding and all(context.recv_hook is None for context in contexts)


def test_without_dbo_receives_directly_without_inactive_hooks(monkeypatch):
    monkeypatch.setattr(async_cam_dbo, "dbo_enabled", lambda: False)
    monkeypatch.setattr(
        async_cam_dbo, "dbo_maybe_run_recv_hook", lambda: pytest.fail("inactive hook")
    )
    monkeypatch.setattr(
        async_cam_dbo, "dbo_register_recv_hook", lambda _: pytest.fail("inactive hook")
    )
    calls = []
    value = torch.ones(2, 1)

    def recv(**kwargs):
        calls.append("recv")
        return value

    connector = SimpleNamespace(
        send_attn_output=lambda *args, **kwargs: calls.append("send"),
        recv_ffn_output=recv,
    )
    receive = async_cam_dbo.send_async_cam_remote_call(connector, value, _context(0))
    assert calls == ["send"]
    assert receive() is value and receive() is value
    assert calls == ["send", "recv"]


def test_failed_receive_is_never_resubmitted(monkeypatch):
    monkeypatch.setattr(async_cam_dbo, "dbo_enabled", lambda: False)
    calls = []

    def fail(**kwargs):
        calls.append("recv")
        raise ValueError("wire failure")

    connector = SimpleNamespace(
        send_attn_output=lambda *a, **k: None, recv_ffn_output=fail
    )
    receive = async_cam_dbo.send_async_cam_remote_call(
        connector, torch.ones(2, 1), _context(0)
    )
    with pytest.raises(ValueError, match="wire failure"):
        receive()
    with pytest.raises(RuntimeError, match="restart"):
        receive()
    assert calls == ["recv"]
