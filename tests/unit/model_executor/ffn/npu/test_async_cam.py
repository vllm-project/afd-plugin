# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Async CAM model-side layout and native DBO handoff contracts."""

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.v1.worker import ubatching  # noqa: E402

from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata  # noqa: E402
from afd_plugin.model_executor.ffn.npu import async_cam  # noqa: E402


def test_plain_tp_cam_boundary_shards_and_restores_replicated_tokens(monkeypatch):
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    router_logits = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    padded_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100

    monkeypatch.setattr(
        async_cam,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_output,
    )

    expected_hidden_rows = (
        hidden_states[:3],
        torch.cat((hidden_states[3:], hidden_states.new_zeros((1, 2)))),
    )
    for tp_rank in range(2):
        tp_group.rank_in_group = tp_rank
        payload = async_cam.prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=False,
        )

        assert torch.equal(payload.hidden_states, expected_hidden_rows[tp_rank])
        assert payload.hidden_states.shape[0] == 3
        assert payload.topk_weights.shape[0] == 3
        assert payload.topk_ids.shape[0] == 3
        assert payload.router_logits is not None
        assert payload.router_logits.shape[0] == 3
        assert payload.layout.parent_tokens == 5
        assert payload.layout.padded_tokens == 6
        assert payload.layout.requires_tp_all_gather is True

        local_output = padded_output[tp_rank * 3 : (tp_rank + 1) * 3]
        restored = async_cam.restore_cam_dispatch_output(
            local_output,
            payload.layout,
        )
        assert torch.equal(restored, padded_output[:5])


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
    first = async_cam.send_async_cam_remote_call(connector, hidden[0], _context(0))
    registry[identity] = 1
    second = async_cam.send_async_cam_remote_call(connector, hidden[1], _context(1))
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
    monkeypatch.setattr(async_cam, "dbo_enabled", lambda: False)
    monkeypatch.setattr(
        async_cam, "dbo_maybe_run_recv_hook", lambda: pytest.fail("inactive hook")
    )
    monkeypatch.setattr(
        async_cam, "dbo_register_recv_hook", lambda _: pytest.fail("inactive hook")
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
    receive = async_cam.send_async_cam_remote_call(connector, value, _context(0))
    assert calls == ["send"]
    assert receive() is value and receive() is value
    assert calls == ["send", "recv"]


def test_failed_receive_is_never_resubmitted(monkeypatch):
    monkeypatch.setattr(async_cam, "dbo_enabled", lambda: False)
    calls = []

    def fail(**kwargs):
        calls.append("recv")
        raise ValueError("wire failure")

    connector = SimpleNamespace(
        send_attn_output=lambda *a, **k: None, recv_ffn_output=fail
    )
    receive = async_cam.send_async_cam_remote_call(
        connector, torch.ones(2, 1), _context(0)
    )
    with pytest.raises(ValueError, match="wire failure"):
        receive()
    with pytest.raises(RuntimeError, match="restart"):
        receive()
    assert calls == ["recv"]
