# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native CAM message contracts and ownership through the public connector."""

import sys
import weakref
from types import SimpleNamespace

import pytest

from tests.unit.fakes.async_cam import (
    _FakeTensor,
    _topk_payload,
    recording_cam_connector,
)
from tests.unit.fakes.async_cam_startup import MemoryStore, startup_world
from tests.unit.fakes.async_ffn import load_async_modules


@pytest.fixture
def cam(monkeypatch):
    _, connector, _ = load_async_modules(monkeypatch)
    return SimpleNamespace(
        create=lambda **kwargs: recording_cam_connector(
            monkeypatch, connector, **kwargs
        ),
        connector_module=connector,
        metadata=sys.modules["afd_plugin.connectors.metadata"],
    )


def test_async_connector_calls_cam_shaped_ops(cam):
    h = cam.create(pcp_size=3)
    connector, fake_torch = h.connector, h.torch
    hidden_states = _FakeTensor((3, 16))
    metadata = cam.metadata.AFDTransferMetadata.create_attention_metadata(
        layer_idx=2,
        stage_idx=0,
        seq_len=3,
    )
    context = cam.metadata.AFDTransferContext(metadata=metadata)

    output = connector.send_attn_output(
        hidden_states,
        context,
        **_topk_payload(3),
    )
    combined = connector.recv_ffn_output(
        ref_tensor=hidden_states,
        ubatch_idx=0,
    )

    assert output is None
    assert combined.shape == (3, 16)
    assert fake_torch.ops.afd_ascend.calls[0][0] == "dispatch_send"
    assert fake_torch.ops.afd_ascend.calls[1][0] == "combine_recv"
    assert fake_torch.ops.afd_ascend.calls[0][1][3] == cam.connector_module.CAM_COMM_ID
    assert fake_torch.ops.afd_ascend.calls[1][1][4] == cam.connector_module.CAM_COMM_ID
    assert fake_torch.ops.afd_ascend.calls[0][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.afd_ascend.calls[1][1][5:11] == (3, 16, 2, 2, 4, 4)
    assert fake_torch.ops.afd_ascend.calls[0][1][14] == 4
    assert isinstance(context.states, cam.connector_module.AFDAsyncTransferState)
    assert isinstance(context.states, cam.metadata.AFDTransferState)


def test_async_explicit_receive_consumes_matching_pending_dispatch(cam):
    h = cam.create()
    connector, fake_torch = h.connector, h.torch
    hidden_states = _FakeTensor((3, 16))
    contexts = []
    payloads = []
    for layer_idx in (2, 3):
        context = cam.metadata.AFDTransferContext(
            metadata=cam.metadata.AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx, stage_idx=0, seq_len=3
            )
        )
        payload = _topk_payload(3)
        connector.send_attn_output(hidden_states, context, **payload)
        contexts.append(context)
        payloads.append(payload)

    with pytest.raises(RuntimeError, match="does not match the next pending"):
        connector.recv_ffn_output(hidden_states, ubatch_idx=0, context=contexts[1])
    assert len(fake_torch.ops.afd_ascend.calls) == 2

    connector.recv_ffn_output(
        hidden_states,
        ubatch_idx=0,
        context=contexts[0],
        **payloads[1],
    )
    assert fake_torch.ops.afd_ascend.calls[-1][1][1] is payloads[0]["topk_ids"]

    connector.recv_ffn_output(hidden_states, ubatch_idx=0)
    assert fake_torch.ops.afd_ascend.calls[-1][1][1] is payloads[1]["topk_ids"]


@pytest.mark.parametrize("mega", [False, True])
def test_protocol_owns_stage_queue_or_exclusive_window(cam, mega):
    h = cam.create(mega=mega)
    hidden = _FakeTensor((3, 16))
    contexts = [
        cam.metadata.AFDTransferContext(
            cam.metadata.AFDTransferMetadata.create_attention_metadata(
                layer_idx=2, stage_idx=stage, seq_len=3
            )
        )
        for stage in (0, 1)
    ]
    h.connector.send_attn_output(hidden, contexts[0], **_topk_payload(3))
    if mega:
        with pytest.raises(RuntimeError, match="Pull window is pending"):
            h.connector.send_attn_output(hidden, contexts[1], **_topk_payload(3))
        assert len(h.torch.ops.afd_ascend.calls) == 1
        h.connector.recv_ffn_output(hidden, ubatch_idx=0, context=contexts[0])
        h.connector.send_attn_output(hidden, contexts[1], **_topk_payload(3))
    else:
        h.connector.send_attn_output(hidden, contexts[1], **_topk_payload(3))
    h.connector.recv_ffn_output(hidden, ubatch_idx=1, context=contexts[1])
    if mega:
        assert h.torch.ops.afd_ascend.calls[-1][0] == "mega_combine_recv"
        assert h.torch.ops.afd_ascend.calls[-1][1][4] == 0x41464401
        assert h.torch.ops.afd_ascend.calls[-1][1][-2:] == (16, 4)
    else:
        assert h.connector.has_pending_attention
        h.connector.recv_ffn_output(hidden, ubatch_idx=0, context=contexts[0])
    assert not h.connector.has_pending_attention


@pytest.mark.parametrize("mega", [False, True])
def test_failed_combine_retains_original_routing_for_retry(cam, monkeypatch, mega):
    h = cam.create(mega=mega)
    hidden = _FakeTensor((3, 16))
    context = cam.metadata.AFDTransferContext(
        cam.metadata.AFDTransferMetadata.create_attention_metadata(
            layer_idx=2, stage_idx=0, seq_len=3
        )
    )
    payload = _topk_payload(3)
    h.connector.send_attn_output(hidden, context, **payload)
    operation = "afd_mega_combine_recv" if mega else "afd_async_combine_recv"
    with monkeypatch.context() as patch:

        def fail(**kwargs):
            raise RuntimeError("combine launch failed")

        patch.setattr(h.torch.ops.afd_ascend, operation, fail)
        with pytest.raises(RuntimeError, match="combine launch failed"):
            h.connector.recv_ffn_output(hidden, context=context)
    assert h.connector.has_pending_attention
    h.connector.recv_ffn_output(hidden, context=context)
    assert h.torch.ops.afd_ascend.calls[-1][1][1] is payload["topk_ids"]
    assert not h.connector.has_pending_attention


def test_pull_ffn_rejects_separate_receive_and_send(cam):
    h = cam.create(mega=True, role="ffn")
    with pytest.raises(RuntimeError, match="Mega executor"):
        h.connector.recv_attn_output()
    with pytest.raises(RuntimeError, match="Mega executor"):
        h.connector.send_ffn_output(_FakeTensor((3, 16)), None)
    assert not h.torch.ops.afd_ascend.calls


def test_async_ffn_side_dispatch_recv_and_combine_send(cam):
    h = cam.create(role="ffn", tp_size=2, pcp_size=2)
    connector, fake_torch = h.connector, h.torch
    recv_output = connector.recv_attn_output(batch_size=4, layer_idx=1)
    connector.send_ffn_output(recv_output.hidden_states, recv_output.context)

    states = recv_output.context.states
    assert recv_output.hidden_states.shape == (connector.max_num_batched_tokens, 16)
    assert states.dynamic_scales.shape == (connector.max_num_batched_tokens,)
    assert states.group_list.shape == (connector.topology.expert_per_rank,)
    assert fake_torch.ops.afd_ascend.calls[0][0] == "dispatch_recv"
    assert fake_torch.ops.afd_ascend.calls[1][0] == "combine_send"
    assert fake_torch.ops.afd_ascend.calls[0][1][11] == 2
    assert fake_torch.ops.afd_ascend.calls[1][1][12] == 2
    assert fake_torch.ops.afd_ascend.calls[1][1][2] is states.token_nums_rankid_layeridx


def test_async_combine_send_requires_dispatch_recv_token_metadata(cam):
    h = cam.create(role="ffn")
    connector = h.connector
    metadata = cam.metadata.AFDTransferMetadata.create_ffn_metadata(
        layer_idx=1,
        stage_idx=0,
        seq_lens=[4],
    )
    # token_nums_rankid_layeridx is left unset (None) so combine send must fail.
    states = cam.connector_module.AFDAsyncTransferState(
        batch_size=4,
        hidden_size=16,
        topk=2,
        layer_idx=1,
        group_list=_FakeTensor((4,), dtype="int64"),
    )
    context = cam.metadata.AFDTransferContext(metadata=metadata, states=states)

    with pytest.raises(RuntimeError, match="TokenNums_Rankid_Layeridx"):
        connector.send_ffn_output(_FakeTensor((4, 16)), context)


@pytest.mark.parametrize("mega", [False, True])
def test_dispatch_failure_preserves_protocol_window_ownership(cam, monkeypatch, mega):
    h = cam.create(mega=mega)
    connector, fake_torch = h.connector, h.torch
    context = cam.metadata.AFDTransferContext(
        metadata=cam.metadata.AFDTransferMetadata.create_attention_metadata(
            layer_idx=2,
            stage_idx=0,
            seq_len=3,
        )
    )

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(
        fake_torch.ops.afd_ascend,
        "afd_mega_dispatch_send" if mega else "afd_async_dispatch_send",
        fail_dispatch,
    )
    with pytest.raises(RuntimeError, match="dispatch failed"):
        connector.send_attn_output(_FakeTensor((3, 16)), context, **_topk_payload(3))
    if mega:
        with pytest.raises(RuntimeError, match="pending"):
            connector.close()
    else:
        connector.close()
        assert not connector.is_initialized


def test_close_releases_every_stage_routing(cam):
    h = cam.create()
    connector, fake_torch = h.connector, h.torch
    references: list[weakref.ReferenceType] = []
    for stage in (0, 1):
        context = cam.metadata.AFDTransferContext(
            metadata=cam.metadata.AFDTransferMetadata.create_attention_metadata(
                layer_idx=2, stage_idx=stage, seq_len=3
            )
        )
        payload = _topk_payload(3)
        references.extend(weakref.ref(tensor) for tensor in payload.values())
        connector.send_attn_output(_FakeTensor((3, 16)), context, **payload)
    del payload, context
    fake_torch.ops.afd_ascend.calls.clear()
    assert all(reference() is not None for reference in references)
    connector.close()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "operation",
    ["send_attn_output", "recv_ffn_output", "recv_attn_output", "send_ffn_output"],
)
def test_closed_connector_rejects_transfers_before_native_launch(cam, operation):
    h = cam.create()
    h.connector.close()
    hidden = _FakeTensor((1, 16))
    context = cam.metadata.AFDTransferContext(
        metadata=cam.metadata.AFDTransferMetadata.create_attention_metadata(
            layer_idx=2, stage_idx=0, seq_len=1
        )
    )
    arguments = {
        "send_attn_output": (hidden, context),
        "recv_ffn_output": (hidden,),
        "recv_attn_output": (),
        "send_ffn_output": (hidden, context),
    }
    with pytest.raises(RuntimeError, match="not initialized"):
        getattr(h.connector, operation)(*arguments[operation])
    assert h.torch.ops.afd_ascend.calls == []


def test_reinitialization_updates_legacy_window_reader(cam):
    h = cam.create()
    first_window = h.connector.comm_args
    h.connector.close()
    assert h.connector.comm_args is None
    h.connector.ffn_session().initialize("npu:1", "replacement")
    replacement = h.connector.comm_args
    assert replacement is not None and replacement is not first_window
    assert replacement.device == "npu:1"
    with pytest.raises(AttributeError):
        h.connector.comm_args = first_window


@pytest.mark.parametrize("mega", [False, True])
def test_window_allocation_failure_reaches_peer_before_ready(cam, monkeypatch, mega):
    h = cam.create(mega=mega)
    connector = h.connector
    connector._initialized = False
    connector.ffn_session().release()
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "window-failure")
    group = SimpleNamespace(_get_backend=lambda device: backend)
    monkeypatch.setattr(
        cam.connector_module, "init_afd_process_group", lambda **kwargs: group
    )
    store = MemoryStore()
    peer = startup_world(attn=4, ffn=2, store=store)[1]
    connector.graph_startup._store = store
    connector.graph_startup._prefix = peer._prefix

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("window allocation failed")

    monkeypatch.setattr(h.torch, "empty", fail_allocation)
    with pytest.raises(RuntimeError, match="window allocation failed"):
        connector.init_afd_connector()
    with pytest.raises(RuntimeError, match="window allocation failed"):
        peer.check_failure()
    assert not connector.is_initialized and connector.cam_pg is group
    assert connector.comm_args is None
