# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector  # noqa: E402


def _vllm_config(*, enforce_eager=True):
    text_config = SimpleNamespace(hidden_size=4, num_hidden_layers=3)
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=text_config,
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )


def _attention_connector():
    return P2pNcclAFDConnector(
        rank=0,
        local_rank=0,
        vllm_config=_vllm_config(),
        afd_config=AFDConfig(role="attention"),
        role_rank=0,
    )


def _ffn_connector(*, attention_ranks=1, enforce_eager=True):
    return P2pNcclAFDConnector(
        rank=0,
        local_rank=0,
        vllm_config=_vllm_config(enforce_eager=enforce_eager),
        afd_config=AFDConfig(
            role="ffn",
            num_attention_ranks=attention_ranks,
            num_ffn_ranks=1,
        ),
        role_rank=0,
    )


def _deliver_on_wire(monkeypatch, connector, payload):
    """Fake the FFN receive ops: write ``payload`` into the buffers in wire order.

    Returns the list of buffers the connector received into.
    """
    connector.a2e_group = SimpleNamespace(world_size=connector.group_size)
    connector.a2e_comm_id = 0
    incoming = iter(payload)
    buffers = []

    def recv_group(outs, srcs, comm_id):
        for out in outs:
            out.copy_(next(incoming))
            buffers.append(out)

    monkeypatch.setattr(torch.ops.vllm, "afd_p2p_recv_group", recv_group, raising=False)
    monkeypatch.setattr(
        torch.ops.vllm,
        "afd_p2p_recv",
        lambda out, src, comm_id: recv_group([out], [src], comm_id),
        raising=False,
    )
    return buffers


def _context():
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=2,
            stage_idx=1,
            seq_len=2,
        ),
    )


def test_attention_gate_reuses_hidden_state_send_for_router_logits(monkeypatch):
    connector = _attention_connector()
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    router_logits = torch.ones((2, 3), dtype=torch.float32)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    connector.send_attn_output(
        hidden_states,
        _context(),
        router_logits=router_logits,
    )

    assert sent == [hidden_states, router_logits]


def test_ffn_gate_sends_only_hidden_states(monkeypatch):
    connector = _attention_connector()
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    connector.send_attn_output(
        hidden_states,
        _context(),
    )

    assert sent == [hidden_states]


def test_router_shape_is_validated_before_any_send(monkeypatch):
    connector = _attention_connector()
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    with pytest.raises(ValueError, match="equal token counts"):
        connector.send_attn_output(
            torch.ones((2, 4), dtype=torch.bfloat16),
            _context(),
            router_logits=torch.ones((3, 3), dtype=torch.float32),
        )

    assert sent == []


def test_attention_gate_fan_in_preserves_peer_order(monkeypatch):
    connector = _ffn_connector(attention_ranks=2)
    connector.tensor_metadata_list[1] = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        size=torch.Size([4, 4]),
    )
    for src in (1, 2):
        connector._recv_attn_tensor_metadata_list[(1, src)] = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            size=torch.Size([2, 4]),
        )

    hidden_peer_1 = torch.full((2, 4), 1, dtype=torch.bfloat16)
    router_peer_1 = torch.full((2, 3), 10, dtype=torch.float32)
    hidden_peer_2 = torch.full((2, 4), 2, dtype=torch.bfloat16)
    router_peer_2 = torch.full((2, 3), 20, dtype=torch.float32)
    _deliver_on_wire(
        monkeypatch,
        connector,
        [hidden_peer_1, router_peer_1, hidden_peer_2, router_peer_2],
    )

    payload = connector.recv_attn_output(
        ubatch_idx=1,
        routing_spec=AFDExpertRoutingSpec(
            router_logits_width=3,
            router_logits_dtype=torch.float32,
        ),
    )

    assert torch.equal(
        payload.hidden_states,
        torch.cat([hidden_peer_1, hidden_peer_2]),
    )
    assert payload.context.metadata.layer_idx == 0
    assert payload.context.metadata.stage_idx == 1
    assert payload.context.metadata.seq_lens == [2, 2]
    assert torch.equal(
        payload.router_logits,
        torch.cat([router_peer_1, router_peer_2]),
    )
    assert payload.context.states is None


def test_ffn_gate_receive_does_not_expect_router_logits(monkeypatch):
    connector = _ffn_connector()
    connector.tensor_metadata_list[0] = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        size=torch.Size([2, 4]),
    )
    connector._recv_attn_tensor_metadata_list[(0, 1)] = connector.tensor_metadata_list[
        0
    ]
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    received = _deliver_on_wire(monkeypatch, connector, [hidden_states])

    payload = connector.recv_attn_output()

    assert len(received) == 1
    assert torch.equal(payload.hidden_states, hidden_states)
    assert payload.router_logits is None
    assert payload.context.states is None


def test_attention_send_orders_token_ids_after_hidden_states(monkeypatch):
    connector = _attention_connector()
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    input_ids = torch.tensor([11, 13], dtype=torch.int32)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    connector.send_attn_output(
        hidden_states,
        _context(),
        input_ids=input_ids,
    )

    assert sent == [hidden_states, input_ids]


def test_ffn_fan_in_preserves_input_id_peer_order_and_dtype(monkeypatch):
    connector = _ffn_connector(attention_ranks=2)
    connector.tensor_metadata_list[1] = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        size=torch.Size([4, 4]),
    )
    for src in (1, 2):
        connector._recv_attn_tensor_metadata_list[(1, src)] = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            size=torch.Size([2, 4]),
        )
    hidden_peer_1 = torch.full((2, 4), 1, dtype=torch.bfloat16)
    input_ids_peer_1 = torch.tensor([11, 13], dtype=torch.int32)
    hidden_peer_2 = torch.full((2, 4), 2, dtype=torch.bfloat16)
    input_ids_peer_2 = torch.tensor([17, 19], dtype=torch.int32)
    _deliver_on_wire(
        monkeypatch,
        connector,
        [hidden_peer_1, input_ids_peer_1, hidden_peer_2, input_ids_peer_2],
    )

    payload = connector.recv_attn_output(
        ubatch_idx=1,
        recv_input_ids=True,
    )

    assert torch.equal(
        payload.hidden_states,
        torch.cat([hidden_peer_1, hidden_peer_2]),
    )
    assert torch.equal(
        payload.input_ids,
        torch.cat([input_ids_peer_1, input_ids_peer_2]),
    )
    assert payload.input_ids.dtype is torch.int32
    assert payload.context.metadata.seq_lens == [2, 2]


def test_ffn_graph_receive_reuses_input_ids_buffer(monkeypatch):
    connector = _ffn_connector(enforce_eager=False)
    metadata = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        size=torch.Size([2, 4]),
    )
    connector.tensor_metadata_list[0] = metadata
    connector._recv_attn_tensor_metadata_list[(0, 1)] = metadata
    hidden_ref = torch.empty((2, 4), dtype=torch.bfloat16)
    input_ids_ref = torch.empty(2, dtype=torch.int32)
    connector._recv_attn_buffers[(0, 1, (2, 4))] = hidden_ref
    connector._recv_attn_input_ids_buffers[(0, 1, (2,))] = input_ids_ref
    refs = _deliver_on_wire(
        monkeypatch,
        connector,
        [torch.ones((2, 4), dtype=torch.bfloat16), torch.tensor([3, 5])],
    )

    payload = connector.recv_attn_output(recv_input_ids=True)

    assert refs[0] is hidden_ref
    assert refs[1] is input_ids_ref
    assert payload.hidden_states is hidden_ref
    assert payload.input_ids is input_ids_ref
