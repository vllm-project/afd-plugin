from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransportSpec,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector  # noqa: E402


def _vllm_config():
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            hf_config=SimpleNamespace(hidden_size=4, num_hidden_layers=3),
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
    )


def _ffn_connector(*, attention_ranks=1):
    return P2pNcclAFDConnector(
        rank=0,
        local_rank=0,
        vllm_config=_vllm_config(),
        afd_config=AFDConfig(
            role="ffn",
            num_attention_ranks=attention_ranks,
            num_ffn_ranks=1,
        ),
    )


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
    received = iter(
        [hidden_peer_1, router_peer_1, hidden_peer_2, router_peer_2],
    )
    monkeypatch.setattr(
        connector,
        "_recv_hidden_states",
        lambda *args, **kwargs: next(received),
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
    received = []

    def recv_hidden_states(*args, **kwargs):
        received.append((args, kwargs))
        return hidden_states

    monkeypatch.setattr(connector, "_recv_hidden_states", recv_hidden_states)

    payload = connector.recv_attn_output()

    assert len(received) == 1
    assert payload.hidden_states is hidden_states
    assert payload.router_logits is None
    assert payload.context.states is None


def test_attention_send_orders_token_ids_after_hidden_states(monkeypatch):
    connector = _attention_connector()
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    input_ids = torch.tensor([11, 13], dtype=torch.int64)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    connector.send_attn_output(
        hidden_states,
        _context(),
        transport_spec=AFDA2FTransportSpec(),
        input_ids=input_ids,
    )

    assert sent == [hidden_states, input_ids]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"version": 2}, "unsupported AFD A2F transport version"),
        (
            {"input_ids_dtype": torch.int32},
            "version 1 input_ids must use torch.int64",
        ),
    ],
)
def test_a2f_transport_spec_rejects_unsupported_schema(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AFDA2FTransportSpec(**kwargs)


@pytest.mark.parametrize(
    "send_kwargs",
    [
        {"input_ids": torch.tensor([11, 13], dtype=torch.int64)},
        {"transport_spec": AFDA2FTransportSpec()},
    ],
)
def test_attention_send_rejects_schema_field_mismatch_before_send(
    monkeypatch,
    send_kwargs,
):
    connector = _attention_connector()
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_hidden_states",
        lambda tensor, *args: sent.append(tensor),
    )

    with pytest.raises(ValueError, match="presence must exactly match"):
        connector.send_attn_output(
            torch.ones((2, 4), dtype=torch.bfloat16),
            _context(),
            **send_kwargs,
        )

    assert sent == []


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
    input_ids_peer_1 = torch.tensor([11, 13], dtype=torch.int64)
    hidden_peer_2 = torch.full((2, 4), 2, dtype=torch.bfloat16)
    input_ids_peer_2 = torch.tensor([17, 19], dtype=torch.int64)
    received = iter(
        [hidden_peer_1, input_ids_peer_1, hidden_peer_2, input_ids_peer_2],
    )
    monkeypatch.setattr(
        connector,
        "_recv_hidden_states",
        lambda *args, **kwargs: next(received),
    )

    payload = connector.recv_attn_output(
        ubatch_idx=1,
        transport_spec=AFDA2FTransportSpec(input_ids_dtype=torch.int64),
    )

    assert torch.equal(
        payload.hidden_states,
        torch.cat([hidden_peer_1, hidden_peer_2]),
    )
    assert torch.equal(
        payload.input_ids,
        torch.cat([input_ids_peer_1, input_ids_peer_2]),
    )
    assert payload.input_ids.dtype is torch.int64
    assert payload.context.metadata.seq_lens == [2, 2]
