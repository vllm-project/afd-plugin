from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import AFDA2FTransportSpec  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


class _FakeConnector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor * 0.25


def test_remote_v4_ffn_sends_only_ffn_tensor_spec_and_token_ids(monkeypatch):
    events = []
    connector = _FakeConnector(events)
    afd_metadata = SimpleNamespace(
        connector=connector,
        stage_idx=9,
    )
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=2),
    )

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", record_yield)
    model = object.__new__(adapter.AFDDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=4)
    transport_spec = model.get_afd_transport_spec(3)
    proxy = adapter.RemoteDeepseekV4FFN(
        layer_idx=3,
        transport_spec=transport_spec,
    )
    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)
    input_ids = torch.tensor([11, 13], dtype=torch.int32)

    output = proxy(hidden_states, input_ids)

    assert transport_spec is adapter._DEEPSEEK_V4_TRANSPORT_SPEC
    assert transport_spec == AFDA2FTransportSpec()
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    sent_context = events[0][2]
    assert sent_context.metadata.layer_idx == 3
    assert sent_context.metadata.stage_idx == 2
    assert sent_context.metadata.seq_lens == [2]
    assert sent_context.states is None
    assert set(events[0][3]) == {"transport_spec", "input_ids"}
    assert events[0][3]["transport_spec"] is transport_spec
    assert torch.equal(
        events[0][3]["input_ids"],
        input_ids.to(dtype=torch.int64),
    )
    assert events[0][3]["input_ids"].dtype is torch.int64
    assert events[1][2] == "attention"
    assert events[2][2] == 2
    assert afd_metadata.stage_idx == 2
    assert torch.equal(output, hidden_states * 0.25)


@pytest.mark.parametrize("layer_idx", [-1, 4])
def test_v4_transport_plan_rejects_out_of_range_layer(layer_idx):
    model = object.__new__(adapter.AFDDeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=4)

    with pytest.raises(IndexError, match="invalid DeepSeek-V4 layer index"):
        model.get_afd_transport_spec(layer_idx)


@pytest.mark.parametrize(
    "input_ids",
    [None, torch.ones((2, 1), dtype=torch.int64), torch.ones(3, dtype=torch.int64)],
    ids=["missing", "not-one-dimensional", "token-count-mismatch"],
)
def test_remote_v4_ffn_validates_token_ids_before_metadata_lookup(
    monkeypatch,
    input_ids,
):
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: pytest.fail("metadata lookup must follow token-id validation"),
    )
    proxy = adapter.RemoteDeepseekV4FFN(
        layer_idx=0,
        transport_spec=AFDA2FTransportSpec(),
    )

    error = RuntimeError if input_ids is None else ValueError
    message = (
        "requires input_ids"
        if input_ids is None
        else "one-dimensional and token-aligned"
    )
    with pytest.raises(error, match=message):
        proxy(torch.ones((2, 4)), input_ids)


def test_v4_decoder_forward_rejects_ffn_role(monkeypatch):
    class FakeMissingLayer(torch.nn.Module):
        pass

    monkeypatch.setattr(adapter.native, "PPMissingLayer", FakeMissingLayer)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.attn = FakeMissingLayer()

    with pytest.raises(RuntimeError, match="decoder forward is Attention-owned"):
        layer.forward(
            torch.ones((2, 4)),
            positions=torch.arange(2),
            input_ids=None,
        )


def test_v4_ffn_compute_rejects_attention_role_before_input_ids():
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.ffn = adapter.RemoteDeepseekV4FFN(
        layer_idx=0,
        transport_spec=AFDA2FTransportSpec(),
    )

    with pytest.raises(RuntimeError, match="FFN compute is FFN-role only"):
        layer.compute_ffn_output(torch.ones((2, 4)), input_ids=None)


def test_v4_ffn_compute_requires_input_ids(monkeypatch):
    class FakeMoE(torch.nn.Module):
        pass

    monkeypatch.setattr(adapter.native, "DeepseekV4MoE", FakeMoE)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.ffn = FakeMoE()

    with pytest.raises(RuntimeError, match="FFN compute requires input_ids"):
        layer.compute_ffn_output(torch.ones((2, 4)), input_ids=None)
