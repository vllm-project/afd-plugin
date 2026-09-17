# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.config import AFD_ASYNC_NPU_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402


class _FakeConnector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor * 0.25


class _PassthroughNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _FakeAttention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states


def _install_fake_forward_context(monkeypatch, events, *, stage_idx=2):
    connector = _FakeConnector(events)
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=9)
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    # The stage is the DBO ubatch this thread is running, which vLLM tracks by
    # thread rather than on the forward context. Off DBO the helper returns
    # None and the metadata's own stage stands.
    monkeypatch.setattr(adapter, "current_dbo_ubatch_id", lambda: stage_idx)

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", record_yield)
    return afd_metadata


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["dense", "moe"])
def test_native_decoder_forward_calls_remote_proxy_once(
    monkeypatch,
    layer_idx,
):
    events: list[tuple] = []
    afd_metadata = _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(adapter.native, "DeepseekAttention", _FakeAttention)

    layer = object.__new__(adapter.AFDDeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.use_mha = True
    layer.use_sequence_parallel_moe = False
    layer.routed_scaling_factor = 4.0
    layer.input_layernorm = _PassthroughNorm()
    layer.self_attn = _FakeAttention()
    layer.post_attention_layernorm = _PassthroughNorm()
    layer.mlp = adapter.RemoteFFNProxy(layer_idx=layer_idx)

    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)
    output, residual = layer(
        torch.arange(2),
        hidden_states,
        None,
    )

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    sent_metadata = events[0][2].metadata
    assert sent_metadata.layer_idx == layer_idx
    assert sent_metadata.stage_idx == 2
    assert sent_metadata.seq_lens == [2]
    assert events[1][2] == "attention"
    assert events[2][2] == 2
    assert afd_metadata.stage_idx == 2
    assert torch.equal(output, hidden_states * 0.25)
    assert torch.equal(residual, hidden_states)


@pytest.mark.parametrize(
    ("mlp_type", "expected_scale"),
    [("dense", 0.25), ("moe", 1.0)],
)
def test_ffn_compute_applies_dense_fp16_scaling_once(
    monkeypatch,
    mlp_type,
    expected_scale,
):
    class FakeDenseMLP(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.clone()

    class FakeMoE(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.clone()

    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", FakeDenseMLP)
    layer = object.__new__(adapter.AFDDeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.compute_gate_on_attention = False
    layer.routed_scaling_factor = 4.0
    layer.mlp = FakeDenseMLP() if mlp_type == "dense" else FakeMoE()
    hidden_states = torch.full((2, 4), 8.0, dtype=torch.float16)

    output = layer.compute_ffn_output(hidden_states)

    assert torch.equal(output, hidden_states * expected_scale)


def test_gate_proxy_sends_routing_payload(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    topk_weights = torch.tensor([[0.75, 0.25]])
    topk_ids = torch.tensor([[1, 3]])
    router_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    gate_calls = []

    def compute_gate_topk(**kwargs):
        gate_calls.append(kwargs)
        return topk_weights, topk_ids, router_logits

    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "compute_gate_topk",
        compute_gate_topk,
    )
    proxy = object.__new__(adapter.GateOnlyRemoteMoE)
    adapter.RemoteFFNProxy.__init__(proxy, layer_idx=3)
    proxy.gate = nn.Linear(4, 4, bias=False)
    proxy.vllm_config = object()
    proxy.config = object()
    proxy.top_k = 2
    hidden_states = torch.ones(1, 4)

    output = proxy(hidden_states)

    assert len(gate_calls) == 1
    assert gate_calls[0]["gate"] is proxy.gate
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    send_kwargs = events[0][3]
    assert send_kwargs["router_logits"] is router_logits
    assert send_kwargs["topk_ids"] is topk_ids
    assert send_kwargs["topk_weights"] is topk_weights
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_experts_proxy_sends_router_logits(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    proxy = adapter.AFDAttentionFusedMoE(
        layer_idx=3,
        is_internal_router=False,
    )
    hidden_states = torch.ones(1, 4)
    router_logits = torch.ones(1, 8)

    output = proxy(hidden_states, router_logits)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    context = events[0][2]
    assert context.metadata.layer_idx == 3
    assert context.metadata.stage_idx == 1
    assert context.states is None
    assert events[0][3]["router_logits"] is router_logits
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_proxy_requires_forward_metadata(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
        adapter.RemoteFFNProxy(layer_idx=0)(torch.ones(1, 4))


def test_remote_proxy_exchanges_cam_during_profile(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    hidden_states = torch.ones(2, 4)

    output = adapter.RemoteFFNProxy(layer_idx=0)(hidden_states)

    assert torch.equal(output, hidden_states * 0.25)
    assert [event[0] for event in events] == ["send", "yield", "recv"]


def test_synchronous_model_forward_delegates_to_native(monkeypatch):
    calls = []
    expected = torch.ones(1, 4)

    def native_forward(instance, *args):
        calls.append((instance, args))
        return expected

    monkeypatch.setattr(adapter.native.DeepseekV2Model, "forward", native_forward)
    model = object.__new__(adapter.AFDDeepseekV2Model)
    nn.Module.__init__(model)
    model.afd_config = AFDConfig(role="attention")
    positions = torch.arange(1)

    output = adapter.AFDDeepseekV2Model.forward(
        model,
        None,
        positions,
        None,
    )

    assert output is expected
    assert calls == [(model, (None, positions, None, None))]


def test_async_connector_dispatches_to_schedule_adapter(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward

    expected = torch.ones(1, 4)
    calls = []

    def async_forward(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "run_model_forward",
        async_forward,
    )
    monkeypatch.setattr(
        adapter.native.DeepseekV2Model,
        "forward",
        lambda *_args: pytest.fail("native synchronous forward was called"),
    )
    model = object.__new__(adapter.AFDDeepseekV2Model)
    nn.Module.__init__(model)
    model.afd_config = AFDConfig(
        role="attention",
        connector=AFD_ASYNC_NPU_CONNECTOR,
    )
    positions = torch.arange(1)

    output = adapter.AFDDeepseekV2Model.forward(
        model,
        None,
        positions,
        None,
    )

    assert output is expected
    assert calls == [(model, None, positions, None, None)]


def test_decoder_inherits_native_forward_without_override():
    assert "forward" not in adapter.AFDDeepseekV2DecoderLayer.__dict__
    assert (
        adapter.AFDDeepseekV2DecoderLayer.forward
        is adapter.native.DeepseekV2DecoderLayer.forward
    )
