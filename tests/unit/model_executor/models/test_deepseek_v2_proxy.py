# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.config import AFD_ASYNC_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor import remote_moe  # noqa: E402
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
        remote_moe,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    monkeypatch.setattr(
        remote_moe,
        "get_forward_context",
        lambda: SimpleNamespace(ubatch_idx=stage_idx),
    )

    def record_yield(hidden_states, *, role):
        events.append(("yield", hidden_states, role))
        return hidden_states

    monkeypatch.setattr(remote_moe, "maybe_apply_dbo_yield", record_yield)
    return afd_metadata


def _forward_only_remote_moe(layer_idx):
    runner = object.__new__(remote_moe.AFDRemoteMoERunner)
    nn.Module.__init__(runner)
    runner.layer_name = f"model.layers.{layer_idx}.mlp.experts"
    shell = object.__new__(adapter.AFDDeepseekV2RemoteExpertsMoE)
    nn.Module.__init__(shell)
    shell.is_sequence_parallel = False
    shell.routed_scaling_factor = 2.5
    shell.gate = None
    shell.shared_experts = None
    shell.experts = runner
    return shell


@pytest.mark.parametrize(
    ("layer_idx", "use_remote_runner"),
    [(0, False), (1, False), (1, True)],
    ids=["dense-proxy", "moe-proxy", "moe-runner"],
)
def test_native_decoder_forward_calls_remote_proxy_once(
    monkeypatch,
    layer_idx,
    use_remote_runner,
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
    layer.mlp = (
        _forward_only_remote_moe(layer_idx)
        if use_remote_runner
        else adapter.RemoteFFNProxy(layer_idx=layer_idx)
    )

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


@pytest.mark.parametrize("num_tokens", [1, 7])
def test_native_moe_boundary_preserves_legacy_transfer(monkeypatch, num_tokens):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    shell = _forward_only_remote_moe(layer_idx=1)
    hidden_states = torch.arange(num_tokens * 7, dtype=torch.bfloat16).view(
        num_tokens,
        7,
    )

    legacy_output = adapter.RemoteFFNProxy(layer_idx=1)(hidden_states)
    output = shell(hidden_states)

    assert shell.forward.__func__ is adapter.native.DeepseekV2MoE.forward
    assert [event[0] for event in events] == ["send", "yield", "recv"] * 2
    for sent in (events[0], events[3]):
        assert sent[1].shape == hidden_states.shape
        assert sent[1].data_ptr() == hidden_states.data_ptr()
        assert sent[2].metadata.layer_idx == 1
        assert sent[2].metadata.stage_idx == 2
        assert sent[2].metadata.seq_lens == [num_tokens]
        assert sent[3] == {}
    assert torch.equal(output, legacy_output)


@pytest.mark.npu
def test_npu_v4_legacy_proxy_passes_ids_through_common_exchange(monkeypatch):
    pytest.importorskip("vllm_ascend")
    from afd_plugin.model_executor.models.npu import deepseek_v4 as npu_v4
    from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate

    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    input_ids = torch.tensor([11, 13], dtype=torch.int32)
    hidden_states = torch.ones(2, 7, dtype=torch.bfloat16)
    forward_context = SimpleNamespace(ubatch_idx=2)
    monkeypatch.setattr(npu_v4, "get_forward_context", lambda: forward_context)

    def token_ids(*, forward_context: object, router_tokens: int):
        assert router_tokens == 2
        return input_ids

    monkeypatch.setattr(
        deepseek_v4_attention_gate, "hash_input_ids_from_context", token_ids
    )
    output = npu_v4.AFDDeepseekV4RemoteMoE(layer_idx=3)(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert events[0][2].metadata.layer_idx == 3
    assert events[0][2].metadata.stage_idx == 2
    assert set(events[0][3]) == {"input_ids"}
    assert events[0][3]["input_ids"] is input_ids
    assert torch.equal(output, hidden_states * 0.25)


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


def test_cam_decoder_returns_attention_without_routing():
    hidden_states = torch.ones(1, 4)

    def unexpected_routing(_states):
        raise AssertionError("Attention must leave routing to the complete MoE call")

    layer = object.__new__(adapter.AFDDeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.compute_gate_on_attention = True
    layer.is_moe_layer = True
    layer.use_mha = True
    layer.input_layernorm = _PassthroughNorm()
    layer.self_attn = _FakeAttention()
    layer.post_attention_layernorm = _PassthroughNorm()
    layer.mlp = SimpleNamespace(
        experts=SimpleNamespace(_route_native=unexpected_routing)
    )
    output, residual = layer.compute_attn_output(
        torch.zeros(1, dtype=torch.long), hidden_states, None
    )

    assert output is hidden_states
    assert residual is not hidden_states
    assert torch.equal(residual, hidden_states)


def test_remote_experts_runner_sends_router_logits(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events, stage_idx=1)
    proxy = object.__new__(remote_moe.AFDExternalRoutingMoERunner)
    nn.Module.__init__(proxy)
    proxy.layer_name = "model.layers.3.mlp.experts"
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
        remote_moe,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
        adapter.RemoteFFNProxy(layer_idx=0)(torch.ones(1, 4))


def test_remote_proxy_exchanges_cam_during_profile(monkeypatch):
    events: list[tuple] = []
    _install_fake_forward_context(monkeypatch, events)
    monkeypatch.setattr(
        remote_moe,
        "get_forward_context",
        lambda: SimpleNamespace(in_profile_run=True, ubatch_idx=0),
    )
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
        connector=AFD_ASYNC_CONNECTOR,
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
