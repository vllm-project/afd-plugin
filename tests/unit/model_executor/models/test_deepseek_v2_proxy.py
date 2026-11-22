# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.config import AFD_ASYNC_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.ffn import remote  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402


class _PassthroughNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _FakeAttention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["dense", "moe"])
def test_native_decoder_forward_calls_remote_proxy_once(
    monkeypatch,
    layer_idx,
):
    events: list[tuple] = []
    afd_metadata = install_proxy_context(monkeypatch, events)
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
    install_proxy_context(monkeypatch, events, stage_idx=1)
    topk_weights = torch.tensor([[0.75, 0.25]])
    topk_ids = torch.tensor([[1, 3]])
    router_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    def compute_gate_topk(**kwargs):
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
    proxy.gate_router = object()
    hidden_states = torch.ones(1, 4)

    output = proxy(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    send_kwargs = events[0][3]
    assert send_kwargs["router_logits"] is router_logits
    assert send_kwargs["topk_ids"] is topk_ids
    assert send_kwargs["topk_weights"] is topk_weights
    assert torch.equal(output, hidden_states * 0.25)


def test_remote_proxy_requires_forward_metadata(monkeypatch):
    monkeypatch.setattr(
        remote,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
        adapter.RemoteFFNProxy(layer_idx=0)(torch.ones(1, 4))


def test_remote_proxy_exchanges_cam_during_profile(monkeypatch):
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    monkeypatch.setattr(
        remote,
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


@pytest.mark.usefixtures("registered_routing_ids_op")
def test_attention_gate_balances_only_profile_routing(monkeypatch):
    pytest.importorskip("vllm_ascend")
    from vllm.forward_context import override_forward_context

    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_attention_gate as gate_module,
    )

    monkeypatch.delenv("AFD_FORCE_BALANCED_TOPK_IDS", raising=False)
    logits = torch.zeros(3, 4)
    weights = torch.full((3, 2), 0.5)
    hot_ids = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)

    class Gate(nn.Module):
        e_score_correction_bias = None

        def forward(self, hidden_states):
            return logits, None

    connector = SimpleNamespace(
        select_experts=lambda **kwargs: (weights.clone(), hot_ids.clone())
    )
    monkeypatch.setattr(
        gate_module,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(connector=connector),
    )
    config = SimpleNamespace(n_routed_experts=4, n_shared_experts=0)
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    for is_profile in (False, True, False):
        context = SimpleNamespace(
            in_profile_run=is_profile,
            additional_kwargs={"in_profile_run": is_profile},
        )
        with override_forward_context(context):
            actual_weights, actual_ids, actual_logits = gate_module.compute_gate_topk(
                gate=Gate(),
                vllm_config=vllm_config,
                config=config,
                top_k=2,
                hidden_states=torch.zeros(3, 4),
            )
        expected_ids = torch.tensor([[0, 1], [2, 3], [0, 1]]) if is_profile else hot_ids
        torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(actual_weights, weights)
        assert actual_logits is logits


def test_default_proxy_is_parameter_free_and_sends_no_routes(monkeypatch):
    """The proxy sends only hidden states and owns no parameters.

    Why: the FFN role runs the whole MLP or MoE block behind the proxy, so any
    extra payload or local weight would break that contract.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    proxy = adapter.RemoteFFNProxy(layer_idx=0)

    proxy(torch.ones(2, 4))

    assert list(proxy.parameters()) == []
    assert events[0][3] == {}
