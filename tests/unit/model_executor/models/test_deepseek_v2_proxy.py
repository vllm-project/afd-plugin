# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

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
