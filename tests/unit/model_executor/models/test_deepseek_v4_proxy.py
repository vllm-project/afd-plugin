# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")
nn = torch.nn

from afd_plugin.model_executor.models import deepseek_v2 as proxy_runtime  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


class _FakeAttention(nn.Module):
    def forward(self, *, positions, hidden_states, llama_4_scaling=None):
        del positions, llama_4_scaling
        return hidden_states * 0.75


class _FakeMoE(nn.Module):
    def __init__(self, *, quantized: bool) -> None:
        super().__init__()
        self.quantized = quantized

    def forward(self, hidden_states, input_ids=None):
        del input_ids
        output = hidden_states.float() * 0.375
        if self.quantized:
            output = torch.round(output * 32.0) / 32.0
        return output.to(hidden_states.dtype)


class _LoopbackConnector:
    yield_after_attn_send = True

    def __init__(self, ffn_layer) -> None:
        self.ffn_layer = ffn_layer
        self.sent = []
        self.received = []

    def send_attn_output(self, hidden_states, context, **kwargs):
        self.sent.append((hidden_states.clone(), context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx, **kwargs):
        self.received.append((ref_tensor, ubatch_idx, kwargs))
        return self.ffn_layer.compute_ffn_output(self.sent[-1][0])


class _IdentityNorm(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


def _install_hc_stubs(layer) -> None:
    layer.hc_attn_fn = nn.Parameter(torch.empty(1))
    layer.hc_attn_scale = nn.Parameter(torch.empty(1))
    layer.hc_attn_base = nn.Parameter(torch.empty(1))
    layer.hc_ffn_fn = nn.Parameter(torch.empty(1))
    layer.hc_ffn_scale = nn.Parameter(torch.empty(1))
    layer.hc_ffn_base = nn.Parameter(torch.empty(1))

    def hc_pre(hidden_states, *_args):
        collapsed = hidden_states.float().mean(dim=1).to(hidden_states.dtype)
        marker = torch.zeros_like(collapsed)
        return collapsed, marker, marker

    def hc_post(hidden_states, residual, _post, _comb):
        return residual + hidden_states.unsqueeze(1) * 0.125

    layer.hc_pre = hc_pre
    layer.hc_post = hc_post


def _attention_layer(*, layer_idx: int, moe: nn.Module):
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "attention"
    layer.layer_idx = layer_idx
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _FakeAttention()
    layer.mlp = moe
    _install_hc_stubs(layer)
    return layer


def _ffn_layer(*, layer_idx: int, quantized: bool):
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "ffn"
    layer.layer_idx = layer_idx
    layer.mlp = _FakeMoE(quantized=quantized)
    return layer


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 42])
@pytest.mark.parametrize(
    ("quantized", "tolerance"),
    [(False, 2e-2), (True, 5e-2)],
    ids=["bf16", "w8a8"],
)
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_loopback_layer_matches_local_reference(
    monkeypatch,
    layer_idx,
    quantized,
    tolerance,
    num_tokens,
):
    torch.manual_seed(1024 + layer_idx + num_tokens)
    hidden_states = torch.randn(num_tokens, 4, 8, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int64)

    reference = _attention_layer(
        layer_idx=layer_idx,
        moe=_FakeMoE(quantized=quantized),
    )
    ffn_layer = _ffn_layer(layer_idx=layer_idx, quantized=quantized)
    connector = _LoopbackConnector(ffn_layer)
    split = _attention_layer(
        layer_idx=layer_idx,
        moe=adapter.AFDDeepseekV4RemoteMoEProxy(layer_idx=layer_idx),
    )

    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)
    monkeypatch.setattr(
        proxy_runtime,
        "get_afd_metadata_from_forward_context",
        lambda: afd_metadata,
    )
    fake_forward_context = SimpleNamespace(
        ubatch_idx=0,
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
    )
    monkeypatch.setattr(
        proxy_runtime,
        "get_forward_context",
        lambda: fake_forward_context,
    )
    monkeypatch.setattr(adapter, "get_forward_context", lambda: fake_forward_context)
    monkeypatch.setattr(
        proxy_runtime,
        "maybe_apply_dbo_yield",
        lambda tensor, *, role: tensor,
    )

    expected, _ = reference(positions, hidden_states.clone(), None)
    actual, _ = split(positions, hidden_states.clone(), None)

    assert actual.shape == (num_tokens, 4, 8)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert len(connector.sent) == 1
    assert len(connector.received) == 1
    assert connector.sent[0][1].metadata.layer_idx == layer_idx
    assert connector.sent[0][1].metadata.seq_lens == [num_tokens]


def test_decoder_layer_forwards_layer_identity_to_remote_receive(monkeypatch):
    calls = []
    proxy = adapter.AFDDeepseekV4RemoteMoEProxy(layer_idx=7)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx = 7
    layer.mlp = proxy
    transfer = SimpleNamespace()
    output = torch.ones((1, 8))

    def receive_remote_ffn(_self, value, *, layer_idx=None):
        calls.append((value, layer_idx))
        return output

    monkeypatch.setattr(
        adapter.AFDDeepseekV4RemoteMoEProxy,
        "receive_remote_ffn",
        receive_remote_ffn,
    )

    assert layer.receive_remote_ffn(transfer) is output
    assert calls == [(transfer, 7)]


def test_mtp_remote_moe_does_not_require_or_send_input_ids(monkeypatch):
    hidden_states = torch.ones((3, 4, 8), dtype=torch.bfloat16)
    ffn_layer = _ffn_layer(layer_idx=0, quantized=False)
    connector = _LoopbackConnector(ffn_layer)
    proxy = adapter.AFDDeepseekV4RemoteMoEProxy(layer_idx=0, phase="mtp")
    proxy.attach_connector(connector)

    monkeypatch.setattr(
        proxy_runtime,
        "get_afd_metadata_from_forward_context",
        lambda: None,
    )
    monkeypatch.setattr(
        proxy_runtime,
        "get_forward_context",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: pytest.fail("MTP phase must not query target input_ids"),
    )
    monkeypatch.setattr(
        proxy_runtime,
        "maybe_apply_dbo_yield",
        lambda tensor, *, role: tensor,
    )

    output = proxy(hidden_states)

    assert output.shape == hidden_states.shape
    assert len(connector.sent) == 1
    _, context, send_kwargs = connector.sent[0]
    assert context.metadata.phase == "mtp"
    assert context.metadata.layer_idx == 0
    assert "input_ids" not in send_kwargs
    assert send_kwargs["num_tokens_across_dp"].tolist() == [3]
    assert connector.received[0][2] == {"phase": "mtp"}


def test_single_mtp_layer_preserves_proposal_step_for_remote_protocol():
    observed_steps = []

    class RecordingLayer(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_index,
        ):
            del input_ids, positions, previous_hidden_states, inputs_embeds
            observed_steps.append(spec_step_index)
            return torch.ones((1, 8))

    predictor = object.__new__(adapter.AFDDeepSeekMultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.afd_role = "attention"
    predictor.num_mtp_layers = 1
    predictor.layers = nn.ModuleDict({"0": RecordingLayer()})

    for speculative_step in range(3):
        predictor(
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int64),
            torch.ones((1, 8)),
            torch.ones((1, 8)),
            spec_step_idx=speculative_step,
        )

    assert observed_steps == [0, 1, 2]


def test_mtp_model_numbers_implicit_merged_draft_steps():
    observed_steps = []

    class RecordingPredictor(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        ):
            del input_ids, positions, inputs_embeds
            observed_steps.append(spec_step_idx)
            return hidden_states

    model = object.__new__(adapter.AFDDeepSeekV4MTP)
    nn.Module.__init__(model)
    model.model = RecordingPredictor()
    model._afd_num_speculative_tokens = 3
    model._afd_next_speculative_step = 0
    inputs = torch.ones((1, 8))

    for _ in range(4):
        model.forward(inputs, inputs, inputs)
    model.forward(inputs, inputs, inputs, spec_step_idx=2)

    assert observed_steps == [0, 1, 2, 0, 2]
