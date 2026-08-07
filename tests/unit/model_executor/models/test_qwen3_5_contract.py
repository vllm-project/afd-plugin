from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from vllm.model_executor.models import qwen3_5 as native  # noqa: E402

from afd_plugin.model_executor.models import qwen3_5 as adapter  # noqa: E402
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDAttentionFusedMoE,
)


def test_qwen_adapter_keeps_native_signatures_and_forward_methods():
    assert inspect.signature(adapter.AFDQwen3_5DecoderLayer.__init__) == (
        inspect.signature(native.Qwen3_5DecoderLayer.__init__)
    )
    assert inspect.signature(adapter.AFDQwen3_5Model.__init__) == inspect.signature(
        native.Qwen3_5Model.__init__
    )
    assert inspect.signature(
        adapter.AFDQwen3_5MoeForConditionalGeneration.__init__
    ) == inspect.signature(native.Qwen3_5MoeForConditionalGeneration.__init__)
    assert adapter.AFDQwen3_5DecoderLayer.forward is native.Qwen3_5DecoderLayer.forward
    assert adapter.AFDQwen3_5Model.forward is native.Qwen3_5Model.forward
    assert (
        adapter.AFDQwen3_5MoeForConditionalGeneration.forward
        is native.Qwen3_5MoeForConditionalGeneration.forward
    )


def test_attention_moe_uses_native_forward_and_parameter_free_proxy():
    assert (
        adapter.AFDQwen3_5RemoteExpertsMoE.forward
        is native.Qwen3NextSparseMoeBlock.forward
    )
    proxy = AFDAttentionFusedMoE(
        layer_idx=7,
        is_internal_router=False,
    )
    assert list(proxy.parameters()) == []


def test_ffn_compute_experts_calls_external_native_runner():
    calls = []

    class FakeExperts(nn.Module):
        is_internal_router = False

        def forward(self, *, hidden_states, router_logits):
            calls.append((hidden_states, router_logits))
            return hidden_states + 1

    moe = object.__new__(native.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(moe)
    moe.experts = FakeExperts()
    layer = object.__new__(adapter.AFDQwen3_5DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "ffn"
    layer.mlp = moe
    hidden_states = torch.zeros(2, 4)
    router_logits = torch.zeros(2, 256)

    output = layer.compute_experts_output(hidden_states, router_logits)

    assert calls == [(hidden_states, router_logits)]
    assert torch.equal(output, hidden_states + 1)


def test_ffn_compute_ffn_output_calls_native_internal_router():
    calls = []

    class FakeInternalMoe(native.Qwen3NextSparseMoeBlock):
        def forward(self, hidden_states):
            calls.append(hidden_states)
            return hidden_states + 1

    moe = object.__new__(FakeInternalMoe)
    nn.Module.__init__(moe)
    moe.experts = type("Experts", (), {"is_internal_router": True})()
    layer = object.__new__(adapter.AFDQwen3_5DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "ffn"
    layer.mlp = moe
    hidden_states = torch.zeros(2, 4)

    output = layer.compute_ffn_output(hidden_states)

    assert calls == [hidden_states]
    assert torch.equal(output, hidden_states + 1)
