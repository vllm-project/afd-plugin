# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""W8A8 scale preparation for the DeepSeek-V4 layered FFN executor."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import load_host_module

torch = pytest.importorskip("torch")


@pytest.fixture
def preparation(monkeypatch):
    context = ModuleType("afd_plugin.forward_context")
    context.__dict__["get_afd_metadata_from_forward_context"] = lambda: None
    monkeypatch.setitem(sys.modules, context.__name__, context)
    quantization = ModuleType("vllm_ascend.quantization.quant_type")
    quant_type = SimpleNamespace(W8A8="w8a8", W4A8="w4a8", NONE="none")
    quantization.__dict__["QuantType"] = quant_type
    monkeypatch.setitem(sys.modules, quantization.__name__, quantization)
    module = load_host_module(
        monkeypatch, "afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate"
    )
    return module.prepare_attention_gate_w8a8_scales, quant_type


def layer_for(quant_type, *, dynamic_eplb=False, scale=None):
    owner = SimpleNamespace(dynamic_eplb=dynamic_eplb)
    if scale is not None:
        owner.w13_weight_scale = scale
    experts = SimpleNamespace(quant_type=quant_type, routed_experts=owner)
    return SimpleNamespace(
        is_moe_layer=True, mlp=SimpleNamespace(experts=experts)
    ), owner


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_w8a8_scale_preparation_preserves_canonical_and_cached_values(
    preparation, dtype
):
    prepare, quantization = preparation
    scale = torch.tensor([[0.5, 1.0]], dtype=dtype)
    layer, owner = layer_for(quantization.W8A8, scale=scale)

    prepare([layer])
    cached = owner.w13_weight_scale_fp32
    prepare([layer])

    assert owner.w13_weight_scale is scale
    assert owner.w13_weight_scale_fp32 is cached
    assert cached.dtype == torch.float32
    torch.testing.assert_close(cached, scale.float())


def test_w8a8_scale_is_rebuilt_after_native_finalization(preparation):
    prepare, quantization = preparation
    layer, owner = layer_for(quantization.W8A8, scale=torch.ones(2))
    prepare([layer])
    old = owner.w13_weight_scale_fp32
    owner.w13_weight_scale.add_(1)
    del owner.w13_weight_scale_fp32

    prepare([layer])

    assert owner.w13_weight_scale_fp32 is not old
    torch.testing.assert_close(owner.w13_weight_scale_fp32, torch.full((2,), 2.0))


@pytest.mark.parametrize("kind", ["dense", "none", "w4a8", "dynamic_eplb"])
def test_scale_preparation_skips_layers_without_stacked_w8a8(preparation, kind):
    prepare, quantization = preparation
    layer, owner = layer_for(
        getattr(quantization, kind.upper(), quantization.W8A8),
        dynamic_eplb=kind == "dynamic_eplb",
    )
    if kind == "dense":
        layer.is_moe_layer = False

    prepare([layer])

    assert not hasattr(owner, "w13_weight_scale_fp32")


def test_w8a8_scale_preparation_keeps_existing_cache(preparation):
    prepare, quantization = preparation
    layer, owner = layer_for(quantization.W8A8)
    owner.w13_weight_scale_fp32 = None

    prepare([layer])

    assert owner.w13_weight_scale_fp32 is None


def test_w8a8_scale_preparation_requires_canonical_scale(preparation):
    prepare, quantization = preparation
    layer, _ = layer_for(quantization.W8A8)

    with pytest.raises(AttributeError, match="w13_weight_scale"):
        prepare([layer])
