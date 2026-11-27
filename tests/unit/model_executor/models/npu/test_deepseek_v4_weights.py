# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4's FFN post-load hook retains the W8A8 scales consumed by GMM."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def post_load_model(ascend_v4_model, monkeypatch):
    ascend_v4_model.native.current_platform = SimpleNamespace(device_type="npu")
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        SimpleNamespace(QuantType=SimpleNamespace(W8A8="w8a8")),
    )
    cls = ascend_v4_model.AFDDeepseekV4ForCausalLM
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.afd_role = "ffn"
    model.afd_config = SimpleNamespace(compute_gate_on_attention=True)
    model.model = SimpleNamespace(layers=[])
    return model


def add_layer(model, quant_type="w8a8", *, is_moe=True, dynamic_eplb=False, scale=None):
    owner = torch.nn.Module()
    owner.dynamic_eplb = dynamic_eplb
    if scale is not None:
        owner.w13_weight_scale = scale
    experts = SimpleNamespace(quant_type=quant_type, routed_experts=owner)
    model.model.layers.append(
        SimpleNamespace(is_moe_layer=is_moe, mlp=SimpleNamespace(experts=experts))
    )
    return owner


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_w8a8_scale_preparation_preserves_canonical_and_cached_values(
    post_load_model, dtype
):
    scale = torch.nn.Parameter(torch.tensor([[0.5, 1.0]], dtype=dtype))
    owner = add_layer(post_load_model, scale=scale)

    post_load_model.process_weights_after_loading()
    cached = owner.w13_weight_scale_fp32
    post_load_model.process_weights_after_loading()

    assert owner.w13_weight_scale is scale
    assert owner.w13_weight_scale_fp32 is cached
    assert set(owner.state_dict()) == {"w13_weight_scale"}
    assert cached.dtype == torch.float32
    torch.testing.assert_close(cached, scale.float())


def test_w8a8_scale_is_rebuilt_after_native_finalization(post_load_model):
    owner = add_layer(post_load_model, scale=torch.ones(2, dtype=torch.float16))
    post_load_model.process_weights_after_loading()
    old = owner.w13_weight_scale_fp32
    owner.w13_weight_scale.add_(1)
    del owner.w13_weight_scale_fp32

    post_load_model.process_weights_after_loading()

    assert owner.w13_weight_scale_fp32 is not old
    torch.testing.assert_close(owner.w13_weight_scale_fp32, torch.full((2,), 2.0))


@pytest.mark.parametrize(
    ("quant_type", "is_moe", "dynamic_eplb"),
    [
        ("w8a8", False, False),
        ("none", True, False),
        ("w4a8", True, False),
        ("w8a8", True, True),
    ],
    ids=["dense", "unquantized", "w4a8", "dynamic-eplb"],
)
def test_scale_preparation_skips_layers_without_stacked_w8a8(
    post_load_model, quant_type, is_moe, dynamic_eplb
):
    owner = add_layer(
        post_load_model, quant_type, is_moe=is_moe, dynamic_eplb=dynamic_eplb
    )
    eligible = add_layer(post_load_model, scale=torch.ones(2, dtype=torch.bfloat16))

    post_load_model.process_weights_after_loading()

    assert not hasattr(owner, "w13_weight_scale_fp32")
    torch.testing.assert_close(eligible.w13_weight_scale_fp32, torch.ones(2))


@pytest.mark.parametrize(
    ("device", "role", "gate_on_attention"),
    [("cpu", "ffn", True), ("npu", "attention", True), ("npu", "ffn", False)],
)
def test_post_load_skips_other_platforms_and_gate_placements(
    post_load_model, ascend_v4_model, device, role, gate_on_attention
):
    ascend_v4_model.native.current_platform.device_type = device
    post_load_model.afd_role = role
    post_load_model.afd_config.compute_gate_on_attention = gate_on_attention
    owner = add_layer(post_load_model)

    post_load_model.process_weights_after_loading()

    assert not hasattr(owner, "w13_weight_scale_fp32")


def test_w8a8_scale_preparation_keeps_existing_cache(post_load_model):
    owner = add_layer(post_load_model)
    owner.w13_weight_scale_fp32 = None

    post_load_model.process_weights_after_loading()

    assert owner.w13_weight_scale_fp32 is None


def test_w8a8_scale_preparation_requires_canonical_scale(post_load_model):
    add_layer(post_load_model)

    with pytest.raises(AttributeError, match="w13_weight_scale"):
        post_load_model.process_weights_after_loading()
