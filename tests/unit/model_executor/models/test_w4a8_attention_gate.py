# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise the CAM W4A8 MLP contract without loading an NPU runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.unit.model_executor.models.attention_gate_test_utils import (
    load_attention_gate_moe_ffn,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dynamic_eplb", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_w4a8_cam_mlp_contract(monkeypatch, dynamic_eplb, per_channel, with_bias, rows):
    calls: list[SimpleNamespace] = []

    def apply_mlp(*, mlp_compute_input, quant_method):
        assert quant_method is owner.quant_method.quant_method
        calls.append(mlp_compute_input)
        return torch.ones((rows, 4), dtype=torch.bfloat16), None

    compute_ffn, quant_type = load_attention_gate_moe_ffn(monkeypatch, apply_mlp)
    owner = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        owner.register_parameter(
            name,
            torch.nn.Parameter(
                torch.ones((2, 4, 4), dtype=torch.int32), requires_grad=False
            ),
        )
    for name in ("w13_weight_scale", "w2_weight_scale"):
        owner.register_parameter(
            name, torch.nn.Parameter(torch.ones((2, 4)), requires_grad=False)
        )
    for name in ("w13_scale_bias", "w2_scale_bias"):
        owner.register_parameter(
            name,
            torch.nn.Parameter(torch.ones((2, 4)), requires_grad=False)
            if with_bias
            else None,
        )
    owner.w13_weight_list = [owner.w13_weight]
    owner.w2_weight_list = [owner.w2_weight]
    owner.w13_weight_scale_list = [owner.w13_weight_scale]
    owner.w2_weight_scale_list = [owner.w2_weight_scale]
    owner.w13_scale_bias_list = [owner.w13_scale_bias] if with_bias else None
    owner.w2_scale_bias_list = [owner.w2_scale_bias] if with_bias else None
    owner.quant_method = SimpleNamespace(
        quant_method=SimpleNamespace(is_per_channel_weight=per_channel)
    )
    owner.dynamic_eplb = dynamic_eplb
    owner.activation = "silu"
    # The native MoE config owns the activation contract.
    owner.swiglu_limit = None
    experts = SimpleNamespace(
        quant_type=quant_type.W4A8,
        routed_experts=owner,
        shared_experts=None,
        moe_config=SimpleNamespace(
            swiglu_limit=10.0,
            swiglu_alpha=1.7,
            swiglu_beta=0.5,
            activation_situ_beta=0.8,
            activation_situ_linear_beta=0.3,
        ),
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=experts, routed_scaling_factor=1.0)
    )
    hidden_states = torch.ones((rows, 4), dtype=torch.int8)
    scales = torch.ones(rows)
    output = compute_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.tensor([rows, rows]),
        dynamic_scales=scales,
        expand_x_shared=None,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=0,
    )
    assert len(calls) == int(rows > 0)
    assert output.routed_output.shape == (rows, 4)
    assert output.routed_output.dtype == torch.bfloat16
    if not rows:
        return
    contract = calls[0]
    assert contract.hidden_states is hidden_states
    assert contract.layer is owner
    assert contract.dynamic_scale is scales
    assert contract.quant.quant_type == quant_type.W4A8
    assert contract.quant.is_per_channel_weight == per_channel
    assert contract.swiglu_limit == 10.0
    assert contract.swiglu_alpha == 1.7
    assert contract.swiglu_beta == 0.5
    assert contract.activation_situ_beta == 0.8
    assert contract.activation_situ_linear_beta == 0.3
    assert contract.weights.w1[0].data_ptr() == owner.w13_weight.data_ptr()
    assert contract.weights.w2[0].data_ptr() == owner.w2_weight.data_ptr()
    assert contract.weights.w1[0].dtype == torch.int32
    assert contract.weights.w1_scale[0] is owner.w13_weight_scale
    assert contract.weights.w2_scale[0] is owner.w2_weight_scale
    assert (contract.weights.w1_scale_bias is not None) == with_bias
    assert (contract.weights.w2_scale_bias is not None) == with_bias
    assert contract.fusion is False
