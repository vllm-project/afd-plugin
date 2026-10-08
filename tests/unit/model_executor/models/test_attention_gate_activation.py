# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checks for the attention-gate MoE activation input contract."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tests.unit.model_executor.models.attention_gate_test_utils import (
    load_attention_gate_moe_ffn,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("quant_name", ["NONE", "W8A8"])
@pytest.mark.parametrize("custom_activation", [False, True])
@pytest.mark.parametrize("routed_scale_applied_in_topk", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_attention_gate_passes_activation_and_scaling_contract(
    monkeypatch,
    quant_name,
    custom_activation,
    routed_scale_applied_in_topk,
    input_dtype,
):
    captured = []

    def apply_mlp(*, mlp_compute_input, quant_method):
        # This is a deterministic CPU reference for the MoEMlpComputeInput
        # contract, not an implementation or validation of an Ascend kernel.
        del quant_method
        captured.append(mlp_compute_input)
        inputs = mlp_compute_input.hidden_states.to(torch.float32)
        gate = inputs[:, :2]
        up = inputs[:, 2:]
        if mlp_compute_input.swiglu_limit > 0:
            limit = mlp_compute_input.swiglu_limit
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        # This CPU reference mirrors the native SILU clamp path only; the other
        # activation fields are checked as forwarded inputs, not interpreted here.
        routed = torch.nn.functional.silu(gate) * up
        return routed, None

    compute_ffn, quant_type = load_attention_gate_moe_ffn(monkeypatch, apply_mlp)
    requested_quant_type = getattr(quant_type, quant_name)

    owner = torch.nn.Module()
    owner.get_eplb_parameter = lambda name: getattr(owner, name)
    owner.w13_weight = torch.ones((2, 4, 4))
    owner.w2_weight = torch.ones((2, 4, 4))
    owner.w13_weight_scale_fp32 = torch.ones((2, 4))
    owner.w2_weight_scale = torch.ones((2, 4))
    owner.quant_method = SimpleNamespace(
        quant_method=SimpleNamespace(is_per_channel_weight=False)
    )
    owner.dynamic_eplb = False
    owner.activation = "silu"

    activation_values: dict[str, float | None]
    if custom_activation:
        activation_values = {
            "swiglu_limit": 2.0,
            "swiglu_alpha": 1.5,
            "swiglu_beta": 0.25,
            "activation_situ_beta": 0.1,
            "activation_situ_linear_beta": 0.2,
        }
    else:
        activation_values = {
            "swiglu_limit": None,
            "swiglu_alpha": None,
            "swiglu_beta": None,
            "activation_situ_beta": None,
            "activation_situ_linear_beta": None,
        }
    moe_config = SimpleNamespace(has_bias=False, **activation_values)
    experts = SimpleNamespace(
        quant_type=requested_quant_type,
        routed_experts=owner,
        shared_experts=SimpleNamespace(
            _layer=lambda inputs: torch.full_like(inputs, 7.0)
        ),
        moe_config=moe_config,
    )
    scaling_factor = 2.5
    layer = SimpleNamespace(
        mlp=SimpleNamespace(experts=experts, routed_scaling_factor=scaling_factor)
    )
    hidden_states = torch.full((1, 4), 3.0, dtype=input_dtype)
    shared_states = torch.ones((1, 4))
    result = compute_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.tensor([1, 1]),
        dynamic_scales=None,
        expand_x_shared=shared_states,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=0,
        routed_scale_applied_in_topk=routed_scale_applied_in_topk,
    )

    assert len(captured) == 1
    activation = captured[0]
    expected_parameters = (
        (2.0, 1.5, 0.25, 0.1, 0.2) if custom_activation else (0.0, 1.0, 0.0, None, None)
    )
    assert (
        activation.swiglu_limit,
        activation.swiglu_alpha,
        activation.swiglu_beta,
        activation.activation_situ_beta,
        activation.activation_situ_linear_beta,
    ) == expected_parameters

    limit = 2.0 if custom_activation else 0.0
    expected_value = 3.0 if limit == 0 else limit
    expected_value = expected_value / (1.0 + math.exp(-expected_value)) * expected_value
    expected_routed = torch.full((1, 2), expected_value)
    if not routed_scale_applied_in_topk and input_dtype != torch.float16:
        expected_routed *= scaling_factor
    torch.testing.assert_close(result.routed_output, expected_routed)
    if custom_activation:
        unclamped_value = 3.0 / (1.0 + math.exp(-3.0)) * 3.0
        assert not math.isclose(expected_value, unclamped_value)

    expected_shared_value = 7.0
    if not routed_scale_applied_in_topk and input_dtype == torch.float16:
        expected_shared_value /= scaling_factor
    torch.testing.assert_close(
        result.shared_output, torch.full((1, 4), expected_shared_value)
    )
