# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise the CAM W4A8 MLP contract without loading an NPU runtime."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dynamic_eplb", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_w4a8_cam_mlp_contract(monkeypatch, dynamic_eplb, per_channel, with_bias, rows):
    # Compile the production function in isolation: importing the model module
    # would initialize optional vLLM/Ascend dependencies on CPU test hosts.
    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py"
    ).read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_attention_gate_moe_ffn"
    )
    namespace = {
        "torch": torch,
        "AFDF2ATransferPayload": SimpleNamespace,
        "_gmmswigluquant_fusion_enabled": lambda: False,
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "<cam-moe>", "exec"), namespace)
    quant_type = SimpleNamespace(NONE="none", W8A8="w8a8", W4A8="w4a8")
    calls = []

    def apply_mlp(*, mlp_compute_input):
        calls.append(mlp_compute_input)
        return torch.ones((rows, 4), dtype=torch.bfloat16), None

    modules = {
        "vllm_ascend.ops.fused_moe.moe_mlp": SimpleNamespace(
            unified_apply_mlp=apply_mlp
        ),
        "vllm_ascend.ops.fused_moe.moe_stage_contracts": SimpleNamespace(
            MoEMlpComputeInput=SimpleNamespace, MoEWeights=SimpleNamespace
        ),
        "vllm_ascend.ops.fused_moe.moe_stage_params": SimpleNamespace(
            MoEQuantParams=SimpleNamespace
        ),
        "vllm_ascend.quantization.quant_type": SimpleNamespace(QuantType=quant_type),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
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
    # The wrapper owns the clamp; do not accidentally read it from the owner.
    owner.swiglu_limit = None
    experts = SimpleNamespace(
        quant_type=quant_type.W4A8,
        dynamic_eplb=dynamic_eplb,
        routed_experts=owner,
        _shared_experts=None,
        activation="silu",
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=experts, swiglu_limit=10.0, routed_scaling_factor=1.0
        )
    )
    hidden_states = torch.ones((rows, 4), dtype=torch.int8)
    scales = torch.ones(rows)
    output = namespace["compute_attention_gate_moe_ffn"](
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
    assert contract.dynamic_scale is scales
    assert contract.quant.quant_type == quant_type.W4A8
    assert contract.quant.is_per_channel_weight == per_channel
    assert contract.swiglu_limit == 10.0
    assert contract.weights.w1[0].data_ptr() == owner.w13_weight.data_ptr()
    assert contract.weights.w2[0].data_ptr() == owner.w2_weight.data_ptr()
    assert contract.weights.w1[0].dtype == torch.int32
    assert contract.weights.w1_scale[0] is owner.w13_weight_scale
    assert contract.weights.w2_scale[0] is owner.w2_weight_scale
    assert (contract.weights.w1_scale_bias is not None) == with_bias
    assert (contract.weights.w2_scale_bias is not None) == with_bias
    assert contract.fusion is False
