# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 eager CAM computation using the native MXFP4 MLP after routed receive."""

from __future__ import annotations

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.quant_type import QuantType


class AsyncCAMMXFPExecutor:
    """Keep the A5 eager host layer choice within main's connector runtime."""

    def __init__(self, layers):
        self.layers = dict(layers)
        if not self.layers or len(self.layers) != len(layers):
            raise ValueError("MXFP CAM requires unique loaded expert layers")
        for moe in self.layers.values():
            experts = moe.experts
            if (
                experts.quant_type != QuantType.W4A8MXFP
                or experts.routed_experts.dynamic_eplb
                or experts.shared_experts is not None
            ):
                raise ValueError("A5 CAM requires static routed-only MXFP4 experts")

    def signature(self):
        return "a5-mxfp4:" + repr(
            [
                (
                    index,
                    tuple(moe.experts.routed_experts.w13_weight.shape),
                    tuple(moe.experts.routed_experts.w2_weight.shape),
                )
                for index, moe in sorted(self.layers.items())
            ]
        )

    def __call__(self, hidden_states, dynamic_scales, group_list, batch_info):
        if hidden_states.dtype != torch.bfloat16 or dynamic_scales is not None:
            raise ValueError("MXFP CAM requires BF16 transport without receive scales")
        # This is the existing eager control copy, not device layer selection.
        layer, rows = torch.stack((batch_info[2], group_list.sum())).cpu().tolist()
        if not 0 <= rows <= hidden_states.shape[0]:
            raise ValueError("CAM routed count exceeds receive capacity")
        hidden_states = hidden_states[:rows]
        if rows == 0:
            return hidden_states
        experts = self.layers[layer].experts
        owner, config = experts.routed_experts, experts.moe_config
        output, _ = apply_moe_mlp(
            mlp_compute_input=MoEMlpComputeInput(
                hidden_states=hidden_states,
                group_list=group_list,
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                weights=MoEWeights(
                    w1=owner.w13_weight,
                    w2=owner.w2_weight,
                    w1_scale=owner.w13_weight_scale,
                    w2_scale=owner.w2_weight_scale,
                ),
                layer=owner,
                quant=MoEQuantParams(
                    quant_type=QuantType.W4A8MXFP,
                    mxfp=MoEMxfpParams(
                        act_quant_type=torch.float8_e4m3fn,
                        weight_quant_type=torch_npu.float4_e2m1fn_x2,
                        scale_dtype=torch_npu.float8_e8m0fnu,
                        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                        use_bf16=True,
                    ),
                    is_per_channel_weight=False,
                ),
                swiglu_limit=config.swiglu_limit or 0.0,
                swiglu_alpha=config.swiglu_alpha or 1.0,
                swiglu_beta=config.swiglu_beta or 0.0,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                fusion=False,
                activation=owner.activation,
                need_trans=False,
                dynamic_eplb=False,
            ),
            quant_method=owner.quant_method.quant_method,
        )
        # Routing weights already carry routed_scaling_factor; combine applies it.
        return output
