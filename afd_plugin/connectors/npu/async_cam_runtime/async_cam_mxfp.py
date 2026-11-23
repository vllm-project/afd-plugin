# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Eager Ascend 950 expert computation on rows already routed by CAM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.quant_type import QuantType

from afd_plugin.connectors.npu.async_cam_runtime import CAM_LAYER_INDEX

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4.model import DeepseekV4MoE


class AsyncCAMMXFPExecutor:
    """Select the received layer and run its native, routed-only W4A8MXFP MLP."""

    def __init__(self, layers: dict[int, DeepseekV4MoE], local_experts: int) -> None:
        if not layers:
            raise ValueError("Ascend 950 Async CAM requires loaded MXFP4 expert layers")
        self.layers = layers
        self.local_experts = local_experts
        signatures = []
        for index, moe in sorted(layers.items()):
            experts = moe.experts
            owner = experts.routed_experts
            if (
                experts.quant_type != QuantType.W4A8MXFP
                or owner.dynamic_eplb
                or experts._shared_experts is not None
            ):
                raise ValueError(
                    f"Ascend 950 Async CAM layer {index} requires static routed-only "
                    "W4A8MXFP experts"
                )
            # Native post-load may retain only Mega's per-expert lists. Eager GMM
            # consumes the finalized non-Mega tensors and their block scales.
            geometry = []
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                weight = owner._parameters.get(name)
                if (
                    weight is None
                    or not weight.numel()
                    or weight.shape[0] != local_experts
                ):
                    raise ValueError(
                        f"Ascend 950 Async CAM layer {index} requires non-Mega "
                        f"{name} for {local_experts} local experts"
                    )
                geometry.append((tuple(weight.shape), str(weight.dtype)))
            signatures.append(
                (
                    index,
                    geometry,
                    str(owner.activation),
                    experts.moe_config.swiglu_limit,
                )
            )
        self.signature = "compact_mxfp4:" + repr(signatures)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        batch_info: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("Ascend 950 Async CAM requires BF16 received activations")
        if group_list.numel() != self.local_experts:
            raise ValueError("CAM group list must contain one count per local expert")
        # Compact DR's first word is rank-wide. Only this chunk's expert counts
        # describe its valid rows. Eager mode allows one host read for both values.
        layer_id, rows = (
            torch.stack(
                (batch_info[CAM_LAYER_INDEX], group_list.sum(dtype=torch.int64))
            )
            .cpu()
            .tolist()
        )
        if layer_id not in self.layers:
            raise ValueError(f"CAM received an unknown FFN layer {layer_id}")
        if not 0 <= rows <= hidden_states.size(0):
            raise ValueError("CAM expert row count exceeds dispatch-receive capacity")
        if rows == 0:
            # CS must still retire the original chunk. Its ABI requires a
            # nonempty floating tensor, and its metadata specifies zero payload.
            return hidden_states.new_zeros((1, hidden_states.size(1)))

        experts = self.layers[layer_id].experts
        owner, config = experts.routed_experts, experts.moe_config
        # See vllm-ascend bd42eea0's w4a8_mxfp4.py::apply_gmm1_act_quant.
        # With dynamic_scale=None the native method derives MXFP8 block scales
        # from BF16. The single scale slot of a quant0 CAM receive is unused.
        output, _ = apply_moe_mlp(
            mlp_compute_input=MoEMlpComputeInput(
                hidden_states=hidden_states[:rows],
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
                ),
                activation=owner.activation,
                swiglu_limit=config.swiglu_limit or 0.0,
                swiglu_alpha=config.swiglu_alpha or 1.0,
                swiglu_beta=config.swiglu_beta or 0.0,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                fusion=False,
            ),
            quant_method=owner.quant_method.quant_method,
        )
        # Attention's top-k weights carry the routed scaling factor. CR applies it.
        return output
