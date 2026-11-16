# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
import torch
# ### PATCH START: AFD standalone MegaMoE extension
from afd_plugin import _C_megamoe as _C
# ### PATCH END: AFD standalone MegaMoE extension
from .common.megamoe_split import create_megamoe_split_buffer, DispatchTicket
from ._library import library

library.define('npu_megamoe_fused_moe(Tensor(a!) context, Tensor[] weight1, Tensor[] weight2, Tensor[] weight_scales1, Tensor[] weight_scales2, int[] configuration, int[] dp_ranks, float[] activation_parameters) -> Tensor')

def meta(context, weight1, weight2, weight_scales1, weight_scales2, configuration, dp_ranks, activation_parameters):
    return context.new_empty((configuration[2] // configuration[1],), dtype=torch.int32)

library.impl("npu_megamoe_fused_moe", meta, "Meta")
library.impl("npu_megamoe_fused_moe", _C.npu_megamoe_fused_moe, "PrivateUse1")

def megamoe_fused_moe(weight1, weight2, weight_scales1, weight_scales2, buffer, dp_ranks, sequence):
    if buffer.rank < buffer.layout.attention_ranks:
        raise ValueError("fused_moe must run on a MoE rank")
    configuration = buffer.configuration(dp_ranks, sequence, buffer.layout.capacity)
    # ### PATCH START: AFD operator namespace
    return torch.ops.afd_ascend.npu_megamoe_fused_moe(
        buffer.context, weight1, weight2, weight_scales1, weight_scales2,
        configuration, dp_ranks, buffer.activation_parameters)
    # ### PATCH END: AFD operator namespace
