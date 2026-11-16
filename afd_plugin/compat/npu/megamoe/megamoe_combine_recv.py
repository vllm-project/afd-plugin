# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
import torch
# ### PATCH START: AFD standalone MegaMoE extension
from afd_plugin import _C_megamoe as _C
# ### PATCH END: AFD standalone MegaMoE extension
from .common.megamoe_split import create_megamoe_split_buffer, DispatchTicket
from ._library import library

library.define('npu_megamoe_combine_recv(Tensor(a!) context, Tensor topk_ids, Tensor topk_weights, Tensor snapshot_data, Tensor snapshot_scales, Tensor[] shared_weight1, Tensor[] shared_weight2, Tensor[] shared_scales1, Tensor[] shared_scales2, int[] configuration, int[] dp_ranks, float[] activation_parameters) -> (Tensor, Tensor)')

def meta(context, topk_ids, topk_weights, snapshot_data, snapshot_scales, shared_weight1, shared_weight2, shared_scales1, shared_scales2, configuration, dp_ranks, activation_parameters):
    return (context.new_empty((configuration[13], configuration[4]), dtype=torch.bfloat16),
            context.new_empty((1,), dtype=torch.int32))

library.impl("npu_megamoe_combine_recv", meta, "Meta")
library.impl("npu_megamoe_combine_recv", _C.npu_megamoe_combine_recv, "PrivateUse1")

def megamoe_combine_recv(ticket, buffer, shared_weight1=None, shared_weight2=None,
                          shared_scales1=None, shared_scales2=None):
    if buffer.rank not in ticket.dp_ranks:
        raise ValueError("combine_recv must consume this Attention rank's dispatch ticket")
    if buffer.shared_experts == 0:
        # The CANN dynamic-input signature needs a dtype even when no dense expert executes.
        dtype = torch.float8_e4m3fn if ticket.configuration[7] == 4 else torch.float8_e5m2
        shared_weight1 = shared_weight2 = [torch.empty(0, device=buffer.context.device, dtype=dtype)]
        shared_scales1 = shared_scales2 = [torch.empty(0, device=buffer.context.device, dtype=torch.float8_e8m0fnu)]
    # ### PATCH START: AFD operator namespace
    output, completion = torch.ops.afd_ascend.npu_megamoe_combine_recv(
        buffer.context, ticket.topk_ids, ticket.topk_weights, ticket.snapshot_data, ticket.snapshot_scales,
        shared_weight1, shared_weight2, shared_scales1, shared_scales2,
        ticket.configuration, ticket.dp_ranks, buffer.activation_parameters)
    # ### PATCH END: AFD operator namespace
    return output
