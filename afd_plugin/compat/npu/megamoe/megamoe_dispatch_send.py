# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
import torch
# ### PATCH START: AFD standalone MegaMoE extension
from afd_plugin import _C_megamoe as _C
# ### PATCH END: AFD standalone MegaMoE extension
from .common.megamoe_split import create_megamoe_split_buffer, DispatchTicket
from ._library import library

library.define('npu_megamoe_dispatch_send(Tensor(a!) context, Tensor x, Tensor topk_ids, Tensor topk_weights, int[] configuration, int[] dp_ranks, float[] activation_parameters, int layer_index=0) -> (Tensor, Tensor, Tensor)')

def meta(context, x, topk_ids, topk_weights, configuration, dp_ranks, activation_parameters, layer_index=0):
    tokens = configuration[13] if configuration[12] else 0
    return (context.new_empty((tokens, configuration[4]), dtype=torch.uint8),
            context.new_empty((tokens, (configuration[4] + 63) // 64 * 2), dtype=torch.uint8), context.new_empty((1,), dtype=torch.int32))

library.impl("npu_megamoe_dispatch_send", meta, "Meta")
library.impl("npu_megamoe_dispatch_send", _C.npu_megamoe_dispatch_send, "PrivateUse1")

def megamoe_dispatch_send(activations, topk_ids, topk_weights, buffer, dp_ranks, sequence, layer_index=0):
    if not isinstance(layer_index, int) or not 0 <= layer_index < 2**24:
        raise ValueError("layer_index must be an integer in [0, 2**24)")
    if buffer.rank not in dp_ranks:
        raise ValueError("dispatch_send must run on an Attention member of this DP")
    configuration = buffer.configuration(dp_ranks, sequence, activations.shape[0])
    topk_weights = topk_weights.float().contiguous()
    # ### PATCH START: AFD operator namespace
    snapshot_data, snapshot_scales, completion = torch.ops.afd_ascend.npu_megamoe_dispatch_send(
        buffer.context, activations, topk_ids, topk_weights, configuration, dp_ranks, buffer.activation_parameters, layer_index)
    # ### PATCH END: AFD operator namespace
    return DispatchTicket(snapshot_data, snapshot_scales, topk_ids, topk_weights,
                          list(dp_ranks), sequence, configuration, completion)
