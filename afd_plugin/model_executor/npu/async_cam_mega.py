# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One MC2 W4A8 launch drains a transaction, splitting only on receive overflow."""

from __future__ import annotations

import torch

from afd_plugin.model_executor.npu.async_cam_w4a8 import AsyncCAMW4A8Executor
from afd_plugin.validation import cam_receive_capacity


class AsyncCAMMegaExecutor:
    def __init__(self, weights: AsyncCAMW4A8Executor, connector) -> None:
        if not weights.per_channel:
            raise ValueError("native MC2 Mega requires per-channel W4A8 weights")
        self.weights = weights
        self.connector = connector
        self.capacity = cam_receive_capacity()
        experts, hidden, packed = weights.w13[0].shape
        intermediate = packed * 4
        self.config = [
            connector.max_num_batched_tokens,
            hidden,
            intermediate,
            connector.topk,
            connector.ffn_size,
            connector.attn_size,
            experts,
            connector.world_rank,
            connector.topology.world_size,
            connector.tp_size,
            self.capacity,
            1,
            1,
            len(weights.layer_ids),
        ]
        device = weights.w13[0].device
        size = torch.ops.afd_ascend.afd_mega_moe_pull_workspace_size(
            self.capacity, hidden, intermediate, experts, connector.tp_size
        )
        self.scratch = torch.empty(size, dtype=torch.uint8, device=device)
        metadata_bytes = 8 * (7 + connector.tp_size * (1 + experts))
        count_offset = (metadata_bytes + 511) // 512 * 512
        self.batch_info = self.scratch[:metadata_bytes].view(torch.int64)
        self.group_list = self.scratch[count_offset : count_offset + 8 * experts].view(
            torch.int64
        )
        self.anchor = torch.empty(1, dtype=torch.bfloat16, device=device)
        mapping = [-1] * (weights.layer_ids[-1] + 1)
        for slot, layer_id in enumerate(weights.layer_ids):
            mapping[layer_id] = slot
        self.layer_map = torch.tensor(mapping, dtype=torch.int64, device=device)
        self.op = torch.ops.afd_ascend.afd_mega_moe_pull_native

    def __call__(self) -> torch.Tensor:
        c, w = self.connector, self.weights
        return self.op(
            self.anchor,
            c.comm_args,
            self.layer_map,
            self.scratch,
            w.w13,
            w.w2,
            w.w13_scale,
            w.w2_scale,
            w.w13_bias,
            w.w2_bias,
            self.config,
            c.group_name,
            w.routed_scaling_factor,
        )
