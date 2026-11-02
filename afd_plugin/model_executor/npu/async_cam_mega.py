# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One MC2 W4A8 launch drains a transaction, splitting only on receive overflow."""

from __future__ import annotations

import torch

from afd_plugin.model_executor.npu.async_cam_w4a8 import AsyncCAMW4A8Executor
from afd_plugin.validation import cam_receive_capacity

# Mirror Pull::MetadataWords in afd_pull_layout.h (all fields are int64_t):
# [total, source, layer, round_begin, round_end], tp source prefixes,
# tp * experts original counts, then [session, sequence].
METADATA_WORD_BYTES = torch.int64.itemsize
METADATA_FIXED_WORDS = 7
METADATA_PREFIX_WORDS_PER_SOURCE = 1
# NativeMegaWorkspace in afd_native_mega_workspace.h aligns scratch regions to
# 512 bytes. Match its counts offset, including padding after batch metadata.
WORKSPACE_ALIGNMENT_BYTES = 512


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
        metadata_words = METADATA_FIXED_WORDS + connector.tp_size * (
            METADATA_PREFIX_WORDS_PER_SOURCE + experts
        )
        metadata_bytes = METADATA_WORD_BYTES * metadata_words
        count_offset = (
            (metadata_bytes + WORKSPACE_ALIGNMENT_BYTES - 1)
            // WORKSPACE_ALIGNMENT_BYTES
            * WORKSPACE_ALIGNMENT_BYTES
        )
        count_bytes = METADATA_WORD_BYTES * experts
        self.batch_info = self.scratch[:metadata_bytes].view(torch.int64)
        self.group_list = self.scratch[count_offset : count_offset + count_bytes].view(
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
