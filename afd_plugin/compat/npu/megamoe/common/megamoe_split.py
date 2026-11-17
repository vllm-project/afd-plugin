# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""Shared registered MTE window and per-request ownership."""

from dataclasses import dataclass

import torch
import torch.distributed as distributed

from afd_plugin._C_megamoe import SPLIT_SEQUENCE_MAX

from .comm_context import CommContextManager
from .megamoe_split_layout import SplitLayout


def split_sequence(iteration):
    """Map a one-based per-DP iteration to the wire generation."""
    if iteration < 1:
        raise ValueError("Split iterations start at 1")
    return (iteration - 1) % SPLIT_SEQUENCE_MAX + 1


class SplitBuffer:
    def __init__(
        self,
        group,
        attention_ranks,
        moe_ranks,
        num_experts,
        capacity,
        hidden,
        intermediate,
        top_k,
        shared_experts=0,
        dispatch_dtype=torch.float8_e4m3fn,
        combine_quant_mode=0,
        prefetch=False,
        activation_mode=0,
        activation_submode=0,
        activation_parameters=(3.4028234663852886e38, 1.702, 1.0),
    ):
        self.layout = SplitLayout(
            attention_ranks,
            moe_ranks,
            num_experts,
            capacity,
            hidden,
            top_k,
            prefetch,
            combine_quant_mode,
        )
        if (
            attention_ranks < 1
            or moe_ranks < 1
            or self.layout.world_size > 768
            or num_experts % moe_ranks
            or distributed.get_world_size(group) != self.layout.world_size
        ):
            raise ValueError(
                "Use a union group ordered [Attention ranks, MoE ranks], "
                "with E divisible by M"
            )
        if dispatch_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                "Split MTE supports FP8 activations (A8W8 or A8W4), not A4W4"
            )
        if (
            not 256 <= intermediate <= 4096
            or intermediate % 128
            or not 0 <= shared_experts <= 4
        ):
            raise ValueError("Invalid intermediate width or shared expert count")
        self.group = group
        self.rank = distributed.get_rank(group)
        self.shared_experts = shared_experts
        self.activation_parameters = list(activation_parameters)
        self.launch = 0
        self._configuration = [
            attention_ranks,
            moe_ranks,
            num_experts,
            capacity,
            hidden,
            intermediate * 2,
            top_k,
            4 if dispatch_dtype == torch.float8_e4m3fn else 3,
            combine_quant_mode,
            int(prefetch),
            self.layout.buffer_bytes,
            0,
            shared_experts,
            0,
            0,
            activation_mode,
            activation_submode,
        ]
        # Complete the warmup before host-side topology queries and channel creation.
        check = torch.zeros(1, device="npu")
        distributed.all_reduce(check, group=group)
        torch.npu.synchronize()
        name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank, init_comm=True
        )
        self.manager = CommContextManager(
            name,
            self.layout.world_size,
            backend="channel",
            opName="megamoe_split_mte",
            customCclBufferSize=self.layout.buffer_bytes,
        )
        self.context = self.manager.create_context()
        if self.manager.topo_type != 0:
            raise ValueError(
                "This split protocol requires a same-supernode MTE topology; "
                "URMA is unchanged"
            )
        torch.npu.synchronize()
        distributed.barrier(group=group)

    def configuration(self, dp_ranks, sequence, tokens):
        if (
            not dp_ranks
            or len(set(dp_ranks)) != len(dp_ranks)
            or any(rank < 0 or rank >= self.layout.attention_ranks for rank in dp_ranks)
        ):
            raise ValueError(
                "dp_ranks must list the distinct Attention union ranks of one DP"
            )
        if not 1 <= sequence <= SPLIT_SEQUENCE_MAX:
            raise ValueError(
                f"Use consecutive per-DP sequences in [1, {SPLIT_SEQUENCE_MAX}], "
                "wrapping to 1"
            )
        self.launch = self.launch % SPLIT_SEQUENCE_MAX + 1
        values = self._configuration.copy()
        values[11], values[13], values[14] = sequence, tokens, self.launch
        return values

    def destroy(self):
        # Caller drains all role streams and synchronizes the union group first.
        self.manager.destroy()


def create_megamoe_split_buffer(*args, **kwargs):
    return SplitBuffer(*args, **kwargs)


@dataclass
class DispatchTicket:
    snapshot_data: torch.Tensor
    snapshot_scales: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    dp_ranks: list
    sequence: int
    configuration: list
    completion: torch.Tensor
