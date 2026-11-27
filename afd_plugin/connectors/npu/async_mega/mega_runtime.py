# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""Shared registered MTE window and per-request ownership."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as distributed

from afd_plugin.compat.npu.ops import ensure_mega_ops_available

from .mega_layout import SplitLayout

ensure_mega_ops_available()

from afd_plugin._C_afd_async_mega import (  # noqa: E402
    SPLIT_SEQUENCE_MAX,
    CommContextManager,
)


def split_sequence(iteration: int) -> int:
    """Map a one-based per-DP iteration to the wire generation."""
    if iteration < 1:
        raise ValueError("Split iterations start at 1")
    return (iteration - 1) % SPLIT_SEQUENCE_MAX + 1


class SplitBuffer:
    def __init__(
        self,
        group: distributed.ProcessGroup,
        attention_ranks: int,
        moe_ranks: int,
        num_experts: int,
        capacity: int,
        hidden: int,
        intermediate: int,
        top_k: int,
        shared_experts: int = 0,
        dispatch_dtype: torch.dtype = torch.float8_e4m3fn,
        combine_quant_mode: int = 0,
        prefetch: bool = False,
        activation_mode: int = 0,
        activation_submode: int = 0,
        activation_parameters: Sequence[float] = (3.4028234663852886e38, 1.702, 1.0),
        attn_ranks_per_dp: int = 1,
    ) -> None:
        self.rank = distributed.get_rank(group)
        self.layout = SplitLayout(
            attention_ranks,
            moe_ranks,
            num_experts,
            capacity,
            hidden,
            top_k,
            prefetch,
            combine_quant_mode,
            is_attention=self.rank < attention_ranks,
            attn_ranks_per_dp=attn_ranks_per_dp,
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
            opName="mega_split_mte",
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

    def configuration(
        self, dp_ranks: list[int], sequence: int, tokens: int
    ) -> list[int]:
        width = self.layout.attn_ranks_per_dp
        start = self.rank // width * width
        if sorted(dp_ranks) != list(range(start, start + width)):
            raise ValueError(
                "dp_ranks must match this Attention rank's contiguous DP group"
            )
        if not 1 <= sequence <= SPLIT_SEQUENCE_MAX:
            raise ValueError(
                f"Use consecutive per-DP sequences in [1, {SPLIT_SEQUENCE_MAX}], "
                "wrapping to 1"
            )
        return self.launch_configuration(sequence, tokens)

    def launch_configuration(self, sequence_or_dp_width: int, tokens: int) -> list[int]:
        self.launch = self.launch % SPLIT_SEQUENCE_MAX + 1
        values = self._configuration.copy()
        values[11], values[13], values[14] = sequence_or_dp_width, tokens, self.launch
        return values

    def destroy(self) -> None:
        # Caller drains all role streams and synchronizes the union group first.
        self.manager.destroy()


def create_mega_split_buffer(
    group: distributed.ProcessGroup,
    attention_ranks: int,
    moe_ranks: int,
    num_experts: int,
    capacity: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    shared_experts: int = 0,
    dispatch_dtype: torch.dtype = torch.float8_e4m3fn,
    combine_quant_mode: int = 0,
    prefetch: bool = False,
    activation_mode: int = 0,
    activation_submode: int = 0,
    activation_parameters: Sequence[float] = (3.4028234663852886e38, 1.702, 1.0),
    attn_ranks_per_dp: int = 1,
) -> SplitBuffer:
    return SplitBuffer(
        group=group,
        attention_ranks=attention_ranks,
        moe_ranks=moe_ranks,
        num_experts=num_experts,
        capacity=capacity,
        hidden=hidden,
        intermediate=intermediate,
        top_k=top_k,
        shared_experts=shared_experts,
        dispatch_dtype=dispatch_dtype,
        combine_quant_mode=combine_quant_mode,
        prefetch=prefetch,
        activation_mode=activation_mode,
        activation_submode=activation_submode,
        activation_parameters=activation_parameters,
        attn_ranks_per_dp=attn_ranks_per_dp,
    )


@dataclass
class DispatchTicket:
    snapshot_data: torch.Tensor
    snapshot_scales: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    dp_ranks: list[int]
    sequence: int
    configuration: list[int]
    completion: torch.Tensor


def atten_send(
    activations: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: SplitBuffer,
    dp_ranks: list[int],
    sequence: int,
    layer_index: int = 0,
) -> DispatchTicket:
    if not isinstance(layer_index, int) or not 0 <= layer_index < 2**24:
        raise ValueError("layer_index must be an integer in [0, 2**24)")
    if buffer.rank not in dp_ranks:
        raise ValueError("dispatch_send must run on an Attention member of this DP")
    configuration = buffer.configuration(dp_ranks, sequence, activations.shape[0])
    topk_weights = topk_weights.float().contiguous()
    send = torch.ops.afd_ascend.afd_async_atten_send
    snapshot_data, snapshot_scales, completion = send(
        buffer.context,
        activations,
        topk_ids,
        topk_weights,
        configuration,
        dp_ranks,
        buffer.activation_parameters,
        layer_index,
    )
    return DispatchTicket(
        snapshot_data,
        snapshot_scales,
        topk_ids,
        topk_weights,
        list(dp_ranks),
        sequence,
        configuration,
        completion,
    )


def atten_recv(
    ticket: DispatchTicket,
    buffer: SplitBuffer,
    shared_weight1: list[torch.Tensor] | None = None,
    shared_weight2: list[torch.Tensor] | None = None,
    shared_scales1: list[torch.Tensor] | None = None,
    shared_scales2: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if buffer.rank not in ticket.dp_ranks:
        raise ValueError(
            "combine_recv must consume this Attention rank's dispatch ticket"
        )
    if buffer.shared_experts == 0:
        # The CANN dynamic-input signature needs a dtype even when no dense
        # expert executes.
        dtype = (
            torch.float8_e4m3fn if ticket.configuration[7] == 4 else torch.float8_e5m2
        )
        empty_dtype = torch.empty(0, device=buffer.context.device, dtype=dtype)
        empty_scale = torch.empty(
            0, device=buffer.context.device, dtype=torch.float8_e8m0fnu
        )
        shared_weight1 = shared_weight2 = [empty_dtype]
        shared_scales1 = shared_scales2 = [empty_scale]
    recv = torch.ops.afd_ascend.afd_async_atten_recv
    output, completion = recv(
        buffer.context,
        ticket.topk_ids,
        ticket.topk_weights,
        ticket.snapshot_data,
        ticket.snapshot_scales,
        shared_weight1,
        shared_weight2,
        shared_scales1,
        shared_scales2,
        ticket.configuration,
        ticket.dp_ranks,
        buffer.activation_parameters,
    )
    return output


def mega_ffn(
    weight1: list[torch.Tensor],
    weight2: list[torch.Tensor],
    weight_scales1: list[torch.Tensor],
    weight_scales2: list[torch.Tensor],
    buffer: SplitBuffer,
) -> torch.Tensor:
    if buffer.rank < buffer.layout.attention_ranks:
        raise ValueError("fused_moe must run on a MoE rank")
    configuration = buffer.launch_configuration(
        buffer.layout.attn_ranks_per_dp, buffer.layout.capacity
    )
    return torch.ops.afd_ascend.afd_async_mega_ffn(
        buffer.context,
        weight1,
        weight2,
        weight_scales1,
        weight_scales2,
        configuration,
        buffer.activation_parameters,
    )
