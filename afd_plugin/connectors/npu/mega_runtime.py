# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""Shared registered MTE window and per-request ownership."""

from dataclasses import dataclass

import torch
import torch.distributed as distributed

from afd_plugin.compat.npu.ops import ensure_mega_ops_available

from .mega_layout import CONTROL_ROW_BYTES, SplitLayout


ensure_mega_ops_available()

from afd_plugin._C_afd_async_mega import SPLIT_SEQUENCE_MAX, CommContextManager  # noqa: E402


PUBLISHED_WORD = 0


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

    def control_rows(self):
        """Return int32 peer rows using the same layout as SplitWord in the kernel.

        The view borrows the registered window and must not outlive this buffer.
        """
        word_bytes = torch.int32.itemsize
        window = self.manager.get_local_buffer_tensor(
            torch.int32, self.layout.control_offset // word_bytes
        )
        words_per_row = CONTROL_ROW_BYTES // word_bytes
        return window[: self.layout.world_size * words_per_row].view(
            self.layout.world_size, words_per_row
        )

    def create_published_reader(self, ranks_per_group=1):
        """Create a reusable host reader; stop using it before destroying the buffer.

        Reading synchronizes only the reader's copy stream. It does not wait for
        fused_moe, which may itself be waiting for a dispatch to be published.
        """
        if ranks_per_group < 1 or self.layout.attention_ranks % ranks_per_group:
            raise ValueError("Attention ranks must divide into complete groups")
        return PublishedReader(self, ranks_per_group)

    def destroy(self):
        # Caller drains all role streams and synchronizes the union group first.
        self.manager.destroy()


def create_mega_split_buffer(*args, **kwargs):
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


class PublishedReader:
    """Read local PUBLISHED words without queuing behind an FFN compute stream."""

    def __init__(self, buffer, ranks_per_group: int):
        self.rows = buffer.control_rows()[: buffer.layout.attention_ranks]
        self.host = torch.empty(self.rows.shape, dtype=torch.int32, pin_memory=True)
        self.stream = torch.npu.Stream(device=self.rows.device)
        self.ranks_per_group = ranks_per_group

    def __call__(self) -> list[list[int]]:
        # Copy contiguous rows with DMA. Strided gathers would launch a device
        # kernel that could contend with the very fused work we are scheduling.
        with torch.npu.stream(self.stream):
            self.host.copy_(self.rows, non_blocking=True)
        self.stream.synchronize()
        return self.host[:, PUBLISHED_WORD].reshape(-1, self.ranks_per_group).tolist()


def atten_send(activations, topk_ids, topk_weights, buffer, dp_ranks, sequence, layer_index=0):
    if not isinstance(layer_index, int) or not 0 <= layer_index < 2**24:
        raise ValueError("layer_index must be an integer in [0, 2**24)")
    if buffer.rank not in dp_ranks:
        raise ValueError("dispatch_send must run on an Attention member of this DP")
    configuration = buffer.configuration(dp_ranks, sequence, activations.shape[0])
    topk_weights = topk_weights.float().contiguous()
    snapshot_data, snapshot_scales, completion = torch.ops.afd_ascend.afd_async_atten_send(
        buffer.context, activations, topk_ids, topk_weights, configuration, dp_ranks, buffer.activation_parameters, layer_index)
    return DispatchTicket(snapshot_data, snapshot_scales, topk_ids, topk_weights,
                          list(dp_ranks), sequence, configuration, completion)


def atten_recv(ticket, buffer, shared_weight1=None, shared_weight2=None,
                          shared_scales1=None, shared_scales2=None):
    if buffer.rank not in ticket.dp_ranks:
        raise ValueError("combine_recv must consume this Attention rank's dispatch ticket")
    if buffer.shared_experts == 0:
        # The CANN dynamic-input signature needs a dtype even when no dense expert executes.
        dtype = torch.float8_e4m3fn if ticket.configuration[7] == 4 else torch.float8_e5m2
        shared_weight1 = shared_weight2 = [torch.empty(0, device=buffer.context.device, dtype=dtype)]
        shared_scales1 = shared_scales2 = [torch.empty(0, device=buffer.context.device, dtype=torch.float8_e8m0fnu)]
    output, completion = torch.ops.afd_ascend.afd_async_atten_recv(
        buffer.context, ticket.topk_ids, ticket.topk_weights, ticket.snapshot_data, ticket.snapshot_scales,
        shared_weight1, shared_weight2, shared_scales1, shared_scales2,
        ticket.configuration, ticket.dp_ranks, buffer.activation_parameters)
    return output


def mega_ffn(weight1, weight2, weight_scales1, weight_scales2, buffer, dp_ranks, sequence):
    if buffer.rank < buffer.layout.attention_ranks:
        raise ValueError("fused_moe must run on a MoE rank")
    configuration = buffer.configuration(dp_ranks, sequence, buffer.layout.capacity)
    return torch.ops.afd_ascend.afd_async_mega_ffn(
        buffer.context, weight1, weight2, weight_scales1, weight_scales2,
        configuration, dp_ranks, buffer.activation_parameters)
