# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FastAFD M2N connector: dispatch and combine with expert-owning FFN ranks.

Attention ranks own routing. For every MoE layer and microbatch lane they run
a dispatch kernel that quantizes the hidden states and hands each FFN rank the
tokens routed to its experts, then a combine kernel that waits for the expert
outputs and reduces the top-k. FFN ranks run one persistent expert kernel per step
(``FastAFDFFNModelRunner``); the two kernels synchronize through one
symmetric buffer per lane on a union group whose Attention ranks come first.

``M2NLaneBuffer`` follows FastAFD's ``MegaMoEM2NSymmBuffer`` at
3c7161949310b6d59d6b4cf9bf997a4935c8113b:
https://github.com/hao-ai-lab/FastAFD/tree/3c7161949310b6d59d6b4cf9bf997a4935c8113b
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from afd_plugin.compat.gpu.fastafd import fastafd_ops
from afd_plugin.config import FASTAFD_CONNECTOR, AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDTransferContext,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Byte range of the phase-0 NVLink barrier signal in the buffer workspace. The
# kernels defer each cleanup-barrier wait into their next launch, so a fresh
# buffer must already show one completed round there.
_BARRIER_SIGNAL_BYTES = slice(20, 24)
# The combine kernel spends most of its time waiting for the expert results, so
# a small grid leaves SMs to the shared expert and other microbatches.
ATTENTION_KERNEL_SMS = 24
# Model families whose MoE layout the connector and the FFN runner support.
FASTAFD_MODEL_TYPES = ("qwen3_5_moe_text", "qwen3_moe")
# Hidden and expert intermediate sizes must be multiples of this: the symmetric
# buffer keeps one UE8M0 scale byte per 32 values of each, and TMA needs those
# rows 16-byte aligned (csrc/apis/mega_m2n.hpp, layout/mega_moe.cuh).
_SHAPE_ALIGNMENT = 512
# The Attention kernel's combine splits each hidden row into at most two chunks
# of at most 4096 elements (sm100_mega_moe_m2n.cuh).
_MAX_HIDDEN_SIZE = 8192
# Checkpoint FP8 block size the FFN weight conversion accepts.
_CHECKPOINT_BLOCK = [128, 128]
# The M2N kernels are built for SM100 (Blackwell) only.
_SUPPORTED_SM_MAJOR = 10
# Microbatch lanes when DBO splits a step into two ubatches.
_DBO_LANES = 2
# The union rank that sends each step's control payload to the FFN ranks.
_CONTROL_SENDER_RANK = 0
# Torch switch that skips the symmetric-memory device-index overlap check.
_SYMM_MEM_ALLOW_OVERLAP_ENV = "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"
# Attention kernel phases: 1 quantizes and sends, 2 waits and reduces.
_DISPATCH_PHASE = 1
_COMBINE_PHASE = 2


@dataclass(frozen=True)
class M2NShape:
    """Static protocol facts every union rank must agree on."""

    num_attention_ranks: int
    num_ffn_ranks: int
    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    max_tokens_per_rank: int

    @property
    def num_experts_per_ffn_rank(self) -> int:
        return self.num_experts // self.num_ffn_ranks


class M2NLaneBuffer:
    """One microbatch lane: a union-symmetric buffer and its peer addresses."""

    def __init__(self, shape: M2NShape, group: ProcessGroup) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        ops = fastafd_ops()
        alignment = int(ops.get_token_alignment_for_mega_moe())
        self.shape = shape
        self.max_tokens_per_rank = (
            (shape.max_tokens_per_rank + alignment - 1) // alignment * alignment
        )
        num_bytes, _, _ = ops.get_symm_buffer_size_for_mega_moe_m2n(
            shape.num_attention_ranks,
            shape.num_ffn_ranks,
            shape.num_experts,
            self.max_tokens_per_rank,
            shape.top_k,
            shape.hidden_size,
            shape.intermediate_size,
        )
        self.buffer = symm_mem.empty(int(num_bytes), dtype=torch.int8, device="cuda")
        handle = symm_mem.rendezvous(self.buffer, group=group)
        self.peer_ptrs = [int(ptr) for ptr in handle.buffer_ptrs]
        self.buffer.zero_()
        self.buffer[_BARRIER_SIGNAL_BYTES].view(torch.int32).fill_(group.size())
        dist.barrier(group=group)
        torch.cuda.synchronize()


def m2n_shape_from_config(vllm_config: VllmConfig, afd_config: AFDConfig) -> M2NShape:
    """M2N protocol shape of a supported model, or an actionable error."""
    model_config = vllm_config.model_config
    config = model_config.hf_text_config
    if config.model_type not in FASTAFD_MODEL_TYPES:
        raise ValueError(
            f"{FASTAFD_CONNECTOR} supports model types {FASTAFD_MODEL_TYPES!r}, "
            f"got {config.model_type!r}",
        )
    if model_config.dtype != torch.bfloat16:
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs bfloat16 activations, got {model_config.dtype}",
        )
    quantization = getattr(model_config.hf_config, "quantization_config", None) or {}
    if (
        quantization.get("quant_method") != "fp8"
        or list(quantization.get("weight_block_size") or []) != _CHECKPOINT_BLOCK
    ):
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs an FP8 checkpoint with "
            f"{_CHECKPOINT_BLOCK} block scales, got quantization_config "
            f"{quantization!r}",
        )
    # Every layer must be a MoE layer: the FFN kernel serves all of them.
    if getattr(config, "mlp_only_layers", None) or (
        int(getattr(config, "decoder_sparse_step", 1)) != 1
    ):
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs a MoE block in every layer",
        )
    shape = M2NShape(
        num_attention_ranks=afd_config.num_attention_ranks,
        num_ffn_ranks=afd_config.num_ffn_ranks,
        num_layers=int(config.num_hidden_layers),
        num_experts=int(config.num_experts),
        top_k=int(config.num_experts_per_tok),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.moe_intermediate_size),
        max_tokens_per_rank=int(vllm_config.scheduler_config.max_num_batched_tokens),
    )
    if (
        shape.hidden_size % _SHAPE_ALIGNMENT
        or shape.intermediate_size % _SHAPE_ALIGNMENT
    ):
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs hidden and MoE intermediate sizes that "
            f"are multiples of {_SHAPE_ALIGNMENT}, got {shape.hidden_size} and "
            f"{shape.intermediate_size}",
        )
    if shape.hidden_size > _MAX_HIDDEN_SIZE:
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs a hidden size of at most "
            f"{_MAX_HIDDEN_SIZE}, got {shape.hidden_size}",
        )
    if shape.num_experts % shape.num_ffn_ranks:
        raise ValueError(
            f"{FASTAFD_CONNECTOR} needs num_ffn_ranks to divide the "
            f"{shape.num_experts} routed experts, got {shape.num_ffn_ranks}",
        )
    return shape


def check_same_across_ranks(
    fields: dict[str, int],
    group: ProcessGroup | None,
    device: torch.device,
) -> None:
    """Raise unless every rank of ``group`` passed the same ``fields``."""
    local = torch.tensor(list(fields.values()), dtype=torch.int64, device=device)
    # One MAX all-reduce over (v, -v) yields every field's max and min.
    bounds = torch.cat([local, -local])
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=group)
    values = bounds.tolist()
    mismatched = {
        name: (-values[len(fields) + index], values[index])
        for index, name in enumerate(fields)
        if values[index] != -values[len(fields) + index]
    }
    if mismatched:
        raise RuntimeError(
            f"{FASTAFD_CONNECTOR} needs the same lanes, layers and buffer shape on "
            f"every Attention and FFN rank; (min, max) differ for {mismatched}",
        )


class FastAFDAFDConnector(AFDConnectorBase):
    """Attention-first M2N connector backed by the FastAFD MegaMoE kernels.

    Attention calls ``dispatch`` and then ``combine`` once per MoE layer and
    lane, through the model's custom ops. FFN ranks borrow ``lanes``,
    ``union_rank`` and ``shape`` to launch the expert kernel; the connector
    owns the union group and the symmetric buffers.
    """

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError(
                f"{FASTAFD_CONNECTOR} connector_extra_config must be a mapping"
            )
        if raw:
            raise ValueError(
                f"{FASTAFD_CONNECTOR} does not support connector_extra_config"
            )
        return ConnectorExtraInfo()

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.attn_size = afd_config.num_attention_ranks
        self.ffn_size = afd_config.num_ffn_ranks
        self.union_rank = (
            role_rank if afd_config.role == "attention" else self.attn_size + role_rank
        )
        parallel_config = vllm_config.parallel_config
        self.num_lanes = (
            _DBO_LANES
            if parallel_config.enable_dbo or parallel_config.use_ubatching
            else 1
        )
        if parallel_config.tensor_parallel_size != 1:
            raise ValueError(
                f"{FASTAFD_CONNECTOR} supports tensor_parallel_size=1 on both roles",
            )
        self.shape = m2n_shape_from_config(vllm_config, afd_config)
        self.group: ProcessGroup | None = None
        self.lanes: list[M2NLaneBuffer] = []
        self.control_plane = FastAFDControlPlane(self)

    def close(self) -> None:
        """Release the lane buffers, then the union group. Safe to repeat."""
        self.lanes = []
        if self.group is not None:
            dist.destroy_process_group(self.group)
            self.group = None

    def init_afd_connector(self) -> None:
        """Create the union group and the lane buffers (collective on all ranks)."""
        self.group = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.attn_size + self.ffn_size,
            rank=self.union_rank,
            group_name="fastafd",
            timeout=timedelta(seconds=self.afd_config.afd_process_group_timeout_s),
        )
        device = torch.device(f"cuda:{self.local_rank}")
        major, minor = torch.cuda.get_device_capability(device)
        if major != _SUPPORTED_SM_MAJOR:
            raise RuntimeError(
                f"{FASTAFD_CONNECTOR} runs on SM100 (Blackwell) GPUs, got "
                f"compute capability {major}.{minor}",
            )
        # Each rank sizes its symmetric buffers from its own config, so a
        # mismatch (DBO on one role only, a different max_num_batched_tokens or
        # layer count) would otherwise hang or corrupt the kernels.
        check_same_across_ranks(
            {"num_lanes": self.num_lanes, **asdict(self.shape)},
            self.group,
            device,
        )
        self._allow_isolated_devices_in_symmetric_memory()
        self.lanes = [
            M2NLaneBuffer(self.shape, self.group) for _ in range(self.num_lanes)
        ]

    def _allow_isolated_devices_in_symmetric_memory(self) -> None:
        """Let symmetric memory span role processes that each see one GPU.

        A role process usually sees only its own GPUs, so every rank can report
        ``cuda:0`` and torch's rendezvous rejects that as overlapping devices.
        Check physical identity (GPU UUID) here instead, then lift the
        device-index check.
        """
        device = torch.device(f"cuda:{self.local_rank}")
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        uuids: list[str | None] = [None] * (self.attn_size + self.ffn_size)
        dist.all_gather_object(uuids, uuid, group=self.group)
        if len(set(uuids)) != len(uuids):
            raise RuntimeError(
                f"{FASTAFD_CONNECTOR} needs a distinct GPU per Attention and FFN "
                f"rank; got device UUIDs {uuids}",
            )
        os.environ.setdefault(_SYMM_MEM_ALLOW_OVERLAP_ENV, "1")

    @property
    def is_initialized(self) -> bool:
        return bool(self.lanes)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        lane: int,
    ) -> None:
        """Send one MoE layer's routed tokens to the FFN ranks (Attention side).

        ``topk_ids`` are int32 or int64 expert ids and ``topk_weights`` the
        fp32 (renormalized) routing weights. The kernel quantizes and pushes
        the tokens, then exits, so local work can run while the FFN ranks
        compute. Every dispatch must be followed by exactly one ``combine``
        with the same inputs on the same lane, and every Attention rank must
        run the same layers and lanes in the same order each step. The lane
        buffers serve only these split launches: the fused kernel keeps
        different deferred-barrier state, and mixing the two on one buffer can
        hang.
        """
        # The dispatch phase does not write the output, but the launcher
        # checks its shape.
        self._launch_attention_kernel(
            torch.empty_like(hidden_states),
            hidden_states,
            topk_ids,
            topk_weights,
            lane,
            _DISPATCH_PHASE,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        lane: int,
    ) -> torch.Tensor:
        """Wait for a ``dispatch``'s expert outputs and reduce the top-k."""
        output = torch.empty_like(hidden_states)
        self._launch_attention_kernel(
            output, hidden_states, topk_ids, topk_weights, lane, _COMBINE_PHASE
        )
        return output

    def _launch_attention_kernel(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        lane: int,
        phase: int,
    ) -> None:
        shape = self.shape
        buffer = self.lanes[lane]
        fastafd_ops().mega_moe_m2n_ag(
            output,
            hidden_states.contiguous(),
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            None,
            False,
            buffer.buffer,
            buffer.peer_ptrs,
            self.union_rank,
            shape.num_attention_ranks,
            shape.num_ffn_ranks,
            buffer.max_tokens_per_rank,
            shape.num_experts,
            shape.top_k,
            shape.intermediate_size,
            ATTENTION_KERNEL_SMS,
            phase,
            False,
            False,
            shape.top_k,
            shape.num_experts,
            0.0,
            None,
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            f"{FASTAFD_CONNECTOR} exchanges tokens inside its fused kernels; "
            "use dispatch and combine",
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{FASTAFD_CONNECTOR} exchanges tokens inside its fused kernels; "
            "use dispatch and combine",
        )

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        raise NotImplementedError(
            f"{FASTAFD_CONNECTOR} FFN ranks run the expert kernel through "
            "FastAFDFFNModelRunner",
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            f"{FASTAFD_CONNECTOR} FFN ranks run the expert kernel through "
            "FastAFDFFNModelRunner",
        )


class FastAFDControlPlane(AFDControlPlane):
    """Per-step plans from Attention union rank 0 to every FFN rank.

    The plan is the Attention runner's ``AFDControlPayload``: its stages are
    the lanes and its DP token counts bound the step. Tokens never travel on
    this path; it only tells FFN ranks which expert kernel to launch.
    """

    def __init__(self, connector: FastAFDAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(self, payload: AFDControlPayload) -> None:
        if len(payload.dp_metadata_list) > self.connector.num_lanes:
            raise RuntimeError(
                f"{FASTAFD_CONNECTOR} has {self.connector.num_lanes} lanes, got a "
                f"{len(payload.dp_metadata_list)}-stage step",
            )

    def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if connector.union_rank != _CONTROL_SENDER_RANK:
            return
        assert connector.group is not None
        send_control_payload(
            payload,
            dst=list(
                range(connector.attn_size, connector.attn_size + connector.ffn_size)
            ),
            group=connector.group,
            device=torch.device(f"cuda:{connector.local_rank}"),
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        assert connector.group is not None
        return recv_control_payload(
            src=_CONTROL_SENDER_RANK,
            group=connector.group,
            device=torch.device(f"cuda:{connector.local_rank}"),
        )


__all__ = [
    "ATTENTION_KERNEL_SMS",
    "FASTAFD_MODEL_TYPES",
    "FastAFDAFDConnector",
    "FastAFDControlPlane",
    "M2NLaneBuffer",
    "M2NShape",
    "check_same_across_ranks",
    "m2n_shape_from_config",
]
