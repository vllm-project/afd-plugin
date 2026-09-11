# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NVSHMEM-backed asynchronous connector for CUDA AFD.

``GpuAsyncAFDConnector`` is the CUDA counterpart of ``CAMAsyncAFDConnector``:
Attention ranks run MoE routing, write routed tokens one-sided into the FFN
ranks' symmetric windows, and later reduce the weighted expert output; FFN ranks
poll their window, run their local experts, and write the result back. There is
no DP metadata control plane (``control_plane`` stays ``None``) and FFN work is
driven directly by the connector receive loop, so Attention DP replicas never
wait for each other.

The world is Attention-first, ``[A0, A1, ..., F0, F1, ...]``, matching
``CAMAsyncAFDConnector``. Every Attention rank routes to every FFN rank, so an
FFN window holds one region per Attention rank and vice versa.

The Attention-side data path is CUDA-graph capturable, under the same
``FULL_DECODE_ONLY`` policy the rest of AFD accepts. Two things make it so, and
both are visible in the flag protocol: the dispatch sequence number lives in a
device tensor the graph advances, so a replay stamps a fresh number even though
it runs no Python; and a reply signals with the constant ``FLAG_REPLY_READY``,
which the Attention rank resets inside the graph once it has consumed the slot,
because a captured stream wait can only ever compare against the value that was
live when it was recorded. The FFN side is driven by a host poll loop and stays
eager, so nothing there needs capturing.

See ``docs/design/rfc_async_gpu_connector.md``. Supported deployment requires
``async=true``, ``compute_gate_on_attention=true``, and a single node.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_positive_int,
    coerce_extra_str,
)
from afd_plugin.connectors.async_topology import (
    ASYNC_MOE_REQUEST_SPLIT,
    build_async_topology,
)
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.gpu.symm_window import (
    FLAG_SHUTDOWN_BIT,
    H_ROUTED_TOKENS,
    H_SEGMENT_START,
    HEADER_FIXED_WORDS,
    SlotLayout,
    SymmWindow,
    encode_header,
    fill_header_prefix,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup
    from vllm.config import VllmConfig

AFD_ASYNC_GPU_GROUP_NAME = "afd_async_gpu"

# What an FFN reply stamps on the flag the Attention rank is waiting for, and
# what that rank puts back once it has consumed the slot.
#
# It is a constant because the wait has to survive CUDA graph capture. The
# awaited value becomes an immediate inside the captured graph, so a reply that
# signalled with a rising sequence number would be waited for with whatever
# number happened to be live at capture time -- and on the second replay the
# flag still holds it, so the wait falls straight through and the combine reads
# the previous layer's data. A constant marker plus an in-graph reset makes
# every replay identical, which is exactly what a graph needs.
FLAG_REPLY_READY: Final[int] = 1

_GPU_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "attn_ranks_per_dp",
        "ring_depth",
        "recv_poll_timeout_ms",
        "async_moe_ubatching",
        "async_moe_num_ubatches",
        "async_moe_split",
    },
)

# Name the logger inside vLLM's tree: vLLM installs its handler on the "vllm"
# logger only, so a bare ``afd_plugin.*`` logger propagates to a handler-less
# root and every line is dropped -- which is how the window summary, the only
# report of a multi-GiB allocation, stayed invisible.
logger = init_logger(f"vllm.{__name__}")

# Bring-up instrument: setting AFD_ASYNC_DEBUG=1 logs magnitude checks for the
# first MoE layers on both roles, pinning where a zero output first appears on
# the async data path (received dispatch, FFN reply, Attention combine).
AFD_ASYNC_DEBUG_ENV: Final[str] = "AFD_ASYNC_DEBUG"
AFD_ASYNC_DEBUG_LAYERS: Final[int] = 3


def _debug_norm(name: str, tensor: Tensor | None) -> str:
    if tensor is None or tensor.numel() == 0:
        return f"{name}=empty"
    return f"{name}={float(tensor.float().abs().mean()):.6e}"


@dataclass(frozen=True)
class GpuAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async GPU connector configuration.

    Attributes:
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        ring_depth: Slots per peer region. Derived from the send-then-recv
            invariant, not a performance knob: an Attention rank has at most one
            in-flight request per ``(peer, stage)``, so ``num_stages`` suffices.
            The default here counts only this connector's own MoE stages; the
            connector raises it for vLLM's ubatching, which it cannot see from
            the extra config alone.
        ring_depth_pinned: Whether the extra config named a ring depth. An
            explicit one is honoured as given; an implied one is free to grow.
        recv_poll_timeout_ms: Idle poll timeout on the FFN loop; bounds shutdown
            response time.
        async_moe_ubatching: Whether request-boundary async MoE ubatching is used.
        async_moe_num_ubatches: Number of stages used by async MoE ubatching.
        async_moe_split: Boundary at which async MoE work is split.
    """

    attn_ranks_per_dp: int = 1
    ring_depth: int = 0
    ring_depth_pinned: bool = False
    recv_poll_timeout_ms: int = 50
    async_moe_ubatching: bool = False
    async_moe_num_ubatches: int = 2
    async_moe_split: str = ASYNC_MOE_REQUEST_SPLIT

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> GpuAsyncExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _GPU_ASYNC_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown AFD async GPU connector_extra_config field(s): "
                + ", ".join(unknown),
            )

        ubatching = coerce_extra_bool(
            raw.get("async_moe_ubatching", False),
            field_name="async_moe_ubatching",
        )
        num_ubatches = coerce_extra_positive_int(
            raw.get("async_moe_num_ubatches", 2),
            field_name="async_moe_num_ubatches",
        )
        # Ring depth follows the number of live stages unless pinned explicitly.
        ring_depth = coerce_extra_positive_int(
            raw.get("ring_depth", num_ubatches if ubatching else 1),
            field_name="ring_depth",
        )
        return cls(
            attn_ranks_per_dp=coerce_extra_positive_int(
                raw.get("attn_ranks_per_dp", 1),
                field_name="attn_ranks_per_dp",
            ),
            ring_depth=ring_depth,
            ring_depth_pinned="ring_depth" in raw,
            recv_poll_timeout_ms=coerce_extra_positive_int(
                raw.get("recv_poll_timeout_ms", 50),
                field_name="recv_poll_timeout_ms",
            ),
            async_moe_ubatching=ubatching,
            async_moe_num_ubatches=num_ubatches,
            async_moe_split=coerce_extra_str(
                raw.get("async_moe_split", ASYNC_MOE_REQUEST_SPLIT),
                field_name="async_moe_split",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
            "ring_depth": self.ring_depth,
            "recv_poll_timeout_ms": self.recv_poll_timeout_ms,
            "async_moe_ubatching": self.async_moe_ubatching,
            "async_moe_num_ubatches": self.async_moe_num_ubatches,
            "async_moe_split": self.async_moe_split,
        }


class ConnectorShutdown(RuntimeError):  # noqa: N818
    """Raised on the FFN loop when a peer announced shutdown."""


@dataclass(slots=True)
class GpuAsyncTransferState(AFDTransferState):
    """FFN-side state carried from dispatch recv through combine send.

    ``region``/``ring`` locate the window slot so ``send_ffn_work_item_output``
    can write back to the originating Attention rank and release the slot.

    ``group_list`` is a device view of the arrived header's trailing words,
    which is what the grouped GEMM reads; the host-side copy of those counts
    went away with the combine header that used to carry them back.

    ``expand_idx`` and ``weights`` describe this rank's own run of partials:
    which token each one reads on the way in, and what to weight it by when the
    expert output is reduced back to one row per token on the way out.
    """

    region: int = 0
    ring: int = 0
    src_role_rank: int = 0
    layer_idx: int = 0
    stage_idx: int = 0
    num_tokens: int = 0
    routed_tokens: int = 0
    shared_tokens: int = 0
    group_list: Tensor | None = None
    # True when recv_attn_output gathered straight into the caller's buffer, so
    # the rows are already staged and must not be copied again.
    staged_routed: bool = False
    expand_idx: Tensor | None = None
    weights: Tensor | None = None
    expand_x_shared: Tensor | None = None


@dataclass(slots=True)
class GpuAsyncFFNWorkItem:
    """Normalized FFN-side work item produced by a window arrival."""

    hidden_states: Tensor
    context: AFDTransferContext
    recv_output: AFDA2FTransferPayload
    layer_idx: int
    stage_idx: int
    num_tokens: int
    total_num_tokens: int
    shared_num_tokens: int


@dataclass(slots=True)
class _PendingDispatch:
    """Attention-side record of one in-flight layer, popped by combine recv.

    Every reply carries a full batch of rows, already weighted and summed over
    that rank's experts, so combine is a plain add at matching row positions and
    needs no index from the dispatch.

    ``shared_slices[r]`` is the contiguous token range whose shared-expert
    output that rank returns. It is recorded here because combine must know the
    shape of a reply *before* it arrives: that is what lets the wait happen on a
    stream instead of on the host, which cannot then be told what turned up.
    """

    context: AFDTransferContext
    shared_slices: list[slice]
    num_tokens: int
    ring: int
    expected_ffn: list[int]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    """One layer's routing plan, already in the order every consumer wants.

    Every field is indexed by partial -- one entry per ``(token, topk slot)`` --
    and sorted by global expert, so a destination's partials are one contiguous
    run, grouped by local expert inside it, which feeds the grouped GEMM
    directly.

    Nothing here is ever read back to the host. The three per-destination
    vectors go into the slot headers on the device, and the arrays are shipped
    whole, so the sender never needs to know how the routing came out.

    Attributes:
        counts: Partials per global expert, padded to
            ``ffn_size * expert_per_rank``. These are the header's group list.
        routed_per_rank: Partials each FFN rank receives.
        segment_start: Where each FFN rank's run of partials begins.
        expand_idx: Per partial, the token it carries.
        weights: Per partial, its topk weight.
    """

    counts: Tensor
    routed_per_rank: Tensor
    segment_start: Tensor
    expand_idx: Tensor
    weights: Tensor


def plan_dispatch(
    topk_ids: Tensor,
    topk_weights: Tensor,
    *,
    ffn_size: int,
    expert_per_rank: int,
    expert_bounds_probe: Tensor | None = None,
) -> DispatchPlan:
    """Cluster ``(token, topk_slot)`` partials by destination expert.

    Sorting by the global expert id groups partials by destination rank and, in
    the same pass, by local expert inside each destination, which is every
    grouping any consumer needs. One sort and a bounds lookup is the whole plan.

    Every step is a whole-tensor op and the host learns nothing here, by
    design: a readback of the routing would block the send path behind the
    device once per MoE layer.
    """
    num_slots = topk_ids.shape[1]
    # Sort the expert ids at their own width. Widening to int64 first cost a
    # full copy of the partials and then doubled the bytes the sort moves; the
    # ids index at most ffn_size * expert_per_rank experts, which never needs
    # 64 bits. Measured on the real shape (2048 tokens, topk 6): 0.129ms per
    # call for the widened sort against 0.087ms for this one.
    flat = topk_ids.reshape(-1)
    sorted_experts, order = torch.sort(flat, stable=True)

    # Where each expert's partials start and stop, read off the sorted ids.
    # torch.bincount would do this in one call, but it sizes its output from the
    # data's maximum and so copies that maximum to the host -- a pageable
    # readback measured at 783us per call, once per MoE layer, which was the
    # largest single host cost on the Attention rank. searchsorted needs no such
    # thing, because the expert count is known, and it hands back the offsets
    # that would otherwise be a second pass.
    # The probe is the same every call, so the caller keeps one rather than
    # allocating and filling an arange per MoE layer.
    if expert_bounds_probe is None:
        expert_bounds_probe = torch.arange(
            ffn_size * expert_per_rank + 1,
            device=flat.device,
            dtype=flat.dtype,
        )
    bounds = torch.searchsorted(sorted_experts, expert_bounds_probe)
    offsets = bounds[:-1]
    counts = bounds[1:] - offsets

    # A destination reads the whole batch out of its slot, so a partial names
    # its token directly and needs no rebasing onto shipped rows.
    expand_idx = torch.div(order, num_slots, rounding_mode="floor").to(torch.int32)
    weights = topk_weights.reshape(-1)[order].to(torch.float32)

    # A rank owns a contiguous block of experts, so its partials begin where its
    # first expert's do and run to the end of its last.
    return DispatchPlan(
        counts=counts,
        routed_per_rank=counts.view(ffn_size, expert_per_rank).sum(1),
        segment_start=offsets.view(ffn_size, expert_per_rank)[:, 0],
        expand_idx=expand_idx,
        weights=weights,
    )


class GpuAsyncAFDConnector(AFDConnectorBase):
    """NVSHMEM symmetric-window asynchronous connector for CUDA AFD."""

    control_plane = None

    # The base builds this in __init__ from parse_extra_config; the narrowed
    # annotation is what lets mypy see the async connector's own fields.
    extra_info: GpuAsyncExtraInfo

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> GpuAsyncExtraInfo:
        return GpuAsyncExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = hf_config.hidden_size
        self.topk = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        # Combine has to know whether a reply carries shared-expert rows before
        # it arrives, and only the model config says so.
        self.has_shared_experts = bool(hf_config.n_shared_experts)
        self.payload_dtype = vllm_config.model_config.dtype
        self.max_seq_len = vllm_config.scheduler_config.max_num_batched_tokens
        self.tp_size = self.extra_info.attn_ranks_per_dp

        self.topology = build_async_topology(
            afd_config,
            role_rank,
            num_routed_experts=self.num_routed_experts,
        )
        self.world_rank = self.topology.world_rank
        self.attn_size = self.topology.attn_size
        self.ffn_size = self.topology.ffn_size
        self.expert_per_rank = self.topology.expert_per_rank
        self.is_attention = afd_config.role == "attention"

        # Stages that can be in flight at once, from either splitter. vLLM's
        # own ubatching (DBO) drives two forwards concurrently and stamps each
        # with its ubatch index, which lands here as the stage -- so it needs
        # rings just as much as this connector's async_moe_ubatching does.
        # Counting only the latter gave both DBO ubatches ring 0, where the
        # second dispatch overwrote the first's slot and flag and the reply the
        # first was waiting for never came: the two ubatch threads then
        # deadlocked, one inside recv_ffn_output and one waiting to be yielded
        # to.
        parallel_config = vllm_config.parallel_config
        vllm_stages = (
            max(1, int(parallel_config.num_ubatches))
            if bool(parallel_config.use_ubatching)
            else 1
        )
        connector_stages = (
            max(1, self.extra_info.async_moe_num_ubatches)
            if self.extra_info.async_moe_ubatching
            else 1
        )
        self.num_stages = max(vllm_stages, connector_stages)
        # An implied ring depth grows to fit; a pinned one is taken as given so
        # a deliberate choice still fails loudly below rather than being
        # silently overruled.
        self.ring_depth = (
            self.extra_info.ring_depth
            if self.extra_info.ring_depth_pinned
            else max(self.extra_info.ring_depth, self.num_stages)
        )
        if self.ring_depth < self.num_stages:
            raise ValueError(
                f"ring_depth {self.ring_depth} cannot serve "
                f"{self.num_stages} async MoE stages; each stage needs a slot "
                "of its own",
            )
        # Every Attention rank routes to every FFN rank, so a window carries one
        # region per opposite-role peer. Both roles allocate the larger of the
        # two so the symmetric allocation matches.
        self.num_regions = max(self.attn_size, self.ffn_size)
        # Payload rows are distinct tokens, so a batch is their bound no matter
        # how skewed the gate is. Only the 4-byte-per-partial index arrays need
        # the every-partial-to-one-rank worst case.
        self.token_cap = max(1, self.max_seq_len)
        self.partial_cap = max(1, self.max_seq_len * self.topk)
        # Shared-expert rows are split across the FFN ranks, so a slot holds a
        # fraction of the batch, not all of it -- and none at all when the model
        # has no shared experts. Both roles derive this from the same config, so
        # the symmetric allocation still matches.
        self.shared_cap = (
            -(-self.token_cap // self.ffn_size) if self.has_shared_experts else 0
        )
        self.layout = SlotLayout.build(
            expert_per_rank=self.expert_per_rank,
            partial_cap=self.partial_cap,
            token_cap=self.token_cap,
            shared_cap=self.shared_cap,
            hidden_size=self.hidden_size,
            payload_itemsize=torch.empty(0, dtype=self.payload_dtype).element_size(),
        )

        # Device headers, one buffer per (layer_idx, num_tokens) dispatch shape.
        self._header_device: dict[tuple[int, int], Tensor] = {}
        self._seq_device: Tensor | None = None
        # The searchsorted probe for the dispatch plan: identical every call,
        # so it is built once instead of per MoE layer.
        self._expert_bounds_probe: Tensor | None = None

        self.pg: ProcessGroup | None = None
        self.window: SymmWindow | None = None
        self._pending: dict[int, list[_PendingDispatch]] = {}
        # Dispatch payloads parked by afd_async_dispatch for the deferred
        # afd_async_recv one layer later, keyed by connector stage. The two
        # DBO ubatch threads share the connector, so the stage key is what
        # keeps one half's payload out of the other's receive.
        self.pending_cam_dispatches: dict[int, object] = {}
        self._free_rings: dict[int, list[int]] = {}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _rings_for_stage(self, stage_idx: int) -> list[int]:
        """Ring slots this stage owns.

        A ring names a window slot, so stages must not share one: two stages of
        the same layer are in flight at the same time, and handing both the same
        slot means the second dispatch overwrites the first's payload and flag,
        after which the reply the first is waiting for never arrives.
        """
        first = stage_idx % self.num_stages
        return list(range(first, self.ring_depth, self.num_stages))

    def init_afd_connector(self) -> None:
        """Collectively create the AFD world group and the symmetric window.

        All Attention and FFN ranks must call this with identical rendezvous and
        topology settings; the window allocation is symmetric, so a mismatched
        size fails here rather than corrupting a later transfer.
        """
        if self._initialized:
            return

        self.pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.topology.world_size,
            rank=self.world_rank,
            group_name=AFD_ASYNC_GPU_GROUP_NAME,
            timeout=timedelta(minutes=30),
        )
        device = torch.device("cuda", self.local_rank)
        # The dispatch sequence number lives on the device so that a CUDA graph
        # can advance it. An FFN rank recognizes an arrival by its flag holding
        # something other than the value it last consumed, so the number has to
        # rise on every dispatch -- including on every replay of a captured
        # graph, which runs no Python at all and would otherwise stamp the one
        # value that was live when the graph was recorded.
        self._seq_device = torch.zeros((), dtype=torch.int32, device=device)
        self.window = SymmWindow(
            num_regions=self.num_regions,
            ring_depth=self.ring_depth,
            layout=self.layout,
            payload_dtype=self.payload_dtype,
            device=device,
            group=self.pg,
            rank=self.world_rank,
            world_size=self.topology.world_size,
        )
        for stage in range(self.num_stages):
            self._free_rings[stage] = self._rings_for_stage(stage)
        logger.info(
            "AFD async GPU window ready: role=%s role_rank=%d world_rank=%d/%d "
            "regions=%d rings=%d partial_cap=%d slot=%.1fMiB total=%.1fMiB",
            self.afd_config.role,
            self.role_rank,
            self.world_rank,
            self.topology.world_size,
            self.num_regions,
            self.ring_depth,
            self.partial_cap,
            self.layout.slot_bytes / 2**20,
            self.window.total_bytes / 2**20,
        )
        self._initialized = True

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
        self.window = None
        if self.pg is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.pg)
        self.pg = None
        self._pending.clear()
        self._free_rings.clear()
        self._header_device.clear()
        self._seq_device = None
        self._initialized = False

    def select_experts(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Run vLLM's grouped top-k on the Attention side.

        ``compute_gate_topk`` delegates expert selection to the connector so the
        CAM and CUDA paths can share one gate; this is the CUDA half.
        """
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        if kwargs.get("mix_placement"):
            raise RuntimeError(
                "AFD async GPU connector does not support mix_placement",
            )
        return grouped_topk(
            hidden_states=kwargs["hidden_states"],
            gating_output=kwargs["router_logits"],
            topk=kwargs["top_k"],
            renormalize=kwargs["renormalize"],
            num_expert_group=kwargs.get("num_expert_group", 0),
            topk_group=kwargs.get("topk_group", 0),
            scoring_func=kwargs.get("scoring_func", "softmax"),
            routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
            e_score_correction_bias=kwargs.get("e_score_correction_bias"),
        )

    def _require_initialized(self) -> SymmWindow:
        if not self._initialized or self.window is None:
            raise RuntimeError("AFD async GPU connector is not initialized")
        return self.window

    def _shared_slice(self, ffn_rank: int, num_tokens: int) -> slice:
        """Token range whose shared-expert output ``ffn_rank`` owns.

        Contiguous chunks, so a rank's slice is a view: no index to build, none
        to gather through, and none to put on the wire. Empty when the model has
        no shared experts, which is what keeps the payload off the wire and the
        field out of the slot.

        The header's ``shared_tokens`` and the rows actually written have to
        agree, and they are produced by different callers, so both come from
        here rather than from two copies of the arithmetic.
        """
        if not self.has_shared_experts:
            return slice(0, 0)
        return slice(
            ffn_rank * num_tokens // self.ffn_size,
            (ffn_rank + 1) * num_tokens // self.ffn_size,
        )

    def _headers_for_shape(
        self,
        *,
        layer_idx: int,
        num_tokens: int,
    ) -> Tensor:
        """Device headers for one dispatch shape, prefix already filled.

        The prefix -- magic, version, layer, batch size, flags -- is fixed by
        ``(layer_idx, num_tokens)``, so it is written once here and then left
        alone. A dispatch only rewrites the routing tail.

        That the prefix never changes is what lets it leave the layer path. It
        used to be copied down from the host on every dispatch, not to carry
        anything new but to overwrite the previous layer's prefix in a buffer
        every layer shared. One buffer per shape removes the overwrite, and
        with it a strided host-to-device copy per MoE layer -- and, under a
        CUDA graph, a copy node that re-shipped a constant on every replay.

        Costs ``ffn_size * header_words`` int32s per distinct shape: MoE layers
        times the token counts they run at, a few KiB in practice.
        """
        key = (layer_idx, num_tokens)
        headers = self._header_device.get(key)
        if headers is not None:
            return headers

        assert self.window is not None
        device = self.window.device
        # torch.compiler.is_compiling() first, and not merely for speed: under
        # Dynamo the capture query below is a torch.* op returning a bool,
        # which cannot be traced into an FX graph and aborts compilation. It
        # folds to a constant while tracing, so the branch drops out there and
        # still guards the eager capture path, which is the one that can hit it.
        if (
            not torch.compiler.is_compiling()
            and device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            # Allocating here would come from the graph's private pool and the
            # prefix copy would be recorded against a host buffer that is gone
            # by the first replay. Warmup runs the same shapes capture does, so
            # reaching this means the shape was never warmed.
            raise RuntimeError(
                "AFD async GPU dispatch shape "
                f"(layer={layer_idx}, num_tokens={num_tokens}) was first seen "
                "during CUDA graph capture; warm it before capturing",
            )
        # Outside inference mode on purpose. This cache outlives the call that
        # fills it, and the first call for a shape can land inside vLLM's
        # inference-mode forward -- which brands the tensor an inference tensor,
        # so the next dispatch's write to the routing tail raises "Inplace
        # update to inference tensor outside InferenceMode". That is exactly
        # what a compiled prefill hits, because AOT compilation moves the first
        # touch of each shape inside the compiled region.
        with torch.inference_mode(False):
            headers = torch.empty(
                (self.ffn_size, self.layout.header_words),
                dtype=torch.int32,
                device=device,
            )
        # Written on the device, not copied down. The copy this replaces was
        # from pageable memory and therefore synchronous, and a blocking CUDA
        # call here deadlocks under vLLM's ubatching: this runs inside a ubatch
        # thread, which cannot block before reaching its next yield without
        # stranding the peer thread waiting to be yielded to.
        fill_header_prefix(
            headers,
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            flags=0,
        )
        self._header_device[key] = headers
        return headers

    def _headers_for(
        self,
        *,
        layer_idx: int,
        num_tokens: int,
        plan: DispatchPlan,
    ) -> Tensor:
        """Assemble one dispatch header per FFN rank, on the device.

        Only the routing tail is written here, straight from the plan, without
        ever leaving the device. Reading the routing back to encode it on the
        host instead was the last synchronize on the layer path, and a profile
        put it at 392us a call once per MoE layer -- not the copy, but the host
        waiting for everything queued ahead of it, which capped how far ahead of
        the device the host could ever get.
        """
        headers = self._headers_for_shape(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
        )
        headers[:, H_ROUTED_TOKENS] = plan.routed_per_rank
        headers[:, H_SEGMENT_START] = plan.segment_start
        headers[:, HEADER_FIXED_WORDS:] = plan.counts.view(
            self.ffn_size,
            self.expert_per_rank,
        )
        return headers

    # ==================================================================
    # Attention-side data path
    # ==================================================================

    def send_attn_output(
        self,
        hidden_states: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Route this layer's tokens and write them into every FFN window.

        ``topk_ids``/``topk_weights`` come from the Attention-side gate. Weights
        stay local -- only the routed activations and their route table go on the
        wire, and the weighting happens in ``recv_ffn_output``.
        """
        window = self._require_initialized()
        topk_ids: Tensor | None = kwargs.get("topk_ids")
        topk_weights: Tensor | None = kwargs.get("topk_weights")
        if topk_ids is None or topk_weights is None:
            raise RuntimeError(
                "AFD async GPU send_attn_output requires topk_ids and "
                "topk_weights from the Attention-side gate",
            )
        if (
            os.environ.get(AFD_ASYNC_DEBUG_ENV)
            and context.metadata.layer_idx < AFD_ASYNC_DEBUG_LAYERS
        ):
            logger.info(
                "AFD debug A%d send layer=%d tokens=%d %s %s",
                self.role_rank,
                context.metadata.layer_idx,
                int(hidden_states.shape[0]),
                _debug_norm("hidden", hidden_states),
                _debug_norm("w", topk_weights),
            )
        metadata = context.metadata
        num_tokens = metadata.total_tokens
        if hidden_states.shape[0] != num_tokens:
            raise ValueError(
                f"hidden_states has {hidden_states.shape[0]} rows but metadata "
                f"expects {num_tokens}",
            )
        expected_shape = (num_tokens, self.topk)
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                f"topk_ids shape must be {expected_shape}, got {tuple(topk_ids.shape)}",
            )
        # The weights are flattened alongside the ids to give each partial its
        # own weight, so a mismatched shape would silently misalign them.
        if tuple(topk_weights.shape) != expected_shape:
            raise ValueError(
                f"topk_weights shape must be {expected_shape}, "
                f"got {tuple(topk_weights.shape)}",
            )

        stage_idx = metadata.stage_idx
        rings = self._free_rings.setdefault(
            stage_idx,
            self._rings_for_stage(stage_idx),
        )
        if not rings:
            raise RuntimeError(
                f"AFD async GPU ring exhausted on stage {stage_idx}; the "
                "send-then-recv invariant was violated or the topology config "
                "does not match the actual peer count",
            )
        ring = rings.pop(0)
        # Bump the counter on the device, not on the host. Under CUDA graph
        # capture the host runs this function once and every later replay runs
        # only the recorded kernels, so a Python counter would freeze at the
        # captured value and stamp every FFN flag with it -- and an FFN rank
        # notices a dispatch precisely by its flag changing.
        assert self._seq_device is not None
        self._seq_device += 1

        if self._expert_bounds_probe is None:
            self._expert_bounds_probe = torch.arange(
                self.ffn_size * self.expert_per_rank + 1,
                device=topk_ids.device,
                dtype=topk_ids.dtype,
            )
        plan = plan_dispatch(
            topk_ids,
            topk_weights,
            ffn_size=self.ffn_size,
            expert_per_rank=self.expert_per_rank,
            expert_bounds_probe=self._expert_bounds_probe,
        )
        # Every FFN rank gets a slot even when routing sends it nothing, and it
        # replies to every slot, so a reply is expected from all of them.
        # Expecting only the ranks that received data leaves the empty rank's
        # reply unmatched and its ring slot never released -- which is what a
        # single-token decode hits, since both the routed segment and the
        # round-robin shared slice can come out empty for one rank.
        expected_ffn = list(range(self.ffn_size))
        shared_slices: list[slice] = []
        headers = self._headers_for(
            layer_idx=metadata.layer_idx,
            num_tokens=num_tokens,
            plan=plan,
        )
        for ffn_rank in range(self.ffn_size):
            # Round-robin needed an arange, a gather and a whole slot field per
            # peer per layer to achieve the balance this contiguous split gets
            # from a view.
            shared = self._shared_slice(ffn_rank, num_tokens)
            shared_slices.append(shared)
            # Everything but the shared slice goes out whole. The index arrays
            # cost a fraction of a percent of the slot, and the payload rows a
            # destination does not need are the few tokens none of whose topk
            # slots landed on it -- 1.6% of them at 2A2F. Sizing either to the
            # routing is what used to make the host wait for the device here.
            window.write_slot(
                peer=self.attn_size + ffn_rank,
                region=self.role_rank,
                ring=ring,
                header=headers[ffn_rank],
                expand_idx=plan.expand_idx,
                weights=plan.weights,
                routed_x=hidden_states,
                shared_x=hidden_states[shared],
                flag_value=self._seq_device,
            )

        logger.debug(
            "AFD dispatch sent: A%d layer=%d stage=%d tokens=%d ring=%d "
            "partials=%d awaiting_ffn=%s",
            self.role_rank,
            metadata.layer_idx,
            stage_idx,
            num_tokens,
            ring,
            num_tokens * self.topk,
            expected_ffn,
        )
        self._pending.setdefault(stage_idx, []).append(
            _PendingDispatch(
                context=context,
                shared_slices=shared_slices,
                num_tokens=num_tokens,
                ring=ring,
                expected_ffn=expected_ffn,
            ),
        )

    def recv_ffn_output(
        self,
        ref_tensor: Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        """Queue this layer's combine and return, without waiting for the data.

        The waiting happens on the stream: every reply is answered into a known
        slot, carries a known number of rows, and stamps the dispatch sequence
        it answers, so the whole combine can be enqueued before any of it has
        arrived. The host goes straight on to the next layer, which is the point
        -- polling for the reply here left the GPU with nothing queued behind
        the wait, and a profile found 86% of kernel launches executing within
        5us of being issued because of it.

        Nothing reports what turned up, so there is no per-arrival header check
        any more. The stream wait replaces it: it blocks on one specific slot
        being marked ready, where the poll took whatever had landed and had to
        check afterwards that it was the right thing.
        """

        window = self._require_initialized()
        queue = self._pending.get(ubatch_idx)
        if not queue:
            raise RuntimeError(
                f"AFD async GPU recv_ffn_output has no pending dispatch on "
                f"stage {ubatch_idx}",
            )
        pending = queue.pop(0)

        # Accumulate in the payload dtype. Each row takes at most one
        # contribution per FFN rank plus its shared one -- the topk sum already
        # happened on the FFN side -- so there is little left for a wider
        # accumulator to protect, and float32 cost a widening pass over every
        # arriving block plus a narrowing one on the way out.
        accumulator = torch.zeros(
            (pending.num_tokens, self.hidden_size),
            dtype=self.payload_dtype,
            device=ref_tensor.device,
        )
        for ffn_rank in pending.expected_ffn:
            # An FFN rank replies into the region it owns, on the ring the
            # dispatch used, and marks it with FLAG_REPLY_READY.
            window.stream_wait(ffn_rank, pending.ring, FLAG_REPLY_READY)
            # A reply is a whole batch, already weighted and summed over that
            # rank's experts, with a zero row wherever the rank held none of a
            # token's experts. Row i answers token i, so this is a plain add:
            # the scatter it replaces was the second largest kernel on the rank.
            accumulator += window.local_routed(
                ffn_rank,
                pending.ring,
                pending.num_tokens,
            )
            shared = pending.shared_slices[ffn_rank]
            shared_tokens = shared.stop - shared.start
            if self.has_shared_experts and shared_tokens:
                accumulator[shared] += window.local_shared(
                    ffn_rank,
                    pending.ring,
                    shared_tokens,
                )
            # Arm the flag for the next use of this slot, after the reads above
            # and before any dispatch that could reuse the ring -- all on this
            # stream, in that order. The peer cannot re-raise it early either:
            # it only replies to a dispatch, and that dispatch is enqueued
            # behind this reset.
            window.clear_flag(ffn_rank, pending.ring)
        if (
            os.environ.get(AFD_ASYNC_DEBUG_ENV)
            and pending.context.metadata.layer_idx < AFD_ASYNC_DEBUG_LAYERS
        ):
            logger.info(
                "AFD debug A%d combine layer=%d ring=%d tokens=%d %s",
                self.role_rank,
                pending.context.metadata.layer_idx,
                pending.ring,
                pending.num_tokens,
                _debug_norm("acc", accumulator),
            )
        logger.debug(
            "AFD combine queued: A%d layer=%d stage=%d ring=%d from=%s",
            self.role_rank,
            pending.context.metadata.layer_idx,
            ubatch_idx,
            pending.ring,
            pending.expected_ffn,
        )

        self._free_rings.setdefault(ubatch_idx, []).append(pending.ring)
        return accumulator

    # ==================================================================
    # FFN-side data path
    # ==================================================================

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        routed_out: Tensor | None = None,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Block until one Attention rank's routed tokens arrive.

        The layer index, token counts, and per-expert group list all come from
        the arrived slot header; the FFN side knows none of them beforehand.

        The slot holds the sender's whole batch and every sender's partials, so
        this rank takes the run of partials the header points it at and gathers
        the tokens they name -- a local gather that replaces both the duplicate
        rows the sender used to put on the wire and the readback it needed to
        size a per-destination slice.

        ``routed_out`` is where that gather lands. A caller that already has a
        destination -- a CUDA graph's input buffer, which must be written in
        place -- passes it and gets the rows staged for free, because the
        gather had to write somewhere regardless. Without it the gather
        allocates, as it always did. Too small a buffer is not an error: the
        gather allocates and the caller sees rows it did not stage, which is
        the same fallback an oversized work item takes anyway.
        """
        window = self._require_initialized()
        timeout_ms = int(kwargs.get("timeout_ms", 0))
        arrived = window.wait(timeout_s=timeout_ms / 1000.0 if timeout_ms else None)
        if arrived is None:
            raise TimeoutError("AFD async GPU dispatch recv timed out")

        header = arrived.header
        # Four fields the wire no longer carries, because this side can work
        # them out. The slot's own region names the Attention rank that wrote
        # it; ``_rings_for_stage`` hands stage s the rings congruent to s, so
        # the ring names the stage; and the shared-token split is the same
        # function of the batch size on both sides, evaluated here for this
        # rank's own slice.
        src_role_rank = arrived.region
        stage_idx = arrived.ring % self.num_stages
        shared = self._shared_slice(self.role_rank, header.num_tokens)
        shared_tokens = shared.stop - shared.start
        if header.is_shutdown:
            raise ConnectorShutdown(
                f"Attention rank {src_role_rank} announced shutdown",
            )
        # A slot is peer-written memory, so its routing tail gets one sanity
        # check before the grouped GEMM trusts it. Checking the decoded host
        # values here is free; the equivalent check on the device tensor cost a
        # synchronize per work item.
        if sum(header.expert_counts) != header.routed_tokens:
            raise RuntimeError(
                f"AFD async GPU dispatch header from A{src_role_rank} "
                f"has expert counts summing to {sum(header.expert_counts)} but "
                f"declares {header.routed_tokens} routed tokens",
            )

        expand_idx = window.local_expand_idx(
            arrived.region,
            arrived.ring,
            header.routed_tokens,
            header.segment_start,
        ).to(torch.int64)
        states = GpuAsyncTransferState(
            region=arrived.region,
            ring=arrived.ring,
            src_role_rank=src_role_rank,
            layer_idx=header.layer_idx,
            stage_idx=stage_idx,
            num_tokens=header.num_tokens,
            routed_tokens=header.routed_tokens,
            shared_tokens=shared_tokens,
            group_list=window.local_expert_counts(arrived.region, arrived.ring),
            expand_idx=expand_idx,
            weights=window.local_weights(
                arrived.region,
                arrived.ring,
                header.routed_tokens,
                header.segment_start,
            ),
            expand_x_shared=window.local_shared(
                arrived.region,
                arrived.ring,
                shared_tokens,
            ),
        )
        logger.debug(
            "AFD dispatch recv: F%d <- A%d layer=%d stage=%d routed=%d shared=%d "
            "region=%d ring=%d",
            self.role_rank,
            src_role_rank,
            header.layer_idx,
            stage_idx,
            header.routed_tokens,
            shared_tokens,
            arrived.region,
            arrived.ring,
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=header.layer_idx,
            stage_idx=stage_idx,
            seq_lens=[max(1, header.routed_tokens)],
        )
        arrived_rows = window.local_routed(
            arrived.region,
            arrived.ring,
            header.num_tokens,
        )
        if routed_out is not None and header.routed_tokens <= routed_out.shape[0]:
            gathered = routed_out[: header.routed_tokens]
            torch.index_select(arrived_rows, 0, expand_idx, out=gathered)
            states.staged_routed = True
        else:
            gathered = arrived_rows.index_select(0, expand_idx)
        return AFDA2FTransferPayload(
            hidden_states=gathered,
            context=AFDTransferContext(metadata=metadata, states=states),
        )

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Reduce expert output back to one row per token and write it back.

        Every partial of a token that landed on this rank is weighted and summed
        here, into the token's own row of a full batch. Tokens this rank held no
        expert for keep their zero row, which is what lets the Attention side
        add replies together without an index. Doing the reduction on this side
        keeps the duplicates off the wire.

        Both the weighting and the sum stay in the payload dtype. Widening to
        float32 first cost two extra passes over ``[partials, hidden]`` and made
        the scatter move twice the bytes, which a profile showed costing almost
        as much GPU time as the expert GEMM itself -- to protect a sum of at
        most ``topk`` terms whose result goes on the wire narrowed anyway.
        """
        window = self._require_initialized()
        states = context.states
        if not isinstance(states, GpuAsyncTransferState):
            raise RuntimeError(
                "AFD async GPU send_ffn_output requires GpuAsyncTransferState",
            )
        if states.expand_idx is None or states.weights is None:
            raise RuntimeError(
                "AFD async GPU send_ffn_output requires the dispatch expansion",
            )
        reduced = torch.zeros(
            (states.num_tokens, self.hidden_size),
            dtype=self.payload_dtype,
            device=ffn_output.device,
        )
        if states.routed_tokens:
            weighted = ffn_output * states.weights.unsqueeze(1).to(ffn_output.dtype)
            reduced.index_add_(0, states.expand_idx, weighted)

        shared_output: Tensor | None = kwargs.get("shared_output")
        if (
            os.environ.get(AFD_ASYNC_DEBUG_ENV)
            and states.layer_idx < AFD_ASYNC_DEBUG_LAYERS
        ):
            logger.info(
                "AFD debug F%d reply layer=%d ring=%d routed_rows=%d %s %s",
                self.role_rank,
                states.layer_idx,
                states.ring,
                states.routed_tokens,
                _debug_norm("routed", reduced),
                _debug_norm("shared", shared_output),
            )
        # ``reduced`` is float32 and the slot is the payload dtype; the copy
        # inside write_slot casts on its way into the peer window, so the
        # narrowing costs no extra pass over the rows.
        window.write_slot(
            peer=states.src_role_rank,
            region=self.role_rank,
            ring=states.ring,
            # No header. The rank waiting for this reply knows its shape before
            # it exists -- that is the premise of waiting on a stream rather
            # than polling -- so it never reads one, and every field a header
            # could carry it either chose itself or can derive. Writing one was
            # a copy per peer per layer that nothing consumed.
            header=None,
            expand_idx=None,
            weights=None,
            routed_x=reduced,
            shared_x=shared_output,
            # A constant marker: the Attention rank has to name the value it
            # waits for before the reply exists, and a captured graph freezes
            # that name, so it can only ever be a constant.
            flag_value=FLAG_REPLY_READY,
        )

    # ==================================================================
    # Connector-driven FFN loop
    # ==================================================================

    def recv_ffn_work_item(
        self,
        *,
        stage_idx: int,
        max_num_tokens: int,
        routed_out: Tensor | None = None,
    ) -> GpuAsyncFFNWorkItem:
        """Receive and normalize one connector-driven FFN dispatch item."""
        recv_output = self.recv_attn_output(
            ubatch_idx=stage_idx,
            routed_out=routed_out,
            timeout_ms=self.extra_info.recv_poll_timeout_ms,
        )
        states = recv_output.context.states
        assert isinstance(states, GpuAsyncTransferState)
        if (
            os.environ.get(AFD_ASYNC_DEBUG_ENV)
            and states.layer_idx < AFD_ASYNC_DEBUG_LAYERS
        ):
            logger.info(
                "AFD debug F%d recv layer=%d ring=%d routed=%d shared=%d %s",
                self.role_rank,
                states.layer_idx,
                states.ring,
                states.routed_tokens,
                states.shared_tokens,
                _debug_norm("hidden", recv_output.hidden_states),
            )
        return GpuAsyncFFNWorkItem(
            hidden_states=recv_output.hidden_states,
            context=recv_output.context,
            recv_output=recv_output,
            layer_idx=states.layer_idx,
            stage_idx=states.stage_idx,
            num_tokens=states.routed_tokens,
            total_num_tokens=states.num_tokens,
            shared_num_tokens=states.shared_tokens,
        )

    def send_ffn_work_item_output(
        self,
        work_item: GpuAsyncFFNWorkItem,
        ffn_output: Tensor | AFDF2ATransferPayload,
    ) -> Tensor:
        """Return one work item's expert output to its Attention rank."""
        if isinstance(ffn_output, AFDF2ATransferPayload):
            routed = ffn_output.routed_output
            shared = ffn_output.shared_output
        else:
            routed = ffn_output
            shared = None
        self.send_ffn_output(routed, work_item.context, shared_output=shared)
        return routed

    def announce_shutdown(self) -> None:
        """Tell every opposite-role peer to leave its receive loop."""
        window = self._require_initialized()
        peers = (
            range(self.attn_size, self.attn_size + self.ffn_size)
            if self.is_attention
            else range(self.attn_size)
        )
        # The same counter the dispatches stamp: a peer spots this exactly the
        # way it spots a dispatch, by the flag differing from what it consumed
        # last, so the two must not hand out the same number twice.
        assert self._seq_device is not None
        self._seq_device += 1
        header = encode_header(
            self.layout,
            layer_idx=0,
            num_tokens=0,
            routed_tokens=0,
            flags=FLAG_SHUTDOWN_BIT,
            expert_counts=[0] * self.expert_per_rank,
        )
        for peer in peers:
            # Header only: the shutdown bit is the whole message, so every
            # payload field is empty and write_slot skips it.
            window.write_slot(
                peer=peer,
                region=self.role_rank,
                ring=0,
                header=header,
                expand_idx=None,
                weights=None,
                routed_x=None,
                shared_x=None,
                flag_value=self._seq_device,
            )


__all__ = [
    "AFD_ASYNC_GPU_GROUP_NAME",
    "ConnectorShutdown",
    "DispatchPlan",
    "GpuAsyncAFDConnector",
    "GpuAsyncExtraInfo",
    "GpuAsyncFFNWorkItem",
    "GpuAsyncTransferState",
    "plan_dispatch",
]
