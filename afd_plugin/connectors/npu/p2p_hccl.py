# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""HCCL Send/Recv connector for Attention/FFN disaggregation on NPU.

``P2pHcclAFDConnector`` deliberately uses the public PyTorch distributed
point-to-point interface for its data path. Hidden states, FFN outputs, and
DeepSeek-V4 input IDs are transferred with ``torch.distributed.send`` and
``torch.distributed.recv`` over HCCL process groups. It does not load or call
the CAMP2P A2E/E2A custom operators.

The connector supports either consecutive Attention peers per FFN rank
(``A = k * F``) or consecutive FFN peers per Attention rank (``F = k * A``).
Each DBO stage owns an independent HCCL group so stages cannot
consume each other's messages. A Gloo control group carries stage token counts
before the FFN side posts receives and prepares its aggregate receive buffers.

U2 keeps the same point-to-point protocol but orders decoder transfers with
connector-owned NPU streams and events. DeepSeek-V4 drives its two stages from
one layer-major host loop, avoiding cross-thread DBO handoff without changing
the HCCL message protocol or introducing ``isend/irecv``. Eager execution uses
``dist.send/recv`` on dedicated communication streams. During Graph
warmup/capture, graph-visible A2F sends and F2A receives remain on the parent
capture stream, while model compute runs on an event-connected side stream.
FFN releases each F2A send from its own side stream as soon as the matching
compute completes, so Attention can close stage 0 before advancing to stage 1.
MTP remains a merged stage-0 exchange; for
``A = k * F`` each FFN rank aggregates its ``k`` Attention peers before running
the draft MoE and splits the result back to the same peers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.logger import init_logger
from vllm_ascend.utils import is_dspark_config

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import build_rank_mapping, init_afd_process_group
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# torch-npu 2.10.0.post2 source:
# torch_npu/dynamo/npugraph_ex/ops/_hcom_send_recv.py.
# Its dist.send/recv tracing wrappers pass torch.Size directly, which the target
# vLLM 0.23 compiler first flattens and then specializes despite a dynamic token
# dimension. Keep the same HCCL lowering without the optional, unused shape.
def _graph_hccl_send(
    tensor: torch.Tensor,
    *,
    dst: int,
    group: ProcessGroup,
) -> None:
    ranks = dist.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)
    # ### PATCH START: avoid the torch-npu dynamic-shape tracing guard
    torch.ops.npu_define._send.default(
        tensor,
        dst,
        ranks,
        pg_tag,
        0,
        None,
        None,
    )
    # ### PATCH END: avoid the torch-npu dynamic-shape tracing guard


def _graph_hccl_recv(
    tensor: torch.Tensor,
    *,
    src: int,
    group: ProcessGroup,
) -> None:
    ranks = dist.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)
    # ### PATCH START: avoid the torch-npu dynamic-shape tracing guard
    received = torch.ops.npu_define._recv.default(
        tensor,
        src,
        ranks,
        pg_tag,
        0,
        None,
        None,
    )
    tensor.copy_(received)
    # ### PATCH END: avoid the torch-npu dynamic-shape tracing guard


def _ensure_graph_hccl_ops_registered() -> None:
    if hasattr(torch.ops.npu_define, "_send") and hasattr(
        torch.ops.npu_define,
        "_recv",
    ):
        return

    # torch-npu 2.10.0.post2 registers these ops from its npugraph_ex package.
    # Service compilation normally imports it first, but connector initialization
    # must not depend on that incidental order.
    from torch_npu.dynamo.npugraph_ex.ops import (  # noqa: F401
        _hcom_send_recv,
    )

    if not hasattr(torch.ops.npu_define, "_send") or not hasattr(
        torch.ops.npu_define,
        "_recv",
    ):
        raise RuntimeError(
            "torch-npu did not register npu_define HCCL send/recv Graph ops",
        )


@dataclass(slots=True)
class HCCLP2PTransferState(AFDTransferState):
    """State retained between one FFN receive and its matching send."""

    stage_idx: int
    num_tokens: int
    peer_ranks: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    peer_slices: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class HCCLP2PStageLayout:
    """Immutable peer layout parsed once from a stage control payload."""

    peer_ranks: tuple[int, ...]
    seq_lens: tuple[int, ...]
    peer_slices: tuple[tuple[int, int, int], ...]
    num_tokens: int


@dataclass(frozen=True, slots=True)
class HCCLMTPHeader:
    """MTP phase shape and DP layout sent before the draft hidden tensor."""

    num_tokens: int
    speculative_step: int
    num_tokens_across_dp: torch.Tensor


@dataclass(frozen=True, slots=True)
class HCCLAttentionPipelineEvents:
    """NPU events ordering one Attention compute/send/receive exchange."""

    compute_done: Any
    send_done: Any
    recv_done: Any


@dataclass(frozen=True, slots=True)
class HCCLAttentionGraphEvents:
    """Events connecting one logical Graph stage across physical streams."""

    ready: Any
    compute_done: Any
    send_done: Any
    recv_done: Any


@dataclass(frozen=True, slots=True)
class HCCLAttentionGraphStreamPlan:
    """Map logical Attention Graph work onto physical NPU streams.

    A ``None`` communication stream means that operation remains on the parent
    capture stream. The plan keeps the logical DAG independent from whether
    Graph communication uses the parent stream or dedicated send/receive
    streams.
    """

    compute_stream: Any
    send_stream: Any | None = None
    recv_stream: Any | None = None

    def resolve(
        self,
        operation: Literal["compute", "send", "recv"],
        parent_stream: Any,
    ) -> Any:
        if operation == "compute":
            return self.compute_stream
        if operation == "send":
            return self.send_stream if self.send_stream is not None else parent_stream
        if operation == "recv":
            return self.recv_stream if self.recv_stream is not None else parent_stream
        raise ValueError(f"unknown Attention Graph stream operation: {operation}")


@dataclass(frozen=True, slots=True)
class HCCLPendingAttentionTransfer:
    """Attention stream state retained between proxy send and receive calls."""

    layer_idx: int
    events: HCCLAttentionPipelineEvents


@dataclass(frozen=True, slots=True)
class HCCLAttentionReceiveDependency:
    """F2A receive that a layer-major Attention stage has not consumed yet."""

    tensor: torch.Tensor
    event: Any


_MTP_HEADER_MAGIC = 0x4D545031
_MTP_HEADER_PREFIX_SIZE = 4
_EAGER_U2_STREAM_OVERLAP_ENV = "AFD_HCCL_EAGER_U2_STREAM_OVERLAP"
_STAGE_DIAGNOSTICS_ENV = "AFD_HCCL_STAGE_DIAGNOSTICS"
_FFN_COMPUTE_SYNC_DIAGNOSTICS_ENV = "AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS"
_GRAPH_U2_COMPUTE_OVERLAP_ENV = "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"
_GRAPH_U2_HYBRID_DAG_ENV = "AFD_HCCL_GRAPH_U2_HYBRID_DAG"
_GRAPH_U2_ATTENTION_THREE_STREAM_ENV = "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM"
_GRAPH_U2_FFN_RECV_STREAM_ENV = "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM"
_GRAPH_U2_FFN_CROSS_LAYER_ENV = "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER"


def _strict_binary_env_enabled(name: str, *, default: str = "1") -> bool:
    raw = os.environ.get(name, default).strip()
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def _graph_u2_compute_overlap_enabled() -> bool:
    return _strict_binary_env_enabled(_GRAPH_U2_COMPUTE_OVERLAP_ENV)


def _eager_u2_stream_overlap_enabled() -> bool:
    return _strict_binary_env_enabled(_EAGER_U2_STREAM_OVERLAP_ENV)


def _stage_diagnostics_enabled() -> bool:
    return _strict_binary_env_enabled(_STAGE_DIAGNOSTICS_ENV, default="0")


def _ffn_compute_sync_diagnostics_enabled() -> bool:
    return _strict_binary_env_enabled(
        _FFN_COMPUTE_SYNC_DIAGNOSTICS_ENV,
        default="0",
    )


def _graph_u2_hybrid_dag_enabled() -> bool:
    return _strict_binary_env_enabled(_GRAPH_U2_HYBRID_DAG_ENV)


def _graph_u2_attention_three_stream_enabled() -> bool:
    return _strict_binary_env_enabled(
        _GRAPH_U2_ATTENTION_THREE_STREAM_ENV,
        default="1",
    )


def _graph_u2_ffn_recv_stream_enabled() -> bool:
    return _strict_binary_env_enabled(
        _GRAPH_U2_FFN_RECV_STREAM_ENV,
        default="1",
    )


def _graph_u2_ffn_cross_layer_enabled() -> bool:
    return _strict_binary_env_enabled(
        _GRAPH_U2_FFN_CROSS_LAYER_ENV,
        default="1",
    )


class P2pHcclAFDConnector(AFDConnectorBase):
    """Move AFD tensors with standard HCCL point-to-point operations."""

    yield_after_attn_send = False

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("HCCL P2P connector_extra_config must be a mapping")
        if raw:
            raise ValueError(
                "P2pHcclAFDConnector does not support connector_extra_config",
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
        self.mapping = build_rank_mapping(afd_config, role_rank)
        # FFN token-count routing uses the connector topology contract shared
        # with CAMP2P. Here the generic rank mapping is the complete topology.
        self.topology = self.mapping

        self._initialized = False
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.attention_fans_out = self.mapping.attention_fans_out
        self.tensor_parallel_size = int(
            vllm_config.parallel_config.tensor_parallel_size,
        )
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dtype = vllm_config.model_config.dtype
        self.max_num_batched_tokens = int(
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        architectures = vllm_config.model_config.hf_config.architectures or []
        self.requires_input_ids = any(
            architecture in {"DeepseekV4ForCausalLM", "AFDDeepseekV4ForCausalLM"}
            for architecture in architectures
        )
        speculative_config = getattr(vllm_config, "speculative_config", None)
        self.requires_mtp = (
            speculative_config is not None
            and getattr(speculative_config, "method", None) == "mtp"
            and not is_dspark_config(vllm_config)
        )
        self.num_speculative_tokens = (
            int(getattr(speculative_config, "num_speculative_tokens", 1))
            if self.requires_mtp
            else 0
        )
        self.mtp_draft_graph_enabled = bool(
            self.requires_mtp
            and not bool(
                getattr(speculative_config, "enforce_eager", True),
            )
        )
        self.vocab_size = int(vllm_config.model_config.hf_config.vocab_size)
        self.num_stages = max(
            1,
            int(vllm_config.parallel_config.num_ubatches),
        )
        self.stream_overlap_enabled = self.num_stages > 1
        self.eager_u2_stream_overlap_enabled = bool(
            self.stream_overlap_enabled and _eager_u2_stream_overlap_enabled()
        )
        self.stage_diagnostics_enabled = _stage_diagnostics_enabled()
        self.ffn_compute_sync_diagnostics_enabled = (
            _ffn_compute_sync_diagnostics_enabled()
        )
        self.graph_u2_compute_overlap_enabled = _graph_u2_compute_overlap_enabled()
        self.graph_u2_hybrid_dag_enabled = _graph_u2_hybrid_dag_enabled()
        self.graph_u2_attention_three_stream_enabled = (
            _graph_u2_attention_three_stream_enabled()
        )
        self.graph_u2_ffn_recv_stream_enabled = _graph_u2_ffn_recv_stream_enabled()
        self.graph_u2_ffn_cross_layer_enabled = _graph_u2_ffn_cross_layer_enabled()
        if (
            self.graph_u2_ffn_cross_layer_enabled
            and not self.graph_u2_ffn_recv_stream_enabled
        ):
            raise RuntimeError(
                f"{_GRAPH_U2_FFN_CROSS_LAYER_ENV}=1 requires "
                f"{_GRAPH_U2_FFN_RECV_STREAM_ENV}=1"
            )
        if self.stage_diagnostics_enabled or self.ffn_compute_sync_diagnostics_enabled:
            logger.warning(
                "AFD HCCL diagnostics configuration: role=%s world_rank=%s "
                "role_rank=%s stages=%s eager_u2_stream_overlap=%s "
                "ffn_compute_sync=%s "
                "graph_u2_compute_overlap=%s graph_u2_hybrid_dag=%s "
                "graph_u2_attention_three_stream=%s "
                "graph_u2_ffn_recv_stream=%s graph_u2_ffn_cross_layer=%s",
                self.afd_config.role,
                self.world_rank,
                self.role_rank,
                self.num_stages,
                int(self.eager_u2_stream_overlap_enabled),
                int(self.ffn_compute_sync_diagnostics_enabled),
                int(self.graph_u2_compute_overlap_enabled),
                int(self.graph_u2_hybrid_dag_enabled),
                int(self.graph_u2_attention_three_stream_enabled),
                int(self.graph_u2_ffn_recv_stream_enabled),
                int(self.graph_u2_ffn_cross_layer_enabled),
            )

        self.data_pg_list: list[ProcessGroup] = []
        self.ids_pg_list: list[ProcessGroup] = []
        self.p2p_pg: ProcessGroup | None = None
        self.input_ids_buffers: list[torch.Tensor] = []
        self.mtp_header_buffers: list[torch.Tensor] = []
        self.mtp_header_buffers_by_peer: list[dict[int, torch.Tensor]] = []
        self.hidden_recv_buffers: dict[int, torch.Tensor] = {}
        self.mtp_hidden_recv_buffers: dict[int, torch.Tensor] = {}
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.stage_layouts: dict[int, HCCLP2PStageLayout] = {}
        self.mtp_stage_layouts: dict[int, HCCLP2PStageLayout] = {}
        self.mtp_graph_header_values: tuple[int, ...] | None = None
        self.mtp_graph_header_values_by_peer: dict[int, tuple[int, ...]] = {}
        self.mtp_phase_control_enabled = False
        self.mtp_phase_graph_replay = False
        self.target_graph_replay: bool | None = None
        self.is_graph_capturing = False
        self.is_warmup = False
        self.a2f_send_stream = None
        self.f2a_recv_stream = None
        self.attention_graph_compute_stream = None
        self.attention_graph_stream_plan: HCCLAttentionGraphStreamPlan | None = None
        self.attention_pipeline_events: dict[
            tuple[int, int], HCCLAttentionPipelineEvents
        ] = {}
        self.attention_graph_events: dict[
            tuple[int, int], HCCLAttentionGraphEvents
        ] = {}
        self.pending_attention_transfers: dict[int, HCCLPendingAttentionTransfer] = {}
        self.attention_receive_dependencies: dict[
            int, HCCLAttentionReceiveDependency
        ] = {}
        self.control_plane = P2pHcclAFDControlPlane(self)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        """Create one data group per stage and the separate control/ID groups."""
        if self._initialized:
            return

        # Importing torch_npu registers the HCCL backend. This connector does
        # not import or require the afd_ascend custom-op extension.
        import torch_npu  # noqa: F401

        if not bool(getattr(self.vllm_config.model_config, "enforce_eager", True)):
            _ensure_graph_hccl_ops_registered()

        timeout = timedelta(minutes=30)
        try:
            for stage_idx in range(self.num_stages):
                suffix = "" if stage_idx == 0 else f"_{stage_idx}"
                data_group = init_afd_process_group(
                    backend="hccl",
                    init_method=(
                        f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                    ),
                    world_size=self.ffn_size + self.attn_size,
                    rank=self.world_rank,
                    group_name=f"afd_hccl_p2p{suffix}",
                    timeout=timeout,
                )
                self.data_pg_list.append(data_group)

                if self.requires_input_ids or self.requires_mtp:
                    ids_group = init_afd_process_group(
                        backend="hccl",
                        init_method=(
                            f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                        ),
                        world_size=self.ffn_size + self.attn_size,
                        rank=self.world_rank,
                        group_name=f"afd_hccl_p2p_ids{suffix}",
                        timeout=timeout,
                    )
                    self.ids_pg_list.append(ids_group)
                    self.input_ids_buffers.append(
                        torch.empty(
                            max(self.max_num_batched_tokens, self.ratio),
                            dtype=torch.int32,
                            device=f"npu:{self.local_rank}",
                        ),
                    )
                    self.mtp_header_buffers.append(
                        torch.empty(
                            _MTP_HEADER_PREFIX_SIZE + self.ffn_size,
                            dtype=torch.int32,
                            device=f"npu:{self.local_rank}",
                        ),
                    )
                    self.mtp_header_buffers_by_peer.append(
                        {
                            peer_rank: torch.empty(
                                _MTP_HEADER_PREFIX_SIZE + self.ffn_size,
                                dtype=torch.int32,
                                device=f"npu:{self.local_rank}",
                            )
                            for peer_rank in self.mapping.ffn_peer_ranks
                        }
                    )

            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.p2p_rank,
                group_name="afd_hccl_p2p_control",
                timeout=timeout,
            )
            if self.stream_overlap_enabled and self.afd_config.role == "attention":
                self._initialize_attention_stream_pipeline()
        except BaseException:
            self.close()
            raise
        self._initialized = True

    def close(self) -> None:
        """Destroy every process group and discard connector-owned buffers."""
        groups = [self.p2p_pg, *self.ids_pg_list, *self.data_pg_list]
        # Detach owned state before process-group teardown. Worker shutdown can
        # enter close() again after a first destroy reports a device-side error;
        # the second entry must not destroy the same group twice.
        self.p2p_pg = None
        self.ids_pg_list = []
        self.data_pg_list = []
        destroyed: set[int] = set()
        first_error: Exception | None = None
        for group in groups:
            if group is None or id(group) in destroyed:
                continue
            destroyed.add(id(group))
            try:
                dist.destroy_process_group(group)
            except Exception as error:
                if first_error is None:
                    first_error = error

        self.input_ids_buffers = []
        self.mtp_header_buffers = []
        self.mtp_header_buffers_by_peer = []
        self.hidden_recv_buffers = {}
        self.mtp_hidden_recv_buffers = {}
        self.dp_metadata_list = {}
        self.stage_layouts = {}
        self.mtp_stage_layouts = {}
        self.mtp_graph_header_values = None
        self.mtp_graph_header_values_by_peer = {}
        self.mtp_phase_control_enabled = False
        self.mtp_phase_graph_replay = False
        self.target_graph_replay = None
        self.a2f_send_stream = None
        self.f2a_recv_stream = None
        self.attention_graph_compute_stream = None
        self.attention_graph_stream_plan = None
        self.attention_pipeline_events = {}
        self.attention_graph_events = {}
        self.pending_attention_transfers = {}
        self.attention_receive_dependencies = {}
        self.control_plane.reset_pending_payload()
        self._initialized = False
        if first_error is not None:
            raise first_error

    @property
    def attention_stream_pipeline_ready(self) -> bool:
        return bool(
            self.stream_overlap_enabled
            and self.afd_config.role == "attention"
            and self.a2f_send_stream is not None
            and self.f2a_recv_stream is not None
            and self.attention_pipeline_events
        )

    def _attention_stream_pipeline_active(self) -> bool:
        # Dynamo lowers the synchronous HCCL path into graph ops. Stream/event
        # scheduling belongs to the later NPUGraph warmup/capture phase.
        if torch.compiler.is_compiling():
            return False
        if not self.attention_stream_pipeline_ready:
            return False
        if not self.eager_u2_stream_overlap_enabled:
            return False
        try:
            forward_context = get_forward_context()
        except AssertionError:
            # Standalone connector component tests intentionally run without
            # a vLLM model forward context and must retain the synchronous path.
            return False
        graph_ubatching = bool(getattr(forward_context, "afd_graph_ubatching", False))
        # Graph U2 uses a dedicated path below. Keep this eager pipeline
        # disabled so the Graph stream plan owns all physical stream mapping.
        if graph_ubatching:
            return False
        return bool(
            getattr(forward_context, "dbo_enabled", False)
            and int(getattr(forward_context, "num_ubatches", 1)) > 1
        )

    @staticmethod
    def _layer_major_attention_pipeline_active() -> bool:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(getattr(forward_context, "afd_layer_major_u2", False))

    def _initialize_attention_stream_pipeline(self) -> None:
        """Create eager communication and Graph compute streams/events."""
        device = torch.device("npu", self.local_rank)
        self.a2f_send_stream = torch.npu.Stream(device=device)
        self.f2a_recv_stream = torch.npu.Stream(device=device)
        self.attention_graph_compute_stream = torch.npu.Stream(device=device)
        self.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
            compute_stream=self.attention_graph_compute_stream,
            send_stream=(
                self.a2f_send_stream
                if self.graph_u2_attention_three_stream_enabled
                else None
            ),
            recv_stream=(
                self.f2a_recv_stream
                if self.graph_u2_attention_three_stream_enabled
                else None
            ),
        )
        num_layers = int(self.vllm_config.model_config.hf_config.num_hidden_layers)
        self.attention_pipeline_events = {
            (layer_idx, stage_idx): HCCLAttentionPipelineEvents(
                compute_done=torch.npu.Event(),
                send_done=torch.npu.Event(),
                recv_done=torch.npu.Event(),
            )
            for layer_idx in range(num_layers)
            for stage_idx in range(self.num_stages)
        }
        self.attention_graph_events = {
            (layer_idx, stage_idx): HCCLAttentionGraphEvents(
                ready=torch.npu.Event(),
                compute_done=torch.npu.Event(),
                send_done=torch.npu.Event(),
                recv_done=torch.npu.Event(),
            )
            for layer_idx in range(num_layers + 1)
            for stage_idx in range(self.num_stages)
        }

    def attention_graph_compute_pipeline_active(self) -> bool:
        """Return whether Graph U2 should run model compute on its side stream."""
        # Keep Python container inspection out of the Dynamo full-graph path.
        # The synchronous HCCL ops are lowered while compiling; stream/event
        # scheduling is selected later during NPUGraph warmup and capture.
        if torch.compiler.is_compiling():
            return False
        graph_pipeline_ready = bool(
            self.stream_overlap_enabled
            and self.afd_config.role == "attention"
            and self.attention_graph_stream_plan is not None
            and self.attention_graph_events
        )
        if not graph_pipeline_ready:
            return False
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(
            self.graph_u2_compute_overlap_enabled
            and getattr(forward_context, "afd_graph_ubatching", False)
            and getattr(forward_context, "afd_layer_major_u2", False)
            and getattr(forward_context, "dbo_enabled", False)
            and int(getattr(forward_context, "num_ubatches", 1)) > 1
        )

    def attention_graph_hybrid_dag_active(self) -> bool:
        """Return whether Graph U2 uses per-stage receive-ready dependencies."""
        return bool(
            self.graph_u2_hybrid_dag_enabled
            and self.attention_graph_compute_pipeline_active()
        )

    def _attention_graph_stream(
        self,
        operation: Literal["compute", "send", "recv"],
        parent_stream: Any,
    ) -> Any:
        plan = self.attention_graph_stream_plan
        if plan is None:
            raise RuntimeError("HCCL P2P Attention Graph stream plan is unavailable")
        return plan.resolve(operation, parent_stream)

    @contextmanager
    def _use_attention_graph_stream(
        self,
        operation: Literal["send", "recv"],
        parent_stream: Any,
    ) -> Iterator[Any]:
        operation_stream = self._attention_graph_stream(operation, parent_stream)
        if operation_stream is parent_stream:
            yield operation_stream
            return
        with torch.npu.stream(operation_stream):
            yield operation_stream

    @contextmanager
    def attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
        wait_for_receive_layer_idx: int | None = None,
    ) -> Iterator[None]:
        """Fork Graph compute from the parent HCCL capture stream."""
        if not self.attention_graph_compute_pipeline_active():
            raise RuntimeError("HCCL P2P Attention Graph compute pipeline is inactive")
        events = self._attention_graph_events(layer_idx, stage_idx)
        parent_stream = torch.npu.current_stream()
        if wait_for_receive_layer_idx is None:
            events.ready.record(parent_stream)
            ready_event = events.ready
        else:
            ready_event = self._attention_graph_events(
                wait_for_receive_layer_idx,
                stage_idx,
            ).recv_done
        compute_stream = self._attention_graph_stream("compute", parent_stream)
        with torch.npu.stream(compute_stream):
            ready_event.wait(compute_stream)
            for tensor in tensors:
                self._record_stream(tensor, compute_stream)
            yield
            events.compute_done.record(compute_stream)

    def wait_for_attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Release one logical send after the matching Graph compute."""
        if not self.attention_graph_compute_pipeline_active():
            raise RuntimeError("HCCL P2P Attention Graph compute pipeline is inactive")
        parent_stream = torch.npu.current_stream()
        events = self._attention_graph_events(layer_idx, stage_idx)
        if self._graph_parent_transport_active():
            events.compute_done.wait(parent_stream)
            for tensor in tensors:
                self._record_stream(tensor, parent_stream)
            return
        send_stream = self._attention_graph_stream("send", parent_stream)
        events.compute_done.wait(send_stream)
        for tensor in tensors:
            self._record_stream(tensor, send_stream)

    def wait_for_attention_graph_receive(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensor: torch.Tensor,
    ) -> None:
        """Join a Graph F2A receive before non-hybrid next-layer compute."""
        if not self.attention_graph_compute_pipeline_active():
            return
        parent_stream = torch.npu.current_stream()
        events = self._attention_graph_events(layer_idx, stage_idx)
        events.recv_done.wait(parent_stream)
        self._record_stream(tensor, parent_stream)

    def join_attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Join final side-stream compute back to the parent capture stream."""
        if not self.attention_graph_compute_pipeline_active():
            raise RuntimeError("HCCL P2P Attention Graph compute pipeline is inactive")
        parent_stream = torch.npu.current_stream()
        events = self._attention_graph_events(layer_idx, stage_idx)
        events.compute_done.wait(parent_stream)
        for tensor in tensors:
            self._record_stream(tensor, parent_stream)

    def _attention_graph_events(
        self,
        layer_idx: int,
        stage_idx: int,
    ) -> HCCLAttentionGraphEvents:
        try:
            return self.attention_graph_events[(layer_idx, stage_idx)]
        except KeyError as exc:
            raise RuntimeError(
                "HCCL P2P Attention Graph event is not initialized: "
                f"layer={layer_idx} stage={stage_idx}"
            ) from exc

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"HCCL P2P metadata token count {metadata.total_tokens}",
            )

        if metadata.phase == "mtp":
            self._validate_mtp_scope(metadata)
            expected_shape = (metadata.total_tokens, self.hidden_size)
            if (
                not torch.compiler.is_compiling()
                and tuple(hidden_states.shape) != expected_shape
            ):
                raise ValueError(
                    "DSV4 MTP HCCL transfer requires post-HC MoE input shape "
                    f"{expected_shape}, got {tuple(hidden_states.shape)}"
                )
            num_tokens_across_dp = kwargs.get("num_tokens_across_dp")
            if num_tokens_across_dp is None:
                raise RuntimeError(
                    "DSV4 MTP HCCL transfer requires num_tokens_across_dp"
                )
            self.send_mtp_header(
                num_tokens=metadata.total_tokens,
                speculative_step=metadata.speculative_step,
                num_tokens_across_dp=num_tokens_across_dp,
                stage_idx=metadata.stage_idx,
            )
        elif self.requires_input_ids:
            input_ids: torch.Tensor | None = kwargs.get("input_ids")
            if metadata.layer_idx == 0:
                if input_ids is None:
                    raise RuntimeError("DSV4 HCCL P2P layer 0 requires input_ids")
                pretransferred = bool(
                    getattr(
                        get_forward_context(),
                        "afd_input_ids_pretransferred",
                        False,
                    ),
                )
                if not pretransferred:
                    if self._attention_stream_pipeline_active():
                        raise RuntimeError(
                            "HCCL P2P stream overlap requires input_ids to be "
                            "pretransferred before the Attention ubatch threads"
                        )
                    self.send_input_ids(input_ids, ubatch_idx=metadata.stage_idx)
                    maybe_apply_dbo_yield(input_ids, role="attention")
            elif input_ids is not None:
                raise RuntimeError("DSV4 HCCL P2P sends input_ids only at layer 0")

        group = self._data_group(metadata.stage_idx)
        if self._attention_stream_pipeline_active() and metadata.phase == "decoder":
            self._enqueue_attention_send(
                hidden_states,
                layer_idx=metadata.layer_idx,
                stage_idx=metadata.stage_idx,
                group=group,
            )
            return
        if (
            metadata.phase == "decoder"
            and self.attention_graph_compute_pipeline_active()
        ):
            events = self._attention_graph_events(
                metadata.layer_idx,
                metadata.stage_idx,
            )
            parent_stream = torch.npu.current_stream()
            # Graph-visible HCCL must stay on the stream that owns the active
            # NPUGraph's parent stream.  A configured side send stream can
            # leave pending HCCL work before capture and an unjoined graph
            # task during capture.  Keep the model compute fork independent.
            if self._graph_parent_transport_active():
                # The compute fork may be a side stream.  Join it before the
                # parent-stream HCCL operation in both warmup and capture.
                events.compute_done.wait(parent_stream)
                self._send_attention_tensor(
                    hidden_states,
                    group=group,
                )
                events.send_done.record(parent_stream)
                return
            with self._use_attention_graph_stream(
                "send",
                parent_stream,
            ) as send_stream:
                self._send_attention_tensor(
                    hidden_states,
                    group=group,
                    stream=(None if send_stream is parent_stream else send_stream),
                )
                events.send_done.record(send_stream)
            return
        self._send_attention_tensor(
            hidden_states,
            group=group,
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_initialized()
        group = self._data_group(ubatch_idx)
        phase = str(kwargs.get("phase", "decoder"))
        if self._attention_stream_pipeline_active() and phase == "decoder":
            return self._enqueue_attention_receive(
                ref_tensor,
                stage_idx=ubatch_idx,
                group=group,
            )
        if phase == "decoder" and self.attention_graph_compute_pipeline_active():
            layer_idx = kwargs.get("layer_idx")
            if layer_idx is None:
                raise RuntimeError(
                    "HCCL P2P Attention Graph receive requires layer_idx"
                )
            events = self._attention_graph_events(int(layer_idx), ubatch_idx)
            parent_stream = torch.npu.current_stream()
            if self._graph_parent_transport_active():
                events.send_done.wait(parent_stream)
                self._recv_attention_tensor(
                    ref_tensor,
                    group=group,
                )
                events.recv_done.record(parent_stream)
                return ref_tensor
            with self._use_attention_graph_stream(
                "recv",
                parent_stream,
            ) as recv_stream:
                events.send_done.wait(recv_stream)
                self._recv_attention_tensor(
                    ref_tensor,
                    group=group,
                    stream=(None if recv_stream is parent_stream else recv_stream),
                )
                events.recv_done.record(recv_stream)
            return ref_tensor
        self._recv_attention_tensor(
            ref_tensor,
            group=group,
        )
        # FFN processes layers in layer-major order (stage 0, then stage 1),
        # while each Attention DBO thread would otherwise continue directly
        # to the next layer. Yield after the blocking receive so the peer
        # stage can send the current layer before this stage sends layer + 1.
        if phase == "decoder":
            maybe_apply_dbo_yield(ref_tensor, role="attention")
        return ref_tensor

    def _enqueue_attention_send(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_idx: int,
        stage_idx: int,
        group: ProcessGroup,
    ) -> None:
        if stage_idx in self.pending_attention_transfers:
            raise RuntimeError(
                "HCCL P2P Attention stage already has a pending receive: "
                f"stage={stage_idx}"
            )
        events = self._attention_events(layer_idx, stage_idx)
        compute_stream = torch.npu.current_stream()
        events.compute_done.record(compute_stream)
        assert self.a2f_send_stream is not None
        with torch.npu.stream(self.a2f_send_stream):
            events.compute_done.wait(self.a2f_send_stream)
            self._send_attention_tensor(
                hidden_states,
                group=group,
                stream=self.a2f_send_stream,
            )
            events.send_done.record(self.a2f_send_stream)
        self.pending_attention_transfers[stage_idx] = HCCLPendingAttentionTransfer(
            layer_idx=layer_idx,
            events=events,
        )

    def _enqueue_attention_receive(
        self,
        ref_tensor: torch.Tensor,
        *,
        stage_idx: int,
        group: ProcessGroup,
    ) -> torch.Tensor:
        pending = self.pending_attention_transfers.pop(stage_idx, None)
        if pending is None:
            raise RuntimeError(
                "HCCL P2P Attention receive has no matching streamed send: "
                f"stage={stage_idx}"
            )
        recv_tensor = torch.empty_like(ref_tensor)
        assert self.f2a_recv_stream is not None
        with torch.npu.stream(self.f2a_recv_stream):
            pending.events.send_done.wait(self.f2a_recv_stream)
            self._recv_attention_tensor(
                recv_tensor,
                group=group,
                stream=self.f2a_recv_stream,
            )
            pending.events.recv_done.record(self.f2a_recv_stream)

        if self._layer_major_attention_pipeline_active():
            if stage_idx in self.attention_receive_dependencies:
                raise RuntimeError(
                    "HCCL P2P Attention stage has an unconsumed F2A receive: "
                    f"stage={stage_idx}"
                )
            self.attention_receive_dependencies[stage_idx] = (
                HCCLAttentionReceiveDependency(
                    tensor=recv_tensor,
                    event=pending.events.recv_done,
                )
            )
            return recv_tensor

        # Let the peer ubatch enqueue its Attention work while this receive is
        # progressing on the connector-owned stream.
        maybe_apply_dbo_yield(recv_tensor, role="attention")
        compute_stream = torch.npu.current_stream()
        pending.events.recv_done.wait(compute_stream)
        self._record_stream(recv_tensor, compute_stream)
        return recv_tensor

    def wait_for_attention_stage_receive(
        self,
        *,
        stage_idx: int,
        tensor: torch.Tensor,
    ) -> None:
        """Make the current compute stream consume one deferred F2A receive."""
        # Graph U2 owns its receive dependency in attention_graph_events, so
        # it never consumes eager attention_receive_dependencies here.
        if not self._attention_stream_pipeline_active():
            return
        dependency = self.attention_receive_dependencies.pop(stage_idx, None)
        if dependency is None:
            raise RuntimeError(
                "HCCL P2P Attention stage has no deferred F2A receive: "
                f"stage={stage_idx}"
            )
        if dependency.tensor is not tensor:
            raise RuntimeError(
                "HCCL P2P Attention stage received an unexpected tensor: "
                f"stage={stage_idx}"
            )
        compute_stream = torch.npu.current_stream()
        dependency.event.wait(compute_stream)
        self._record_stream(tensor, compute_stream)

    def require_attention_pipeline_idle(self) -> None:
        """Reject a new layer-major step when a prior step left stale state."""
        if self.pending_attention_transfers or self.attention_receive_dependencies:
            raise RuntimeError(
                "HCCL P2P Attention pipeline is not idle: "
                f"pending={tuple(self.pending_attention_transfers)} "
                f"deferred={tuple(self.attention_receive_dependencies)}"
            )

    def reset_attention_pipeline_state(self) -> None:
        """Discard per-step stream bookkeeping after a failed model forward."""
        self.pending_attention_transfers.clear()
        self.attention_receive_dependencies.clear()

    def _attention_events(
        self,
        layer_idx: int,
        stage_idx: int,
    ) -> HCCLAttentionPipelineEvents:
        try:
            return self.attention_pipeline_events[(layer_idx, stage_idx)]
        except KeyError as exc:
            raise RuntimeError(
                "HCCL P2P Attention pipeline event is not initialized: "
                f"layer={layer_idx} stage={stage_idx}"
            ) from exc

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_initialized()
        layer_idx = int(kwargs.get("layer_idx", 0))
        phase = str(kwargs.get("phase", "decoder"))
        speculative_step = int(kwargs.get("speculative_step", 0))
        stream = kwargs.get("_stream")
        if phase == "mtp":
            explicit_num_tokens = int(kwargs.get("num_tokens", 0))
            layout = self.mtp_stage_layouts.pop(ubatch_idx, None)
            if layout is None:
                raise RuntimeError(
                    "DSV4 MTP hidden receive requires a preceding header: "
                    f"stage={ubatch_idx}"
                )
            if layout.num_tokens != explicit_num_tokens:
                raise ValueError(
                    "DSV4 MTP hidden token count does not match the aggregated "
                    f"header layout: {explicit_num_tokens} != {layout.num_tokens}"
                )
            metadata_probe = AFDTransferMetadata.create_ffn_metadata(
                layer_idx=layer_idx,
                stage_idx=ubatch_idx,
                seq_lens=layout.seq_lens,
                phase=phase,
                speculative_step=speculative_step,
            )
            self._validate_mtp_scope(metadata_probe)
        else:
            layout = self._stage_layout(
                ubatch_idx,
                fallback=int(kwargs.get("max_num_tokens", 1)),
            )
        num_tokens = layout.num_tokens
        input_ids = None
        if phase == "decoder" and self.requires_input_ids and layer_idx == 0:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                input_ids = self.recv_input_ids(num_tokens, ubatch_idx=ubatch_idx)

        hidden_states = (
            self._mtp_hidden_recv_buffer(ubatch_idx, num_tokens)
            if phase == "mtp"
            else self._hidden_recv_buffer(ubatch_idx, num_tokens)
        )
        group = self._data_group(ubatch_idx)
        for source_rank, start, end in layout.peer_slices:
            peer_slice = hidden_states[start:end]
            self._recv_tensor(
                peer_slice,
                src=source_rank,
                group=group,
                stream=stream,
            )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=layout.seq_lens,
            phase=phase,
            speculative_step=speculative_step,
        )
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=AFDTransferContext(
                metadata=metadata,
                states=HCCLP2PTransferState(
                    stage_idx=ubatch_idx,
                    num_tokens=num_tokens,
                    peer_ranks=layout.peer_ranks,
                    seq_lens=layout.seq_lens,
                    peer_slices=layout.peer_slices,
                ),
            ),
            input_ids=input_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match HCCL P2P "
                f"metadata token count {metadata.total_tokens}",
            )
        stage_idx = int(kwargs.get("ubatch_idx", metadata.stage_idx))
        stream = kwargs.get("_stream")
        state = context.states
        if not isinstance(state, HCCLP2PTransferState):
            raise RuntimeError(
                "HCCL P2P FFN output requires its matching receive state",
            )
        if state.stage_idx != stage_idx:
            raise ValueError(
                "HCCL P2P FFN output stage does not match receive state: "
                f"{stage_idx} != {state.stage_idx}",
            )
        if tuple(metadata.seq_lens) != state.seq_lens:
            raise ValueError(
                "HCCL P2P FFN output sequence lengths do not match receive state",
            )
        if len(state.peer_ranks) != len(state.seq_lens):
            raise RuntimeError("HCCL P2P receive state has an invalid peer layout")

        group = self._data_group(stage_idx)
        peer_slices = state.peer_slices or _make_peer_slices(
            state.peer_ranks,
            state.seq_lens,
        )
        for destination_rank, start, end in peer_slices:
            peer_slice = ffn_output[start:end]
            self._send_tensor(
                peer_slice,
                dst=destination_rank,
                group=group,
                stream=stream,
            )

    def recv_attn_output_streamed(
        self,
        *,
        ubatch_idx: int,
        recv_stream,
        wait_event,
        done_event,
        **kwargs: Any,
    ) -> tuple[AFDA2FTransferPayload, Any]:
        """Enqueue an FFN-side receive while retaining synchronous recv APIs."""
        # ``recv_attn_output`` is invoked from this helper while the FFN
        # capture stream is current.  Keep Graph U2 HCCL on that parent during
        # both warmup and capture; eager requests may use the dedicated recv
        # stream.  This avoids pending warmup work and unjoined A5 graph tasks
        # while preserving the compute-stream overlap below.
        if self._graph_parent_transport_active():
            parent_stream = torch.npu.current_stream()
            if wait_event is not None:
                wait_event.wait(parent_stream)
            payload = self.recv_attn_output(
                ubatch_idx=ubatch_idx,
                **kwargs,
            )
            self._record_stream(payload.hidden_states, parent_stream)
            if payload.input_ids is not None:
                self._record_stream(payload.input_ids, parent_stream)
            done_event.record(parent_stream)
            return payload, done_event
        with torch.npu.stream(recv_stream):
            if wait_event is not None:
                wait_event.wait(recv_stream)
            payload = self.recv_attn_output(
                ubatch_idx=ubatch_idx,
                _stream=recv_stream,
                **kwargs,
            )
            self._record_stream(payload.hidden_states, recv_stream)
            if payload.input_ids is not None:
                self._record_stream(payload.input_ids, recv_stream)
            done_event.record(recv_stream)
        return payload, done_event

    def send_ffn_output_streamed(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        *,
        ubatch_idx: int,
        send_stream,
        wait_event,
        done_event,
    ) -> Any:
        """Enqueue an FFN-side send while retaining synchronous send APIs."""
        # See ``recv_attn_output_streamed`` above.  The FFN compute event is
        # the only cross-stream dependency needed here; Graph U2 HCCL belongs
        # to the parent stream in both warmup and capture.
        if self._graph_parent_transport_active():
            parent_stream = torch.npu.current_stream()
            wait_event.wait(parent_stream)
            self._record_stream(ffn_output, parent_stream)
            self.send_ffn_output(
                ffn_output,
                context,
                ubatch_idx=ubatch_idx,
            )
            done_event.record(parent_stream)
            return done_event
        with torch.npu.stream(send_stream):
            wait_event.wait(send_stream)
            self._record_stream(ffn_output, send_stream)
            self.send_ffn_output(
                ffn_output,
                context,
                ubatch_idx=ubatch_idx,
                _stream=send_stream,
            )
            done_event.record(send_stream)
        return done_event

    def _attention_peer_slices(
        self,
        num_tokens: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if self.afd_config.role != "attention":
            raise RuntimeError("only Attention owns FFN peer slices")
        transport_tokens = (
            max(num_tokens, self.ratio) if self.attention_fans_out else num_tokens
        )
        split_sizes = _balanced_split_sizes(
            transport_tokens,
            len(self.mapping.ffn_peer_ranks),
        )
        return _make_peer_slices(self.mapping.ffn_peer_ranks, split_sizes)

    def _pad_attention_tensor_for_fanout(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling() and self.attention_fans_out:
            # vLLM reuses this graph without shape guards. Always retain enough
            # padding for a one-token batch: tracing a conditional pad at a
            # larger size would omit it. Peer slices trim the unused tail.
            padding_shape = (self.ratio - 1, *tensor.shape[1:])
            return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=0)
        peer_slices = self._attention_peer_slices(int(tensor.shape[0]))
        transport_tokens = peer_slices[-1][2]
        if transport_tokens == int(tensor.shape[0]):
            return tensor
        padding_shape = (transport_tokens - int(tensor.shape[0]), *tensor.shape[1:])
        return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=0)

    def _send_attention_tensor(
        self,
        tensor: torch.Tensor,
        *,
        group: ProcessGroup,
        stream=None,
    ) -> None:
        transport = self._pad_attention_tensor_for_fanout(tensor)
        peer_slices = self._attention_peer_slices(int(tensor.shape[0]))
        for destination_rank, start, end in peer_slices:
            shard = (
                tensor
                if len(peer_slices) == 1 and transport is tensor
                else transport[start:end]
            )
            kwargs = {} if stream is None else {"stream": stream}
            self._send_tensor(shard, dst=destination_rank, group=group, **kwargs)

    def _recv_attention_tensor(
        self,
        ref_tensor: torch.Tensor,
        *,
        group: ProcessGroup,
        stream=None,
    ) -> torch.Tensor:
        peer_slices = self._attention_peer_slices(int(ref_tensor.shape[0]))
        transport_tokens = peer_slices[-1][2]
        if torch.compiler.is_compiling() and self.attention_fans_out:
            # The receive graph must also retain capacity for padding when
            # reused for a batch smaller than the FFN peer count.
            transport = ref_tensor.new_empty(
                (ref_tensor.shape[0] + self.ratio - 1, *ref_tensor.shape[1:]),
            )
        else:
            transport = (
                ref_tensor
                if transport_tokens == int(ref_tensor.shape[0])
                else ref_tensor.new_empty((transport_tokens, *ref_tensor.shape[1:]))
            )
        for source_rank, start, end in peer_slices:
            shard = (
                ref_tensor
                if len(peer_slices) == 1 and transport is ref_tensor
                else transport[start:end]
            )
            kwargs = {} if stream is None else {"stream": stream}
            self._recv_tensor(shard, src=source_rank, group=group, **kwargs)
        if transport is not ref_tensor:
            ref_tensor.copy_(transport[: ref_tensor.shape[0]])
        return ref_tensor

    def send_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        ubatch_idx: int,
    ) -> None:
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        flat_ids = input_ids.reshape(-1)
        num_tokens = int(flat_ids.numel())
        self._validate_input_ids(flat_ids, num_tokens)
        peer_slices = self._attention_peer_slices(num_tokens)
        transport_tokens = peer_slices[-1][2]
        buffer[:num_tokens].copy_(flat_ids, non_blocking=False)
        if transport_tokens > num_tokens:
            buffer[num_tokens:transport_tokens].zero_()
        for destination_rank, start, end in peer_slices:
            dist.send(buffer[start:end], dst=destination_rank, group=group)

    def recv_input_ids(
        self,
        num_tokens: int,
        *,
        ubatch_idx: int,
    ) -> torch.Tensor:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        layout = self._stage_layout(
            ubatch_idx,
            fallback=num_tokens,
        )
        if layout.num_tokens != num_tokens:
            raise ValueError(
                "DSV4 input_ids token count does not match HCCL P2P peer layout: "
                f"{num_tokens} != {layout.num_tokens}",
            )
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        input_ids = buffer[:num_tokens]
        for source_rank, start, end in layout.peer_slices:
            peer_slice = input_ids[start:end]
            dist.recv(peer_slice, src=source_rank, group=group)
        return input_ids

    def send_mtp_header(
        self,
        *,
        num_tokens: int,
        speculative_step: int,
        num_tokens_across_dp: torch.Tensor,
        stage_idx: int,
    ) -> None:
        if self.mtp_draft_graph_enabled:
            self._send_prepared_mtp_graph_header(
                num_tokens=num_tokens,
                speculative_step=speculative_step,
                stage_idx=stage_idx,
            )
            return
        counts = self._mtp_ffn_token_counts(num_tokens_across_dp)
        peer_slices = self._attention_peer_slices(num_tokens)
        for destination_rank, start, end in peer_slices:
            peer_tokens = end - start
            self._validate_mtp_header_values(
                num_tokens=peer_tokens,
                speculative_step=speculative_step,
                num_tokens_across_dp=counts,
            )
            buffer, group = self._mtp_header_buffer_for_peer(
                stage_idx,
                destination_rank,
            )
            peer_counts = counts.reshape(-1).to(
                device=buffer.device,
                dtype=torch.int32,
            )
            buffer[0] = _MTP_HEADER_MAGIC
            buffer[1] = speculative_step
            buffer[2] = peer_tokens
            buffer[3] = self.ffn_size
            buffer[_MTP_HEADER_PREFIX_SIZE:].copy_(peer_counts, non_blocking=False)
            self._send_tensor(buffer, dst=destination_rank, group=group)

    def prepare_mtp_header_for_graph(
        self,
        dp_metadata_list: Mapping[int, DPMetadata | AFDDPMetadata],
    ) -> None:
        """Prepare the draft header before entering the Attention ACL Graph."""

        if self.afd_config.role != "attention" or not self.mtp_draft_graph_enabled:
            return
        self._validate_mtp_topology()
        attention_peer_counts = [0] * self.attn_size
        for metadata in dp_metadata_list.values():
            stage_counts = _attention_token_counts(
                metadata,
                attention_size=self.attn_size,
                fallback=1,
            )
            for peer_index, count in enumerate(stage_counts):
                attention_peer_counts[peer_index] += int(count)

        ffn_counts = _project_attention_counts_to_ffn(
            attention_peer_counts,
            ffn_size=self.ffn_size,
        )
        local_tokens = attention_peer_counts[self.mapping.role_rank]
        self._validate_receive_capacity(local_tokens)
        peer_slices = self._attention_peer_slices(local_tokens)
        header_values_by_peer = {
            peer_rank: (
                _MTP_HEADER_MAGIC,
                0,
                end - start,
                self.ffn_size,
                *ffn_counts,
            )
            for peer_rank, start, end in peer_slices
        }
        if header_values_by_peer == self.mtp_graph_header_values_by_peer:
            return

        # ### PATCH START: prepare host values outside the draft graph.
        # Upstream torch-npu 2.10.0.post2 rejects synchronous host-to-device
        # copies while an NPU stream is being captured.  The scheduler control
        # payload owns this exact layout, so populate the stable header buffer
        # before the model enters ACLGraphWrapper.  Synchronize only when a
        # live layout changes and could otherwise overwrite a prior replay.
        if (
            self.mtp_graph_header_values is not None
            and not self.is_graph_capturing
            and not self.is_warmup
        ):
            torch.npu.current_stream().synchronize()
        for peer_rank, header_values in header_values_by_peer.items():
            buffer, _ = self._mtp_header_buffer_for_peer(0, peer_rank)
            buffer.copy_(
                torch.tensor(
                    header_values,
                    dtype=torch.int32,
                    device=buffer.device,
                ),
                non_blocking=False,
            )
        self.mtp_graph_header_values_by_peer = header_values_by_peer
        self.mtp_graph_header_values = header_values_by_peer[
            self.mapping.ffn_peer_ranks[0]
        ]
        # ### PATCH END: prepare host values outside the draft graph.

    def _send_prepared_mtp_graph_header(
        self,
        *,
        num_tokens: int,
        speculative_step: int,
        stage_idx: int,
    ) -> None:
        header_values_by_peer = self.mtp_graph_header_values_by_peer
        if not header_values_by_peer:
            raise RuntimeError(
                "DSV4 MTP draft Graph header was not prepared by the control plane",
            )
        if stage_idx != 0:
            raise RuntimeError("DSV4 MTP draft Graph uses merged stage 0")
        if not 0 <= speculative_step < self.num_speculative_tokens:
            raise RuntimeError(
                "DSV4 MTP draft Graph speculative step must be in "
                f"[0, {self.num_speculative_tokens}), got {speculative_step}"
            )
        # The merged MTP graph reuses this header buffer across its N draft
        # iterations. The fixed header carries step 0; live ordering is
        # enforced by one control-plane phase marker per merged replay.
        peer_slices = self._attention_peer_slices(num_tokens)
        if not torch.compiler.is_compiling() and any(
            header_values_by_peer[peer_rank][2] != end - start
            for peer_rank, start, end in peer_slices
        ):
            raise RuntimeError(
                "DSV4 MTP draft Graph local token count does not match its "
                "prepared fan-out headers",
            )
        for peer_rank, _start, _end in peer_slices:
            buffer, group = self._mtp_header_buffer_for_peer(stage_idx, peer_rank)
            self._send_tensor(buffer, dst=peer_rank, group=group)

    def recv_mtp_header(self, *, stage_idx: int) -> HCCLMTPHeader:
        if self.afd_config.role != "ffn":
            raise RuntimeError("only the FFN role receives MTP headers")
        self._validate_mtp_topology()
        if stage_idx in self.mtp_stage_layouts:
            raise RuntimeError(
                f"DSV4 MTP stage has an unconsumed header layout: stage={stage_idx}"
            )
        buffer, group = self._mtp_header_buffer_and_group(stage_idx)
        peer_ranks = self._attention_peer_world_ranks()
        seq_lens: list[int] = []
        speculative_step: int | None = None
        expected_counts: tuple[int, ...] | None = None
        for source_rank in peer_ranks:
            dist.recv(buffer, src=source_rank, group=group)
            values = [int(value) for value in buffer.cpu().tolist()]
            if values[0] != _MTP_HEADER_MAGIC:
                raise RuntimeError(f"invalid DSV4 MTP HCCL header magic: {values[0]}")
            if values[3] != self.ffn_size:
                raise RuntimeError(
                    "DSV4 MTP HCCL header DP size does not match FFN world: "
                    f"{values[3]} != {self.ffn_size}"
                )
            counts_tuple = tuple(values[_MTP_HEADER_PREFIX_SIZE:])
            counts = torch.tensor(counts_tuple, dtype=torch.int32)
            self._validate_mtp_header_values(
                num_tokens=values[2],
                speculative_step=values[1],
                num_tokens_across_dp=counts,
            )
            if speculative_step is None:
                speculative_step = values[1]
                expected_counts = counts_tuple
            elif values[1] != speculative_step or counts_tuple != expected_counts:
                raise RuntimeError(
                    "DSV4 MTP peer headers disagree on speculative step or "
                    "FFN token counts"
                )
            seq_lens.append(values[2])

        assert speculative_step is not None and expected_counts is not None
        num_tokens = sum(seq_lens)
        self._validate_receive_capacity(num_tokens)
        expected_local_tokens = expected_counts[self.mapping.role_rank]
        if num_tokens != expected_local_tokens:
            raise RuntimeError(
                "DSV4 MTP peer token total does not match the FFN count vector: "
                f"{num_tokens} != {expected_local_tokens}"
            )
        layout = HCCLP2PStageLayout(
            peer_ranks=peer_ranks,
            seq_lens=tuple(seq_lens),
            peer_slices=_make_peer_slices(peer_ranks, seq_lens),
            num_tokens=num_tokens,
        )
        self.mtp_stage_layouts[stage_idx] = layout
        counts = torch.tensor(expected_counts, dtype=torch.int32)
        return HCCLMTPHeader(
            num_tokens=num_tokens,
            speculative_step=speculative_step,
            num_tokens_across_dp=counts,
        )

    def recv_mtp_header_for_graph(
        self,
        *,
        stage_idx: int,
        attention_peer_counts: tuple[int, ...],
    ) -> HCCLMTPHeader:
        """Capture MTP header receives using a scheduler-owned static layout.

        Header values cannot be copied to CPU and parsed inside an NPUGraph.
        The same padded token layout is already carried by the target decoder's
        DP metadata, so capture consumes the wire messages while using that
        metadata as the graph contract.  Live replay is isolated by the exact
        peer-layout key and therefore does not trust values from another shape.
        """

        if self.afd_config.role != "ffn":
            raise RuntimeError("only the FFN role receives MTP headers")
        self._validate_mtp_topology()
        if len(attention_peer_counts) != self.attn_size:
            raise ValueError(
                "DSV4 MTP graph layout must match Attention world size: "
                f"{len(attention_peer_counts)} != {self.attn_size}",
            )
        if stage_idx in self.mtp_stage_layouts:
            raise RuntimeError(
                f"DSV4 MTP stage has an unconsumed header layout: stage={stage_idx}"
            )

        attention_role_ranks = tuple(
            rank - self.ffn_size for rank in self.mapping.attention_peer_ranks
        )
        if self.attention_fans_out:
            attention_tokens = int(attention_peer_counts[attention_role_ranks[0]])
            fanout_index = self.mapping.role_rank % self.ratio
            seq_lens = (
                _balanced_split_sizes(
                    max(attention_tokens, self.ratio),
                    self.ratio,
                )[fanout_index],
            )
        else:
            seq_lens = tuple(
                int(attention_peer_counts[role_rank])
                for role_rank in attention_role_ranks
            )
        peer_ranks = self._attention_peer_world_ranks()
        layout = HCCLP2PStageLayout(
            peer_ranks=peer_ranks,
            seq_lens=seq_lens,
            peer_slices=_make_peer_slices(peer_ranks, seq_lens),
            num_tokens=sum(seq_lens),
        )
        self._validate_receive_capacity(layout.num_tokens)

        buffer, group = self._mtp_header_buffer_and_group(stage_idx)
        for source_rank in peer_ranks:
            self._recv_tensor(buffer, src=source_rank, group=group)
        self.mtp_stage_layouts[stage_idx] = layout

        ffn_counts = _project_attention_counts_to_ffn(
            attention_peer_counts,
            ffn_size=self.ffn_size,
        )
        return HCCLMTPHeader(
            num_tokens=layout.num_tokens,
            speculative_step=0,
            num_tokens_across_dp=torch.tensor(ffn_counts, dtype=torch.int32),
        )

    def prepare_stage_buffer(self, stage_idx: int, num_tokens: int) -> None:
        """Ensure the FFN receive buffer is allocated before posting recv."""
        if self.afd_config.role != "ffn":
            return
        self._hidden_recv_buffer(stage_idx, num_tokens)

    def _send_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dst: int,
        group: ProcessGroup,
        stream=None,
    ) -> None:
        # Dynamo marks the MTP token dimension dynamic. Querying contiguity in
        # Python specializes that dimension during full draft compilation even
        # though the model-owned buffers are already contiguous.
        send_tensor = (
            tensor
            if torch.compiler.is_compiling() or tensor.is_contiguous()
            else tensor.contiguous()
        )
        if stream is not None:
            self._record_stream(send_tensor, stream)
        if self._graph_transport_active():
            _graph_hccl_send(send_tensor, dst=dst, group=group)
            return
        dist.send(send_tensor, dst=dst, group=group)

    def _recv_tensor(
        self,
        tensor: torch.Tensor,
        *,
        src: int,
        group: ProcessGroup,
        stream=None,
    ) -> None:
        if stream is not None:
            self._record_stream(tensor, stream)
        if self._graph_transport_active():
            _graph_hccl_recv(tensor, src=src, group=group)
            return
        dist.recv(tensor, src=src, group=group)

    def _graph_transport_active(self) -> bool:
        """Use graph-visible HCCL ops while tracing or capturing a target graph."""
        if torch.compiler.is_compiling() or (
            self.is_graph_capturing and not self.is_warmup
        ):
            return True
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(
            getattr(forward_context, "afd_graph_ubatching", False)
            and torch.npu.is_current_stream_capturing()
        )

    def _graph_parent_transport_active(self) -> bool:
        """Keep Graph U2 warmup and capture communication on the parent stream.

        Warmup intentionally uses regular ``dist.send``/``dist.recv`` so the
        first graph capture starts with a quiescent communicator.  It still
        follows the Graph U2 stream schedule, so collectives must not be
        launched on the compute/send/receive side streams.
        """
        return bool(self.is_warmup or self._graph_transport_active())

    @staticmethod
    def _record_stream(tensor: torch.Tensor, stream) -> None:
        if tensor.device.type == "npu":
            tensor.record_stream(stream)

    def _hidden_recv_buffer(
        self,
        stage_idx: int,
        num_tokens: int,
    ) -> torch.Tensor:
        self._validate_receive_capacity(num_tokens)
        buffer = self.hidden_recv_buffers.get(stage_idx)
        if (
            buffer is None
            or int(buffer.shape[0]) < num_tokens
            or int(buffer.shape[1]) != self.hidden_size
            or buffer.dtype != self.dtype
        ):
            buffer = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=f"npu:{self.local_rank}",
            )
            self.hidden_recv_buffers[stage_idx] = buffer
        return buffer[:num_tokens]

    def _mtp_hidden_recv_buffer(
        self,
        stage_idx: int,
        num_tokens: int,
    ) -> torch.Tensor:
        self._validate_receive_capacity(num_tokens)
        buffer = self.mtp_hidden_recv_buffers.get(stage_idx)
        if (
            buffer is None
            or int(buffer.shape[0]) < num_tokens
            or tuple(buffer.shape[1:]) != (self.hidden_size,)
            or buffer.dtype != self.dtype
        ):
            buffer = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=f"npu:{self.local_rank}",
            )
            self.mtp_hidden_recv_buffers[stage_idx] = buffer
        return buffer[:num_tokens]

    def _peer_token_counts_for_stage(
        self,
        stage_idx: int,
        *,
        fallback: int,
    ) -> list[int]:
        return list(self._stage_layout(stage_idx, fallback=fallback).seq_lens)

    def _stage_layout(
        self,
        stage_idx: int,
        *,
        fallback: int,
    ) -> HCCLP2PStageLayout:
        cached = self.stage_layouts.get(stage_idx)
        if cached is not None:
            return cached

        dp_metadata = self.dp_metadata_list.get(stage_idx)
        if dp_metadata is None:
            if self.afd_config.role == "ffn" and self.attn_size != self.ffn_size:
                raise RuntimeError(
                    "HCCL P2P FFN requires DP metadata for unequal topology",
                )
            seq_lens = (max(1, fallback),)
        else:
            attention_counts = _attention_token_counts(
                dp_metadata,
                attention_size=self.attn_size,
                fallback=fallback,
            )
            attention_role_ranks = tuple(
                rank - self.ffn_size for rank in self.mapping.attention_peer_ranks
            )
            if self.attention_fans_out:
                attention_tokens = int(attention_counts[attention_role_ranks[0]])
                fanout_index = self.mapping.role_rank % self.ratio
                seq_lens = (
                    _balanced_split_sizes(
                        max(attention_tokens, self.ratio),
                        self.ratio,
                    )[fanout_index],
                )
            else:
                seq_lens = tuple(
                    max(
                        1,
                        int(
                            attention_counts[role_rank]
                            if role_rank < len(attention_counts)
                            else fallback
                        ),
                    )
                    for role_rank in attention_role_ranks
                )

        peer_ranks = self._attention_peer_world_ranks()
        layout = HCCLP2PStageLayout(
            peer_ranks=peer_ranks,
            seq_lens=seq_lens,
            peer_slices=_make_peer_slices(peer_ranks, seq_lens),
            num_tokens=sum(seq_lens),
        )
        if dp_metadata is not None:
            self.stage_layouts[stage_idx] = layout
        return layout

    def _attention_peer_world_ranks(self) -> tuple[int, ...]:
        if self.afd_config.role != "ffn":
            raise RuntimeError("only an FFN rank owns Attention peer ranks")
        return self.mapping.attention_peer_ranks

    def _validate_receive_capacity(self, num_tokens: int) -> None:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "HCCL P2P aggregate token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}; "
                "increase FFN max_num_batched_tokens for this A/F ratio",
            )

    def _validate_input_ids(
        self,
        flat_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        if torch.compiler.is_compiling():
            return
        if flat_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "DSV4 input_ids must use int32 or int64 before HCCL transfer; "
                f"got {flat_ids.dtype}",
            )
        # NPU IDs originate from vLLM's validated scheduler/tokenizer path.
        # Reading min/max back to the host here adds two device-wide syncs to
        # every decode step, so retain the value-domain check for CPU/Mock
        # boundaries and keep the production data path device-local.
        if flat_ids.device.type != "cpu":
            return
        min_id = int(flat_ids.min().item())
        max_id = int(flat_ids.max().item())
        if min_id < -1 or max_id >= self.vocab_size:
            raise ValueError(
                "DSV4 input_ids must contain -1 padding or IDs in "
                f"[0, {self.vocab_size}); got [{min_id}, {max_id}]",
            )

    def _data_group(self, stage_idx: int) -> ProcessGroup:
        if stage_idx < 0 or stage_idx >= len(self.data_pg_list):
            raise RuntimeError(
                f"HCCL P2P data stage {stage_idx} is not initialized",
            )
        return self.data_pg_list[stage_idx]

    def _ids_buffer_and_group(
        self,
        stage_idx: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        if stage_idx < 0 or stage_idx >= len(self.ids_pg_list):
            raise RuntimeError(
                f"HCCL P2P input IDs stage {stage_idx} is not initialized",
            )
        return self.input_ids_buffers[stage_idx], self.ids_pg_list[stage_idx]

    def _mtp_header_buffer_and_group(
        self,
        stage_idx: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        if stage_idx < 0 or stage_idx >= len(self.mtp_header_buffers):
            raise RuntimeError(
                f"HCCL P2P MTP header stage {stage_idx} is not initialized",
            )
        return self.mtp_header_buffers[stage_idx], self.ids_pg_list[stage_idx]

    def _mtp_header_buffer_for_peer(
        self,
        stage_idx: int,
        peer_rank: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        _, group = self._mtp_header_buffer_and_group(stage_idx)
        if stage_idx < len(self.mtp_header_buffers_by_peer):
            peer_buffers = self.mtp_header_buffers_by_peer[stage_idx]
            if peer_rank in peer_buffers:
                return peer_buffers[peer_rank], group
        if len(self.mapping.ffn_peer_ranks) == 1:
            return self.mtp_header_buffers[stage_idx], group
        raise RuntimeError(
            "HCCL P2P MTP fan-out header buffers are not initialized for "
            f"stage={stage_idx} peer={peer_rank}"
        )

    def _validate_mtp_scope(self, metadata: AFDTransferMetadata) -> None:
        if metadata.phase != "mtp":
            raise ValueError("MTP scope validation requires phase=mtp")
        if not self.requires_mtp:
            raise RuntimeError("HCCL P2P MTP transfer is not enabled")
        self._validate_mtp_topology()
        if metadata.layer_idx != 0 or not (
            0 <= metadata.speculative_step < self.num_speculative_tokens
        ):
            raise RuntimeError(
                "DSV4 HCCL P2P MTP requires layer 0 and speculative step in "
                f"[0, {self.num_speculative_tokens})"
            )
        if metadata.stage_idx != 0:
            raise RuntimeError("DSV4 HCCL P2P MTP uses the merged proposal on stage 0")
        self._validate_receive_capacity(metadata.total_tokens)

    def _validate_mtp_topology(self) -> None:
        if len(self.data_pg_list) not in (1, 2):
            raise RuntimeError("DSV4 HCCL P2P MTP supports target decoder U1 or U2")

    def _mtp_ffn_token_counts(
        self,
        num_tokens_across_dp: torch.Tensor,
    ) -> torch.Tensor:
        """Project Attention DP token counts onto the FFN rank topology."""
        flat_counts = num_tokens_across_dp.reshape(-1)
        count_size = int(flat_counts.numel())
        if (
            count_size < self.attn_size
            and count_size * self.tensor_parallel_size == self.attn_size
        ):
            flat_counts = flat_counts.repeat_interleave(self.tensor_parallel_size)
            count_size = int(flat_counts.numel())
        if count_size != self.attn_size:
            raise ValueError(
                "DSV4 MTP token-count vector must match Attention world size: "
                f"{count_size} != {self.attn_size}"
            )
        if not torch.compiler.is_compiling():
            counts = [int(value) for value in flat_counts.tolist()]
            if any(value < 0 for value in counts):
                raise ValueError("DSV4 MTP token counts cannot be negative")
        if self.ratio == 1:
            return flat_counts
        if not self.attention_fans_out:
            return flat_counts.reshape(self.ffn_size, self.ratio).sum(dim=1)
        transport_counts = torch.clamp(flat_counts, min=self.ratio)
        base_counts = torch.div(
            transport_counts,
            self.ratio,
            rounding_mode="floor",
        )
        remainders = transport_counts.remainder(self.ratio)
        offsets = torch.arange(
            self.ratio,
            dtype=base_counts.dtype,
            device=base_counts.device,
        )
        return (base_counts[:, None] + (offsets < remainders[:, None])).reshape(-1)

    def _validate_mtp_header_values(
        self,
        *,
        num_tokens: int,
        speculative_step: int,
        num_tokens_across_dp: torch.Tensor,
    ) -> None:
        self._validate_mtp_topology()
        self._validate_receive_capacity(num_tokens)
        if not 0 <= speculative_step < self.num_speculative_tokens:
            raise RuntimeError(
                "DSV4 HCCL P2P MTP speculative step must be in "
                f"[0, {self.num_speculative_tokens}), got {speculative_step}"
            )
        count_size = int(num_tokens_across_dp.numel())
        if count_size != self.ffn_size:
            raise ValueError(
                "DSV4 MTP token-count vector must match FFN world size: "
                f"{count_size} != {self.ffn_size}"
            )
        # ### PATCH START: keep eager MTP traceable beside a target ACL graph.
        # torch.compile traces the draft model even when speculative-config
        # enforce_eager disables draft ACL graph capture. Tensor-to-list value
        # checks are data dependent and cannot be part of that full graph; the
        # scheduler-owned DP counts are validated on every non-compiled entry.
        if torch.compiler.is_compiling():
            return
        # ### PATCH END: keep eager MTP traceable beside a target ACL graph.
        counts = [int(value) for value in num_tokens_across_dp.reshape(-1).tolist()]
        if any(value < 0 for value in counts):
            raise ValueError("DSV4 MTP token counts cannot be negative")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("HCCL P2P connector is not initialized")


class P2pHcclAFDControlPlane(AFDControlPlane):
    """Stage metadata transport for ``P2pHcclAFDConnector``."""

    def __init__(self, connector: P2pHcclAFDConnector) -> None:
        self.connector = connector
        self._pending_payload: AFDControlPayload | None = None

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if int(payload.tensor_parallel_size) != connector.tensor_parallel_size:
            raise RuntimeError(
                "DeepSeek-V4 AFD requires matching Attention/FFN tensor "
                "parallel sizes: "
                f"attention={int(payload.tensor_parallel_size)}, "
                f"ffn={connector.tensor_parallel_size}"
            )
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.stage_layouts = {}
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup
        if connector.afd_config.role == "attention":
            connector.prepare_mtp_header_for_graph(payload.dp_metadata_list)
            return
        for stage_idx in payload.dp_metadata_list:
            layout = connector._stage_layout(int(stage_idx), fallback=1)
            connector.prepare_stage_buffer(int(stage_idx), layout.num_tokens)

    def reset_pending_payload(self) -> None:
        self._pending_payload = None

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        self._send_payload(payload)

    def send_mtp_phase_ready(self, *, graph_replay: bool) -> None:
        """Tell FFN that the target entered draft and how it will execute."""
        self._send_payload(
            AFDControlPayload(
                dp_metadata_list={},
                is_graph_capturing=False,
                is_warmup=False,
                mtp_phase_ready=True,
                mtp_phase_graph_replay=bool(graph_replay),
            ),
        )

    def _send_payload(self, payload: AFDControlPayload) -> None:
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if (
            connector.afd_config.role != "attention"
            or connector.world_rank != connector.mapping.attention_peer_ranks[0]
        ):
            return
        for destination_rank in connector.mapping.ffn_peer_ranks:
            send_control_payload(
                payload,
                dst=destination_rank,
                group=connector.p2p_pg,
                device=torch.device("cpu"),
            )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        if self._pending_payload is not None:
            payload = self._pending_payload
            self._pending_payload = None
        else:
            payload = self._recv_payload()
        if payload.mtp_phase_ready:
            raise RuntimeError(
                "unexpected DSV4 MTP phase marker before target metadata",
            )
        self.connector.mtp_phase_control_enabled = bool(
            payload.mtp_phase_control_enabled,
        )
        self.connector.target_graph_replay = payload.target_graph_replay
        self.connector.mtp_phase_graph_replay = False
        return payload

    def recv_mtp_phase_ready(self) -> bool:
        """Consume a draft marker or retain the next target/shutdown payload."""
        if self._pending_payload is not None:
            raise RuntimeError("HCCL P2P control plane already has a pending payload")
        payload = self._recv_payload()
        if payload.mtp_phase_ready:
            self.connector.mtp_phase_graph_replay = bool(
                payload.mtp_phase_graph_replay,
            )
            return True
        self.connector.mtp_phase_graph_replay = False
        self._pending_payload = payload
        return False

    def _recv_payload(self) -> AFDControlPayload:
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError(
                "HCCL P2P DP metadata process group is not initialized",
            )
        source_rank = connector.mapping.attention_peer_ranks[0]
        return recv_control_payload(
            src=source_rank,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )


def _balanced_split_sizes(num_tokens: int, num_peers: int) -> tuple[int, ...]:
    if num_tokens < 0:
        raise ValueError(f"token count must be non-negative, got {num_tokens}")
    if num_peers <= 0:
        raise ValueError(f"peer count must be positive, got {num_peers}")
    # Keep the split branch-free. Python comparisons against the remainder
    # specialize the traced parity, but vLLM reuses that graph without guards.
    return tuple(
        (num_tokens + num_peers - 1 - offset) // num_peers
        for offset in range(num_peers)
    )


def _project_attention_counts_to_ffn(
    attention_counts: Sequence[int],
    *,
    ffn_size: int,
) -> list[int]:
    attention_size = len(attention_counts)
    if attention_size >= ffn_size:
        ratio = attention_size // ffn_size
        return [
            sum(attention_counts[index * ratio : (index + 1) * ratio])
            for index in range(ffn_size)
        ]
    ratio = ffn_size // attention_size
    ffn_counts: list[int] = []
    for count in attention_counts:
        ffn_counts.extend(_balanced_split_sizes(max(int(count), ratio), ratio))
    return ffn_counts


def _num_tokens_for_attention_rank(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_rank: int,
    attention_size: int,
    fallback: int = 1,
) -> int:
    counts = _attention_token_counts(
        dp_metadata,
        attention_size=attention_size,
        fallback=fallback,
    )
    if 0 <= attention_rank < len(counts):
        return counts[attention_rank]
    return max(1, fallback)


def _attention_token_counts(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_size: int,
    fallback: int,
) -> list[int]:
    counts = dp_metadata.num_tokens_across_dp_cpu.flatten().tolist()
    if not counts:
        return [max(1, fallback)] * attention_size
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[index // tp_size] for index in range(attention_size)]
    return [max(1, int(count)) for count in counts]


def _make_peer_slices(
    peer_ranks: tuple[int, ...],
    seq_lens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    offset = 0
    slices = []
    for peer_rank, peer_tokens in zip(peer_ranks, seq_lens, strict=True):
        end = offset + peer_tokens
        slices.append((peer_rank, offset, end))
        offset = end
    return tuple(slices)


__all__ = [
    "HCCLMTPHeader",
    "HCCLP2PTransferState",
    "P2pHcclAFDConnector",
    "P2pHcclAFDControlPlane",
]
