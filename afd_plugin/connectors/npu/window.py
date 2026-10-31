# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Window-based AFD connector initialization for Ascend NPU.

The connector owns the communication resources and the Window A2F/F2A data
path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cann_ops_transformer as cot
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import coerce_extra_int, coerce_extra_positive_int
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferMetadata,
    AFDTransferState,
    AFDTransferContext,
)
from afd_plugin.distributed import (
    build_window_expert_layout,
    build_window_rank_mapping,
    init_afd_process_group,
)

logger = init_logger(__name__)

_COMM_CONTEXT_WINDOW_ALIGNMENT = 2 * 1024 * 1024


def _record_npu_stream(tensor: torch.Tensor, stream: Any) -> None:
    if tensor.device.type == "npu":
        tensor.record_stream(stream)


@dataclass(frozen=True, slots=True)
class WindowAttentionPipelineEvents:
    """One Attention layer/stage's compute, A2F, and Combine events."""

    ready: Any
    compute_done: Any
    send_ready: Any
    send_done: Any
    recv_done: Any


@dataclass(frozen=True, slots=True)
class WindowAttentionReceiveDependency:
    """A deferred Combine result consumed by the next Attention layer."""

    tensor: torch.Tensor
    event: Any


@dataclass(slots=True)
class WindowAFDTransferState(AFDTransferState):
    """Operator-produced routing metadata for one A2F exchange."""

    expert_scales: torch.Tensor
    group_list: torch.Tensor | None = None
    dynamic_scale: torch.Tensor | None = None
    session_ids: torch.Tensor | None = None
    micro_batch_ids: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    actual_token_num: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class WindowAFDExtraInfo(ConnectorExtraInfo):
    """Window protocol options.

    One or two Window micro-batch slots are supported. Scheduling is lock-step
    or independent per Attention session according to ``async_dp``.
    """

    micro_batch_num: int = 1
    quant_mode: int = 4

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                "WindowAFDConnector connector_extra_config must be a mapping",
            )
        allowed = {"micro_batch_num", "quant_mode"}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "unknown WindowAFDConnector connector_extra_config field(s): "
                + ", ".join(unknown),
            )
        quant_mode = coerce_extra_int(
            raw.get("quant_mode", 4),
            field_name="quant_mode",
        )
        if quant_mode not in (0, 2, 4):
            raise ValueError(
                "WindowAFDConnector quant_mode must be 0, 2, or 4, "
                f"got {quant_mode}",
            )
        return cls(
            micro_batch_num=coerce_extra_positive_int(
                raw.get("micro_batch_num", 1),
                field_name="micro_batch_num",
            ),
            quant_mode=quant_mode,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "micro_batch_num": self.micro_batch_num,
            "quant_mode": self.quant_mode,
        }


def _align_up(value: int, alignment: int = 512) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _window_sizes(
    *,
    attention_size: int,
    micro_batch_num: int,
    micro_batch_size: int,
    selected_expert_num: int,
    hidden_size: int,
    quant_mode: int,
) -> tuple[int, int, int, int]:
    """Return Window sizes and per-token byte strides for both directions.

    The result is ``(attn_window_bytes, ffn_window_bytes,
    a2f_token_bytes, f2a_token_bytes)``.  Window sizes include both metadata
    and token data, with the final capacity aligned to 2 MiB for the external
    CommContext Window.  The formulas mirror ref/local_window_utils.py and are
    evaluated once from the configured maximum batch capacity.
    """

    if quant_mode == 2:
        # H INT8 bytes plus one FP32 scale, padded to a 512-byte record.
        a2f_token_bytes = _align_up(hidden_size + 4, 512)
    elif quant_mode == 0:
        # H fp16/bfloat16 values, each occupying two bytes.
        a2f_token_bytes = hidden_size * 2
    elif quant_mode == 4:
        # MXFP8 stores one FP8 E4M3FN byte per value followed by one E8M0
        # scale byte per 32 values. The scale region is two-byte aligned.
        quant_data_bytes = _align_up(hidden_size, 256)
        scale_bytes = _align_up((hidden_size + 31) // 32, 2)
        # ScheduleContext requires the source-row stride to be 512-byte
        # aligned. The operator writes the payload first and leaves row-tail
        # padding unused.
        a2f_token_bytes = _align_up(quant_data_bytes + scale_bytes, 512)
    else:
        raise ValueError(f"unsupported Window quant_mode={quant_mode}")
    # F2A always returns H fp16/bfloat16 values.
    f2a_token_bytes = hidden_size * 2

    attention_window_info_bytes = _align_up(
        4 * selected_expert_num * micro_batch_size * micro_batch_num,
    )
    attention_window_data_bytes = (
        f2a_token_bytes
        * selected_expert_num
        * micro_batch_size
        * micro_batch_num
    )

    ffn_window_info_bytes = _align_up(
        4
        * (selected_expert_num * micro_batch_size + 2)
        * micro_batch_num
        * attention_size,
    )
    ffn_window_data_bytes = (
        a2f_token_bytes
        * selected_expert_num
        * micro_batch_size
        * micro_batch_num
        * attention_size
    )
    # The external Window passed to CommContextManager must satisfy the same
    # 2 MiB capacity alignment used by the operators' ccl_buffer_size helpers.
    attention_window_bytes = _align_up(
        attention_window_info_bytes + attention_window_data_bytes,
        _COMM_CONTEXT_WINDOW_ALIGNMENT,
    )
    ffn_window_bytes = _align_up(
        ffn_window_info_bytes + ffn_window_data_bytes,
        _COMM_CONTEXT_WINDOW_ALIGNMENT,
    )
    return (
        attention_window_bytes,
        ffn_window_bytes,
        a2f_token_bytes,
        f2a_token_bytes,
    )


class WindowAFDConnector(AFDConnectorBase):
    """Create the M2N communication context, Window, and schedule context."""

    yield_after_attn_send = True
    supports_connector_driven_loop = True
    is_window_connector = True
    requires_lockstep_dp_sync = True

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        return WindowAFDExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: Any,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.mapping = build_window_rank_mapping(afd_config, role_rank)
        self.world_rank = self.mapping.world_rank
        self.world_size = self.mapping.world_size
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.async_mode = bool(afd_config.async_dp)
        self.requires_lockstep_dp_sync = not self.async_mode
        self.process_group: ProcessGroup | None = None
        self.hccl_comm_name: str | None = None
        self.window_tensor: torch.Tensor | None = None
        self.comm_buffer: Any | None = None
        self.attn_window_size = 0
        self.ffn_window_size = 0
        self.window_size = 0
        self.window_addr = 0
        self.context_holder: Any | None = None
        self.schedule_context: torch.Tensor | None = None
        self.expert_rank_table: torch.Tensor | None = None
        self.attn_rank_table: torch.Tensor | None = None
        self.local_expert_num = 0
        self._pending_transfers: dict[tuple[int, int], AFDTransferContext] = {}
        self.a2f_send_stream = None
        self.f2a_recv_stream = None
        self.attention_graph_compute_stream = None
        self.attention_pipeline_events: dict[
            tuple[int, int], WindowAttentionPipelineEvents
        ] = {}
        self.attention_receive_dependencies: dict[
            int, WindowAttentionReceiveDependency
        ] = {}
        self.attention_session_id: torch.Tensor | None = None
        self.attention_micro_batch_ids: torch.Tensor | None = None
        self.attention_layer_ids: torch.Tensor | None = None
        self.attention_sync_layer_id: torch.Tensor | None = None
        self._initialized = False

        hf_config = vllm_config.model_config.hf_config
        self.num_layers = int(hf_config.num_hidden_layers)
        self.hidden_size = int(hf_config.hidden_size)
        routed_topk = int(hf_config.num_experts_per_tok)
        shared_expert_num = int(hf_config.n_shared_experts)
        self.shared_expert_num = shared_expert_num
        self.routed_expert_num = int(hf_config.n_routed_experts)
        self.selected_expert_num = routed_topk + shared_expert_num
        self.expert_num = self.routed_expert_num + shared_expert_num
        self.micro_batch_num = self.extra_info.micro_batch_num
        self.stream_overlap_enabled = bool(
            self.async_mode and self.micro_batch_num > 1
        )
        # M is an independent Window slot dimension, not a divisor of BS.
        # Keep the operator's per-slot capacity equal to scheduler capacity.
        self.micro_batch_size = int(
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        if self._initialized:
            return
        if not self.afd_config.compute_gate_on_attention:
            raise ValueError(
                "WindowAFDConnector requires compute_gate_on_attention=true "
                "for the ref-style Attention-to-FFN route",
            )
        if self.micro_batch_num not in (1, 2):
            raise ValueError(
                "WindowAFDConnector supports micro_batch_num=1 or 2, "
                f"got {self.micro_batch_num}",
            )
        parallel_config = self.vllm_config.parallel_config
        runtime_micro_batch_num = (
            int(parallel_config.num_ubatches)
            if parallel_config.enable_dbo and parallel_config.use_ubatching
            else 1
        )
        if self.micro_batch_num != runtime_micro_batch_num:
            raise ValueError(
                "Window micro_batch_num must match the vLLM DBO mode: "
                f"window={self.micro_batch_num} "
                f"runtime={runtime_micro_batch_num}"
            )
        if self.micro_batch_size > 512:
            raise ValueError(
                "WindowAFDConnector requires max_num_batched_tokens <= 512 "
                "for the current AttentionToFfn operator, "
                f"got {self.micro_batch_size}",
            )
        routed_topk = self.selected_expert_num - self.shared_expert_num
        if routed_topk > 16:
            raise ValueError(
                "WindowAFDConnector requires num_experts_per_tok <= 16, "
                f"got {routed_topk}",
            )
        if self.shared_expert_num != 1:
            raise ValueError(
                "WindowAFDConnector currently supports exactly one shared expert, "
                f"got {self.shared_expert_num}",
            )
        routed_ffn_size = self.ffn_size - self.shared_expert_num
        if routed_ffn_size <= 0:
            raise ValueError(
                "WindowAFDConnector requires at least one routed-expert FFN rank"
            )

        (
            self.attn_window_size,
            self.ffn_window_size,
            a2f_token_size,
            f2a_token_size,
        ) = _window_sizes(
            attention_size=self.attn_size,
            micro_batch_num=self.extra_info.micro_batch_num,
            micro_batch_size=self.micro_batch_size,
            selected_expert_num=self.selected_expert_num,
            hidden_size=self.hidden_size,
            quant_mode=self.extra_info.quant_mode,
        )

        timeout = timedelta(minutes=30)
        try:
            self.process_group = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.world_size,
                rank=self.world_rank,
                group_name="afd_window",
                timeout=timeout,
            )
            backend = self.process_group._get_backend(torch.device("npu"))
            getter = getattr(backend, "get_hccl_comm_name", None)
            if getter is None:
                getter = getattr(self.process_group, "get_hccl_comm_name", None)
            if getter is None:
                raise RuntimeError("HCCL ProcessGroup does not expose comm name API")
            self.hccl_comm_name = str(getter(self.world_rank))

            self.window_size = (
                self.attn_window_size
                if self.afd_config.role == "attention"
                else self.ffn_window_size
            )
            self.window_tensor = torch.zeros(
                self.window_size,
                dtype=torch.uint8,
                device=torch.device("npu"),
            )
            self.window_addr = int(self.window_tensor.data_ptr())

            ffn_info, ffn_data, attn_info, attn_data = self._operator_shapes()
            if self.afd_config.role == "attention":
                self.comm_buffer = cot.get_buffer_for_attention_to_ffn(
                    self.process_group,
                    self.world_size,
                    ffn_info,
                    ffn_data,
                    quant_mode=self.extra_info.quant_mode,
                    window_addr=self.window_addr,
                    window_size=self.window_size,
                )
            else:
                self.comm_buffer = cot.get_buffer_for_ffn_to_attention(
                    self.process_group,
                    self.world_size,
                    attn_info,
                    attn_data,
                    window_addr=self.window_addr,
                    window_size=self.window_size,
                )

            context_factory = torch_npu._afd.create_schedule_context_holder
            kwargs = {
                "schedule_mode": 1 if self.afd_config.role == "attention" else 0,
                "session_num": self.attn_size,
                "micro_batch_num": self.extra_info.micro_batch_num,
                "micro_batch_size": self.micro_batch_size,
                "selected_expert_num": self.selected_expert_num,
                "expert_num": self.expert_num,
                "attn_to_ffn_token_size": a2f_token_size,
                "ffn_to_attn_token_size": f2a_token_size,
            }
            if self.afd_config.role == "attention":
                kwargs.update(
                    attention_window=self.window_addr,
                    attention_window_size=self.window_size,
                )
            else:
                kwargs.update(
                    ffn_window=self.window_addr,
                    ffn_window_size=self.window_size,
                )
            self.context_holder = context_factory(**kwargs)
            self.schedule_context = self.context_holder.get_schedule_context_tensor()
            self._build_rank_tables()
            if self.afd_config.role == "attention":
                self._initialize_attention_operator_metadata()
                if self.stream_overlap_enabled:
                    self._initialize_attention_stream_pipeline()
            ffn_kind = ""
            if self.afd_config.role == "ffn":
                ffn_kind = build_window_expert_layout(
                    routed_expert_num=self.routed_expert_num,
                    ffn_size=self.ffn_size,
                    ffn_rank=self.role_rank,
                ).kind
            print(
                "[Window][init] "
                f"role={self.afd_config.role} "
                f"role_rank={self.role_rank} "
                f"world_rank={self.world_rank} "
                f"attn_size={self.attn_size} "
                f"ffn_size={self.ffn_size} "
                f"micro_batch_num={self.extra_info.micro_batch_num} "
                f"micro_batch_size={self.micro_batch_size} "
                f"selected_expert_num={self.selected_expert_num} "
                f"expert_num={self.expert_num} "
                f"local_expert_num={self.local_expert_num} "
                f"ffn_kind={ffn_kind} "
                f"window_size={self.window_size}",
                flush=True,
            )
            self._initialized = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._pending_transfers.clear()
        self.attention_receive_dependencies.clear()
        self.attention_pipeline_events.clear()
        self.attention_session_id = None
        self.attention_micro_batch_ids = None
        self.attention_layer_ids = None
        self.attention_sync_layer_id = None
        self.a2f_send_stream = None
        self.f2a_recv_stream = None
        self.attention_graph_compute_stream = None
        holder = self.context_holder
        self.context_holder = None
        self.schedule_context = None
        if holder is not None:
            try:
                holder.stop_schedule()
            except Exception:
                pass
        comm_buffer = self.comm_buffer
        self.comm_buffer = None
        if comm_buffer is not None:
            try:
                comm_buffer.destroy()
            except Exception:
                pass
        self.window_tensor = None
        self.window_size = 0
        self.window_addr = 0
        self.attn_window_size = 0
        self.ffn_window_size = 0
        group = self.process_group
        self.process_group = None
        if group is not None:
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        self.hccl_comm_name = None
        self._initialized = False

    def _initialize_attention_operator_metadata(self) -> None:
        """Cache immutable operator inputs before ACLGraph capture."""
        device = torch.device("npu", self.local_rank)
        self.attention_session_id = torch.tensor(
            [self.role_rank],
            dtype=torch.int32,
            device=device,
        )
        self.attention_micro_batch_ids = torch.arange(
            self.micro_batch_num,
            dtype=torch.int32,
            device=device,
        )
        self.attention_layer_ids = torch.tensor(
            list(range(self.num_layers)),
            dtype=torch.int32,
            device=device,
        )
        self.attention_sync_layer_id = torch.zeros(
            1,
            dtype=torch.int32,
            device=device,
        )

    def _initialize_attention_stream_pipeline(self) -> None:
        """Create eager communication and ACLGraph compute streams/events."""
        device = torch.device("npu", self.local_rank)
        self.a2f_send_stream = torch.npu.Stream(device=device)
        self.f2a_recv_stream = torch.npu.Stream(device=device)
        self.attention_graph_compute_stream = torch.npu.Stream(device=device)
        self.attention_pipeline_events = {
            (layer_idx, stage_idx): WindowAttentionPipelineEvents(
                ready=torch.npu.Event(),
                compute_done=torch.npu.Event(),
                send_ready=torch.npu.Event(),
                send_done=torch.npu.Event(),
                recv_done=torch.npu.Event(),
            )
            # The final pseudo-layer runs the post-HC continuation.
            for layer_idx in range(self.num_layers + 1)
            for stage_idx in range(self.micro_batch_num)
        }

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
        """Return whether the eager U2 stream pipeline owns this forward."""
        if (
            torch.compiler.is_compiling()
            or not self.attention_stream_pipeline_ready
        ):
            return False
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        if bool(getattr(forward_context, "afd_graph_ubatching", False)):
            return False
        return bool(
            getattr(forward_context, "dbo_enabled", False)
            and int(getattr(forward_context, "num_ubatches", 1)) > 1
        )

    def _attention_events(
        self,
        layer_idx: int,
        stage_idx: int,
    ) -> WindowAttentionPipelineEvents:
        try:
            return self.attention_pipeline_events[(layer_idx, stage_idx)]
        except KeyError as exc:
            raise RuntimeError(
                "Window Attention pipeline event is not initialized: "
                f"layer={layer_idx} stage={stage_idx}"
            ) from exc

    def attention_graph_compute_pipeline_active(self) -> bool:
        """Return whether ACLGraph U2 should use the side compute stream."""
        if torch.compiler.is_compiling():
            return False
        if not (
            self.stream_overlap_enabled
            and self.afd_config.role == "attention"
            and self.attention_graph_compute_stream is not None
            and self.attention_pipeline_events
        ):
            return False
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(
            getattr(forward_context, "afd_graph_ubatching", False)
            and getattr(forward_context, "afd_layer_major_u2", False)
            and getattr(forward_context, "dbo_enabled", False)
            and int(getattr(forward_context, "num_ubatches", 1)) > 1
        )

    def attention_graph_hybrid_dag_active(self) -> bool:
        """Window Graph U2 consumes Combine through the existing dependency."""
        return False

    @contextmanager
    def attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
        wait_for_receive_layer_idx: int | None = None,
    ) -> Iterator[None]:
        """Fork one ACLGraph Attention computation to the compute stream."""
        if not self.attention_graph_compute_pipeline_active():
            raise RuntimeError("Window Attention Graph compute pipeline is inactive")
        events = self._attention_events(layer_idx, stage_idx)
        parent_stream = torch.npu.current_stream()
        if wait_for_receive_layer_idx is None:
            events.ready.record(parent_stream)
            ready_event = events.ready
        else:
            ready_event = self._attention_events(
                wait_for_receive_layer_idx,
                stage_idx,
            ).recv_done
        assert self.attention_graph_compute_stream is not None
        with torch.npu.stream(self.attention_graph_compute_stream):
            ready_event.wait(self.attention_graph_compute_stream)
            for tensor in tensors:
                _record_npu_stream(tensor, self.attention_graph_compute_stream)
            yield
            events.compute_done.record(self.attention_graph_compute_stream)

    def wait_for_attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Make the parent capture stream wait for one Attention compute."""
        if not self.attention_graph_compute_pipeline_active():
            raise RuntimeError("Window Attention Graph compute pipeline is inactive")
        parent_stream = torch.npu.current_stream()
        self._attention_events(layer_idx, stage_idx).compute_done.wait(parent_stream)
        for tensor in tensors:
            _record_npu_stream(tensor, parent_stream)

    def join_attention_graph_compute(
        self,
        *,
        layer_idx: int,
        stage_idx: int,
        tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Join final post-HC compute back to the parent capture stream."""
        self.wait_for_attention_graph_compute(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            tensors=tensors,
        )

    def _build_rank_tables(self) -> None:
        """Build a balanced routed table plus one shared-first FFN rank."""
        if self.schedule_context is None:
            raise RuntimeError("ScheduleContext must be created before rank tables")
        device = self.schedule_context.device
        table = torch.zeros(
            (1, self.expert_num, 3),
            dtype=torch.int32,
            device=device,
        )
        for ffn_rank in range(self.ffn_size):
            layout = build_window_expert_layout(
                routed_expert_num=self.routed_expert_num,
                ffn_size=self.ffn_size,
                ffn_rank=ffn_rank,
            )
            if layout.is_shared:
                table[0, self.routed_expert_num, 0] = 1
                table[0, self.routed_expert_num, 1] = ffn_rank
                table[0, self.routed_expert_num, 2] = 0
                continue
            for local_id in range(layout.local_expert_count):
                expert_id = layout.local_expert_start + local_id
                table[0, expert_id, 0] = 1
                table[0, expert_id, 1] = ffn_rank
                table[0, expert_id, 2] = local_id
        # All layers share the same deployment. Sync A2F indexes layer 0;
        # async A2F indexes the real model layer in this same table.
        table = table.expand(self.num_layers, -1, -1).contiguous()
        self.expert_rank_table = table
        if self.afd_config.role == "ffn":
            self.local_expert_num = build_window_expert_layout(
                routed_expert_num=self.routed_expert_num,
                ffn_size=self.ffn_size,
                ffn_rank=self.role_rank,
            ).local_expert_count
        else:
            self.local_expert_num = 0
        self.attn_rank_table = (
            torch.arange(
                self.attn_size,
                dtype=torch.int32,
                device=device,
            )
            + self.ffn_size
        )

    def _operator_shapes(self) -> tuple[list[int], list[int], list[int], list[int]]:
        batch_size = self.micro_batch_size
        micro_batch_num = self.micro_batch_num
        quant_mode = self.extra_info.quant_mode
        # This is the last dimension of one A2F token record in the FFN
        # Window, not its byte size in every mode.  For H=7168 it is 7168
        # fp16/bfloat16 elements in mode 0, or 7680 one-byte storage elements
        # in mode 2: align(7168 INT8 bytes + 4 scale bytes, 512) = 7680.
        if quant_mode == 2:
            a2f_token_data_dim = _align_up(self.hidden_size + 4, 512)
        elif quant_mode == 4:
            # One FP8 byte per hidden value plus an E8M0 scale for every
            # 32-value block. The operator rounds the scale count to even.
            quant_data_dim = _align_up(self.hidden_size, 256)
            scale_dim = _align_up((self.hidden_size + 31) // 32, 2)
            a2f_token_data_dim = _align_up(
                quant_data_dim + scale_dim,
                512,
            )
        else:
            a2f_token_data_dim = self.hidden_size
        ffn_info = [
            self.attn_size,
            micro_batch_num,
            2 + batch_size * self.selected_expert_num,
        ]
        ffn_data = [
            self.attn_size,
            micro_batch_num,
            batch_size,
            self.selected_expert_num,
            a2f_token_data_dim,
        ]
        attn_info = [micro_batch_num, batch_size, self.selected_expert_num]
        attn_data = [
            micro_batch_num,
            batch_size,
            self.selected_expert_num,
            self.hidden_size,
        ]
        return ffn_info, ffn_data, attn_info, attn_data

    def _token_dtype(self) -> int:
        if self.extra_info.quant_mode == 2:
            return 2
        if self.extra_info.quant_mode == 4:
            return 4
        return 1 if self.vllm_config.model_config.dtype == torch.bfloat16 else 0

    @staticmethod
    def _token_dtype_for_tensor(tensor: torch.Tensor) -> int:
        if tensor.dtype == torch.bfloat16:
            return 1
        if tensor.dtype == torch.float16:
            return 0
        raise RuntimeError(
            "Window combine requires float16 or bfloat16 reference tensor, "
            f"got {tensor.dtype}",
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        expert_ids = kwargs.get("expert_ids")
        expert_scales = kwargs.get("expert_scales")
        if expert_ids is None or expert_scales is None:
            raise RuntimeError("Window A2F requires expert_ids and expert_scales")
        batch_size = int(hidden_states.shape[0])
        if batch_size <= 0:
            raise RuntimeError("Window A2F requires at least one token")
        if batch_size > self.micro_batch_size:
            raise RuntimeError(
                "Window A2F batch exceeds the configured capacity: "
                f"batch={batch_size} capacity={self.micro_batch_size}",
            )
        expert_ids = expert_ids.to(torch.int32).reshape(batch_size, -1)
        expert_scales = expert_scales.to(torch.float32).reshape(batch_size, -1)

        # A2F receives only routed top-k IDs. The shared-expert slot is
        # represented by selected_expert_num (K + shared) in the Window
        # layout and rank table, not by an extra expert_ids column.
        routed_topk = self.selected_expert_num - self.shared_expert_num
        if expert_ids.shape[1] != routed_topk:
            raise RuntimeError(
                "Window A2F received an unexpected routed expert ID width: "
                f"got {expert_ids.shape[1]}, expected {routed_topk}",
            )
        if expert_scales.shape[1] != routed_topk:
            raise RuntimeError(
                "Window A2F received an unexpected routed expert scale width: "
                f"got {expert_scales.shape[1]}, expected {routed_topk}",
            )
        # Window buffers keep the configured capacity shape, while active_mask
        # marks only the rows belonging to this request as valid.
        x = hidden_states.new_zeros(
            (1, self.micro_batch_size, self.hidden_size),
        )
        x[0, :batch_size].copy_(
            hidden_states.reshape(batch_size, self.hidden_size),
        )
        padded_expert_ids = torch.zeros(
            (1, self.micro_batch_size, routed_topk),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        padded_expert_ids[0, :batch_size].copy_(expert_ids)
        active_mask = torch.zeros(
            (1, self.micro_batch_size),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        active_mask[0, :batch_size] = True
        # The Window token/data buffers use the configured capacity as the
        # stride between microbatch slots. Keep the combine input at that
        # capacity and snapshot the current scales; the operator uses its
        # first dimension for both tiling and the microbatch offset.
        combine_scales = expert_scales.new_zeros(
            (self.micro_batch_size, routed_topk),
        )
        combine_scales[:batch_size].copy_(expert_scales)
        _, _, attn_info, _ = self._operator_shapes()
        stage_idx = int(context.metadata.stage_idx)
        if stage_idx < 0 or stage_idx >= self.micro_batch_num:
            raise RuntimeError(
                "Window A2F received an out-of-range micro batch: "
                f"micro_batch_id={stage_idx} micro_batch_num={self.micro_batch_num}"
            )
        model_layer_idx = int(context.metadata.layer_idx)
        if model_layer_idx < 0 or model_layer_idx >= self.num_layers:
            raise RuntimeError(
                "Window A2F received an out-of-range model layer: "
                f"layer={model_layer_idx} num_layers={self.num_layers}"
            )
        assert self.attention_session_id is not None
        assert self.attention_micro_batch_ids is not None
        assert self.attention_layer_ids is not None
        assert self.attention_sync_layer_id is not None
        session_id = self.attention_session_id
        micro_batch_id = self.attention_micro_batch_ids[stage_idx : stage_idx + 1]
        layer_id = (
            self.attention_layer_ids[model_layer_idx : model_layer_idx + 1]
            if self.async_mode
            else self.attention_sync_layer_id
        )

        def enqueue_a2f() -> None:
            cot.attention_to_ffn(
                self.comm_buffer,
                x,
                session_id,
                micro_batch_id,
                layer_id,
                padded_expert_ids,
                self.expert_rank_table,
                attn_info,
                self.routed_expert_num,
                # 0 notifies every FFN rank; 1 only notifies ranks receiving tokens.
                sync_flag=1 if self.async_mode else 0,
                ffn_start_rank_id=0,
                active_mask=active_mask,
            )

        if self.attention_graph_compute_pipeline_active():
            if stage_idx in self.attention_receive_dependencies:
                raise RuntimeError(
                    "Window Attention stage has an unconsumed Combine result: "
                    f"stage={stage_idx}"
            )
            events = self._attention_events(model_layer_idx, stage_idx)
            parent_stream = torch.npu.current_stream()
            # Gate/top-k and fixed-capacity A2F inputs are produced on the
            # parent capture stream after Attention compute has joined it.
            # A distinct event prevents the send stream from reading them
            # as soon as the earlier compute_done event fires.
            events.send_ready.record(parent_stream)
            assert self.a2f_send_stream is not None
            with torch.npu.stream(self.a2f_send_stream):
                events.send_ready.wait(self.a2f_send_stream)
                for tensor in (
                    x,
                    session_id,
                    micro_batch_id,
                    layer_id,
                    padded_expert_ids,
                    active_mask,
                ):
                    _record_npu_stream(tensor, self.a2f_send_stream)
                enqueue_a2f()
                events.send_done.record(self.a2f_send_stream)
        elif self._attention_stream_pipeline_active():
            if stage_idx in self.attention_receive_dependencies:
                raise RuntimeError(
                    "Window Attention stage has an unconsumed Combine result: "
                    f"stage={stage_idx}"
                )
            events = self._attention_events(model_layer_idx, stage_idx)
            compute_stream = torch.npu.current_stream()
            events.compute_done.record(compute_stream)
            assert self.a2f_send_stream is not None
            with torch.npu.stream(self.a2f_send_stream):
                events.compute_done.wait(self.a2f_send_stream)
                for tensor in (
                    x,
                    session_id,
                    micro_batch_id,
                    layer_id,
                    padded_expert_ids,
                    active_mask,
                ):
                    _record_npu_stream(tensor, self.a2f_send_stream)
                enqueue_a2f()
                events.send_done.record(self.a2f_send_stream)
        else:
            enqueue_a2f()
        logger.debug(
            "Window A2F sent layer=%d stage=%d batch=%d topk=%d",
            context.metadata.layer_idx,
            context.metadata.stage_idx,
            batch_size,
            expert_ids.shape[-1],
        )
        transfer_key = (
            int(context.metadata.stage_idx),
            int(context.metadata.layer_idx),
        )
        self._pending_transfers[transfer_key] = context
        state = WindowAFDTransferState(expert_scales=combine_scales)
        context.states = state

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_data_path()
        key = (int(ubatch_idx), int(kwargs.get("layer_idx", 0)))
        context = self._pending_transfers.pop(key, None)
        if context is None or not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError(f"Window F2A has no pending transfer for {key}")

        def enqueue_combine(layer_id: torch.Tensor) -> torch.Tensor:
            output, _ = torch_npu.npu_attention_worker_combine(
                self.schedule_context,
                context.states.expert_scales,
                layer_id,
                self.hidden_size,
                # ``token_dtype=2`` is only the INT8 payload mode of
                # ``ffn_worker_batching``.  ``attention_worker_combine`` accepts
                # only the output dtype modes: 0=FP16 and 1=BF16.  Use the same
                # dtype as the Attention continuation/residual, as P2P does.
                token_dtype=self._token_dtype_for_tensor(ref_tensor),
                need_schedule=1,
            )
            return output[: ref_tensor.shape[0]].reshape_as(ref_tensor)

        graph_pipeline_active = self.attention_graph_compute_pipeline_active()
        eager_pipeline_active = self._attention_stream_pipeline_active()
        if not (graph_pipeline_active or eager_pipeline_active):
            assert self.attention_layer_ids is not None
            layer_id = self.attention_layer_ids[key[1] : key[1] + 1]
            return enqueue_combine(layer_id)

        events = self._attention_events(key[1], key[0])
        assert self.f2a_recv_stream is not None
        assert self.attention_layer_ids is not None
        layer_id = self.attention_layer_ids[key[1] : key[1] + 1]
        with torch.npu.stream(self.f2a_recv_stream):
            events.send_done.wait(self.f2a_recv_stream)
            _record_npu_stream(
                context.states.expert_scales,
                self.f2a_recv_stream,
            )
            output = enqueue_combine(layer_id)
            _record_npu_stream(output, self.f2a_recv_stream)
            events.recv_done.record(self.f2a_recv_stream)
        if key[0] in self.attention_receive_dependencies:
            raise RuntimeError(
                "Window Attention stage already has a deferred Combine result: "
                f"stage={key[0]}"
            )
        self.attention_receive_dependencies[key[0]] = (
            WindowAttentionReceiveDependency(
                tensor=output,
                event=events.recv_done,
            )
        )
        return output

    def require_attention_pipeline_idle(self) -> None:
        """Reject a new layer-major step if prior stream state is stale."""
        if self._pending_transfers or self.attention_receive_dependencies:
            raise RuntimeError(
                "Window Attention pipeline is not idle: "
                f"pending={tuple(self._pending_transfers)} "
                f"deferred={tuple(self.attention_receive_dependencies)}"
            )

    def wait_for_attention_stage_receive(
        self,
        *,
        stage_idx: int,
        tensor: torch.Tensor,
    ) -> None:
        """Make the current compute stream consume one deferred Combine."""
        if not (
            self.attention_graph_compute_pipeline_active()
            or self._attention_stream_pipeline_active()
        ):
            return
        dependency = self.attention_receive_dependencies.pop(stage_idx, None)
        if dependency is None:
            raise RuntimeError(
                "Window Attention stage has no deferred Combine result: "
                f"stage={stage_idx}"
            )
        if dependency.tensor is not tensor:
            raise RuntimeError(
                "Window Attention stage received an unexpected tensor: "
                f"stage={stage_idx}"
            )
        compute_stream = torch.npu.current_stream()
        dependency.event.wait(compute_stream)
        _record_npu_stream(tensor, compute_stream)

    def reset_attention_pipeline_state(self) -> None:
        """Discard per-step stream bookkeeping after a failed forward."""
        self._pending_transfers.clear()
        self.attention_receive_dependencies.clear()

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_data_path()
        batch_size = self.micro_batch_size
        # The operator expects the logical dimensions [A, BS, K+1, H].
        # Its tiling validates K+1 independently (currently <= 64); the
        # product A*BS*(K+1) is computed internally for the output rows.
        max_out_shape = [
            self.attn_size,
            batch_size,
            self.selected_expert_num,
            self.hidden_size,
        ]
        batching_expert_num = self.local_expert_num * (
            self.num_layers if self.async_mode else 1
        )
        if batching_expert_num > 8192:
            raise RuntimeError(
                "Window batching expert count exceeds the operator limit: "
                f"expert_num={batching_expert_num}"
            )
        if self.async_mode:
            outputs = cot.ffn_worker_batching(
                self.schedule_context,
                batching_expert_num,
                max_out_shape,
                token_dtype=self._token_dtype(),
                need_schedule=1,
                layer_num=self.num_layers,
                sync_flag=1,
                # ACLGraph cannot represent E8M0 tensors directly. Expose the
                # same scale bytes as uint8 in both warmup and replay; the
                # Global FFN supplies their logical E8M0 dtype to GMM.
                mx_output_uint8=self.extra_info.quant_mode == 4,
            )
        else:
            outputs = torch_npu.npu_ffn_worker_batching(
                self.schedule_context,
                batching_expert_num,
                max_out_shape,
                token_dtype=self._token_dtype(),
                need_schedule=1,
                layer_num=0,
            )
        (
            hidden_states,
            group_list,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            dynamic_scale,
            actual_token_num,
        ) = outputs
        if actual_token_num.numel() != 1:
            raise RuntimeError(
                "Window batching returned actual_token_num with unexpected "
                f"shape {tuple(actual_token_num.shape)}",
            )
        if group_list.shape != (batching_expert_num, 2):
            raise RuntimeError(
                "Window batching returned group_list with unexpected shape: "
                f"got={tuple(group_list.shape)} "
                f"expected={(batching_expert_num, 2)}",
            )
        if self.async_mode:
            # Keep the compact type-2 table and scalar token count on device.
            # The global Window FFN converts it to the GMM representation
            # without selecting Python layer modules or synchronizing to host.
            return self._build_ffn_transfer_payload(
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scale=dynamic_scale,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
                layer_idx=int(kwargs.get("layer_idx", 0)),
                ubatch_idx=ubatch_idx,
            )
        actual_num = int(actual_token_num.item())
        if actual_num < 0 or actual_num > hidden_states.shape[0]:
            raise RuntimeError(
                "Window batching returned invalid actual_token_num: "
                f"actual={actual_num} capacity={hidden_states.shape[0]}",
            )
        # On A3 the batching kernel writes a compact type-2 group list followed
        # by one [0, 0] sentinel, but does not clear the rest of the fixed-size
        # output.  Locate the valid prefix using actual_token_num, then convert
        # it to the dense cumulative type-0 form consumed by the native P2P
        # W8A8 MoE MLP path.  This also discards the stale fixed-buffer suffix.
        if actual_num == 0:
            group_list = torch.zeros(
                (batching_expert_num,),
                dtype=group_list.dtype,
                device=group_list.device,
            )
        else:
            group_counts = group_list[:, 1]
            cumulative_counts = torch.cumsum(group_counts, dim=0)
            prefix_ends = torch.nonzero(
                cumulative_counts == actual_num,
                as_tuple=False,
            ).flatten()
            if prefix_ends.numel() == 0:
                raise RuntimeError(
                    "Window batching group_list has no valid prefix matching "
                    f"actual_token_num={actual_num}",
                )
            valid_row_num = int(prefix_ends[0].item()) + 1
            if bool(torch.any(group_counts[:valid_row_num] <= 0).item()):
                raise RuntimeError(
                    "Window batching valid group_list prefix contains a "
                    "non-positive expert token count",
                )
            valid_expert_ids = group_list[:valid_row_num, 0]
            if bool(
                torch.any(
                    (valid_expert_ids < 0)
                    | (valid_expert_ids >= batching_expert_num)
                ).item()
            ):
                raise RuntimeError(
                    "Window batching valid group_list prefix contains an "
                    "out-of-range local expert ID",
                )
            if valid_row_num > 1 and bool(
                torch.any(valid_expert_ids[1:] <= valid_expert_ids[:-1]).item()
            ):
                raise RuntimeError(
                    "Window batching valid group_list expert IDs are not "
                    "strictly increasing",
                )
            expert_counts = torch.zeros(
                (batching_expert_num,),
                dtype=group_list.dtype,
                device=group_list.device,
            )
            expert_counts.scatter_(
                0,
                valid_expert_ids.to(torch.long),
                group_counts[:valid_row_num],
            )
            group_list = torch.cumsum(expert_counts, dim=0)

        group_sum = int(group_list[-1].item()) if group_list.numel() else 0
        if group_sum != actual_num:
            raise RuntimeError(
                "Window batching cumulative group_list does not match "
                f"actual_token_num: group_sum={group_sum} actual={actual_num}",
            )
        logger.debug(
            "Window FFN batching completed layer=%d stage=%d",
            int(kwargs.get("layer_idx", 0)),
            ubatch_idx,
        )
        return self._build_ffn_transfer_payload(
            hidden_states=hidden_states,
            group_list=group_list,
            dynamic_scale=dynamic_scale,
            session_ids=session_ids,
            micro_batch_ids=micro_batch_ids,
            token_ids=token_ids,
            expert_offsets=expert_offsets,
            actual_token_num=actual_token_num,
            layer_idx=int(kwargs.get("layer_idx", 0)),
            ubatch_idx=ubatch_idx,
        )

    @staticmethod
    def _build_ffn_transfer_payload(
        *,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor,
        session_ids: torch.Tensor,
        micro_batch_ids: torch.Tensor,
        token_ids: torch.Tensor,
        expert_offsets: torch.Tensor,
        actual_token_num: torch.Tensor,
        layer_idx: int,
        ubatch_idx: int,
    ) -> AFDA2FTransferPayload:
        """Package one Batching result without reading device values on Host."""
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=layer_idx,
                stage_idx=int(ubatch_idx),
                seq_lens=[int(hidden_states.shape[0])],
            ),
            states=WindowAFDTransferState(
                expert_scales=torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                group_list=group_list,
                dynamic_scale=dynamic_scale,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
            ),
        )
        # Keep the static batching capacity Y. Async mode keeps the compact
        # type-2 group list; synchronous mode supplies cumulative type-0.
        # actual_token_num identifies the valid prefix consumed by FFN and F2A.
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=context,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        if not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError("Window F2A requires batching state")
        state = context.states
        if any(
            value is None
            for value in (
                state.session_ids,
                state.micro_batch_ids,
                state.token_ids,
                state.expert_offsets,
                state.actual_token_num,
            )
        ):
            raise RuntimeError(
                "Window batching did not return complete routing metadata"
            )

        # FFNWorkerBatching returns fixed-capacity tensors.  actual_token_num
        # identifies their valid prefix; FfnToAttention consumes the same
        # capacity Y and ignores the suffix after that prefix.
        actual_token_num = state.actual_token_num.reshape(-1)
        if actual_token_num.numel() != 1:
            raise RuntimeError(
                "Window batching returned actual_token_num with unexpected "
                f"shape {tuple(state.actual_token_num.shape)}"
            )
        routed_output = getattr(ffn_output, "routed_output", ffn_output)
        if routed_output.dim() != 2:
            raise RuntimeError(
                "Window F2A output must be two-dimensional: "
                f"output_shape={tuple(routed_output.shape)}"
            )

        metadata = (
            state.session_ids,
            state.micro_batch_ids,
            state.token_ids,
            state.expert_offsets,
        )
        if any(
            value.dim() != 1 or value.shape[0] != routed_output.shape[0]
            for value in metadata
        ):
            raise RuntimeError(
                "Window F2A output and metadata capacities do not match: "
                f"output_shape={tuple(routed_output.shape)} metadata_shapes="
                f"{[tuple(value.shape) for value in metadata]}"
            )

        cot.ffn_to_attention(
            self.comm_buffer,
            routed_output,
            state.session_ids,
            state.micro_batch_ids,
            state.token_ids,
            state.expert_offsets,
            actual_token_num,
            attn_rank_table=self.attn_rank_table,
        )
        logger.debug(
            "Window F2A sent stage=%d actual_tokens=%s",
            context.metadata.stage_idx,
            state.actual_token_num,
        )

    def _require_data_path(self) -> None:
        if not self._initialized:
            raise RuntimeError("WindowAFDConnector data path is not initialized")

    def select_experts(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        return select_experts(**kwargs)


__all__ = ["WindowAFDConnector", "WindowAFDExtraInfo", "WindowAFDTransferState"]
