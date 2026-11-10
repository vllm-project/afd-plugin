# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM connector for Attention/FFN disaggregation.

The connector validates payloads and manages the Attention-first HCCL world
``[A0, A1, ..., F0, F1, ...]``. Push/Pull sessions own transfer state and
protocol rules; the FFN ModelRunner owns eager/FULL execution. Attention
microbatching uses native DBO. Native operators carry routing/count metadata
without a separate Gloo control plane.

See ``docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`` for configuration, rank
derivation and supported paths.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import Tensor
from vllm.logger import init_logger

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
    coerce_optional_extra_positive_int,
)
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    ConnectorExtraInfo,
    FFNExecutionSession,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_session import (
    CAM_COMM_ID,
    PullCAMSession,
    PushCAMSession,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
    GRAPH_STARTUP_TIMEOUT_SECONDS,
    POLL_SECONDS,
    AsyncCAMGraphStartup,
)
from afd_plugin.distributed import (
    create_hccl_process_group_options,
    init_afd_process_group,
)
from afd_plugin.validation import cam_receive_capacity, validate_graph_capacity

if TYPE_CHECKING:
    from torch.distributed import Store
    from torch.distributed.distributed_c10d import ProcessGroup
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe import FusedMoERouter

logger = init_logger(__name__)

AFD_ASYNC_CAM_GROUP_NAME = "afd_async_cam"
ATTN_RANKS_PER_DP_CONFIG_KEY = "attn_ranks_per_dp"

_AFD_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "dynamicQuant",
        "ffn_mega_moe",
        "attn_ranks_per_dp",
        "hccl_buffer_size",
    },
)


def select_cam_experts(
    *,
    hidden_states: Tensor,
    router_logits: Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    scoring_func: str,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    e_score_correction_bias: Tensor | None,
    mix_placement: bool,
    num_logical_experts: int,
    num_shared_experts: int,
    num_experts: int,
    router: FusedMoERouter | None = None,
) -> tuple[Tensor, Tensor]:
    """Select routed-only CAM experts using Ascend's native router policy."""
    if mix_placement:
        raise RuntimeError(
            "Async CAM uses routed-only expert IDs and does not support mix_placement"
        )
    # No EPLB state or fused shared experts: CAM dispatch consumes logical
    # routed IDs. Ascend owns fused/fallback selection and weight scaling.
    if router is None:
        from vllm_ascend.ops.fused_moe.router.router_factory import (
            create_ascend_fused_moe_router,
        )

        router = create_ascend_fused_moe_router(
            top_k=top_k,
            global_num_experts=router_logits.shape[-1],
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
    topk_weights, topk_ids = router.select_experts(
        hidden_states,
        router_logits,
        topk_indices_dtype=torch.int32,
    )
    return topk_weights, topk_ids


@dataclass(frozen=True)
class AFDAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async CAM connector configuration.

    Attributes:
        dynamic_quant: Dynamic quantization mode accepted by CAM operators.
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        hccl_buffer_size: Optional buffer size in MB for the CAM HCCL domain.
        ffn_mega_moe: Enable Mega FFN and its pull protocol on every A/F rank.
    """

    ffn_mega_moe: bool = False
    dynamic_quant: int = 0
    attn_ranks_per_dp: int = 1
    hccl_buffer_size: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> AFDAsyncExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _AFD_ASYNC_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown AFD async connector_extra_config field(s): "
                + ", ".join(unknown),
            )

        result = cls(
            ffn_mega_moe=coerce_extra_bool(
                raw.get("ffn_mega_moe", False), field_name="ffn_mega_moe"
            ),
            dynamic_quant=coerce_extra_int(
                raw.get("dynamicQuant", 0),
                field_name="dynamicQuant",
            ),
            attn_ranks_per_dp=coerce_extra_positive_int(
                raw.get("attn_ranks_per_dp", 1),
                field_name="attn_ranks_per_dp",
            ),
            hccl_buffer_size=coerce_optional_extra_positive_int(
                raw.get("hccl_buffer_size"),
                field_name="hccl_buffer_size",
            ),
        )

        if result.ffn_mega_moe and result.dynamic_quant != 1:
            raise ValueError("ffn_mega_moe requires dynamicQuant=1")
        return result

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dynamicQuant": self.dynamic_quant,
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
        }
        if self.ffn_mega_moe:
            result["ffn_mega_moe"] = True
        if self.hccl_buffer_size is not None:
            result["hccl_buffer_size"] = self.hccl_buffer_size
        return result


@dataclass(frozen=True, slots=True)
class AFDAsyncTopology:
    """Role-local and HCCL-world rank information for one CAM participant."""

    role: str
    role_rank: int
    world_rank: int
    attn_size: int
    ffn_size: int
    expert_per_rank: int

    @property
    def world_size(self) -> int:
        """Return the total number of Attention and FFN ranks."""
        return self.attn_size + self.ffn_size


@dataclass(slots=True)
class AFDAsyncTransferState(AFDTransferState):
    """Routed activations, counts and untouched compact combine metadata."""

    batch_size: int = 1
    hidden_size: int = 1
    topk: int = 1
    layer_idx: int = 0
    token_nums_rankid_layeridx: Tensor | None = None
    group_list: Tensor | None = None
    dynamic_scales: Tensor | None = None
    send_completion: Tensor | None = None


class CAMAsyncAFDConnector(AFDConnectorBase):
    """CAM-backed asynchronous connector for Ascend NPU AFD.

    Attention ranks occupy the first part of the HCCL world and FFN ranks the
    second. CAM dispatch/combine operators own both the collective data motion
    and its routing metadata, so this connector has no DP metadata control
    plane (``control_plane`` stays ``None``) and FFN work is triggered directly
    by the connector receive loop.
    """

    control_plane = None
    extra_info: AFDAsyncExtraInfo

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> AFDAsyncExtraInfo:
        return AFDAsyncExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        """Derive CAM topology, tensor dimensions, and connector state.

        Communication resources are created collectively by
        ``init_afd_connector``. ``role_rank`` is resolved before connector
        construction; ``attn_ranks_per_dp`` is used as the CAM Attention TP
        width.
        """
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = hf_config.hidden_size
        self.activation_dtype = vllm_config.model_config.dtype
        self.topk = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        extra_info = cast(AFDAsyncExtraInfo, self.extra_info)
        self.dynamic_quant = extra_info.dynamic_quant
        self.hccl_buffer_size_mb = extra_info.hccl_buffer_size
        self.group_name = ""
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.tp_size = extra_info.attn_ranks_per_dp
        self.cam_pg: ProcessGroup | None = None
        self.topology = build_async_topology(
            afd_config,
            role_rank,
            num_routed_experts=self.num_routed_experts,
        )
        self.world_rank = self.topology.world_rank
        self.attn_size = self.topology.attn_size
        self.ffn_size = self.topology.ffn_size
        self.expert_per_rank = self.topology.expert_per_rank
        self.graph_startup = AsyncCAMGraphStartup(
            self.world_rank, self.attn_size, self.topology.world_size
        )
        self._session: PushCAMSession | PullCAMSession = (
            PullCAMSession(
                self.graph_startup,
                self.topology,
                hidden_size=self.hidden_size,
                topk=self.topk,
                max_num_batched_tokens=self.max_num_batched_tokens,
                tp_size=self.tp_size,
                receive_capacity=cam_receive_capacity(),
            )
            if extra_info.ffn_mega_moe
            else PushCAMSession(self.activation_dtype)
        )

    @property
    def comm_args(self) -> Tensor | None:
        """Expose the selected protocol's live window to Mega compute."""
        return self._session.comm_args

    @property
    def comm_id(self) -> int:
        return self._session.comm_id

    @property
    def has_pending_attention(self) -> bool:
        return self._session.has_pending_attention

    @property
    def is_initialized(self) -> bool:
        """Return whether the CAM HCCL group and operator buffers are ready."""
        return self._initialized

    def init_afd_connector(self) -> None:
        """Collectively initialize the CAM HCCL world and operator buffers.

        All Attention and FFN ranks must call this method with identical
        rendezvous and topology settings. Missing/duplicate ranks or mismatched
        world sizes fail or time out during the 30-minute rendezvous.
        """
        if self._initialized:
            return

        ensure_cam_async_ops_available()
        # Same generation-uniqueness rationale as the CAMP2p connector: the
        # HCCL communicator is resolved by group *name* in the native ops, so
        # a rebuilt connector must not reuse the retired generation's name.
        self.cam_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.topology.world_size,
            rank=self.world_rank,
            group_name=f"{AFD_ASYNC_CAM_GROUP_NAME}_{self.afd_config.port}",
            timeout=timedelta(minutes=30),
            pg_options=create_hccl_process_group_options(
                self.hccl_buffer_size_mb,
            ),
            store_callback=self._initialize_graph_startup,
        )
        backend = self.cam_pg._get_backend(torch.device("npu"))
        self.group_name = str(backend.get_hccl_comm_name(self.world_rank))
        try:
            self._session.initialize(f"npu:{self.local_rank}", self.group_name)
        except Exception as exc:
            self.graph_startup.fail(exc)
            raise
        self._initialized = True
        if self.graph_startup.mode == "full" and self.world_rank < self.attn_size:
            self._run_attention_graph_startup()

    def _initialize_graph_startup(self, store: Store) -> None:
        signature = repr(
            (
                self.attn_size,
                self.ffn_size,
                self.hidden_size,
                self.topk,
                self.num_routed_experts,
                str(self.activation_dtype),
                self.dynamic_quant,
                self.tp_size,
                self.max_num_batched_tokens,
            )
        )
        signature = self._session.signature_prefix + signature
        self.graph_startup.initialize(store, signature)
        if self.graph_startup.mode == "full":
            try:
                if not self.vllm_config.use_v2_model_runner:
                    raise ValueError("Async FFN FULL requires MRV2 on every rank")
                validate_graph_capacity(
                    self.max_num_batched_tokens,
                    self.tp_size,
                    cam_receive_capacity(),
                    allow_row_slicing=self._session.allow_row_slicing,
                )
                self.graph_startup.phase("validated")
            except Exception as exc:
                self.graph_startup.fail(exc)
                raise

    def wait_graph_device(self, event: torch.npu.Event) -> None:
        deadline = time.monotonic() + GRAPH_STARTUP_TIMEOUT_SECONDS
        while not event.query():
            self.graph_startup.check_failure()
            if time.monotonic() >= deadline:
                raise TimeoutError("Async graph warmup timed out; retain resources")
            time.sleep(POLL_SECONDS)

    def graph_startup_chunks(self) -> int:
        """Every A TP rank sends one row to experts [0, topk)."""
        if isinstance(self._session, PullCAMSession):
            # A fused launch drains every receive round of the selected source.
            return 1
        local_experts = max(
            0,
            min(
                self.expert_per_rank,
                self.topk - (self.world_rank - self.attn_size) * self.expert_per_rank,
            ),
        )
        experts_per_chunk = cam_receive_capacity() // self.tp_size
        return max(1, (local_experts + experts_per_chunk - 1) // experts_per_chunk)

    def _run_attention_graph_startup(self) -> None:
        assert self.graph_startup.warmup_layer_id is not None
        device = torch.device(f"npu:{self.local_rank}")
        hidden = torch.ones(
            (1, self.hidden_size), dtype=self.activation_dtype, device=device
        )
        ids = torch.arange(self.topk, dtype=torch.int32, device=device).view(1, -1)
        weights = torch.full((1, self.topk), 1 / self.topk, device=device)
        try:
            # Every A/F rank participates in cold MC2 warmup before capture.
            context = AFDTransferContext(
                AFDTransferMetadata(self.graph_startup.warmup_layer_id, 0, [1])
            )
            self.send_attn_output(hidden, context, topk_ids=ids, topk_weights=weights)
            output = self.recv_ffn_output(hidden, context=context)
            done = torch.npu.Event()
            done.record()
            self.wait_graph_device(done)
            del output
            self.graph_startup.phase("warmup")
            self.graph_startup.phase("captured")
            self.graph_startup.wait_service_ready()
        except Exception as exc:
            self.graph_startup.fail(exc)
            raise

    def ffn_session(self) -> FFNExecutionSession:
        return self._session

    def attention_step(self) -> AbstractContextManager[None]:
        return self._session.attention_step()

    def close(self) -> None:
        """Destroy the HCCL process group and clear pending transfer states."""
        self._session.prepare_close(self._initialized)
        if self.cam_pg is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.cam_pg)
        self.cam_pg = None
        self._session.release()
        self._initialized = False

    def select_experts(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Run the target vLLM-Ascend router on Attention."""
        return select_cam_experts(**kwargs)

    def send_attn_output(
        self,
        hidden_states: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Validate routed activations, then submit through the selected protocol."""
        self._require_initialized()
        metadata = context.metadata
        if not metadata.validate_tensor_shape(tuple(hidden_states.shape)):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD async metadata token count {metadata.total_tokens}",
            )
        states = AFDAsyncTransferState(
            batch_size=metadata.total_tokens,
            hidden_size=self.hidden_size,
            topk=self.topk,
            layer_idx=metadata.layer_idx,
        )
        context.states = states
        topk_ids = kwargs.get("topk_ids")
        topk_weights = kwargs.get("topk_weights")
        if topk_ids is None or topk_weights is None:
            raise RuntimeError(
                "CAMAsyncAFDConnector send_attn_output requires topk_ids/topk_weights",
            )
        topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=states.batch_size,
            topk=states.topk,
        )
        if states.batch_size > self.max_num_batched_tokens // self.tp_size:
            raise ValueError(
                "Async CAM batch exceeds max_num_batched_tokens / TP capacity"
            )
        hidden_states = hidden_states.contiguous()
        args = self._cam_args(states) | dict(
            x=hidden_states,
            expert_ids=topk_ids,
            max_seq_len=self.max_num_batched_tokens,
            batch_size=states.batch_size,
            attn_rank_id=self.world_rank,
            layer_index=states.layer_idx,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
        )
        _log_cam_op_values("async_dispatch_send", "inputs", **args)
        self._session.dispatch((context, topk_ids, topk_weights), **args)

    def recv_ffn_output(
        self,
        ref_tensor: Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        """Combine the oldest matching Push dispatch or the exclusive Pull window."""
        self._require_initialized()
        context = kwargs.get("context")
        payload = self._session.pending_for(ubatch_idx)
        if payload is None:
            raise RuntimeError(
                "CAMAsyncAFDConnector recv_ffn_output is missing pending "
                "Attention metadata",
            )
        if context is not None and context is not payload[0]:
            raise RuntimeError(
                "CAMAsyncAFDConnector recv_ffn_output context does not match "
                "the next pending Attention dispatch",
            )
        context, topk_ids, topk_weights = payload

        states = _require_async_transfer_state(context)
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=states.batch_size,
            topk=states.topk,
        )
        placeholder = ref_tensor.new_empty((1,))
        args = self._cam_args(states) | dict(
            expand_x=placeholder,
            expert_ids=topk_ids,
            expert_scales=topk_weights,
            batch_size=states.batch_size,
            attn_rank_id=self.world_rank,
        )
        _log_cam_op_values("async_combine_recv", "inputs", **args)
        output = self._session.combine(ubatch_idx, **args)
        _log_cam_op_values("async_combine_recv", "outputs", output=output)
        return output

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive a Push work item and preserve its compact return metadata."""
        session = self._require_push_session()
        batch_size = int(kwargs.get("batch_size", self.max_num_batched_tokens) or 1)
        layer_idx = int(kwargs.get("layer_idx", 0) or 0)
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[batch_size],
        )
        states = AFDAsyncTransferState(
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            topk=self.topk,
            layer_idx=layer_idx,
        )
        context = AFDTransferContext(
            metadata=metadata,
            states=states,
        )
        placeholder = kwargs.get("placeholder", session.placeholder)
        args = self._cam_args(states) | dict(
            x=placeholder,
            max_seq_len=self.max_num_batched_tokens,
            moe_rank_id=self.world_rank,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
        )
        _log_cam_op_values("async_dispatch_recv", "inputs", **args)
        outputs = torch.ops.afd_ascend.afd_async_dispatch_recv(**args)
        hidden_states, dynamic_scales, batch_info, expert_token_nums = outputs
        _log_cam_op_values(
            "async_dispatch_recv",
            "outputs",
            hidden_states=hidden_states,
            dynamic_scales=dynamic_scales,
            batch_info=batch_info,
            expert_token_nums=expert_token_nums,
        )
        states.token_nums_rankid_layeridx = batch_info
        states.group_list = expert_token_nums
        states.dynamic_scales = dynamic_scales
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=context,
        )

    def send_ffn_output(
        self,
        ffn_output: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Return a Push work item with its original dispatch metadata."""
        self._require_push_session()
        states = _require_async_transfer_state(context)
        token_nums_rankid_layeridx = states.token_nums_rankid_layeridx
        if token_nums_rankid_layeridx is None:
            token_nums_rankid_layeridx = kwargs.get("token_nums_rankid_layeridx")
        if token_nums_rankid_layeridx is None:
            raise RuntimeError(
                "AFD async CAM combine send requires "
                "TokenNums_Rankid_Layeridx from async_dispatch_recv",
            )
        args = self._cam_args(states) | dict(
            expand_x=ffn_output,
            batch_info=token_nums_rankid_layeridx,
            max_seq_len=self.max_num_batched_tokens,
            moe_rank_id=self.world_rank,
            tp_size=self.tp_size,
        )
        _log_cam_op_values("async_combine_send", "inputs", **args)
        states.send_completion = torch.ops.afd_ascend.afd_async_combine_send(**args)

    def _require_push_session(self) -> PushCAMSession:
        self._require_initialized()
        if not isinstance(self._session, PushCAMSession):
            raise RuntimeError(
                "Pull FFN receive/compute/send belongs to the Mega executor"
            )
        return self._session

    def _cam_args(
        self, states: AFDAsyncTransferState
    ) -> dict[str, Tensor | int | str | None]:
        """Fields shared by the four registered CAM operator schemas."""
        return dict(
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            hidden_size=states.hidden_size,
            top_k=states.topk,
            moe_rank_num=self.ffn_size,
            attn_rank_num=self.attn_size,
            route_expert_num_per_moe=self.expert_per_rank,
            world_size=self.topology.world_size,
            group_name=self.group_name,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CAMAsyncAFDConnector is not initialized")


def build_async_topology(
    afd_config: AFDConfig,
    role_rank: int,
    *,
    num_routed_experts: int | None = None,
) -> AFDAsyncTopology:
    """Validate role-local rank settings and derive HCCL world rank.

    The world is Attention-first: Attention role rank ``i`` maps to world rank
    ``i`` and FFN role rank ``j`` maps to
    ``num_attention_ranks + j``. Routed experts are evenly distributed across FFN
    ranks, so the routed-expert count must divide by the FFN rank count.
    """
    attn_size = afd_config.num_attention_ranks
    ffn_size = afd_config.num_ffn_ranks
    if attn_size <= 0 or ffn_size <= 0:
        raise ValueError("AFD async topology sizes must be positive")
    if role_rank < 0:
        raise ValueError(f"AFD async role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attn_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attn_size})",
            )
        world_rank = role_rank
    elif afd_config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = attn_size + role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    expert_count = num_routed_experts if num_routed_experts is not None else ffn_size
    if expert_count % ffn_size:
        raise ValueError("Async CAM requires routed experts divisible by FFN ranks")
    expert_per_rank = expert_count // ffn_size
    return AFDAsyncTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        attn_size=attn_size,
        ffn_size=ffn_size,
        expert_per_rank=expert_per_rank,
    )


def _require_async_transfer_state(
    context: AFDTransferContext,
) -> AFDAsyncTransferState:
    states = context.states
    if not isinstance(states, AFDAsyncTransferState):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires AFDAsyncTransferState in the "
            "transfer context",
        )
    return states


_CAM_LOG_SKIPPED_ARGS = frozenset({"comm_args", "comm_id", "group_name"})
_CAM_OP_IO_LOG_ENV = "AFD_CAM_OP_IO_LOG"


def _log_cam_op_values(op_name: str, label: str, **kwargs: object) -> None:
    if os.environ.get(_CAM_OP_IO_LOG_ENV, "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    lines: list[str] = []
    for name, value in kwargs.items():
        if name in _CAM_LOG_SKIPPED_ARGS:
            continue
        if isinstance(value, Tensor):
            description = (
                f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)}, "
                f"device={value.device})"
            )

        else:
            description = repr(value)
        lines.append(f"  {name}={description}")
    logger.warning("AFD CAM %s %s:\n%s", op_name, label, "\n".join(lines))


def _validate_topk_payload(
    topk_ids: Tensor,
    topk_weights: Tensor | None,
    *,
    batch_size: int,
    topk: int,
) -> None:
    if tuple(topk_ids.shape) != (batch_size, topk):
        raise ValueError(
            f"topk_ids shape must match ({batch_size}, {topk}), got {topk_ids.shape!r}",
        )
    if topk_weights is None:
        raise ValueError("topk_weights is required")
    if tuple(topk_weights.shape) != (batch_size, topk):
        raise ValueError(
            "topk_weights shape must match "
            f"({batch_size}, {topk}), "
            f"got {topk_weights.shape!r}",
        )


__all__ = [
    "AFD_ASYNC_CAM_GROUP_NAME",
    "CAMAsyncAFDConnector",
    "AFDAsyncTransferState",
    "AFDAsyncTopology",
    "ATTN_RANKS_PER_DP_CONFIG_KEY",
    "CAM_COMM_ID",
    "build_async_topology",
    "select_cam_experts",
]
