# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM asynchronous connector for Attention/FFN disaggregation.

``CAMAsyncAFDConnector`` supports eager Attention and eager or FULL FFN.
Attention ranks run MoE routing, submit activations with CAM async
dispatch-send, and receive combined expert output with combine-recv. FFN ranks
receive routed-expert activations with dispatch-recv, execute their
local experts, and return the results with combine-send.

The connector creates one HCCL world ordered as
``[A0, A1, ..., F0, F1, ...]``. Attention world ranks therefore equal their
role ranks, while FFN world ranks start at ``num_attention_ranks``. Unlike the
synchronous connectors, CAM async carries routing and token-count metadata in
the CAM operator payload and does not create a separate Gloo DP-metadata
control plane.

The supported deployment requires ``async=true``, Ascend CAM
operator packages, and matching topology/configuration on every rank. Regular
prefill and autoregressive decode steps use the same connector. Attention
microbatching uses the native DBO runtime with Ascend adaptations. FFN FULL
requires MRV2 and accepts Attention-side DBO. Native DBO owns microbatch
splitting and scheduling. See ``docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`` for
configuration, rank
derivation, launch guidance, and the full limitations.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import torch
from torch import Tensor
from vllm.logger import init_logger

from afd_plugin.async_cam_startup import (
    GRAPH_STARTUP_TIMEOUT_SECONDS,
    POLL_SECONDS,
    AsyncCAMGraphStartup,
)
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
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
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

AFD_ASYNC_CAM_GROUP_NAME = "afd_async_cam"
CAM_COMM_ID = 0
ATTN_RANKS_PER_DP_CONFIG_KEY = "attn_ranks_per_dp"

_AFD_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "dynamicQuant",
        "attn_ranks_per_dp",
        "hccl_buffer_size",
        "ffn_two_stream",
        "ffn_compute_aic_num",
        "ffn_compute_aiv_num",
        "ffn_combine_aiv_num",
    },
)

logger = init_logger(__name__)


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
    """

    dynamic_quant: int = 0
    attn_ranks_per_dp: int = 1
    hccl_buffer_size: int | None = None
    ffn_two_stream: bool = False
    ffn_compute_aic_num: int = 12
    ffn_compute_aiv_num: int = 24
    ffn_combine_aiv_num: int = 24

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

        return cls(
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
            ffn_two_stream=coerce_extra_bool(
                raw.get("ffn_two_stream", False), field_name="ffn_two_stream"
            ),
            ffn_compute_aic_num=coerce_extra_positive_int(
                raw.get("ffn_compute_aic_num", 12), field_name="ffn_compute_aic_num"
            ),
            ffn_compute_aiv_num=coerce_extra_positive_int(
                raw.get("ffn_compute_aiv_num", 24), field_name="ffn_compute_aiv_num"
            ),
            ffn_combine_aiv_num=coerce_extra_positive_int(
                raw.get("ffn_combine_aiv_num", 24), field_name="ffn_combine_aiv_num"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dynamicQuant": self.dynamic_quant,
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
        }
        if self.hccl_buffer_size is not None:
            result["hccl_buffer_size"] = self.hccl_buffer_size
        if self.ffn_two_stream or (
            self.ffn_compute_aic_num,
            self.ffn_compute_aiv_num,
            self.ffn_combine_aiv_num,
        ) != (12, 24, 24):
            result.update(
                ffn_two_stream=self.ffn_two_stream,
                ffn_compute_aic_num=self.ffn_compute_aic_num,
                ffn_compute_aiv_num=self.ffn_compute_aiv_num,
                ffn_combine_aiv_num=self.ffn_combine_aiv_num,
            )
        return result


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


@dataclass(slots=True)
class AFDAsyncFFNWorkItem:
    """Normalized FFN-side work item produced by async CAM dispatch recv."""

    hidden_states: Tensor
    context: AFDTransferContext
    recv_output: AFDA2FTransferPayload
    layer_idx: int
    stage_idx: int
    num_tokens: int
    total_num_tokens: int


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

    layered_gmm_enabled: bool = False

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
        self.comm_id = CAM_COMM_ID
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
        self.comm_args: Tensor | None = None
        self._placeholder: Tensor | None = None
        self._pending_attention_payloads: dict[
            int,
            list[tuple[AFDTransferContext, Tensor, Tensor]],
        ] = {}
        self.graph_startup = AsyncCAMGraphStartup(
            self.world_rank, self.attn_size, self.topology.world_size
        )

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
        self.cam_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.topology.world_size,
            rank=self.world_rank,
            group_name=AFD_ASYNC_CAM_GROUP_NAME,
            timeout=timedelta(minutes=30),
            pg_options=create_hccl_process_group_options(
                self.hccl_buffer_size_mb,
            ),
            store_callback=self._initialize_graph_startup,
        )
        backend = self.cam_pg._get_backend(torch.device("npu"))
        self.group_name = str(backend.get_hccl_comm_name(self.world_rank))
        device = f"npu:{self.local_rank}"
        self.comm_args = torch.empty((1,), dtype=torch.float16, device=device)
        self._placeholder = torch.empty(
            (1,),
            dtype=self.activation_dtype,
            device=device,
        )
        self._initialized = True
        if self.graph_startup.mode == "full" and self.world_rank < self.attn_size:
            self._run_attention_graph_startup()

    def _initialize_graph_startup(self, store: Store) -> None:
        self.graph_startup.validate_attention_streams = (
            self._validate_two_stream_attention
        )
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
        self.graph_startup.initialize(store, signature)
        if self.graph_startup.mode == "full":
            try:
                if not self.vllm_config.use_v2_model_runner:
                    raise ValueError("Async FFN FULL requires MRV2 on every rank")
                validate_graph_capacity(
                    self.max_num_batched_tokens, self.tp_size, cam_receive_capacity()
                )
                self.graph_startup.phase("validated")
            except Exception as exc:
                self.graph_startup.fail(exc)
                raise

    def _validate_two_stream_attention(self) -> None:
        config = self.vllm_config
        parallel = config.parallel_config
        if (
            not config.use_v2_model_runner
            or not config.model_config.enforce_eager
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
            or os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower()
            in ("1", "true")
            or config.speculative_config is not None
        ):
            raise ValueError(
                "FFN two-stream requires eager MRV2 Attention without PP/CP/SP"
            )

    def wait_graph_device(self, event: torch.npu.Event) -> None:
        deadline = time.monotonic() + GRAPH_STARTUP_TIMEOUT_SECONDS
        while not event.query():
            self.graph_startup.check_failure()
            if time.monotonic() >= deadline:
                raise TimeoutError("Async graph warmup timed out; retain resources")
            time.sleep(POLL_SECONDS)

    def graph_startup_chunks(self) -> int:
        """Every A TP rank sends one row to experts [0, topk)."""
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

    def close(self) -> None:
        """Destroy the HCCL process group and clear pending transfer states."""
        if self.cam_pg is not None:
            import torch.distributed as dist

            dist.destroy_process_group(self.cam_pg)
        self.cam_pg = None
        self.comm_args = None
        self._placeholder = None
        self._pending_attention_payloads.clear()
        self._initialized = False

    def select_experts(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Run the target vLLM-Ascend router on Attention."""
        return select_cam_experts(**kwargs)

    def recv_ffn_work_item(
        self,
        *,
        stage_idx: int,
        max_num_tokens: int,
    ) -> AFDAsyncFFNWorkItem:
        """Receive and normalize one connector-driven FFN dispatch item.

        CAM metadata supplies the actual layer and routed token counts;
        returned tensors are sliced from operator capacity to those counts.
        """
        recv_output = self.recv_attn_output(
            stage_idx=stage_idx,
            layer_idx=0,
            batch_size=max(1, self.max_num_batched_tokens or max_num_tokens),
            ubatch_idx=stage_idx,
        )
        context = recv_output.context
        metadata = context.metadata
        states = _require_async_transfer_state(context)
        token_nums_rankid_layeridx = states.token_nums_rankid_layeridx
        if token_nums_rankid_layeridx is None:
            raise RuntimeError(
                "AFD async CAM FFN work item requires "
                "TokenNums_Rankid_Layeridx from async_dispatch_recv",
            )
        expert_token_nums = states.group_list
        assert expert_token_nums is not None
        # Preserve the full compact metadata for combine-send. Its total is
        # rank-wide, while counts describe only this receive chunk.
        control = torch.stack(
            (
                token_nums_rankid_layeridx[0],
                token_nums_rankid_layeridx[2],
                expert_token_nums.sum(dtype=torch.int64),
            )
        )
        total_num_tokens, layer_idx, num_tokens = control.cpu().tolist()
        if not 0 <= num_tokens <= recv_output.hidden_states.shape[0]:
            raise RuntimeError(
                "AFD async CAM routed chunk count is outside dispatch-recv capacity"
            )

        metadata.layer_idx = layer_idx
        metadata.stage_idx = stage_idx
        metadata.seq_lens = [num_tokens]

        hidden_states = recv_output.hidden_states[:num_tokens]
        if states.dynamic_scales is not None:
            states.dynamic_scales = states.dynamic_scales[:num_tokens]

        return AFDAsyncFFNWorkItem(
            hidden_states=hidden_states,
            context=context,
            recv_output=recv_output,
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            num_tokens=num_tokens,
            total_num_tokens=total_num_tokens,
        )

    def _send_ffn_output_payload(
        self,
        ffn_output: Tensor | AFDF2ATransferPayload,
        context: AFDTransferContext,
        *,
        stage_idx: int,
    ) -> None:
        if not isinstance(ffn_output, AFDF2ATransferPayload):
            self.send_ffn_output(
                ffn_output,
                context,
                ubatch_idx=stage_idx,
            )
            return

        self.send_ffn_output(
            ffn_output.routed_output,
            context,
            ubatch_idx=stage_idx,
        )

    def send_ffn_work_item_output(
        self,
        work_item: AFDAsyncFFNWorkItem,
        ffn_output: Tensor | AFDF2ATransferPayload,
    ) -> Tensor | AFDF2ATransferPayload:
        """Return one FFN work item's routed output through CAM.

        A one-token floating routed placeholder is used when a rank receives no
        routed tokens because CAM combine-send cannot consume the dynamic
        quantized dispatch buffer as an empty routed result.
        """
        if work_item.num_tokens > 0:
            self._send_ffn_output_payload(
                ffn_output,
                work_item.context,
                stage_idx=work_item.stage_idx,
            )
            return ffn_output

        # Temporary NPU workaround: CAM combine-send does not accept the int8
        # dispatch-recv buffer as routed output. When this FFN rank has no
        # routed tokens, send one fake floating routed token instead of reusing the
        # dynamicQuant int8 input buffer.
        fake_routed_output = torch.zeros(
            (1, work_item.recv_output.hidden_states.shape[-1]),
            dtype=self.activation_dtype,
            device=work_item.recv_output.hidden_states.device,
        )
        if isinstance(ffn_output, AFDF2ATransferPayload):
            ffn_output = AFDF2ATransferPayload(
                routed_output=fake_routed_output,
            )
        else:
            ffn_output = fake_routed_output
        self._send_ffn_output_payload(
            ffn_output,
            work_item.context,
            stage_idx=work_item.stage_idx,
        )
        return ffn_output

    def send_attn_output(
        self,
        hidden_states: Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Dispatch Attention activations and top-k routing to FFN ranks.

        The matching context and routing tensors are queued per async stage
        for ``recv_ffn_output``. The input token count and top-k tensor shapes
        must match the transfer metadata and model configuration.
        """
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

        _log_cam_op_values(
            "async_dispatch_send",
            "inputs",
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            max_seq_len=self.max_num_batched_tokens,
            batch_size=states.batch_size,
            hidden_size=states.hidden_size,
            topk=states.topk,
            ffn_size=self.ffn_size,
            attn_size=self.attn_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            layer_idx=states.layer_idx,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
            group_name=self.group_name,
        )
        torch.ops.afd_ascend.afd_async_dispatch_send(
            hidden_states,
            topk_ids,
            self.comm_args,
            self.comm_id,
            self.max_num_batched_tokens,
            states.batch_size,
            states.hidden_size,
            states.topk,
            self.ffn_size,
            self.attn_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            states.layer_idx,
            self.tp_size,
            self.dynamic_quant,
            self.group_name,
        )
        self._pending_attention_payloads.setdefault(
            context.metadata.stage_idx, []
        ).append(
            (context, topk_ids, topk_weights),
        )

        return None

    def recv_ffn_output(
        self,
        ref_tensor: Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        """Receive and combine expert output on an Attention rank.

        The optional context must match the oldest pending dispatch for the
        stage. Routing tensors always come from that dispatch. ``ref_tensor``
        selects the output device and dtype for the CAM receive placeholder.

        Args:
            ref_tensor: Tensor from which device/dtype are taken for the CAM
                combine-recv placeholder.
            ubatch_idx: Stage/microbatch index of the pending Attention
                dispatch payload. Defaults to ``0``.
            **kwargs: Optional ``context`` from the matching dispatch. Uses
                FIFO order when omitted.
        """
        self._require_initialized()
        context = kwargs.get("context")
        payloads = self._pending_attention_payloads.get(ubatch_idx)
        if not payloads:
            raise RuntimeError(
                "CAMAsyncAFDConnector recv_ffn_output is missing pending "
                "Attention metadata",
            )
        if context is not None and context is not payloads[0][0]:
            raise RuntimeError(
                "CAMAsyncAFDConnector recv_ffn_output context does not match "
                "the next pending Attention dispatch",
            )
        context, topk_ids, topk_weights = payloads.pop(0)
        if not payloads:
            self._pending_attention_payloads.pop(ubatch_idx, None)

        states = _require_async_transfer_state(context)
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=states.batch_size,
            topk=states.topk,
        )
        placeholder = ref_tensor.new_empty((1,))
        _log_cam_op_values(
            "async_combine_recv",
            "inputs",
            placeholder=placeholder,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            batch_size=states.batch_size,
            hidden_size=states.hidden_size,
            topk=states.topk,
            ffn_size=self.ffn_size,
            attn_size=self.attn_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            group_name=self.group_name,
        )
        output = torch.ops.afd_ascend.afd_async_combine_recv(
            placeholder,
            topk_ids,
            topk_weights,
            self.comm_args,
            self.comm_id,
            states.batch_size,
            states.hidden_size,
            states.topk,
            self.ffn_size,
            self.attn_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.group_name,
        )
        _log_cam_op_values("async_combine_recv", "outputs", output=output)
        return output

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive routed activations and preserve their compact metadata."""
        self._require_initialized()
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
        placeholder = kwargs.get("placeholder", self._placeholder)
        _log_cam_op_values(
            "async_dispatch_recv",
            "inputs",
            placeholder=placeholder,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            max_seq_len=self.max_num_batched_tokens,
            batch_size=states.batch_size,
            hidden_size=states.hidden_size,
            topk=states.topk,
            ffn_size=self.ffn_size,
            attn_size=self.attn_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            tp_size=self.tp_size,
            dynamic_quant=self.dynamic_quant,
            group_name=self.group_name,
        )
        outputs = torch.ops.afd_ascend.afd_async_dispatch_recv(
            placeholder,
            self.comm_args,
            self.comm_id,
            self.max_num_batched_tokens,
            states.hidden_size,
            states.topk,
            self.ffn_size,
            self.attn_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.tp_size,
            self.dynamic_quant,
            self.group_name,
        )
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
        """Send routed expert outputs for CAM combination.

        ``TokenNums_Rankid_Layeridx`` from the matching dispatch-recv is
        mandatory because CAM uses it to return results to Attention ranks.
        """
        self._require_initialized()
        states = _require_async_transfer_state(context)
        token_nums_rankid_layeridx = states.token_nums_rankid_layeridx
        if token_nums_rankid_layeridx is None:
            token_nums_rankid_layeridx = kwargs.get("token_nums_rankid_layeridx")
        if token_nums_rankid_layeridx is None:
            raise RuntimeError(
                "AFD async CAM combine send requires "
                "TokenNums_Rankid_Layeridx from async_dispatch_recv",
            )
        _log_cam_op_values(
            "async_combine_send",
            "inputs",
            ffn_output=ffn_output,
            comm_args=self.comm_args,
            token_nums_rankid_layeridx=token_nums_rankid_layeridx,
            metadata_values=not self.layered_gmm_enabled,
            comm_id=self.comm_id,
            max_seq_len=self.max_num_batched_tokens,
            batch_size=states.batch_size,
            hidden_size=states.hidden_size,
            topk=states.topk,
            ffn_size=self.ffn_size,
            attn_size=self.attn_size,
            expert_per_rank=self.expert_per_rank,
            rank=self.world_rank,
            world_size=self.topology.world_size,
            tp_size=self.tp_size,
            group_name=self.group_name,
        )
        states.send_completion = torch.ops.afd_ascend.afd_async_combine_send(
            ffn_output,
            self.comm_args,
            token_nums_rankid_layeridx,
            self.comm_id,
            self.max_num_batched_tokens,
            states.hidden_size,
            states.topk,
            self.ffn_size,
            self.attn_size,
            self.expert_per_rank,
            self.world_rank,
            self.topology.world_size,
            self.tp_size,
            self.group_name,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CAMAsyncAFDConnector is not initialized")


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


def _log_cam_op_values(
    op_name: str, label: str, *, metadata_values: bool = True, **kwargs: object
) -> None:
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
            if metadata_values and name == "token_nums_rankid_layeridx":
                try:
                    first5: object = value.detach().flatten()[:5].cpu().tolist()
                except Exception as exc:  # pragma: no cover - defensive logging helper
                    first5 = f"<unavailable: {type(exc).__name__}>"
                description = f"{description}, first5={first5!r}"
        else:
            description = repr(value)
        lines.append(f"  {name}={description}")
    logger.warning("AFD CAM %s %s:\n%s", op_name, label, "\n".join(lines))


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


def _validate_topk_payload(
    topk_ids: Tensor,
    topk_weights: Tensor | None,
    *,
    batch_size: int,
    topk: int,
    require_weights: bool = True,
) -> None:
    if tuple(topk_ids.shape) != (batch_size, topk):
        raise ValueError(
            f"topk_ids shape must match ({batch_size}, {topk}), got {topk_ids.shape!r}",
        )
    if not require_weights and topk_weights is None:
        return
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
    "AFDAsyncFFNWorkItem",
    "AFDAsyncTopology",
    "ATTN_RANKS_PER_DP_CONFIG_KEY",
    "CAM_COMM_ID",
    "build_async_topology",
    "select_cam_experts",
]
