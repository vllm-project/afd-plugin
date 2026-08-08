# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend CAM asynchronous connector for Attention/FFN disaggregation.

``CAMAsyncAFDConnector`` is the eager-only Ascend prefill data path. Attention
ranks run MoE routing, submit activations with CAM async dispatch-send, and
receive combined expert output with combine-recv. FFN ranks receive routed and
shared-expert activations with dispatch-recv, execute their local experts, and
return the results with combine-send.

The connector creates one HCCL world ordered as
``[A0, A1, ..., F0, F1, ...]``. Attention world ranks therefore equal their
role ranks, while FFN world ranks start at ``num_attention_ranks``. Unlike the
synchronous connectors, CAM async carries routing and token-count metadata in
the CAM operator payload and does not create a separate Gloo DP-metadata
control plane.

The supported deployment requires ``async=true``, eager execution, Ascend CAM
operator packages, and matching topology/configuration on every rank. vLLM
native DBO, ACL graph execution, and decode are not supported.
Optional AFD-managed MoE ubatching is a separate two-stage pipeline using
request boundaries or token-balanced stages for DP+TP/SP. See
``docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`` for configuration, rank
derivation, launch guidance, and the full limitations.
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

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
    coerce_extra_str,
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
from afd_plugin.distributed import init_afd_process_group

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup
    from vllm.config import VllmConfig

AFD_ASYNC_CAM_GROUP_NAME = "afd_async_cam"
CAM_COMM_ID = 0
ATTN_RANKS_PER_DP_CONFIG_KEY = "attn_ranks_per_dp"
ASYNC_MOE_NUM_STAGES = 2
ASYNC_MOE_REQUEST_SPLIT = "request"
ASYNC_MOE_TOKEN_SPLIT = "token"

_AFD_ASYNC_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "dynamicQuant",
        "attn_ranks_per_dp",
        "async_moe_ubatching",
        "async_moe_num_ubatches",
        "async_moe_split",
    },
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class AFDAsyncExtraInfo(ConnectorExtraInfo):
    """Typed async CAM connector configuration.

    Attributes:
        dynamic_quant: Dynamic quantization mode accepted by CAM operators.
        attn_ranks_per_dp: Number of Attention ranks in each data-parallel group.
        async_moe_ubatching: Whether two-stage async MoE ubatching is used.
        async_moe_num_ubatches: Number of stages used by async MoE ubatching.
        async_moe_split: ``"request"`` for request boundaries or ``"token"``
            for token-balanced DP+TP/SP stages.
    """

    dynamic_quant: int = 0
    attn_ranks_per_dp: int = 1
    async_moe_ubatching: bool = False
    async_moe_num_ubatches: int = ASYNC_MOE_NUM_STAGES
    async_moe_split: str = ASYNC_MOE_REQUEST_SPLIT

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
            async_moe_ubatching=coerce_extra_bool(
                raw.get("async_moe_ubatching", False),
                field_name="async_moe_ubatching",
            ),
            async_moe_num_ubatches=coerce_extra_positive_int(
                raw.get("async_moe_num_ubatches", ASYNC_MOE_NUM_STAGES),
                field_name="async_moe_num_ubatches",
            ),
            async_moe_split=coerce_extra_str(
                raw.get("async_moe_split", ASYNC_MOE_REQUEST_SPLIT),
                field_name="async_moe_split",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dynamicQuant": self.dynamic_quant,
            "attn_ranks_per_dp": self.attn_ranks_per_dp,
            "async_moe_ubatching": self.async_moe_ubatching,
            "async_moe_num_ubatches": self.async_moe_num_ubatches,
            "async_moe_split": self.async_moe_split,
        }


@dataclass(slots=True)
class AFDAsyncTransferState(AFDTransferState):
    """CAM-side transfer state carried from dispatch recv to combine send.

    ``batch_size``, ``hidden_size``, ``topk`` and ``layer_idx`` size the CAM
    operators, while ``token_nums_rankid_layeridx`` and
    ``expert_token_nums_shared`` are the dispatch-recv outputs the matching send
    and FFN token accounting need. ``group_list``, ``dynamic_scales``,
    ``expand_x_shared`` and ``dynamic_scales_shared`` are the routed/shared MoE
    compute payloads the FFN model runner feeds into ``compute_ffn_output``.
    """

    batch_size: int = 1
    hidden_size: int = 1
    topk: int = 1
    layer_idx: int = 0
    token_nums_rankid_layeridx: Tensor | None = None
    expert_token_nums_shared: Tensor | None = None
    group_list: Tensor | None = None
    dynamic_scales: Tensor | None = None
    expand_x_shared: Tensor | None = None
    dynamic_scales_shared: Tensor | None = None


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
    shared_num_tokens: int


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
        self.topk = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        self.dynamic_quant = self.extra_info.dynamic_quant
        self.group_name = ""
        self.max_seq_len = vllm_config.scheduler_config.max_num_batched_tokens
        self.comm_id = CAM_COMM_ID
        self.tp_size = self.extra_info.attn_ranks_per_dp
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
        )
        backend = self.cam_pg._get_backend(torch.device("npu"))
        self.group_name = str(backend.get_hccl_comm_name(self.world_rank))
        device = f"npu:{self.local_rank}"
        self.comm_args = torch.empty((1,), dtype=torch.float16, device=device)
        self._placeholder = torch.empty(
            (1,),
            dtype=torch.bfloat16,
            device=device,
        )
        self._initialized = True

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
        """Run the pinned vLLM-Ascend expert selector on Attention."""
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        return select_experts(**kwargs)

    def recv_ffn_work_item(
        self,
        *,
        stage_idx: int,
        max_num_tokens: int,
    ) -> AFDAsyncFFNWorkItem:
        """Receive and normalize one connector-driven FFN dispatch item.

        CAM metadata supplies the actual layer and routed/shared token counts;
        returned tensors are sliced from operator capacity to those counts.
        """
        recv_output = self.recv_attn_output(
            stage_idx=stage_idx,
            layer_idx=0,
            batch_size=max(1, self.max_seq_len or max_num_tokens),
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
        total_num_tokens = max(1, int(token_nums_rankid_layeridx[0].item()))
        layer_idx = int(token_nums_rankid_layeridx[2].item())

        expert_token_nums_shared = states.expert_token_nums_shared
        if expert_token_nums_shared is None:
            raise RuntimeError(
                "AFD async CAM FFN work item requires "
                "expert_token_nums_shared from async_dispatch_recv",
            )
        shared_num_tokens = max(0, int(expert_token_nums_shared[0].item()))

        expert_token_nums = states.group_list
        if expert_token_nums is None:
            raise RuntimeError(
                "AFD async CAM FFN work item requires expert_token_nums "
                "from async_dispatch_recv",
            )
        num_tokens = max(
            0,
            int(expert_token_nums.to(torch.int64).sum().item()),
        )

        metadata.layer_idx = layer_idx
        metadata.stage_idx = stage_idx
        metadata.seq_lens = [num_tokens]

        hidden_states = recv_output.hidden_states[:num_tokens]
        if states.expand_x_shared is not None:
            states.expand_x_shared = states.expand_x_shared[:shared_num_tokens]
        if states.dynamic_scales is not None:
            states.dynamic_scales = states.dynamic_scales[:num_tokens]
        if states.dynamic_scales_shared is not None:
            states.dynamic_scales_shared = states.dynamic_scales_shared[
                :shared_num_tokens
            ]

        return AFDAsyncFFNWorkItem(
            hidden_states=hidden_states,
            context=context,
            recv_output=recv_output,
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            num_tokens=num_tokens,
            total_num_tokens=total_num_tokens,
            shared_num_tokens=shared_num_tokens,
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

        kwargs: dict[str, object] = {"ubatch_idx": stage_idx}
        if ffn_output.shared_output is not None:
            kwargs["expand_x_shared"] = ffn_output.shared_output
        self.send_ffn_output(
            ffn_output.routed_output,
            context,
            **kwargs,
        )

    def send_ffn_work_item_output(
        self,
        work_item: AFDAsyncFFNWorkItem,
        ffn_output: Tensor | AFDF2ATransferPayload,
    ) -> Tensor | AFDF2ATransferPayload:
        """Return one FFN work item's routed/shared outputs through CAM.

        A one-token BF16 routed placeholder is used when a rank receives no
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
        # routed tokens, send one fake bf16 routed token instead of reusing the
        # dynamicQuant int8 input buffer.
        fake_routed_output = torch.zeros(
            (1, work_item.recv_output.hidden_states.shape[-1]),
            dtype=torch.bfloat16,
            device=work_item.recv_output.hidden_states.device,
        )
        if isinstance(ffn_output, AFDF2ATransferPayload):
            ffn_output = AFDF2ATransferPayload(
                routed_output=fake_routed_output,
                shared_output=ffn_output.shared_output,
            )
        else:
            ffn_output = fake_routed_output
        work_item.context.metadata.seq_lens = [1]
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
        _validate_topk_payload(
            topk_ids,
            topk_weights,
            batch_size=states.batch_size,
            topk=states.topk,
        )
        self._pending_attention_payloads.setdefault(
            context.metadata.stage_idx, []
        ).append(
            (context, topk_ids, topk_weights),
        )

        _log_cam_op_values(
            "async_dispatch_send",
            "inputs",
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            comm_args=self.comm_args,
            comm_id=self.comm_id,
            max_seq_len=self.max_seq_len,
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
        torch.ops.umdk_cam_op_lib.async_dispatch_send(
            hidden_states,
            topk_ids,
            self.comm_args,
            self.comm_id,
            self.max_seq_len,
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
        return None

    def recv_ffn_output(
        self,
        ref_tensor: Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        """Receive and combine expert output on an Attention rank.

        Routing tensors may be supplied explicitly via ``kwargs`` or recovered
        FIFO from the matching stage's pending dispatch. ``ref_tensor``
        selects the output device and dtype used to construct the CAM receive
        placeholder.

        Args:
            ref_tensor: Tensor from which device/dtype are taken for the CAM
                combine-recv placeholder.
            ubatch_idx: Stage/microbatch index used to pop the pending
                Attention dispatch payload when routing tensors are not
                supplied explicitly. Defaults to ``0``.
            **kwargs: Optional CAM-specific arguments:

                * ``context``: ``AFDTransferContext`` from the matching
                  dispatch. Recovered from the pending queue when omitted.
                * ``topk_ids``: Expert routing indices. Recovered from the
                  pending queue when omitted.
                * ``topk_weights``: Expert routing weights. Recovered from
                  the pending queue when omitted.
        """
        self._require_initialized()
        context = kwargs.get("context")
        topk_ids = kwargs.get("topk_ids")
        topk_weights = kwargs.get("topk_weights")
        if context is None or topk_ids is None or topk_weights is None:
            payloads = self._pending_attention_payloads.get(ubatch_idx)
            if not payloads:
                raise RuntimeError(
                    "CAMAsyncAFDConnector recv_ffn_output is missing pending "
                    "Attention metadata",
                )
            (
                pending_context,
                pending_topk_ids,
                pending_topk_weights,
            ) = payloads.pop(0)
            if not payloads:
                self._pending_attention_payloads.pop(ubatch_idx, None)
            if context is None:
                context = pending_context
            if topk_ids is None:
                topk_ids = pending_topk_ids
            if topk_weights is None:
                topk_weights = pending_topk_weights

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
        output = torch.ops.umdk_cam_op_lib.async_combine_recv(
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
        """Receive CAM-dispatched routed/shared activations on an FFN rank.

        The returned payload preserves dynamic-quant scales, per-expert token
        counts, shared-expert activations, and CAM token/rank/layer metadata so
        local expert execution and the subsequent combine-send use the same
        routing contract.
        """
        self._require_initialized()
        batch_size = int(kwargs.get("batch_size", self.max_seq_len) or 1)
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
        outputs = torch.ops.umdk_cam_op_lib.async_dispatch_recv(
            placeholder,
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
            self.tp_size,
            self.dynamic_quant,
            self.group_name,
        )
        (
            hidden_states,
            expand_x_shared,
            dynamic_scales,
            dynamic_scales_shared,
            token_nums_rankid_layeridx,
            expert_token_nums,
            expert_token_nums_shared,
        ) = outputs
        _log_cam_op_values(
            "async_dispatch_recv",
            "outputs",
            hidden_states=hidden_states,
            expand_x_shared=expand_x_shared,
            dynamic_scales=dynamic_scales,
            dynamic_scales_shared=dynamic_scales_shared,
            token_nums_rankid_layeridx=token_nums_rankid_layeridx,
            expert_token_nums=expert_token_nums,
            expert_token_nums_shared=expert_token_nums_shared,
        )
        states.token_nums_rankid_layeridx = token_nums_rankid_layeridx
        states.expert_token_nums_shared = expert_token_nums_shared
        states.group_list = expert_token_nums
        states.dynamic_scales = dynamic_scales
        states.expand_x_shared = expand_x_shared
        states.dynamic_scales_shared = dynamic_scales_shared
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
        """Send routed and optional shared-expert outputs for CAM combination.

        ``TokenNums_Rankid_Layeridx`` from the matching dispatch-recv is
        mandatory because CAM uses it to return results to Attention ranks.
        """
        self._require_initialized()
        states = _require_async_transfer_state(context)
        expand_x_shared = kwargs.get("expand_x_shared")
        if expand_x_shared is None:
            expand_x_shared = ffn_output
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
            expand_x_shared=expand_x_shared,
            comm_args=self.comm_args,
            token_nums_rankid_layeridx=token_nums_rankid_layeridx,
            comm_id=self.comm_id,
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
        torch.ops.umdk_cam_op_lib.async_combine_send(
            ffn_output,
            expand_x_shared,
            self.comm_args,
            token_nums_rankid_layeridx,
            self.comm_id,
            states.batch_size,
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
            description = f"Tensor(dtype={value.dtype}, shape={tuple(value.shape)})"
            if name == "token_nums_rankid_layeridx":
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
    """Validate role-local rank settings and derive the CAM HCCL world rank.

    The world is Attention-first: Attention role rank ``i`` maps to world rank
    ``i`` and FFN role rank ``j`` maps to
    ``num_attention_ranks + j``. Routed experts are distributed across FFN
    ranks using a ceiling division; production model layouts should keep the
    routed-expert count divisible by the FFN rank count.
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

    expert_count = num_routed_experts or 1
    expert_per_rank = (expert_count + ffn_size - 1) // ffn_size
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
    "ASYNC_MOE_NUM_STAGES",
    "ASYNC_MOE_REQUEST_SPLIT",
    "ASYNC_MOE_TOKEN_SPLIT",
    "CAM_COMM_ID",
    "build_async_topology",
]
