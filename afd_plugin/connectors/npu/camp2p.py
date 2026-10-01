# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous CAMP2p connector for Attention-FFN Disaggregation on NPU.

``CAMP2pAFDConnector`` exchanges hidden states and FFN outputs through Ascend
CAMP2p custom operators backed by HCCL. A separate Gloo group carries DP
metadata from Attention to FFN so each FFN rank can determine the tensor size
for its mapped Attention ranks.

The data path supports eager execution and ``FULL_DECODE_ONLY`` ACL graphs.

See ``docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md`` for configuration and launch
examples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.config import CUDAGraphMode
from vllm.forward_context import DPMetadata, get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.a2e_layout import (
    attention_rank_token_counts,
    ffn_rank_for_attention_rank,
    ffn_receive_rows,
    padded_tile_rows,
)
from afd_plugin.compat.npu import ensure_cam_p2p_ops_available
from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
    coerce_optional_extra_positive_int,
)
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
from afd_plugin.distributed import (
    create_hccl_process_group_options,
    init_afd_process_group,
    topology_from_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_CAMP2P_CUSTOM_OPS_REGISTERED = False

# Padding value for token ids that only exist to fill a padded transfer. The
# receiving FFN maps it to token 0 before routing, so a padded row can never
# index outside the token-to-expert table.
_PAD_HASH_TOKEN_ID: Final[int] = -1

_CAMP2P_EXTRA_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "core_num",
        "attn_core_num",
        "ffn_core_num",
        "compute_gate_on_attention",
        "hccl_buffer_size",
        "quant_mode",
    },
)


@dataclass(frozen=True)
class CAMP2PExtraInfo(ConnectorExtraInfo):
    """Typed CAMP2P connector configuration.

    Attributes:
        core_num: Default number of AIV cores used by each AFD role.
        attn_core_num: Optional Attention-role override for ``core_num``.
        ffn_core_num: Optional FFN-role override for ``core_num``.
        compute_gate_on_attention: Whether Attention computes MoE gate outputs.
        hccl_buffer_size: Optional buffer size in MB for CAMP2P HCCL domains.
        quant_mode: CAM quantization mode; the current runtime supports only 0.
    """

    core_num: int = 8
    attn_core_num: int | None = None
    ffn_core_num: int | None = None
    compute_gate_on_attention: bool = False
    hccl_buffer_size: int | None = None
    quant_mode: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> CAMP2PExtraInfo:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{cls.__name__} connector_extra_config must be a mapping, "
                f"got {type(raw).__name__}",
            )
        unknown = sorted(
            str(key) for key in raw if key not in _CAMP2P_EXTRA_CONFIG_FIELDS
        )
        if unknown:
            raise ValueError(
                "unknown CAMP2P connector_extra_config field(s): " + ", ".join(unknown),
            )

        return cls(
            core_num=coerce_extra_positive_int(
                raw.get("core_num", 8),
                field_name="core_num",
            ),
            attn_core_num=coerce_optional_extra_positive_int(
                raw.get("attn_core_num"),
                field_name="attn_core_num",
            ),
            ffn_core_num=coerce_optional_extra_positive_int(
                raw.get("ffn_core_num"),
                field_name="ffn_core_num",
            ),
            compute_gate_on_attention=coerce_extra_bool(
                raw.get("compute_gate_on_attention", False),
                field_name="compute_gate_on_attention",
            ),
            hccl_buffer_size=coerce_optional_extra_positive_int(
                raw.get("hccl_buffer_size"),
                field_name="hccl_buffer_size",
            ),
            quant_mode=coerce_extra_int(
                raw.get("quant_mode", 0),
                field_name="quant_mode",
            ),
        )

    def aiv_num_for_role(self, role: str) -> int:
        if role == "attention" and self.attn_core_num is not None:
            return self.attn_core_num
        if role == "ffn" and self.ffn_core_num is not None:
            return self.ffn_core_num
        return self.core_num

    def validate_supported(self) -> None:
        if self.compute_gate_on_attention:
            raise RuntimeError(
                "AFD NPU runtime does not support compute_gate_on_attention=true yet",
            )
        if self.quant_mode != 0:
            raise RuntimeError("AFD NPU runtime currently supports only quant_mode=0")

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "core_num": self.core_num,
            "compute_gate_on_attention": self.compute_gate_on_attention,
            "quant_mode": self.quant_mode,
        }
        if self.attn_core_num is not None:
            result["attn_core_num"] = self.attn_core_num
        if self.ffn_core_num is not None:
            result["ffn_core_num"] = self.ffn_core_num
        if self.hccl_buffer_size is not None:
            result["hccl_buffer_size"] = self.hccl_buffer_size
        return result


@dataclass(slots=True)
class CAMP2PTransferState(AFDTransferState):
    """CAMP2P payload metadata carried between recv and send phases.

    This class stores what CAMP2p reads back itself while data travels from
    Attention to FFN and then back to Attention. ``aiv_num``, ``batch_size``,
    ``h`` and ``k`` size the CAMP2p operators, and ``atten_batch_size`` saves the
    A2E-returned Attention token count that the FFN-to-Attention send requires.
    ``x_active_mask`` and ``cam_p2p_ep_name`` are the A2E-returned active-token
    mask and HCCL endpoint name captured on the receive path. ``padded_payload``
    is set on the Attention side when the payload was extended to the reported
    token count, which makes the receive return only the rows the model produced.
    """

    aiv_num: int = 8
    batch_size: int = 0
    h: int = 0
    k: int = 1
    atten_batch_size: torch.Tensor | None = None
    x_active_mask: torch.Tensor | None = None
    cam_p2p_ep_name: str | None = None
    padded_payload: bool = False


@dataclass(frozen=True, slots=True)
class _CAMP2PTopology:
    """Describe where one connector process sits in the communication groups.

    ``world_rank`` is the process number in the complete AFD group.
    ``p2p_rank`` is its number in the smaller Gloo group used to exchange metadata.
    For an Attention rank, ``dp_metadata_destinations`` lists the FFN ranks
    that should receive its metadata.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    dp_metadata_destinations: tuple[int, ...]

    @property
    def p2p_world_size(self) -> int:
        """Return the number of FFN and participating Attention metadata ranks."""
        return self.ffn_size + self.min_size

    @property
    def participates_in_p2p_group(self) -> bool:
        """Return whether this process joins the Gloo DP-metadata group."""
        return self.world_rank < self.ffn_size or self.is_attn_top_min_size_rank

    @property
    def is_attn_top_min_size_rank(self) -> bool:
        """Return whether this is an Attention metadata-sender rank."""
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size


def _reported_attention_tokens(forward_context: Any) -> int | None:
    """Return the row count a padded graph step sends, or ``None``.

    Mirror the runner's own condition (``v1/worker/attention_model_runner.py``:
    ``_full_cudagraph_padded_tokens(forward_context) is not None and not
    ubatch_slices``): a padded graph runs the batch at its capture size, so the
    payload of such a step holds that many rows whatever the forward produced.

    ``forward_context.num_tokens`` is *not* a substitute: outside a padded graph it
    does not describe the payload of the current forward (a prefill step can carry
    hundreds of rows while it holds a decode-sized value).

    Returns ``None`` for a step that is not a padded graph, which is the step an
    eager forward produces its own rows for.
    """

    if getattr(forward_context, "ubatch_slices", None):
        return None
    if getattr(forward_context, "cudagraph_runtime_mode", None) != CUDAGraphMode.FULL:
        return None
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    if batch_descriptor is None:
        return None
    return max(1, int(batch_descriptor.num_tokens))


def prepare_token_id_transfer(
    input_ids: torch.Tensor,
    *,
    topk: int,
    expected_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack token ids and inert scales for the A2E ids channel.

    The operator transports an ``int32`` ids tensor and a ``float32`` scales
    tensor of shape ``(batch, topk)``. The id repeats across the columns, which
    is what the receiving side collapses back. The scales carry no routing
    weight: this channel moves token identity, not router output.

    Args:
        input_ids: Token-aligned ids for the local Attention tokens.
        topk: Number of routed experts per token; the operator's column count.
        expected_tokens: Token count the transfer metadata declares.

    Returns:
        The ``(ids, scales)`` pair to hand to the operator.

    Raises:
        ValueError: If the ids do not describe exactly ``expected_tokens``
            tokens, which is the alignment invariant for this channel.
    """

    num_tokens = int(input_ids.numel())
    if num_tokens != expected_tokens:
        raise ValueError(
            f"input_ids token count {num_tokens} does not match the AFD "
            f"transfer token count {expected_tokens}",
        )

    ids = (
        input_ids.reshape(-1)
        .to(dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, topk)
        .contiguous()
    )
    scales = torch.zeros(
        (num_tokens, topk),
        dtype=torch.float32,
        device=input_ids.device,
    )
    return ids, scales


def received_token_ids(
    sim_expert_ids: torch.Tensor,
    *,
    expected_tokens: int,
) -> torch.Tensor:
    """Collapse the A2E ids channel back to a token-aligned id vector.

    Every column of a row repeats that token's id. The operator sizes its ids
    output to the capacity it was given rather than to the rows it writes, so the
    shape alone cannot prove which rows arrived; the caller states the row count
    the FFN rank computes on, and the Attention side pads every peer up to the
    tile A2E reads so that row count is the one the transfer writes.

    Args:
        sim_expert_ids: The operator's ids output.
        expected_tokens: Rows the FFN rank computes on.

    Returns:
        A one-dimensional ``int32`` tensor of length ``expected_tokens``.

    Raises:
        ValueError: If the operator's capacity cannot hold those rows, which
            means the ids channel is not sized for this topology.
    """

    declared_tokens = int(sim_expert_ids.shape[0])
    if declared_tokens < expected_tokens:
        raise ValueError(
            f"A2E declared {declared_tokens} id rows but the FFN rank computes "
            f"on {expected_tokens}; the ids channel is not sized for this "
            "topology",
        )
    # The columns are replicas of the same id, so the first one carries it.
    return sim_expert_ids[:expected_tokens, 0].contiguous()


class CAMP2pAFDConnector(AFDConnectorBase):
    """Move model data between Attention and FFN workers on Ascend NPU.

    The connector owns HCCL process-group setup and CAMP2P custom-op transfers.
    Runtime validation rejects unsupported nonzero quantization modes and
    compute-gate-on-attention settings.

    DP metadata operations do not live on the connector itself: they are
    provided by the pluggable ``CAMP2pAFDControlPlane`` instance created at
    construction time and exposed as ``control_plane``. The connector still
    owns the ``p2p`` process group the control plane transmits over, because
    creating that group is part of the collective ``init_afd_connector``
    ordering.
    """

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> CAMP2PExtraInfo:
        return CAMP2PExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        """Read the configuration and prepare this connector's local state.

        This method calculates the process ranks and reads the model dimensions.
        It does not connect to the other Attention or FFN processes yet;
        :meth:`init_afd_connector` creates those connections later.

        Args:
            rank: Rank provided by the vLLM worker.
            local_rank: NPU device number used by this worker.
            vllm_config: Model, scheduler, and parallel configuration from vLLM.
            afd_config: AFD role, host, port, rank counts, and extra settings.
            role_rank: Runtime rank within the configured AFD role group.
        """
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        self.topology = build_camp2p_topology(afd_config, role_rank)
        self.world_rank = self.topology.world_rank
        self.p2p_rank = self.topology.p2p_rank
        self.attn_size = self.topology.attention_size
        self.ffn_size = self.topology.ffn_size
        self.min_size = self.topology.min_size
        self.ratio = self.attn_size // self.ffn_size
        self.dst_list = list(self.topology.dp_metadata_destinations)
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.afd_pg_list: list[ProcessGroup] = []
        self.afd_pg: ProcessGroup | None = None
        self.p2p_pg: ProcessGroup | None = None
        self.ffn_pg: ProcessGroup | None = None
        self.hccl_comm_name = ""
        self.hccl_comm_name2 = ""
        self.hccl_comm_name3 = ""
        self.hccl_comm_name1 = ""
        self.hccl_comm_name_list: list[str] = []
        extra_info = cast(CAMP2PExtraInfo, self.extra_info)
        self.aiv_num = extra_info.aiv_num_for_role(afd_config.role)
        self.hccl_buffer_size_mb = extra_info.hccl_buffer_size
        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = hf_config.hidden_size
        self.num_experts_per_tok = hf_config.num_experts_per_tok
        self.num_routed_experts = hf_config.n_routed_experts
        # Row count of the model's token-to-expert (Hash) tables: token ids the
        # ids channel carries are indices into them.
        self.vocab_size = int(hf_config.vocab_size)
        # All-rank token count A2E falls back to when a stage has no usable
        # counts. Both roles read it from the same scheduler configuration, so the
        # fallback tile stays equal on the sending and the receiving side.
        self.max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.control_plane = CAMP2pAFDControlPlane(self)
        # Per-stage, per-DP-rank token counts as plain integers. The control plane
        # refreshes them whenever it publishes a payload; see
        # ``_dp_stage_token_counts`` for why the tensors themselves are not read.
        self.dp_token_counts: dict[int, tuple[int, ...]] = {}

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` after all CAMP2p connections have been created."""
        return self._initialized

    def init_afd_connector(self) -> None:
        """Connect this process to the other Attention and FFN processes.

        The method creates one HCCL group for each batch or ubatch. These groups
        carry hidden states and FFN results. FFN processes also create a group
        for MoE communication. A smaller Gloo group carries token counts and
        other batch information from Attention to FFN.

        The method returns immediately if initialization already succeeded.
        Otherwise, it may wait until every required process joins.

        Raises:
            RuntimeError: If the CAMP2p operators are unavailable or a
                communication group cannot be created.
        """
        if self._initialized:
            return
        import torch_npu  # noqa: F401

        ensure_cam_p2p_ops_available()

        _register_camp2p_custom_ops()

        num_ubatches = max(1, self.vllm_config.parallel_config.num_ubatches)
        self.afd_pg_list = []
        self.hccl_comm_name_list = []
        for ubatch_idx in range(num_ubatches):
            group_name = "afd" if ubatch_idx == 0 else f"afd{ubatch_idx}"
            afd_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.world_rank,
                group_name=group_name,
                timeout=timedelta(minutes=30),
                pg_options=create_hccl_process_group_options(
                    self.hccl_buffer_size_mb,
                ),
            )
            self.afd_pg_list.append(afd_pg)
            backend = afd_pg._get_backend(torch.device("npu"))
            self.hccl_comm_name_list.append(
                str(backend.get_hccl_comm_name(self.world_rank)),
            )
        self.afd_pg = self.afd_pg_list[0]
        self.hccl_comm_name = self.hccl_comm_name_list[0]
        self.hccl_comm_name2 = (
            self.hccl_comm_name_list[1] if num_ubatches > 1 else self.hccl_comm_name
        )
        self.hccl_comm_name3 = self.hccl_comm_name_list[2] if num_ubatches > 2 else ""

        if self.afd_config.role == "ffn":
            self.ffn_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size,
                rank=self.world_rank,
                group_name="afd_moe",
                timeout=timedelta(minutes=30),
                pg_options=create_hccl_process_group_options(
                    self.hccl_buffer_size_mb,
                ),
            )
            backend = self.ffn_pg._get_backend(torch.device("npu"))
            self.hccl_comm_name1 = str(
                backend.get_hccl_comm_name(self.world_rank),
            )

        if self.topology.participates_in_p2p_group:
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.topology.p2p_world_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    def close(self) -> None:
        """Close all communication groups created by this connector.

        The method also clears saved HCCL group names and marks the connector as
        uninitialized. It is safe to initialize the connector again afterward.
        """
        groups = [self.p2p_pg, self.ffn_pg, *self.afd_pg_list]
        if self.afd_pg is not None and not self.afd_pg_list:
            groups.append(self.afd_pg)
        destroyed_group_ids: set[int] = set()
        for group in groups:
            if group is not None:
                group_id = id(group)
                if group_id in destroyed_group_ids:
                    continue
                destroyed_group_ids.add(group_id)
                dist.destroy_process_group(group)
        self.p2p_pg = None
        self.ffn_pg = None
        self.afd_pg = None
        self.afd_pg_list = []
        self.hccl_comm_name = ""
        self.hccl_comm_name2 = ""
        self.hccl_comm_name3 = ""
        self.hccl_comm_name1 = ""
        self.hccl_comm_name_list = []
        self._initialized = False

    def _padding_rows_for_step(
        self,
        stage_idx: int,
    ) -> int:
        """Return the tile this Attention rank pads its payload up to.

        A2E reads the same number of rows from every Attention peer of an FFN
        rank, so this rank has to write that many rows. The tile is the largest
        count of this rank's peer group -- divided by the FlashComm v1 shard,
        because a TP worker holds only its own share of the DP rank's rows -- and
        the FFN rank derives the same number from the same counts. Deriving it
        here rather than from this rank's own report is what keeps the payload and
        the receive the same shape: a rank that kept its own shorter payload would
        leave the tail of the peer's tile holding whatever the previous transfer
        left there, which the FFN rank reads as token ids.

        Only the connector's integer snapshot of the counts is read here, never
        the tensors themselves: this runs inside the traced model forward, where
        the DP tensors are symbolic and any comparison against them fails the
        data-dependent guard in ``torch.compile``.

        The same ``max_num_tokens`` fallback the FFN rank uses covers a step whose
        counts cannot describe the peer group, so a step without usable metadata
        still has both sides agree on one tile instead of the receiver guessing a
        larger one.
        """

        stage_token_counts = self.dp_token_counts.get(stage_idx, ())
        ffn_rank = ffn_rank_for_attention_rank(
            self.role_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
        )
        if ffn_rank is None:
            return max(1, self.max_num_tokens)
        return padded_tile_rows(
            stage_token_counts,
            ffn_rank=ffn_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
            fallback=self.max_num_tokens,
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send hidden states from an Attention rank to its FFN rank.

        The ubatch number selects the matching HCCL communication group. The
        method saves the values returned by CAMP2p so Attention can later
        receive the FFN result for the same ubatch.

        Args:
            hidden_states: Model data with shape ``(tokens, hidden_size)``.
            context: Transfer context whose ``metadata`` supplies the layer
                number, ubatch number, and token count for this transfer.
            **kwargs: An optional token-aligned ``input_ids`` tensor. When it is
                supplied, the transfer runs with ``compute_gate=1`` so the ids
                reach the FFN rank through the operator's ids channel, and the
                matching ``recv_attn_output(recv_input_ids=True)`` returns them
                on the payload's ``input_ids`` field.

        Raises:
            RuntimeError: If the communication groups are not ready, or if this
                rank produced more rows than the tile A2E reads per Attention
                peer, which the equal-tile layout cannot represent.
            ValueError: If the number of tokens in ``hidden_states`` does not
                match ``context.metadata`` outside a ``torch.compile`` trace, or
                if a supplied ``input_ids`` tensor is malformed.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"CAMP2P metadata token count {metadata.total_tokens}",
            )
        input_ids = cast(torch.Tensor | None, kwargs.get("input_ids"))
        forward_context = get_forward_context()
        stage_idx = metadata.stage_idx
        wire_rows = self._padding_rows_for_step(stage_idx)
        # Rows this rank's payload holds, from the count snapshot rather than from
        # the payload tensor: a traced or padded step reports its rows through the
        # graph, and reading a symbol there is what makes the compiled path fail.
        own_counts = attention_rank_token_counts(
            self.dp_token_counts.get(stage_idx, ()),
            attention_size=self.attn_size,
        )
        payload_rows = (
            None
            if own_counts is None or self.role_rank >= len(own_counts)
            else own_counts[self.role_rank]
        )
        graph_rows = _reported_attention_tokens(forward_context)
        if graph_rows is not None:
            # A graph step sends the rows its captured graph holds and cannot pad
            # them, so the tile the FFN rank sizes its receive with has to be
            # exactly those rows.
            if graph_rows != int(wire_rows):
                raise RuntimeError(
                    f"CAMP2P Attention rank {self.role_rank} sends the {graph_rows} "
                    f"rows of its graph while this step's A2E tile is {wire_rows} "
                    f"rows (layer={metadata.layer_idx}, stage={stage_idx}, "
                    f"dp_counts={self.dp_token_counts.get(stage_idx, ())}).",
                )
            padded_payload = False
        else:
            if (
                not torch.compiler.is_compiling()
                and payload_rows is not None
                and wire_rows < payload_rows
            ):
                raise RuntimeError(
                    f"CAMP2P Attention rank sends {payload_rows} tokens but this "
                    f"step reports a {wire_rows}-row tile per Attention rank "
                    f"(layer={metadata.layer_idx}, ubatch={stage_idx}). A2E reads one "
                    "equal tile per Attention peer and sends the same tile back, so "
                    "the extra tokens cannot be represented. Align the reported "
                    "token count with the rows the forward produces.",
                )
            # Hand the operator a payload of exactly the tile's rows when the
            # forward produced fewer: A2E reads one equal tile per Attention peer,
            # so a shorter payload would leave the tail of this rank's ids and
            # hidden-state regions unwritten and the receiving FFN would read the
            # neighbouring regions as token ids. A step whose rows already are the
            # tile keeps its payload and its buffers untouched.
            padded_payload = payload_rows is None or wire_rows > payload_rows
            if padded_payload:
                hidden_states = self._wire_payload(
                    hidden_states,
                    wire_rows=wire_rows,
                    fill=0,
                )
                if input_ids is not None:
                    input_ids = self._wire_payload(
                        input_ids.reshape(-1).to(torch.int32),
                        wire_rows=wire_rows,
                        fill=_PAD_HASH_TOKEN_ID,
                    )
                metadata = AFDTransferMetadata.create_attention_metadata(
                    layer_idx=metadata.layer_idx,
                    stage_idx=stage_idx,
                    seq_len=wire_rows,
                )
        expert_ids: torch.Tensor | None = None
        expert_scales: torch.Tensor | None = None
        compute_gate = 0
        if input_ids is not None:
            expert_ids, expert_scales = prepare_token_id_transfer(
                input_ids,
                topk=self.num_experts_per_tok,
                expected_tokens=metadata.total_tokens,
            )
            compute_gate = 1
        transfer_state = CAMP2PTransferState(
            aiv_num=self.aiv_num,
            batch_size=metadata.total_tokens,
            h=self.hidden_size,
            k=self.num_experts_per_tok,
            padded_payload=padded_payload,
        )
        ubatch_idx = metadata.stage_idx
        forward_context.cam_afdtransfer_state = transfer_state
        forward_context.ubatch_idx = ubatch_idx

        torch.ops.vllm.afd_camp2p_send_attn_output(
            hidden_states,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            transfer_state.aiv_num,
            compute_gate,
            expert_ids,
            expert_scales,
        )
        return None

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Receive the processed model data from FFN on an Attention rank.

        Args:
            ref_tensor: Tensor supplying the expected shape and storage for
                the receive operation.
            ubatch_idx: Ubatch to receive. Defaults to ``0``.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            The model data returned by FFN.

        Raises:
            RuntimeError: If communication is not ready or the matching
                Attention send information was lost.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        transfer_state = getattr(get_forward_context(), "cam_afdtransfer_state", None)
        if transfer_state is None:
            raise RuntimeError("CAMP2P Attention side is missing connector data")
        get_forward_context().ubatch_idx = ubatch_idx
        # A padded send makes the FFN return the padded tile, so receive into a
        # tile-sized buffer and hand back only the rows the model produced. The
        # reference tensor carries that row count symbolically, which keeps the
        # trim free of any Python branch on the token count.
        if not transfer_state.padded_payload:
            return torch.ops.vllm.afd_camp2p_recv_ffn_output(
                ref_tensor,
                self.hccl_comm_name,
                self.hccl_comm_name2,
                self.hccl_comm_name3,
                transfer_state.batch_size,
                transfer_state.h,
                transfer_state.k,
                self.ffn_size,
                self.attn_size,
                self.world_rank,
                transfer_state.aiv_num,
            )
        destination = self._padded_receive_buffer(
            ref_tensor,
            transfer_state.batch_size,
        )
        received = torch.ops.vllm.afd_camp2p_recv_ffn_output(
            destination,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            transfer_state.aiv_num,
        )
        return received[: ref_tensor.shape[0]]

    @staticmethod
    def _wire_payload(
        payload: torch.Tensor,
        *,
        wire_rows: int,
        fill: int,
    ) -> torch.Tensor:
        """Return a ``wire_rows``-row buffer carrying ``payload`` in its head.

        A2E reads a fixed number of rows from every Attention peer, so a payload
        shorter than that tile leaves the tail of this rank's regions unwritten.
        Copying into a fixed-size buffer extends the payload without computing a
        pad amount, which would have to read the payload's token count and
        therefore specialize the dimension the compiled model declares dynamic.
        Pad rows keep ``fill``, the sentinel the FFN maps back to token 0 for ids.
        The tile is a fixed size for every step that pads, so the buffer does not
        have to outlive the transfer it feeds.
        """

        buffer = torch.full(
            (wire_rows, *payload.shape[1:]),
            fill,
            dtype=payload.dtype,
            device=payload.device,
        )
        buffer[: payload.shape[0]] = payload
        return buffer

    @staticmethod
    def _padded_receive_buffer(
        ref_tensor: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        """Return a ``rows``-row buffer shaped like ``ref_tensor``.

        The tile is a fixed size for every step that pads, so this buffer never
        has to outlive the transfer it feeds.
        """

        return torch.empty(
            (rows, *ref_tensor.shape[1:]),
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        """Receive hidden states from Attention on an FFN rank.

        The ubatch number selects the expected token count and HCCL group.
        Values returned by CAMP2p are saved so the FFN result can be sent back
        to the correct Attention ranks.

        Args:
            ubatch_idx: Ubatch number, starting from ``0``.
            **kwargs: May provide existing transfer information or the layer
                number needed to create it. ``recv_input_ids`` states that this
                FFN rank expects token ids on the transfer, which selects the
                operator's ids mode and makes the connector validate and expose
                the operator's ids slot on the payload. The sending rank has to
                select the same mode, so only request ids for a run whose
                Attention role transports them.

        Returns:
            The received hidden states, the information FFN needs to process them
            and send the result back, and the transported ``input_ids`` when the
            ids mode was selected.

        Raises:
            RuntimeError: If communication is not ready or transfer information
                is missing.
            ValueError: If ids were requested but do not align with the FFN
                rank's token layout.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        layer_idx: int = kwargs.get("layer_idx", 0)
        recv_input_ids: bool = bool(kwargs.get("recv_input_ids", False))
        compute_gate_mode = 1 if recv_input_ids else 0
        # A2E gives this rank one tile per Attention peer and reads the same number
        # of rows from each, so the operator is sized by whole tiles and both roles
        # derive that tile from the same counts and fallback (see
        # :mod:`afd_plugin.a2e_layout`). The connector's own ``max_num_tokens`` is
        # the fallback rather than a caller-supplied value, because the Attention
        # ranks size the very same fallback and a different number here would make
        # the operator read rows no peer wrote.
        batch_size = ffn_receive_rows(
            self.dp_token_counts.get(ubatch_idx, ()),
            self.role_rank,
            attention_size=self.attn_size,
            ffn_size=self.ffn_size,
            fallback=self.max_num_tokens,
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[batch_size],
        )
        custom_states = CAMP2PTransferState(
            aiv_num=self.aiv_num,
            batch_size=batch_size,
            h=self.hidden_size,
            k=self.num_experts_per_tok,
        )
        context = AFDTransferContext(
            metadata=metadata,
            states=custom_states,
        )

        group_ep = _get_group_ep(
            ubatch_idx,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
        )
        outputs = torch.ops.afd_ascend.a2e(
            torch.tensor([], dtype=torch.bfloat16, device="npu"),
            torch.tensor([], dtype=torch.int32, device="npu"),
            torch.tensor([], dtype=torch.float32, device="npu"),
            custom_states.batch_size,
            custom_states.h,
            custom_states.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            group_ep,
            custom_states.aiv_num,
            compute_gate_mode,
        )
        custom_states.atten_batch_size = outputs[3]
        custom_states.x_active_mask = outputs[4]
        custom_states.cam_p2p_ep_name = self.hccl_comm_name1
        # The ids slot is only written in the operator's ids mode, so the mode has
        # to match the sending rank's ``compute_gate``. It is declared here by the
        # receiving FFN rank through ``recv_input_ids`` rather than inferred from
        # the returned tensor: a genuine single-token layer would otherwise be
        # indistinguishable from the operator's placeholder. Reading the slot in
        # the other mode would hand the model uninitialised device memory as token
        # ids, which a token-keyed router turns into an out-of-range table read.
        received_ids: torch.Tensor | None = None
        if compute_gate_mode == 1:
            received_ids = received_token_ids(
                outputs[1],
                expected_tokens=batch_size,
            )
        return AFDA2FTransferPayload(
            hidden_states=outputs[0],
            context=context,
            input_ids=received_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send processed model data from an FFN rank back to Attention.

        Args:
            ffn_output: Model data produced by the FFN layers.
            context: Transfer context saved when FFN received the Attention
                output; its ``states`` carries the CAMP2p receive-time results.
            **kwargs: An optional ``ubatch_idx``. If omitted, the method uses
                the ubatch number stored in ``context.metadata``.

        Raises:
            RuntimeError: If communication is not ready, required receive
                information is missing, or the ubatch group does not exist.
        """
        if not self._initialized:
            raise RuntimeError("CAMP2P connector is not initialized")
        states = cast(CAMP2PTransferState, context.states)
        if states.atten_batch_size is None:
            raise RuntimeError("CAMP2P FFN side is missing A2E atten_batch_size")
        ubatch_idx = int(kwargs.get("ubatch_idx", context.metadata.stage_idx))
        group_ep = _get_group_ep(
            ubatch_idx,
            self.hccl_comm_name,
            self.hccl_comm_name2,
            self.hccl_comm_name3,
        )
        torch.ops.afd_ascend.e2a(
            ffn_output,
            states.atten_batch_size,
            states.batch_size,
            states.h,
            states.k,
            self.ffn_size,
            self.attn_size,
            self.world_rank,
            group_ep,
            states.aiv_num,
        )
        return None


class CAMP2pAFDControlPlane(AFDControlPlane):
    """DP metadata control plane for ``CAMP2pAFDConnector``.

    Applies DP metadata payloads to the owning connector's state and moves
    them between Attention and FFN ranks over the connector's dedicated
    ``p2p`` gloo process group. The connector creates one instance at
    construction time and exposes it through ``control_plane``; the process
    group itself is created by ``init_afd_connector``.
    """

    def __init__(self, connector: CAMP2pAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.dp_token_counts = _dp_stage_token_counts(payload.dp_metadata_list)
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if not connector.topology.is_attn_top_min_size_rank:
            return
        # The CAMP2P DP-metadata group runs on gloo, so the wire tensors stay on
        # CPU rather than the NPU device.
        device = torch.device("cpu")
        send_control_payload(
            payload,
            dst=connector.dst_list,
            group=connector.p2p_pg,
            device=device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError("CAMP2P metadata process group is not initialized")
        src = connector.p2p_rank % connector.min_size + connector.ffn_size
        return recv_control_payload(
            src=src,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )


def build_camp2p_topology(
    afd_config: AFDConfig,
    role_rank: int,
) -> _CAMP2PTopology:
    """Calculate the communication rank numbers for one process.

    FFN processes come first in the main AFD group, followed by Attention
    processes. All FFN ranks and the first ``min(A, F)`` Attention ranks also
    join the smaller Gloo group that exchanges token counts and batch details.

    Args:
        afd_config: Process role and total Attention/FFN rank counts.
        role_rank: This process's runtime number within its own role.

    Returns:
        This process's rank numbers and metadata destinations.

    Raises:
        ValueError: If the rank counts or this process's role rank are invalid.
    """
    attention_size, ffn_size = topology_from_config(afd_config)
    if attention_size <= 0 or ffn_size <= 0:
        raise ValueError("CAMP2P topology sizes must be positive")
    if attention_size < ffn_size:
        raise ValueError(
            "CAMP2P requires attention_size >= ffn_size, got "
            f"{attention_size} < {ffn_size}",
        )
    if role_rank < 0:
        raise ValueError(f"CAMP2P role rank must be non-negative, got {role_rank}")

    if afd_config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        p2p_rank = role_rank + min(ffn_size, attention_size)
    elif afd_config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        p2p_rank = role_rank
    else:
        raise ValueError(f"unknown AFD role {afd_config.role!r}")

    min_size = min(attention_size, ffn_size)
    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        dst = local_attention_rank
        while dst < ffn_size:
            destinations.append(dst)
            dst += min_size

    return _CAMP2PTopology(
        role=afd_config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        dp_metadata_destinations=tuple(destinations),
    )


def _dp_stage_token_counts(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
) -> dict[int, tuple[int, ...]]:
    """Copy the per-DP-rank token counts of every stage into plain integers.

    The counts are read once, where the control payload is published, because the
    Attention model forward may be traced: inside a trace the DP tensors are
    symbolic, so a connector that compared or summed those symbols would fail the
    data-dependent guard in ``torch.compile``. Everything the connector derives
    from the counts (the A2E tile and the rows an FFN rank computes on) is derived
    from this snapshot instead of from the tensors.
    """

    return {
        int(stage_idx): tuple(
            int(count) for count in metadata.num_tokens_across_dp_cpu.flatten().tolist()
        )
        for stage_idx, metadata in dp_metadata_list.items()
    }


def _get_group_ep(
    ubatch_idx: int,
    hccl_comm_name1: str,
    hccl_comm_name2: str,
    hccl_comm_name3: str,
) -> str:
    if ubatch_idx == 1:
        return hccl_comm_name2 or hccl_comm_name1
    if ubatch_idx == 2:
        if not hccl_comm_name3:
            raise RuntimeError("CAMP2P ubatch 2 requires a third HCCL group")
        return hccl_comm_name3
    if ubatch_idx < 0:
        raise RuntimeError(f"CAMP2P ubatch index must be non-negative: {ubatch_idx}")
    return hccl_comm_name1


def _register_camp2p_custom_ops() -> None:
    """Register the CAMP2P send and receive operations once per process.

    The A2E operation sends Attention output to FFN. The E2A operation sends
    the FFN result back to Attention.
    """
    global _CAMP2P_CUSTOM_OPS_REGISTERED
    if _CAMP2P_CUSTOM_OPS_REGISTERED:
        return

    def send_attn_output_impl(
        hidden_states: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
        compute_gate: int,
        expert_ids: torch.Tensor | None,
        expert_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        transfer_state = getattr(get_forward_context(), "cam_afdtransfer_state", None)
        if transfer_state is None:
            transfer_state = CAMP2PTransferState()
        # This implementation runs once per step with the tensor the compiled
        # forward produced, while ``batch_size`` is the value the traced Python
        # passed and stays frozen at the shape that trace saw. Sizing the transfer
        # from the traced value makes a step whose rows differ from it write too few
        # rows, and A2E then reads this rank's scales and activations as ids on the
        # FFN side. The payload's own row count is the step's row count.
        transfer_state.batch_size = int(hidden_states.shape[0])
        transfer_state.h = hidden_size
        transfer_state.k = topk
        transfer_state.aiv_num = aiv_num
        group_ep = _get_group_ep(
            int(getattr(get_forward_context(), "ubatch_idx", 0)),
            hccl_comm_name,
            hccl_comm_name2,
            hccl_comm_name3,
        )

        outputs = torch.ops.afd_ascend.a2e(
            hidden_states,
            expert_ids,
            expert_scales,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            transfer_state.aiv_num,
            compute_gate,
        )
        transfer_state.atten_batch_size = outputs[3]
        forward_context = get_forward_context()
        forward_context.cam_afdtransfer_state = transfer_state
        return hidden_states

    def send_attn_output_fake_impl(
        hidden_states: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
        compute_gate: int,
        expert_ids: torch.Tensor | None,
        expert_scales: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the input unchanged while PyTorch inspects the send operation."""
        return hidden_states

    def recv_ffn_output_impl(
        ref_tensor: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        transfer_state = getattr(forward_context, "cam_afdtransfer_state", None)
        if transfer_state is None:
            raise RuntimeError(
                "CAMP2P Attention side is missing connector data: the A2E send did "
                "not run in this forward context "
                f"(num_tokens={getattr(forward_context, 'num_tokens', None)}, "
                f"ubatch_idx={getattr(forward_context, 'ubatch_idx', None)}, "
                "runtime_mode="
                f"{getattr(forward_context, 'cudagraph_runtime_mode', None)})",
            )
        atten_batch_size = transfer_state.atten_batch_size
        if atten_batch_size is None:
            # The A2E call runs in the model forward, and a graph execution path
            # does not re-run that Python, so the per-peer row count it publishes
            # can be missing here. The E2A kernel accepts that tensor but never
            # reads it (``csrc/npu/ascend_kernels/e2a/op_kernel/e2a.h`` binds
            # ``attenBatchSize`` and nothing else), so size it from the tile this
            # rank receives instead of failing the step.
            atten_batch_size = torch.full(
                (max(1, -(-int(attn_size) // max(1, int(ffn_size)))),),
                int(batch_size),
                dtype=torch.int32,
                device=ref_tensor.device,
            )
        transfer_state.batch_size = int(ref_tensor.shape[0])
        transfer_state.h = hidden_size
        transfer_state.k = topk
        transfer_state.aiv_num = aiv_num
        group_ep = _get_group_ep(
            int(getattr(forward_context, "ubatch_idx", 0)),
            hccl_comm_name,
            hccl_comm_name2,
            hccl_comm_name3,
        )
        output = torch.ops.afd_ascend.e2a(
            ref_tensor,
            atten_batch_size,
            transfer_state.batch_size,
            transfer_state.h,
            transfer_state.k,
            ffn_size,
            attn_size,
            world_rank,
            group_ep,
            transfer_state.aiv_num,
        )
        return output

    def recv_ffn_output_fake_impl(
        ref_tensor: torch.Tensor,
        hccl_comm_name: str,
        hccl_comm_name2: str,
        hccl_comm_name3: str,
        batch_size: int,
        hidden_size: int,
        topk: int,
        ffn_size: int,
        attn_size: int,
        world_rank: int,
        aiv_num: int,
    ) -> torch.Tensor:
        """Return the reference tensor while PyTorch inspects the receive."""
        return ref_tensor

    send_annotations = {
        "hidden_states": torch.Tensor,
        "hccl_comm_name": str,
        "hccl_comm_name2": str,
        "hccl_comm_name3": str,
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "aiv_num": int,
        "compute_gate": int,
        "expert_ids": torch.Tensor | None,
        "expert_scales": torch.Tensor | None,
        "return": torch.Tensor,
    }
    recv_annotations = {
        "ref_tensor": torch.Tensor,
        "hccl_comm_name": str,
        "hccl_comm_name2": str,
        "hccl_comm_name3": str,
        "batch_size": int,
        "hidden_size": int,
        "topk": int,
        "ffn_size": int,
        "attn_size": int,
        "world_rank": int,
        "aiv_num": int,
        "return": torch.Tensor,
    }
    send_attn_output_impl.__annotations__ = send_annotations
    send_attn_output_fake_impl.__annotations__ = send_annotations
    recv_ffn_output_impl.__annotations__ = recv_annotations
    recv_ffn_output_fake_impl.__annotations__ = recv_annotations

    try:
        direct_register_custom_op(
            op_name="afd_camp2p_send_attn_output",
            op_func=send_attn_output_impl,
            mutates_args=[],
            fake_impl=send_attn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
        direct_register_custom_op(
            op_name="afd_camp2p_recv_ffn_output",
            op_func=recv_ffn_output_impl,
            mutates_args=[],
            fake_impl=recv_ffn_output_fake_impl,
            dispatch_key="PrivateUse1",
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        duplicate = any(
            marker in message
            for marker in ("already", "duplicate", "same name", "defined")
        )
        if not duplicate:
            raise
    _CAMP2P_CUSTOM_OPS_REGISTERED = True


__all__ = [
    "CAMP2pAFDConnector",
    "CAMP2pAFDControlPlane",
    "CAMP2PAFDConnectorData",
    "CAMP2PExtraInfo",
    "CAMP2PTransferState",
    "build_camp2p_topology",
]

CAMP2PAFDConnectorData = CAMP2PTransferState
