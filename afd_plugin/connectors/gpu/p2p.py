# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NCCL-backed point-to-point AFD connector for CUDA deployments.

``P2pNcclAFDConnector`` exchanges hidden states synchronously between
disaggregated Attention and FFN workers through NCCL point-to-point
communication, implemented with vLLM's ``PyNcclCommunicator``. It supports
both prefill and decode; eager mode is supported and CUDA graph support is
currently limited to ``FULL_DECODE_ONLY``.

Topology:
    The connector creates one AFD NCCL world ordered as
    ``[F0, F1, ..., A0, A1, ...]``: FFN ranks first, followed by Attention
    ranks. Each FFN rank owns one subgroup containing itself and one or more
    consecutive Attention ranks, which requires::

        num_attention_ranks >= num_ffn_ranks
        num_attention_ranks % num_ffn_ranks == 0

    Attention sends hidden states to its mapped FFN rank; the FFN rank
    concatenates inputs from its Attention peers, runs FFN work, splits the
    output by the recorded sequence lengths, and sends each slice back to the
    originating Attention rank.

Control and data planes:
    DP metadata handling is a pluggable control plane, not part of the
    connector interface: ``P2pNcclAFDControlPlane`` (an ``AFDControlPlane``
    implementation exposed as ``connector.control_plane``) sends, receives,
    and applies the per-stage token-count payloads that determine wire
    tensor shapes, moving them over a separate NCCL process group. Because the
    connector exposes a ``control_plane``, each FFN-side step is driven by the
    arrival of a control-plane payload. The
    data path stays on the connector and uses
    ``PyNcclCommunicator.send()`` / ``recv()`` on the current CUDA stream,
    wrapped in the ``torch.ops.vllm.afd_p2p_send`` / ``afd_p2p_recv`` custom
    ops so transfers stay usable under ``torch.compile`` and CUDA graph
    capture.

Requirements and limitations:
    - Requires a CUDA-capable PyTorch/vLLM environment with NCCL and vLLM's
      ``PyNcclCommunicator`` available. Do not use it on Ascend NPU
      deployments (use ``CAMP2pAFDConnector`` or ``CAMAsyncAFDConnector``
      there); there is no automatic fallback to another transport.
    - Hidden-state tensors must reside on CUDA devices; CPU tensors are
      rejected.
    - All ranks must agree on ``host``, ``port``, rank counts, model hidden
      size/dtype, and role-rank assignment. Initialization is collective:
      missing ranks, mismatched counts, or duplicate role ranks can cause
      initialization failure or timeout.
    - The rendezvous base ``port`` and the derived subgroup ports
      (``port + subgroup_index + 1``) must be free and reachable.
    - AFD async mode (``async`` / ``async_dp``) is not supported; GPU DBO
      combined with CUDA graphs is limited to exactly two ubatches.
    - Cross-node use is not established by the checked-in recipes and should
      be treated as unverified.

See ``docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`` for the configuration
contract and launch examples, and ``recipe/gpu/p2p_nccl/`` for complete
deployments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.distributed.distributed_c10d import ProcessGroup, _get_default_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.forward_context import DPMetadata
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDA2FTransportSpec,
    AFDControlPayload,
    AFDDPMetadata,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import (
    DefaultProcessGroupSwitcher,
    build_rank_mapping,
    init_afd_process_group,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_AFD_COMMUNICATORS: dict[int, PyNcclCommunicator] = {}
_AFD_COMM_ID_COUNTER = 0
_AFD_CUSTOM_OPS_REGISTERED = False


class _TensorMetadata(NamedTuple):
    """Device, dtype, and shape describing one expected wire tensor."""

    device: torch.device
    dtype: torch.dtype
    size: torch.Size


class P2pNcclAFDConnector(AFDConnectorBase):
    """NCCL-backed Attention <-> FFN connector for CUDA deployments.

    The P2P topology places FFN ranks before Attention ranks in the AFD world
    (``[F0, F1, ..., A0, A1, ...]``), and each FFN rank owns a subgroup with
    one or more consecutive Attention ranks. Within a subgroup, the FFN rank
    is subgroup rank ``0`` and its Attention peers occupy ranks ``1..ratio``.

    Hidden states move through per-subgroup ``PyNcclCommunicator`` instances
    on the current CUDA stream; a separate NCCL process group distributes the
    DP metadata that determines per-stage tensor shapes.

    DP metadata operations do not live on the connector itself: they are
    provided by the pluggable ``P2pNcclAFDControlPlane`` instance created at
    construction time and exposed as ``control_plane``. The connector still
    owns the ``p2p`` process group the control plane transmits over, because
    creating that group is part of the collective ``init_afd_connector``
    ordering. See the module docstring for the topology rules, configuration
    contract, and requirements.
    """

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        # P2P currently has no connector-specific options, so only an empty
        # connector_extra_config is accepted.
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("P2P connector_extra_config must be a mapping")
        if raw:
            raise ValueError(
                "P2pNcclAFDConnector does not support connector_extra_config",
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
        """Derive the P2P rank mapping and prepare per-stage state caches.

        Also creates the ``P2pNcclAFDControlPlane`` instance exposed as
        ``control_plane``. Communication resources are not created here;
        ``init_afd_connector`` performs the collective initialization.

        Args:
            rank: Process rank passed by the owning vLLM worker; not the same
                as the connector-derived AFD ``world_rank``.
            local_rank: CUDA device index for this worker.
            vllm_config: Upstream vLLM config; used for model hidden size,
                dtype, layer count, and eager/graph mode.
            afd_config: Parsed AFD configuration carrying the role, topology
                sizes, and rendezvous host/port.
            role_rank: Runtime rank within the configured AFD role group.
        """
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self._initialized = False
        self.mapping = build_rank_mapping(afd_config, role_rank)
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.group_size = len(self.mapping.subgroup_ranks)
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.num_hidden_layers = (vllm_config.model_config.hf_config.num_hidden_layers,)
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.tensor_metadata_list: dict[int, _TensorMetadata] = {}
        self._recv_attn_tensor_metadata_list: dict[
            tuple[int, int],
            _TensorMetadata,
        ] = {}
        self._recv_attn_buffers: dict[
            tuple[int, int, tuple[int, ...]],
            torch.Tensor,
        ] = {}
        self._recv_attn_input_ids_buffers: dict[
            tuple[int, int, tuple[int, ...], torch.dtype],
            torch.Tensor,
        ] = {}
        self.a2e_group: StatelessProcessGroup | None = None
        self.e2a_group: StatelessProcessGroup | None = None
        self.p2p_pg: ProcessGroup | None = None
        self.a2e_pynccl: PyNcclCommunicator | None = None
        self.e2a_pynccl: PyNcclCommunicator | None = None
        self.a2e_comm_id: int | None = None
        self.e2a_comm_id: int | None = None
        self._p2p_ordering_token: torch.Tensor | None = None
        self.control_plane = P2pNcclAFDControlPlane(self)

    def close(self) -> None:
        """Release NCCL communicators and their custom-op registrations.

        Unregisters this connector's communicators from the module-level
        registry used by the ``afd_p2p_send`` / ``afd_p2p_recv`` custom ops,
        shuts the ``PyNcclCommunicator`` instances down, and marks the
        connector uninitialized. Safe to call repeatedly.
        """
        for comm_id_name in ("a2e_comm_id", "e2a_comm_id"):
            comm_id = getattr(self, comm_id_name, None)
            if comm_id is not None:
                _AFD_COMMUNICATORS.pop(comm_id, None)
                setattr(self, comm_id_name, None)
        for communicator_name in ("a2e_pynccl", "e2a_pynccl"):
            communicator = getattr(self, communicator_name, None)
            shutdown = getattr(communicator, "shutdown", None)
            if callable(shutdown):
                shutdown()
            setattr(self, communicator_name, None)
        self._p2p_ordering_token = None
        self._initialized = False

    def init_afd_connector(self) -> None:
        """Create the AFD NCCL world and per-subgroup communicators.

        This is a collective call: every Attention and FFN rank must invoke
        it with matching ``host``, ``port``, and rank counts, or the
        rendezvous fails or times out. It performs three steps:

        1. Joins the AFD world process group (FFN ranks first, then Attention
           ranks) rendezvoused at ``tcp://host:port``.
        2. Creates this rank's subgroup ``StatelessProcessGroup`` on
           ``port + subgroup_index + 1`` and two ``PyNcclCommunicator``
           instances over it (Attention-to-FFN and FFN-to-Attention), each
           registered for use by the P2P custom ops.
        3. On ranks that participate in the DP metadata control plane, joins
           the ``p2p`` process group that ``control_plane`` uses to
           distribute per-stage token counts.

        Idempotent: returns immediately if already initialized.
        """
        if self._initialized:
            return

        _register_p2p_custom_ops()

        afd_pg = init_afd_process_group(
            backend="nccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.ffn_size + self.attn_size,
            rank=self.world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )

        with DefaultProcessGroupSwitcher(_get_default_group(), afd_pg):
            base_port = self.afd_config.port
            self.a2e_group = StatelessProcessGroup.create(
                host=self.afd_config.host,
                port=base_port + self.mapping.subgroup_index + 1,
                rank=self.mapping.rank_in_subgroup,
                world_size=len(self.mapping.subgroup_ranks),
            )
            self.e2a_group = self.a2e_group
            self.a2e_pynccl = PyNcclCommunicator(
                group=self.a2e_group,
                device=self.local_rank,
            )
            self.a2e_comm_id = _register_comm(self.a2e_pynccl)
            self.e2a_pynccl = PyNcclCommunicator(
                group=self.e2a_group,
                device=self.local_rank,
            )
            self.e2a_comm_id = _register_comm(self.e2a_pynccl)
            self._p2p_ordering_token = torch.zeros(
                1,
                dtype=torch.int64,
                device=torch.device("cuda", self.local_rank),
            )

        if self.mapping.participates_in_dp_metadata_group:
            self.p2p_pg = init_afd_process_group(
                backend="nccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timedelta(minutes=30),
            )

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Return whether the NCCL groups and communicators are ready."""
        return self._initialized

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send Attention hidden states to this rank's mapped FFN rank.

        Args:
            hidden_states: CUDA tensor of shape ``(num_tokens, hidden_size)``
                whose leading dimension matches
                ``context.metadata.total_tokens``.
            context: Per-transfer context describing the token layout.
            **kwargs: Optional ``router_logits`` and token-aligned ``input_ids``
                tensors. The stable version-1 wire order is hidden states,
                router logits when present, then input IDs when present.

        Raises:
            ValueError: If the tensor shape does not match the metadata token
                count (skipped while ``torch.compile`` is tracing) or the
                tensor is on CPU.
            RuntimeError: If the connector is not initialized.
        """
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"AFD metadata token count {metadata.total_tokens}",
            )
        router_logits: torch.Tensor | None = kwargs.get("router_logits")
        input_ids: torch.Tensor | None = kwargs.get("input_ids")
        transport_spec: AFDA2FTransportSpec | None = kwargs.get("transport_spec")
        if (transport_spec is None) != (input_ids is None):
            raise ValueError(
                "input_ids presence must exactly match the AFD transport spec",
            )
        if (
            router_logits is not None
            and router_logits.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "router_logits and hidden_states must have equal token counts",
            )
        if input_ids is not None:
            if input_ids.ndim != 1 or input_ids.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "input_ids must be one-dimensional and token-aligned with "
                    "hidden_states",
                )
            assert transport_spec is not None
            if input_ids.dtype != transport_spec.input_ids_dtype:
                raise ValueError(
                    "P2P AFD input_ids dtype does not match the transport spec",
                )
        self._send_hidden_states(
            hidden_states,
            0,
            self.a2e_group,
            self.a2e_comm_id,
        )
        if router_logits is not None:
            self._send_hidden_states(
                router_logits,
                0,
                self.a2e_group,
                self.a2e_comm_id,
            )
        if input_ids is not None:
            self._send_hidden_states(
                input_ids,
                0,
                self.a2e_group,
                self.a2e_comm_id,
            )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Receive this rank's FFN output slice on the Attention side.

        Args:
            ref_tensor: Preallocated CUDA tensor to receive into. Used as a
                stable buffer for CUDA graph capture and returned directly
                when the subgroup has a single rank and no wire transfer
                occurs.
            ubatch_idx: Stage/microbatch index. Defaults to ``0``.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            The FFN output tensor for the current layer/stage.

        Raises:
            RuntimeError: If the connector is not initialized, or no receive
                is performed for a single-rank subgroup.
        """
        output = self._recv_hidden_states(
            0,
            self.e2a_group,
            self.e2a_comm_id,
            self.tensor_metadata_list[ubatch_idx],
            ref_tensor=ref_tensor,
        )
        if output is None:
            raise RuntimeError(
                "P2P recv_ffn_output requires ref_tensor when no receive is performed",
            )
        return output

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive and concatenate Attention hidden states on the FFN rank.

        Receives one tensor from every Attention peer in this rank's
        subgroup (subgroup ranks ``1..ratio``), concatenates them along the
        token dimension, and records each peer's sequence length in the
        returned metadata so ``send_ffn_output`` can split the FFN output
        back per peer. When CUDA graphs are enabled, receives reuse the
        buffers preallocated by ``control_plane.update_state_from_dp_metadata``.

        Args:
            ubatch_idx: Stage/microbatch index to receive. Defaults to ``0``.
            **kwargs: Optional ``routing_spec`` and versioned ``transport_spec``
                describing model-owned Attention-to-FFN side tensors.

        Returns:
            ``AFDA2FTransferPayload`` with the concatenated hidden states and a
            transfer context whose FFN metadata carries the per-peer sequence
            lengths.

        Raises:
            RuntimeError: If the connector is not initialized or the subgroup
                has no Attention peers.
        """
        routing_spec: AFDExpertRoutingSpec | None = kwargs.get("routing_spec")
        transport_spec: AFDA2FTransportSpec | None = kwargs.get("transport_spec")
        if transport_spec is not None and transport_spec.version != 1:
            raise ValueError(
                f"unsupported AFD A2F transport version {transport_spec.version}",
            )
        if (
            transport_spec is not None
            and transport_spec.input_ids_dtype not in (None, torch.int64)
        ):
            raise ValueError("P2P AFD input_ids transport requires torch.int64")
        hidden_states_list: list[torch.Tensor] = []
        router_logits_list: list[torch.Tensor] = []
        input_ids_list: list[torch.Tensor] = []

        for src in range(1, self.group_size):
            tensor_metadata = self._recv_attn_tensor_metadata_list.get(
                (ubatch_idx, src),
                self.tensor_metadata_list[ubatch_idx],
            )
            ref_tensor = None
            if not self.vllm_config.model_config.enforce_eager:
                ref_tensor = self._recv_attn_buffers.get(
                    (ubatch_idx, src, tuple(tensor_metadata.size)),
                )
            hidden_states_list.append(
                self._recv_hidden_states(
                    src,
                    self.a2e_group,
                    self.a2e_comm_id,
                    tensor_metadata,
                    ref_tensor=ref_tensor,
                ),
            )
            if routing_spec is not None:
                router_metadata = _TensorMetadata(
                    device=tensor_metadata.device,
                    dtype=routing_spec.router_logits_dtype,
                    size=torch.Size(
                        [
                            tensor_metadata.size[0],
                            routing_spec.router_logits_width,
                        ],
                    ),
                )
                router_logits_list.append(
                    self._recv_hidden_states(
                        src,
                        self.a2e_group,
                        self.a2e_comm_id,
                        router_metadata,
                    ),
                )
            if (
                transport_spec is not None
                and transport_spec.input_ids_dtype is not None
            ):
                input_ids_metadata = _TensorMetadata(
                    device=tensor_metadata.device,
                    dtype=transport_spec.input_ids_dtype,
                    size=torch.Size([tensor_metadata.size[0]]),
                )
                input_ids_ref = None
                if not self.vllm_config.model_config.enforce_eager:
                    input_ids_ref = self._recv_attn_input_ids_buffers.get(
                        (
                            ubatch_idx,
                            src,
                            tuple(input_ids_metadata.size),
                            input_ids_metadata.dtype,
                        ),
                    )
                input_ids_list.append(
                    self._recv_hidden_states(
                        src,
                        self.a2e_group,
                        self.a2e_comm_id,
                        input_ids_metadata,
                        ref_tensor=input_ids_ref,
                    ),
                )

        if not hidden_states_list:
            raise RuntimeError("P2P FFN rank has no Attention peers")
        hidden_states = (
            torch.cat(hidden_states_list, dim=0)
            if len(hidden_states_list) > 1
            else hidden_states_list[0]
        )
        router_logits = None
        if routing_spec is not None:
            router_logits = (
                torch.cat(router_logits_list, dim=0)
                if len(router_logits_list) > 1
                else router_logits_list[0]
            )
        input_ids = None
        if input_ids_list:
            input_ids = (
                torch.cat(input_ids_list, dim=0)
                if len(input_ids_list) > 1
                else input_ids_list[0]
            )

        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=0,
            stage_idx=ubatch_idx,
            seq_lens=[tensor.shape[0] for tensor in hidden_states_list],
        )
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=AFDTransferContext(
                metadata=metadata,
            ),
            router_logits=router_logits,
            input_ids=input_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Split the FFN output and send each slice back to its Attention peer.

        With a one-to-one mapping (``ratio == 1``) the whole tensor is sent to
        the single Attention peer. Otherwise the output is split along the
        token dimension by ``metadata.seq_lens`` — falling back to an even
        split when the recorded lengths do not cover every peer — and each
        slice is sent to the Attention rank that originally produced it.

        Args:
            ffn_output: CUDA tensor of shape ``(num_tokens, hidden_size)``
                produced by the FFN computation for the whole subgroup.
            context: Transfer context from the matching ``recv_attn_output``
                call; its ``metadata.seq_lens`` determine the per-peer split
                sizes.
            **kwargs: Unused; accepted for interface compatibility.

        Raises:
            ValueError: If the tensor shape does not match the metadata
                (skipped while ``torch.compile`` is tracing), or the output
                cannot be evenly split across Attention peers when
                ``seq_lens`` is unusable.
            RuntimeError: If the connector is not initialized.
        """
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match metadata",
            )
        if self.ratio == 1:
            self._send_hidden_states(ffn_output, 1, self.e2a_group, self.e2a_comm_id)
            return

        split_sizes = metadata.seq_lens
        if len(split_sizes) != self.ratio:
            total_tokens = ffn_output.shape[0]
            if total_tokens % self.ratio != 0:
                raise ValueError(
                    "cannot evenly split FFN output across Attention peers: "
                    f"tokens={total_tokens}, ratio={self.ratio}",
                )
            tokens_per_attention = total_tokens // self.ratio
            split_sizes = [tokens_per_attention] * self.ratio

        start = 0
        for dst, token_count in zip(
            range(1, self.group_size),
            split_sizes,
            strict=False,
        ):
            end = start + token_count
            self._send_hidden_states(
                ffn_output[start:end],
                dst,
                self.e2a_group,
                self.e2a_comm_id,
            )
            start = end

    def _send_hidden_states(
        self,
        hidden_states: torch.Tensor,
        dst: int,
        process_group: StatelessProcessGroup | None,
        comm_id: int | None,
    ) -> None:
        """Send ``hidden_states`` to subgroup rank ``dst`` via the custom op.

        No-ops for single-rank subgroups. Raises ``RuntimeError`` if the
        connector is not initialized and ``ValueError`` for an out-of-range
        destination or a CPU tensor.
        """
        if process_group is None or comm_id is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            return
        if dst >= process_group.world_size:
            raise ValueError(f"invalid P2P destination rank {dst}")
        if getattr(hidden_states, "is_cpu", False):
            raise ValueError("P2P hidden states must be on GPU")
        if self._p2p_ordering_token is None:
            raise RuntimeError("P2P connector ordering token is not initialized")

        torch.ops.vllm.afd_p2p_send(
            hidden_states,
            self._p2p_ordering_token,
            dst,
            comm_id,
        )
        return

    def _recv_hidden_states(
        self,
        src: int,
        process_group: StatelessProcessGroup | None,
        comm_id: int | None,
        tensor_metadata: _TensorMetadata,
        *,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Receive a tensor from subgroup rank ``src`` via the custom op.

        Receives into ``ref_tensor`` when it matches the expected shape,
        dtype, and device (keeping allocations stable for CUDA graph
        capture); otherwise allocates a fresh tensor from
        ``tensor_metadata``. For single-rank subgroups no transfer happens
        and ``ref_tensor`` is returned as-is (and is therefore required).
        """
        if process_group is None or comm_id is None:
            raise RuntimeError("P2P connector is not initialized")
        if process_group.world_size == 1:
            if ref_tensor is None:
                raise RuntimeError("single-rank P2P recv requires a reference tensor")
            return ref_tensor
        if src >= process_group.world_size:
            raise ValueError(f"invalid P2P source rank {src}")

        size = list(tensor_metadata.size)
        if ref_tensor is not None:
            size[0] = ref_tensor.shape[0]

        if (
            ref_tensor is not None
            and ref_tensor.shape == tuple(size)
            and ref_tensor.dtype == tensor_metadata.dtype
            and ref_tensor.device == tensor_metadata.device
        ):
            hidden_states = ref_tensor
        else:
            hidden_states = torch.empty(
                tuple(size),
                dtype=tensor_metadata.dtype,
                device=tensor_metadata.device,
            )
        if self._p2p_ordering_token is None:
            raise RuntimeError("P2P connector ordering token is not initialized")
        torch.ops.vllm.afd_p2p_recv(
            hidden_states,
            self._p2p_ordering_token,
            src,
            comm_id,
        )
        return hidden_states


class P2pNcclAFDControlPlane(AFDControlPlane):
    """DP metadata control plane for ``P2pNcclAFDConnector``.

    Applies DP metadata payloads to the owning connector's per-stage tensor
    metadata caches and moves payloads between Attention and FFN ranks over
    the connector's dedicated ``p2p`` NCCL process group. The connector
    creates one instance at construction time and exposes it through
    ``control_plane``; the process group itself is created by
    ``init_afd_connector``.
    """

    def __init__(self, connector: P2pNcclAFDConnector) -> None:
        """Bind the control plane to its owning connector.

        Args:
            connector: The P2P connector whose topology, configuration, and
                per-stage state this control plane reads and updates.
        """
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        """Derive per-stage tensor shapes from a DP metadata payload.

        Stores the payload's DP metadata and graph-capturing/warmup flags on
        the connector, then computes the expected wire-tensor metadata for
        each stage:

        - On Attention ranks, the shape of this rank's own send/receive
          tensor.
        - On FFN ranks, one entry per Attention peer in the subgroup plus the
          concatenated total, and (when CUDA graphs are enabled) preallocated
          receive buffers so graph capture and replay reuse stable
          allocations.

        Args:
            payload: DP metadata control-plane payload with per-stage token
                counts and graph-capturing/warmup flags.
        """
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup
        connector.tensor_metadata_list = {}
        connector._recv_attn_tensor_metadata_list = {}
        device = torch.device(f"cuda:{connector.local_rank}")
        dtype = connector.vllm_config.model_config.dtype
        for stage_idx, dp_metadata in payload.dp_metadata_list.items():
            stage_idx = stage_idx
            if connector.afd_config.role == "ffn":
                peer_metadata: list[_TensorMetadata] = []
                for src_rank in range(1, connector.group_size):
                    if src_rank <= 0 or src_rank >= connector.group_size:
                        raise ValueError(f"invalid Attention subgroup rank {src_rank}")
                    attention_rank = (
                        connector.mapping.subgroup_index * connector.ratio
                        + src_rank
                        - 1
                    )

                    tensor_metadata = _TensorMetadata(
                        device,
                        dtype,
                        torch.Size(
                            [
                                _num_tokens_for_attention_rank(
                                    dp_metadata,
                                    attention_rank=attention_rank,
                                    attention_size=connector.attn_size,
                                ),
                                connector.hidden_size,
                            ],
                        ),
                    )
                    connector._recv_attn_tensor_metadata_list[(stage_idx, src_rank)] = (
                        tensor_metadata
                    )
                    peer_metadata.append(tensor_metadata)
                num_tokens = sum(
                    tensor_metadata.size[0] for tensor_metadata in peer_metadata
                )
            else:
                num_tokens = _num_tokens_for_attention_rank(
                    dp_metadata,
                    attention_rank=connector.mapping.role_rank,
                    attention_size=connector.attn_size,
                )
            connector.tensor_metadata_list[stage_idx] = _TensorMetadata(
                device,
                dtype,
                torch.Size([num_tokens, connector.hidden_size]),
            )

        if (
            connector.afd_config.role == "ffn"
            and not connector.vllm_config.model_config.enforce_eager
        ):
            for (
                stage_idx,
                src_rank,
            ), tensor_metadata in connector._recv_attn_tensor_metadata_list.items():
                buffer_key = (stage_idx, src_rank, tuple(tensor_metadata.size))
                if buffer_key not in connector._recv_attn_buffers:
                    connector._recv_attn_buffers[buffer_key] = torch.empty(
                        tuple(tensor_metadata.size),
                        dtype=tensor_metadata.dtype,
                        device=tensor_metadata.device,
                    )
                if payload.transport_spec is not None:
                    input_ids_key = (
                        stage_idx,
                        src_rank,
                        (tensor_metadata.size[0],),
                        payload.transport_spec.input_ids_dtype,
                    )
                    if input_ids_key not in connector._recv_attn_input_ids_buffers:
                        connector._recv_attn_input_ids_buffers[input_ids_key] = (
                            torch.empty(
                                (tensor_metadata.size[0],),
                                dtype=payload.transport_spec.input_ids_dtype,
                                device=tensor_metadata.device,
                            )
                        )

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        """Send the DP metadata payload from Attention to the FFN ranks.

        No-ops on ranks that are not designated DP metadata senders (only the
        first ``min_size`` Attention ranks in the ``p2p`` group transmit) or
        that do not participate in the DP metadata group at all.

        Args:
            payload: DP metadata payload to distribute to the FFN-side
                connector loop.
        """
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if not (
            connector.ffn_size
            <= connector.world_rank
            < connector.ffn_size + connector.min_size
        ):
            return
        # NCCL transport requires the wire tensors to live on the CUDA device.
        device = torch.device(f"cuda:{connector.local_rank}")
        send_control_payload(
            payload,
            dst=connector.dst_list,
            group=connector.p2p_pg,
            device=device,
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        """Receive a DP metadata payload on an FFN rank.

        Blocks on the ``p2p`` process group until the mapped Attention sender
        rank transmits the next payload.

        Returns:
            The DP metadata payload sent by the Attention side.

        Raises:
            RuntimeError: If the DP metadata process group is not initialized
                on this rank.
        """
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError("P2P DP metadata process group is not initialized")

        src = connector.p2p_rank % connector.min_size + connector.ffn_size
        device = torch.device(f"cuda:{connector.local_rank}")
        return recv_control_payload(
            src=src,
            group=connector.p2p_pg,
            device=device,
        )


def _num_tokens_for_attention_rank(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_rank: int,
    attention_size: int,
    fallback: int = 1,
) -> int:
    """Return the token count one Attention rank contributes for a stage.

    Reads the per-DP-rank token counts from ``dp_metadata``. When the counts
    cover only DP ranks but ``attention_size`` includes TP-derived worker
    ranks, each DP count is expanded across its TP peers. Always returns at
    least ``1`` so wire tensors keep a non-empty shape.
    """
    counts = dp_metadata.num_tokens_across_dp_cpu.flatten().tolist()
    if not counts:
        return max(1, fallback)
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[idx // tp_size] for idx in range(attention_size)]

    if 0 <= attention_rank < len(counts):
        return max(1, counts[attention_rank])
    return max(1, fallback)


def _register_comm(communicator: PyNcclCommunicator) -> int:
    """Register a communicator for the P2P custom ops and return its id.

    Custom ops cannot capture Python objects, so send/recv ops look
    communicators up by integer id in the module-level registry.
    """
    global _AFD_COMM_ID_COUNTER

    comm_id = _AFD_COMM_ID_COUNTER
    _AFD_COMMUNICATORS[comm_id] = communicator
    _AFD_COMM_ID_COUNTER += 1
    return comm_id


def _register_p2p_custom_ops() -> None:
    """Register the ordered AFD P2P send and receive custom ops.

    Wrapping ``PyNcclCommunicator.send()`` / ``recv()`` in custom ops with
    fake implementations keeps the transfers traceable by ``torch.compile``
    and capturable by CUDA graphs. Registration happens once per process;
    ops already registered by another connector instance are reused.
    """
    global _AFD_CUSTOM_OPS_REGISTERED

    if _AFD_CUSTOM_OPS_REGISTERED:
        return

    def afd_p2p_send_impl(
        tensor: torch.Tensor,
        ordering_token: torch.Tensor,
        dst: int,
        comm_id: int,
    ) -> None:
        communicator = _AFD_COMMUNICATORS.get(comm_id)
        if communicator is None:
            raise RuntimeError(f"AFD communicator id {comm_id} is not registered")
        ordering_token.add_(1)
        communicator.send(
            tensor,
            dst,
            stream=torch.cuda.current_stream(tensor.device),
        )
        return None

    def afd_p2p_send_fake(
        tensor: torch.Tensor,
        ordering_token: torch.Tensor,
        dst: int,
        comm_id: int,
    ) -> None:
        pass

    def afd_p2p_recv_impl(
        out: torch.Tensor,
        ordering_token: torch.Tensor,
        src: int,
        comm_id: int,
    ) -> None:
        communicator = _AFD_COMMUNICATORS.get(comm_id)
        if communicator is None:
            raise RuntimeError(f"AFD communicator id {comm_id} is not registered")
        ordering_token.add_(1)
        communicator.recv(
            out,
            src,
            stream=torch.cuda.current_stream(out.device),
        )

    def afd_p2p_recv_fake(
        out: torch.Tensor,
        ordering_token: torch.Tensor,
        src: int,
        comm_id: int,
    ) -> None:
        pass

    def register_one(
        op_name: str,
        op_func: Callable[..., None],
        mutates_args: list[str],
        fake_impl: Callable[..., None],
    ) -> None:
        # A prior import of this module (for example after an importlib reload)
        # can leave the op defined in torch's process-global registry while
        # this module's _AFD_CUSTOM_OPS_REGISTERED flag is back to False. Reuse
        # the existing op instead of re-defining it, which torch rejects.
        if hasattr(torch.ops.vllm, op_name):
            return
        direct_register_custom_op(
            op_name=op_name,
            op_func=op_func,
            mutates_args=mutates_args,
            fake_impl=fake_impl,
        )

    register_one(
        op_name="afd_p2p_send",
        op_func=afd_p2p_send_impl,
        mutates_args=["ordering_token"],
        fake_impl=afd_p2p_send_fake,
    )
    register_one(
        op_name="afd_p2p_recv",
        op_func=afd_p2p_recv_impl,
        mutates_args=["out", "ordering_token"],
        fake_impl=afd_p2p_recv_fake,
    )
    _AFD_CUSTOM_OPS_REGISTERED = True


__all__ = ["P2pNcclAFDConnector", "P2pNcclAFDControlPlane"]
