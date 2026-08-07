# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector contract for AFD Attention/FFN communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from afd_plugin.config import AFDConfig, connector_extra_config_from_source
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDTransferContext,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class ConnectorExtraInfo:
    """Base type for connector-owned configuration."""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> ConnectorExtraInfo:
        raise NotImplementedError

    def to_mapping(self) -> dict[str, Any]:
        return {}


class AFDConnectorBase(ABC):
    """Base class for plugin-owned AFD Attention/FFN connectors.

    An AFD connector owns the communication contract between the Attention
    runtime and the FFN runtime. Implementations are responsible for three
    kinds of work:

    1. Initializing and releasing backend communication resources.
    2. Moving hidden states between Attention and FFN ranks.
    3. Applying DP metadata control-plane payloads so later tensor transfers
       can use backend-specific shapes, buffers, or connector data.

    Implementations may use very different transport mechanisms. For example,
    the GPU P2P connector uses explicit point-to-point tensor transfers, while
    the NPU CAMP2P connector carries additional backend state through its
    ``AFDTransferState`` subclass on ``AFDTransferContext.states``. This base
    class documents the common runtime contract shared by those implementations.
    """

    # An optional DP metadata control plane. When set, FFN steps are driven by
    # DP metadata received over the control plane. When ``None``, the connector
    # has no control plane and FFN steps are driven directly by the connector
    # receive loop.
    control_plane: AFDControlPlane | None = None
    attn_size: int = 0
    ffn_size: int = 0

    @classmethod
    @abstractmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        """Parse connector-owned config using this connector's schema."""

        raise NotImplementedError

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        """Initialize common connector context.

        Args:
            rank: The rank value passed by the owning vLLM worker at connector
                construction time. This is the process rank known to the caller
                and is not necessarily the same as a connector-specific
                ``world_rank`` computed by a backend topology.
            local_rank: Device-local rank for selecting the local accelerator.
                GPU implementations generally use this as the CUDA device
                index; NPU implementations use it as the NPU device index.
            vllm_config: Upstream vLLM ``VllmConfig`` for the current worker.
                Connectors use it to read model shape, dtype, parallelism, and
                graph/capture configuration.
            afd_config: Parsed AFD plugin configuration. This contains the AFD
                role, connector name, topology sizes, and host/port.
            role_rank: Runtime rank resolved from this worker's global
                DP/PCP/TP placement within its AFD role group.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.vllm_config = vllm_config
        self.afd_config = afd_config
        self.role_rank = role_rank
        self.extra_info = self.parse_extra_config(
            connector_extra_config_from_source(vllm_config),
        )

    # ==============================
    # Lifecycle methods
    # ==============================

    @abstractmethod
    def close(self) -> None:
        """Release connector-owned communication resources.

        Implementations should destroy process groups, communicators, cached
        handles, and backend-specific state owned by the connector. The method
        should be safe to call during worker shutdown and should leave
        ``is_initialized`` false after successful cleanup.
        """
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        """Initialize backend communication resources.

        This method is called once the owning runtime is ready to create AFD
        communication state. Implementations typically create process groups,
        register custom ops, initialize backend communicators, and precompute
        topology-derived rank lists.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Return whether backend communication resources are initialized."""
        raise NotImplementedError

    # ==============================
    # Attention-side data path
    # ==============================

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send Attention hidden states to the FFN runtime.

        This method is called on Attention-side ranks after an Attention block
        produces hidden states for one layer/stage.

        Args:
            hidden_states: Tensor to send to the FFN side. The leading
                dimension should match ``context.metadata.total_tokens`` unless
                the backend explicitly supports a different compiled/captured
                calling convention.
            context: Per-transfer context describing layer, stage, and token
                layout. Backends may also consume ``context.states``.
            **kwargs: Optional backend-specific arguments. For an
                experts-boundary transfer, ``router_logits`` and
                ``routing_spec`` must be supplied together. The latter is an
                ``AFDExpertRoutingSpec`` shared with the FFN receiver and
                defines the router tensor's wire shape and dtype.

        Raises:
            ValueError: If tensor shape does not match metadata.
            RuntimeError: If the connector is not initialized or required
                backend-specific metadata is missing.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Receive FFN output on the Attention side.

        This method returns the tensor that should continue through the
        Attention-side model execution after FFN computation completes.

        Args:
            ref_tensor: Preallocated tensor for the receive operation. Every
                backend needs it: CAMP2P and CAM async cannot allocate their
                own output tensor, and P2P uses it both as a stable buffer for
                CUDA graph capture and as the result when a single-rank
                subgroup performs no wire transfer.
            ubatch_idx: Stage/microbatch index. Defaults to ``0``.
            **kwargs: Optional backend-specific receive arguments.

        Returns:
            The FFN output tensor for the current layer/stage.

        Raises:
            RuntimeError: If required backend state or arguments are missing.
        """
        raise NotImplementedError

    # ==============================
    # FFN-side data path
    # ==============================

    @abstractmethod
    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        """Receive Attention hidden states on the FFN side.

        This method is called by FFN-side execution before running the FFN
        compute for a layer/stage. It returns both the hidden states and the
        metadata needed by later FFN-side calls.

        Args:
            ubatch_idx: Microbatch/stage index to receive. Defaults to ``0``.
            **kwargs: Optional backend-specific receive arguments.

        Returns:
            ``AFDA2FTransferPayload`` containing received hidden states and the
            transfer context (metadata + backend-produced transfer state)
            for the receive operation.
        """
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Send FFN output back to the Attention runtime.

        Args:
            ffn_output: Tensor produced by the FFN computation. The leading
                dimension should match ``context.metadata.total_tokens`` unless
                the backend explicitly supports a different compiled/captured
                calling convention.
            context: Transfer context associated with the matching
                ``recv_attn_output`` call. Backends may depend on
                ``context.states`` that was prepared before receive or updated
                after receive.
            **kwargs: Optional backend-specific send arguments such as
                ``ubatch_idx`` or op-specific metadata.

        Raises:
            ValueError: If tensor shape does not match metadata.
            RuntimeError: If required backend state or connector data is
                missing.
        """
        raise NotImplementedError


class AFDControlPlane(ABC):
    """DP metadata control plane"""

    @abstractmethod
    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        """Apply a DP metadata payload to local connector state.

        This is a local state update, not a network send. Implementations use it
        to store DP metadata and flags, derive tensor metadata, and prepare
        backend buffers or connector-specific state before data-path calls.

        Args:
            payload: Structured DP metadata control-plane payload. It contains
                per-stage DP token metadata plus graph-capturing and warmup
                flags.
        """
        raise NotImplementedError

    @abstractmethod
    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        """Submit DP metadata control-plane payload from Attention to FFN.

        The caller does not need to decide whether the current rank is a sender.
        Implementations should encapsulate their topology-specific sender-rank
        checks and no-op on ranks that should not transmit DP metadata.

        Args:
            payload: Structured DP metadata payload to make available to the
                FFN-side connector loop.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_dp_metadata_list(self) -> AFDControlPayload:
        """Receive a DP metadata control-plane payload on the FFN side.

        Returns:
            Structured DP metadata payload received from the Attention side.

        Raises:
            RuntimeError: If the control-plane communication group is not
                initialized or unsupported by this connector.
        """
        raise NotImplementedError


__all__ = ["AFDConnectorBase", "AFDControlPlane", "ConnectorExtraInfo"]
