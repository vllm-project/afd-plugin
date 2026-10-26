# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Contract for connectors whose FFN side receives routed expert rows.

Routing (gate and top-k) runs on Attention. The FFN receives one row per
(token, local expert) pair, runs only its local experts and sends the weighted
rows back. Workers and runners use ``AFDRoutedExpertsConnector`` and the types
here instead of any specific connector, and this module imports no connector
backend, so they stay connector-agnostic. ``AFDConnectorBase`` is unchanged.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import AFDA2FTransferPayload

FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class LocalExpertWeights:
    """This FFN rank's experts, in vLLM ``FusedMoE`` layout.

    Attributes:
        w13: ``[E_local, 2 * I, H]`` gate and up projections (gate first),
            BF16 or FP8.
        w2: ``[E_local, H, I]`` down projection, same dtype as ``w13``.
        w13_scale: FP8 only: ``[E_local, 2 * I / 128, H / 128]`` fp32 block
            scales (vLLM ``w13_weight_scale_inv``).
        w2_scale: FP8 only: ``[E_local, H / 128, I / 128]`` fp32 block scales.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.w13.dtype != self.w2.dtype:
            raise ValueError("w13 and w2 must have the same dtype")
        if self.is_fp8 != (self.w13_scale is not None and self.w2_scale is not None):
            raise ValueError(
                "FP8 expert weights need both block scales (and BF16 none)"
            )

    @property
    def num_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def is_fp8(self) -> bool:
        return self.w13.dtype == FP8_DTYPE


@dataclass(frozen=True)
class FFNLocalExperts:
    """One MoE layer's experts on this FFN rank, as vLLM loaded them.

    Attributes:
        expert_ids: Global ids of the experts in ``weights``, in slot order.
        weights: The same experts in the executor's layout.
    """

    expert_ids: tuple[int, ...]
    weights: LocalExpertWeights


class AFDRoutedExpertsConnector(AFDConnectorBase):
    """A connector whose FFN side receives routed (token, local expert) rows.

    Lifecycle: the worker creates the connector and calls
    ``prepare_before_distributed_init`` before vLLM's distributed init (it may
    set process-wide NCCL environment, which NCCL reads when the process
    creates its first communicator), then hands the same instance to its
    runner, which calls ``init_afd_connector`` after weights load, with what
    it loaded (``afd_plugin/v1/worker/connector_lifecycle.py``). The FFN
    worker selects the routed-experts runner for these connectors.
    """

    @abstractmethod
    def prepare_before_distributed_init(self) -> None:
        """Collective: join the AFD world before vLLM's distributed init."""
        raise NotImplementedError

    # Extends AFDConnectorBase.init_afd_connector with optional keywords only,
    # so calls through the base contract stay valid.
    @abstractmethod
    def init_afd_connector(
        self,
        *,
        moe_layer_indices: Sequence[int] | None = None,
        local_experts: Mapping[int, FFNLocalExperts] | None = None,
    ) -> None:
        """Collective: check what every rank loaded, then create resources.

        Args:
            moe_layer_indices: MoE layers this rank's model runs, in order.
                ``None`` skips the post-load checks for this rank.
            local_experts: FFN only: this rank's experts per MoE layer, as
                vLLM loaded them.
        """
        raise NotImplementedError

    @abstractmethod
    def run_experts(
        self,
        payload: AFDA2FTransferPayload,
        weights: LocalExpertWeights,
    ) -> torch.Tensor:
        """Run this rank's experts on the rows ``recv_attn_output`` returned;
        the result is what ``send_ffn_output`` sends back."""
        raise NotImplementedError

    @abstractmethod
    def synchronize_lanes(self) -> None:
        """Join the streams the connector forked into the caller's stream.

        FFN runners call this at the end of each step, before a CUDA-graph
        capture ends.
        """
        raise NotImplementedError


__all__ = [
    "AFDRoutedExpertsConnector",
    "FFNLocalExperts",
    "FP8_DTYPE",
    "LocalExpertWeights",
]
