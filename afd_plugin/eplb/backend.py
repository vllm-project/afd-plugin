# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Narrow platform backend contract for FFN-side EPLB.

All platform-specific mechanisms (Triton or Ascend custom ops, NCCL or HCCL
communicators, graph-specific update hooks) live behind this structural
``Protocol``. Importing this module does not import torch or vLLM at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from vllm.distributed.eplb.eplb_state import EplbState


@runtime_checkable
class EplbBackend(Protocol):
    """Protocol for platform-specific EPLB backends."""

    def initialize_model(
        self,
        model: torch.nn.Module,
        *,
        load_dummy_weights: bool = False,
    ) -> None:
        """Register the model with the backend EPLB controller/state.

        ``load_dummy_weights`` marks elastic-EP scaling workers that register
        before their process groups are live.
        """
        ...

    def prepare_forward(
        self,
        num_unpadded_tokens: int,
        *,
        is_dummy: bool = False,
    ) -> None:
        """Set unpadded token count and recording gating for the next step."""
        ...

    def step(self, *, is_dummy: bool = False) -> bool:
        """Step the rebalancing loop across EP ranks.

        Returns ``True`` when a rebalance happened on this step.
        """
        ...

    def get_logical_to_physical_map(self) -> torch.Tensor | None:
        """Return the current persistent logical-to-physical routing map."""
        ...

    def get_replica_count(self) -> torch.Tensor | None:
        """Return the current logical replica counts."""
        ...

    def get_eplb_state(self) -> EplbState | None:
        """Return the underlying upstream ``EplbState`` object."""
        ...


__all__ = ["EplbBackend"]
