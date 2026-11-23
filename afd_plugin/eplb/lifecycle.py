# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Platform-neutral EPLB lifecycle helper composed into FFN model runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.eplb.eplb_state import EplbState

    from afd_plugin.config import AFDConfig
    from afd_plugin.eplb.backend import EplbBackend


class FFNEplbLifecycle:
    """Platform-neutral EPLB lifecycle manager for FFN model runners.

    The helper owns no device tensors. It counts steps and forwards the
    runner's call points to the configured ``EplbBackend``. When no backend is
    configured every call is a no-op, so runners can call it unconditionally.
    All FFN ranks in the EP group must call ``step`` at identical logical step
    indices, including idle and dummy steps.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        backend: EplbBackend | None,
    ) -> None:
        self.vllm_config = vllm_config
        self.afd_config = afd_config
        self.backend = backend
        self._step_counter = 0

    @property
    def is_enabled(self) -> bool:
        return self.backend is not None

    @property
    def step_count(self) -> int:
        """Number of ``step`` calls seen, including dummy steps."""
        return self._step_counter

    def initialize_model(
        self,
        model: torch.nn.Module,
        *,
        load_dummy_weights: bool = False,
    ) -> None:
        if self.backend is not None:
            self.backend.initialize_model(
                model,
                load_dummy_weights=load_dummy_weights,
            )

    def prepare_forward(
        self,
        num_unpadded_tokens: int,
        *,
        is_dummy: bool = False,
    ) -> None:
        """Called before every forward pass and graph capture/replay.

        ``num_unpadded_tokens`` must exclude padding rows so padding never
        contributes to expert load.
        """
        if self.backend is None:
            return
        self.backend.prepare_forward(
            num_unpadded_tokens,
            is_dummy=is_dummy,
        )

    def step(self, *, is_dummy: bool = False) -> bool:
        """Called after every forward pass and dummy step.

        Dummy steps still advance the step counter so all ranks stay in
        lockstep. Returns ``True`` if a rebalance happened on this step.
        """
        if self.backend is None:
            return False
        self._step_counter += 1
        return self.backend.step(is_dummy=is_dummy)

    def get_eplb_state(self) -> EplbState | None:
        if self.backend is None:
            return None
        return self.backend.get_eplb_state()


__all__ = ["FFNEplbLifecycle"]
