# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU EPLB backend built on vLLM's ``EPLBController``.

This module is device-only: it imports torch and vLLM at module load, so the
platform-neutral ``afd_plugin.eplb`` package never imports it by accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.model_executor.models.interfaces import MixtureOfExperts


class GpuVllmEplbBackend:
    """FFN-side EPLB backend for vLLM's GPU ``EplbState``/``EPLBController``.

    The backend owns the controller and exposes only the narrow operations the
    platform-neutral lifecycle needs, so the FFN runner never touches vLLM's
    EPLB internals directly.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        from vllm.v1.worker.gpu.eplb_utils import EPLBController

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.controller = EPLBController(vllm_config.parallel_config, device)

    def initialize_model(
        self,
        model: MixtureOfExperts,
        *,
        load_dummy_weights: bool = False,
    ) -> None:
        """Create the EPLB state and register the FFN MoE model with it."""
        controller = self.controller
        controller.prepare_load()
        models_added = controller.maybe_register_model(model, self.model_config)
        controller.maybe_start_async_loop(models_added, load_dummy_weights)

    def prepare_forward(
        self,
        num_unpadded_tokens: int,
        *,
        is_dummy: bool = False,
    ) -> None:
        """Publish the step's real token count to the load-record kernel."""
        self.controller.prepare_forward(self.model_config, num_unpadded_tokens)

    def step(self, *, is_dummy: bool = False) -> bool:
        """Step EPLB once per connector step.

        AFD rejects async EPLB, so a rearrangement happens exactly when this
        step reaches the configured interval. The counter is incremented even
        for dummy steps so all FFN ranks stay in lockstep.
        """
        state = self.controller.state
        rebalances = (
            state is not None
            and state.expert_rearrangement_step + 1
            >= state.expert_rearrangement_step_interval
        )
        self.controller.step(is_dummy=is_dummy)
        return rebalances

    def get_logical_to_physical_map(self) -> torch.Tensor | None:
        state = self.controller.state
        if state is None:
            return None
        model_state = next(iter(state.model_states.values()), None)
        if model_state is None:
            return None
        return model_state.logical_to_physical_map

    def get_replica_count(self) -> torch.Tensor | None:
        state = self.controller.state
        if state is None:
            return None
        model_state = next(iter(state.model_states.values()), None)
        if model_state is None:
            return None
        return model_state.logical_replica_count

    def get_eplb_state(self) -> EplbState | None:
        return self.controller.state


def create_gpu_eplb_backend(
    vllm_config: VllmConfig,
    device: torch.device,
) -> GpuVllmEplbBackend | None:
    """Return the GPU backend, or ``None`` when EPLB is disabled."""
    if not vllm_config.parallel_config.enable_eplb:
        return None
    return GpuVllmEplbBackend(vllm_config, device)


__all__ = ["GpuVllmEplbBackend", "create_gpu_eplb_backend"]
