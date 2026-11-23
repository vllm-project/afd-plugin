# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend NPU EPLB backend built on the vLLM-Ascend V2 EPLB stack.

This module is device-only: it imports torch and vLLM-Ascend lazily so the
platform-neutral ``afd_plugin.eplb`` package stays CPU-importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.model_executor.models.interfaces import MixtureOfExperts


class AscendEplbBackend:
    """FFN-side EPLB backend for vLLM-Ascend's ``AscendEPLBController``.

    AFD NPU runs the V2 model-runner stack, where ``dynamic_eplb`` is forced
    off and expert weights are bound through ``use_expert_weight_list`` when
    EPLB is enabled. Load is recorded by the Ascend routed-experts operator
    from its ``expert_tokens`` output, so the backend only has to publish the
    unpadded token count / recording gate and step the state.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.worker.v2.eplb import AscendEPLBController

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.controller = AscendEPLBController(
            parallel_config,
            device,
            get_ascend_config().eplb_config if parallel_config.enable_eplb else None,
        )

    def initialize_model(
        self,
        model: MixtureOfExperts,
        *,
        load_dummy_weights: bool = False,
    ) -> None:
        """Create the Ascend EPLB state and register the FFN MoE model."""
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
        """Publish the step's unpadded count and operator recording gate."""
        self.controller.prepare_forward(self.model_config, num_unpadded_tokens)

    def step(self, *, is_dummy: bool = False) -> bool:
        """Step EPLB once per connector step, including idle/dummy steps."""
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


def create_ascend_eplb_backend(
    vllm_config: VllmConfig,
    device: torch.device,
) -> AscendEplbBackend | None:
    """Return the NPU backend, or ``None`` when EPLB is disabled."""
    if not vllm_config.parallel_config.enable_eplb:
        return None
    return AscendEplbBackend(vllm_config, device)


__all__ = ["AscendEplbBackend", "create_ascend_eplb_backend"]
