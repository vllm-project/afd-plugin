# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AFDAttentionWorker(Worker):
    """Attention worker that constructs the AFD ModelRunnerV2 extension."""

    afd_expected_role = "attention"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )

    # Patch reason: native device initialization does not validate the AFD role
    # or select its model architecture before allocating GPU resources.
    # Patch functionality: validate and select the AFD model before delegating
    # native device, distributed, workspace and runner initialization.
    # Signature: matches vLLM v0.32.0 Worker.init_device exactly: (self).
    # Upstream source: vllm/v1/worker/gpu_worker.py, Worker.init_device;
    # 5df63a7203cdb9328d824a27b682e77c7b9928be.
    # Delegation exception: the large native device/distributed setup remains
    # in super().init_device(); this override contains only AFD preflight.
    # Removal plan: remove when native worker startup supports AFD preflight.
    def init_device(self):
        """Initialize the native GPU worker with the selected AFD runner."""

        # ### PATCH START: validate and select the AFD model.
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        validate_gpu_model_runner_v2_config(
            self.vllm_config,
            expected_role="attention",
            device_type=self.device_config.device_type,
        )

        afd_model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="cuda",
        )
        self.vllm_config.model_config = afd_model_config
        self.model_config = afd_model_config
        # ### PATCH END: validate and select the AFD model.

        super().init_device()
        torch.accelerator.empty_cache()

    # The native factory still annotates V1 and uses a local return-type
    # exception for V2. Keep that signature and the same typing boundary.
    def _make_model_runner(self) -> GPUModelRunner:
        """Construct the AFD runner through the native worker factory hook."""
        return AFDAttentionModelRunnerV2(self.vllm_config, self.device)  # type: ignore[return-value]


__all__ = ["AFDAttentionWorker"]
