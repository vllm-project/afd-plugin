# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_v1
from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner import (
    AFDAttentionModelRunner,
    fail_if_unsupported_ubatching,
)
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)


class AFDAttentionWorker(Worker):
    """Attention worker that injects :class:`AFDAttentionModelRunner`."""

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

    # Patch reason: vLLM 0.26.0 constructs its runner inside Worker.init_device
    # after an internal module import, with no injectable runner factory.
    # Patch functionality: validate the supported V2 contract, select the AFD model
    # identity before construction, and scope the module-class substitution to
    # the delegated native initialization window.
    # Signature: matches vLLM v0.28.0 Worker.init_device exactly: (self).
    # Upstream source: vllm/v1/worker/gpu_worker.py, Worker.init_device;
    # runner local-import/construct seam; commit
    # 2cf0a6915ce544dc493a0990f2ea38d81601128a.
    # Delegation exception: the large native device/distributed setup remains
    # in super().init_device(); only this missing runner-factory seam is local.
    # Removal/upstream plan: delete this substitution when vLLM adds runner
    # factory/class injection; the upstream target is that factory seam.
    # Concurrency invariant: Worker.init_device runs synchronously during startup,
    # with no concurrent runner construction in the worker process.
    def init_device(self):
        """Initialize the native GPU worker with the selected AFD runner."""

        # ### PATCH START: select and directly construct the AFD runner.
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        if self.use_v2_model_runner:
            validate_gpu_model_runner_v2_config(
                self.vllm_config,
                expected_role="attention",
                device_type=self.device_config.device_type,
            )
            afd_runner_cls = AFDAttentionModelRunnerV2
            native_runner_module = gpu_model_runner_v2
        else:
            fail_if_unsupported_ubatching(self.vllm_config)
            afd_runner_cls = AFDAttentionModelRunner
            native_runner_module = gpu_model_runner_v1

        afd_model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="cuda",
        )
        self.vllm_config.model_config = afd_model_config
        self.model_config = afd_model_config

        native_runner_cls = native_runner_module.GPUModelRunner
        native_runner_module.GPUModelRunner = afd_runner_cls
        try:
            super().init_device()
        finally:
            native_runner_module.GPUModelRunner = native_runner_cls
        # ### PATCH END: select and directly construct the AFD runner.

        torch.accelerator.empty_cache()


__all__ = ["AFDAttentionWorker"]
