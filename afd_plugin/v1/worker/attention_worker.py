# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes

from afd_plugin.elastic.config import is_elastic_attention
from afd_plugin.elastic.gpu import (
    AFDElasticGPUExecutor,
    connect,
    finish_attention,
    release_link,
    send_stop,
    update_topology,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
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
        self.afd_runtime_deferred = is_elastic_attention(vllm_config)
        self.afd_kv_cache_config: KVCacheConfig | None = None
        self.afd_kv_initialized_runner: object | None = None
        if self.afd_runtime_deferred:
            self.elastic_ep_executor = AFDElasticGPUExecutor(self)

    # Native explicit KV budgets still run profile. Delay that forward until F
    # is receiving. Signature matches vLLM 0.30.0 Worker.determine_available_memory.
    def determine_available_memory(self) -> int:
        # ### PATCH START: defer AFD forward until cross-role rendezvous.
        if self.afd_runtime_deferred:
            budget = self.cache_config.kv_cache_memory_bytes
            assert budget is not None and budget > 0
            return budget
        # ### PATCH END: defer AFD forward until cross-role rendezvous.
        return super().determine_available_memory()

    # Preserve the complete upstream allocator/KV-zero/input initialization.
    # Signature matches vLLM 0.30.0 Worker.initialize_from_config.
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # ### PATCH START: save config, never retain an old KV tensor allocation.
        if is_elastic_attention(self.vllm_config):
            self.afd_kv_cache_config = deepcopy(kv_cache_config)
        # ### PATCH END: save config, never retain an old KV tensor allocation.
        super().initialize_from_config(kv_cache_config)
        # ### PATCH START: stamp the runner holding THIS config's KV so
        # elastic finish_attention reuses it instead of double-allocating.
        if is_elastic_attention(self.vllm_config):
            self.afd_kv_initialized_runner = self.model_runner
        # ### PATCH END: stamp the applied runner.

    # Return the native contract during deferred EEP initialization; real warmup
    # follows after all target engines reach COMPLETE. Signature unchanged.
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # ### PATCH START: EEP must not forward into stopped F workers.
        if self.afd_runtime_deferred:
            return CompilationTimes(language_model=0.0, encoder=0.0)
        # ### PATCH END: EEP must not forward into stopped F workers.
        return super().compile_or_warm_up_model()

    def afd_send_stop(self) -> None:
        send_stop(self)

    def afd_release_link(self) -> None:
        release_link(self)

    def afd_update_topology(self, afd: dict[str, str | int | bool]) -> None:
        update_topology(self, afd)

    def afd_connect(self) -> None:
        connect(self)

    def afd_finish(self) -> None:
        finish_attention(self)

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
