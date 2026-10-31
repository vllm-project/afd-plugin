# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.model_executor.models.model_utils import (
    get_afd_model_config,
    install_afd_speculative_model_config,
)
from afd_plugin.v1.worker.npu.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        if self.use_v2_model_runner:
            raise RuntimeError(
                "AFD NPU Attention supports only vllm-ascend model runner v1",
            )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
        )
        install_afd_speculative_model_config(self.vllm_config)
        self.model_runner = AFDNPUAttentionModelRunner(
            self.vllm_config,
            self.device,
        )

    # Upstream: vllm_ascend/worker/worker.py at 11ee45653.
    # Patch reason: the AFD runner owns its profiler instead of NPUWorker.
    # Patch functionality: route profile RPCs to the role-aware AFD profiler.
    # Signature: matches upstream; no added parameters.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Control the plugin-owned profiler from vLLM's profile endpoint."""
        # ### PATCH START: AFD profiler control
        del profile_prefix
        if is_start:
            self.model_runner.start_afd_profiler()
        else:
            self.model_runner.stop_afd_profiler()
        # ### PATCH END: AFD profiler control

    # Upstream source: vLLM-Ascend 3da28f94, NPUWorker.execute_dummy_batch.
    # Patch reason: online idle DP ranks also need the MTP phase handshake;
    # FFN must not post draft device receives before target sampling completes.
    # Patch functionality: distinguish serving dummy work from startup capture.
    # Signature: matches upstream; no added parameters.
    def execute_dummy_batch(self) -> None:
        # ### PATCH START: AFD online dummy scope
        self.model_runner._afd_live_dummy_execution = True
        try:
            super().execute_dummy_batch()
        finally:
            self.model_runner._afd_live_dummy_execution = False
        # ### PATCH END: AFD online dummy scope


__all__ = ["AFDNPUAttentionWorker"]
