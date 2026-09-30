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
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
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
            validate_npu_model_runner_v2_config(
                self.vllm_config,
                expected_role="attention",
                device_type="npu",
            )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        # Keep worker resolution independent of the unselected runner.
        if self.use_v2_model_runner:
            from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
                AFDNPUAttentionModelRunnerV2 as AFDModelRunner,
            )
        else:
            from afd_plugin.v1.worker.npu.attention_model_runner import (
                AFDNPUAttentionModelRunner as AFDModelRunner,
            )

        self.model_runner = AFDModelRunner(
            self.vllm_config,
            self.device,
        )


__all__ = ["AFDNPUAttentionWorker"]
