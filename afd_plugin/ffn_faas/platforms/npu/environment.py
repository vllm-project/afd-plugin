# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Own the F-local Ascend device and native compute groups through execution.

CAM owns a distinct A/F process group. Keeping the native F-only EP context
does not turn on EPLB/EEP or route any per-request work through its collectives.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.environment import FFNEnvironmentBase
from afd_plugin.ffn_faas.platforms.npu.settings import (
    FFNConfig,
    validate_static_experts,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

COMPUTE_GROUP_TIMEOUT_SECONDS = 180


# GPU parallel: afd_plugin/ffn_faas/platforms/gpu/environment.py::
#   GPUFFNEnvironment.
# Reuse: PARTIAL - the shared base owns state, launcher checks, ownership
#   flags, the EP-group check and close-state handling; this class owns Ascend
#   device selection, custom operators, HCCL groups and CPU binding.
class AscendFFNEnvironment(FFNEnvironmentBase[FFNConfig]):
    """Never borrow unrelated groups or tear them down on a failed native seam."""

    def __init__(self, settings: FFNConfig) -> None:
        super().__init__(settings, platform_name="Ascend")

    def initialize_device(self) -> None:
        self.require_uninitialized()
        self.require_launcher_environment()
        settings = self.settings
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        import torch
        import torch_npu  # noqa: F401
        from vllm_ascend.utils import enable_custom_op

        from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

        self.require_no_borrowed_native_groups()
        ensure_cam_async_ops_available()
        torch.npu.set_device(settings.device)
        self._device = torch.device(f"npu:{settings.device}")
        # Native NPUModelRunner v1 installs this packing policy at import time.
        # This service does not import that runner; native W4A8 still needs NZ.
        torch.npu.config.allow_internal_format = True
        if not enable_custom_op():
            raise RuntimeError("native V4 W4A8 custom operators are unavailable")

    def initialize_compute(self, config: VllmConfig) -> None:
        self.require_compute_initializable()
        settings = self.settings
        parallel = config.parallel_config
        validate_static_experts(
            enable_eplb=parallel.enable_eplb,
            enable_elastic_ep=parallel.enable_elastic_ep,
            redundant_experts=parallel.eplb_config.num_redundant_experts,
        )
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.distributed.parallel_state import (
            get_ep_group,
        )
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
        from vllm_ascend.utils import register_ascend_customop

        with set_current_vllm_config(config):
            register_ascend_customop(config)
            init_ascend_config(config)
            # Ownership starts before a collective can partially initialize.
            self.mark_groups_owned()
            init_distributed_environment(
                world_size=settings.ffn_ranks,
                rank=settings.rank,
                local_rank=settings.device,
                distributed_init_method="env://",
                backend="hccl",
                timeout=timedelta(seconds=COMPUTE_GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                backend="hccl",
            )
            init_ascend_model_parallel(parallel)
            self._ep_group = get_ep_group()
            self.validate_ep_group(
                error_message="native V4 expert group differs from the F world"
            )

    def activate_thread(self) -> None:
        import torch

        torch.npu.set_device(self.device)

    def bind_host_threads(self, config: VllmConfig) -> None:
        """Reuse the native Worker's post-warmup affinity policy, without a Worker.

        The receiver is created afterward and inherits the main-thread affinity.
        Like native Worker startup, a binding failure is logged, not hidden as a
        successful affinity proof. Qualification records actual /proc affinity.
        """
        from vllm.platforms import current_platform
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.cpu_binding import bind_cpus

        if not get_ascend_config().enable_cpu_binding:
            return
        engram = config.engram_config
        try:
            bind_cpus(
                self.settings.local_rank,
                npu_id=current_platform.device_id_to_physical_device_id(
                    self.settings.device
                ),
                migrate_memory=not (engram is not None and engram.cpu_offload),
            )
        except Exception as error:
            logging.getLogger(__name__).warning(
                "Native CPU binding failed for F rank %s: %s",
                self.settings.rank,
                error,
            )

    def _destroy_groups(self) -> None:
        """Only after CAM/executor completion; failures retain parameter owners."""
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()
