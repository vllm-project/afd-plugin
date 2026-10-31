# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Own the F-local Ascend device and native compute groups through execution.

CAM owns a distinct A/F process group. Keeping the native F-only EP context
does not turn on EPLB/EEP or route any per-request work through its collectives.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.npu.settings import (
    FFNConfig,
    validate_static_experts,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

COMPUTE_GROUP_TIMEOUT_SECONDS = 180


class AscendFFNEnvironment:
    """Never borrow unrelated groups or tear them down on a failed native seam."""

    def __init__(self, settings: FFNConfig) -> None:
        self.settings = settings
        self._device: torch.device | None = None
        self._ep_group: GroupCoordinator | None = None
        self._groups_owned = False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Ascend FFN device has not been initialized")
        return self._device

    @property
    def ep_group(self) -> GroupCoordinator:
        if self._ep_group is None:
            raise RuntimeError("Ascend FFN expert group has not been initialized")
        return self._ep_group

    def initialize_device(self) -> None:
        if self._groups_owned or self._device is not None:
            raise RuntimeError("Ascend FFN environment is already initialized")
        settings = self.settings
        try:
            world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                "Ascend FFN requires an external F-only launcher"
            ) from error
        if world != settings.ffn_ranks or rank != settings.rank:
            raise ValueError("Ascend FFN environment differs from the F-only launcher")
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        import torch
        import torch_npu  # noqa: F401
        from vllm.distributed.parallel_state import model_parallel_is_initialized
        from vllm_ascend.utils import enable_custom_op

        from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

        if torch.distributed.is_initialized() or model_parallel_is_initialized():
            raise RuntimeError("Ascend FFN cannot borrow existing native groups")
        ensure_cam_async_ops_available()
        torch.npu.set_device(settings.device)
        self._device = torch.device(f"npu:{settings.device}")
        # Native NPUModelRunner v1 installs this packing policy at import time.
        # This service does not import that runner; native W4A8 still needs NZ.
        torch.npu.config.allow_internal_format = True
        if not enable_custom_op():
            raise RuntimeError("native V4 W4A8 custom operators are unavailable")

    def initialize_compute(self, config: VllmConfig) -> None:
        if self._device is None or self._groups_owned:
            raise RuntimeError("Ascend FFN compute initialization requires its device")
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
            self._groups_owned = True
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
            if (
                self._ep_group.world_size != settings.ffn_ranks
                or self._ep_group.rank_in_group != settings.rank
            ):
                raise ValueError("native V4 expert group differs from the F world")

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

    def close(self) -> None:
        """Only after CAM/executor completion; failures retain parameter owners."""
        if not self._groups_owned:
            return
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()
        self._groups_owned = False
        self._ep_group = None
        self._device = None
