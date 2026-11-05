# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native A actor with GPU assignment isolated from colocated F ranks."""

from __future__ import annotations

import os

from vllm.config import VllmConfig
from vllm.v1.engine.core import DPMoEEngineCoreActor
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.executor import Executor


def assigned_bundle_devices() -> list[str]:
    import ray
    from vllm.platforms import current_platform

    return ray.get_runtime_context().get_accelerator_ids()[
        current_platform.ray_device_key
    ]


class AFDEngineCoreActor(DPMoEEngineCoreActor):
    """EEP owns creation, run refs, notifications and removal exactly as before."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        from vllm.plugins import load_general_plugins

        # Actor construction precedes EngineCore's normal general-plugin load.
        # Install the post-READY pause and completion utility before entering it.
        load_general_plugins()
        super().__init__(
            vllm_config,
            local_client,
            addresses,
            executor_class,
            log_stats,
            dp_rank,
            local_dp_rank,
        )

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from vllm.platforms import current_platform

        device_key = current_platform.ray_device_key
        # EEP's CPU actor lives in bundle 1 (TP=1). Query reserved device bundle
        # 0 through a short zero-CPU task; Ray's assignment is stable within
        # this one-device bundle. No extra device or placement group is
        # allocated. Ray maps num_gpus to the "GPU" resource only; non-CUDA
        # accelerators (Ray detects Ascend NPUs as "NPU") go through resources.
        device_options: dict[str, object] = (
            {"num_gpus": 1}
            if device_key == "GPU"
            else {"num_gpus": 0, "resources": {device_key: 1}}
        )
        ids = ray.get(
            ray.remote(assigned_bundle_devices)
            .options(
                num_cpus=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=vllm_config.parallel_config.placement_group,
                    placement_group_bundle_index=0,
                ),
                **device_options,
            )
            .remote()
        )
        os.environ[current_platform.device_control_env_var] = ",".join(ids)
        parallel = vllm_config.parallel_config
        parallel.assigned_physical_gpu_ids = [int(device) for device in ids]
        # The actor sees one device. The native used-device count can include
        # F and exceed the global A rank, violating _init_data_parallel's local
        # rank invariant. It is not a device index in this isolated process.
        parallel.data_parallel_rank_local = 0
