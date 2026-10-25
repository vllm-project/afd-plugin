# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA device for connectors created before vLLM's ``Worker.init_device``.

A routed-experts connector bootstraps before vLLM's distributed init, so it is
created before vLLM selects the worker's device. ``connector_device_index``
mirrors that selection; ``check_connector_device`` compares the result with
the device vLLM chose, so a drift after a vLLM upgrade fails at startup.
``check_connector_role_rank`` does the same for the role rank, whose TP part
is derived from the worker rank before vLLM's TP group exists.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import set_assigned_physical_gpu_ids

from afd_plugin.connectors import AFDConnectorBase
from afd_plugin.distributed import resolve_role_rank

# vLLM's GPU worker skips its local data-parallel device offset for these.
_NO_DP_DEVICE_OFFSET_EXECUTORS = ("ray", "external_launcher")


def connector_device_index(vllm_config: VllmConfig, local_rank: int) -> int:
    """CUDA device index vLLM's GPU worker will select for ``local_rank``.

    Mirrors ``Worker.init_device`` (vLLM v0.30.0, ``vllm/v1/worker/
    gpu_worker.py``, commit ced6857): add the local data-parallel offset,
    then map the logical index through ``assigned_physical_gpu_ids``
    (``--device-ids``) and ``CUDA_VISIBLE_DEVICES``. Registering the assigned
    ids here is what vLLM does next; its setter accepts the same value again.
    """
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.distributed_executor_backend
        not in _NO_DP_DEVICE_OFFSET_EXECUTORS
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
    ):
        dp_local_rank = parallel_config.data_parallel_rank_local
        if dp_local_rank is None:
            dp_local_rank = parallel_config.data_parallel_index
        tp_pp_world_size = (
            parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        )
        local_rank += dp_local_rank * tp_pp_world_size
    assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
    if assigned_physical_gpu_ids is not None:
        set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
    return current_platform.logical_device_id_to_visible_device_id(local_rank)


def check_connector_device(
    connector: AFDConnectorBase,
    worker_device: torch.device,
) -> None:
    """Fail when a connector does not use the device vLLM gave the worker."""
    if connector.local_rank != worker_device.index:
        raise RuntimeError(
            f"{type(connector).__name__} uses cuda:{connector.local_rank} but "
            f"vLLM placed this worker on {worker_device}; the device selection "
            "in afd_plugin/v1/worker/connector_device.py no longer matches "
            "vLLM's Worker.init_device",
        )


def check_connector_role_rank(
    connector: AFDConnectorBase,
    vllm_config: VllmConfig,
) -> None:
    """Fail when a connector's role rank differs from vLLM's distributed ranks.

    Call after vLLM's distributed init, which creates the TP group.
    """
    role_rank = resolve_role_rank(vllm_config, connector.afd_config)
    if connector.role_rank != role_rank:
        raise RuntimeError(
            f"{type(connector).__name__} has role rank {connector.role_rank} but "
            f"vLLM's distributed ranks give {role_rank}; the TP rank derived "
            "before distributed init (tp_rank_before_distributed_init) no longer "
            "matches vLLM's rank layout",
        )


__all__ = [
    "check_connector_device",
    "check_connector_role_rank",
    "connector_device_index",
]
