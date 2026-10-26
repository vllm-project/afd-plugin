# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Worker device resolution before vLLM's ``Worker.init_device``.

A connector that joins the AFD world before vLLM's distributed init is created
before vLLM selects the worker's device. These helpers mirror that selection
(vLLM v0.30.0 ``vllm/v1/worker/gpu_worker.py``, ``Worker.init_device``, commit
ced6857); the worker checks the result against the device vLLM then selects.
vLLM is imported inside the functions, so importing this module stays
CPU-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# vLLM's GPU worker skips its local data-parallel device offset for these.
_NO_DP_DEVICE_OFFSET_EXECUTORS = ("ray", "external_launcher")


def register_assigned_physical_gpu_ids(vllm_config: VllmConfig) -> None:
    """Register ``--device-ids`` with vLLM's platform, as ``init_device`` does.

    Side effect: sets vLLM's process-wide assigned physical GPU ids, through
    which vLLM maps logical device indices. Call it before
    ``resolve_worker_device_index``. vLLM's setter accepts the same ids again
    when ``init_device`` registers them later; different ids raise.
    """
    assigned_physical_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
    if assigned_physical_gpu_ids is None:
        return
    from vllm.platforms.interface import set_assigned_physical_gpu_ids

    set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)


def resolve_worker_device_index(vllm_config: VllmConfig, local_rank: int) -> int:
    """CUDA device index vLLM's GPU worker will select for ``local_rank``.

    Adds the local data-parallel offset, then maps the logical index through
    the assigned physical ids (``--device-ids``, registered by
    ``register_assigned_physical_gpu_ids``) and ``CUDA_VISIBLE_DEVICES``.
    """
    from vllm.platforms import current_platform
    from vllm.platforms.interface import get_assigned_physical_gpu_ids

    parallel_config = vllm_config.parallel_config
    assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
    if (
        assigned_physical_gpu_ids is not None
        and get_assigned_physical_gpu_ids() != assigned_physical_gpu_ids
    ):
        raise RuntimeError(
            "call register_assigned_physical_gpu_ids before resolving the "
            "worker device: vLLM maps device indices through those ids"
        )
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
    return current_platform.logical_device_id_to_visible_device_id(local_rank)


__all__ = [
    "register_assigned_physical_gpu_ids",
    "resolve_worker_device_index",
]
