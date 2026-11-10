# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""ParallelConfig seam copied from vLLM 0.30.0, commit ced6857afa0e."""

from __future__ import annotations

import os

import vllm.envs as envs
from vllm.config.parallel import DistributedExecutorBackend, ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list

from afd_plugin.elastic.config import is_elastic_attention_worker

logger = init_logger(__name__)


# Patch reason: AFD A workers do not host EPLB-managed experts.
# Change: scope the EPLB prerequisite exemption to the explicit AFD worker class;
# keep every other upstream validation and normalization step unchanged.
# Signature: exact ParallelConfig.__post_init__(self) -> None.
# Removal/upstream target: configurable EEP worker executor requirements.
def __post_init__(self) -> None:
    # Continue with the rest of the initialization
    self.world_size = (
        self.pipeline_parallel_size
        * self.tensor_parallel_size
        * self.prefill_context_parallel_size
    )

    if self.distributed_executor_backend == "external_launcher":
        logger.info("Using external launcher for distributed inference.")
        self.world_size *= self.data_parallel_size

    if self.enable_elastic_ep:
        # ### PATCH START: AFD A does not own local experts.
        if not self.enable_eplb and not is_elastic_attention_worker(self):
            raise ValueError("Elastic EP is only supported with enable_eplb=True.")
        # ### PATCH END: AFD A does not own local experts.
        if self.pipeline_parallel_size > 1:
            raise ValueError(
                "Elastic EP is not supported with pipeline parallelism "
                f"(pipeline_parallel_size={self.pipeline_parallel_size})."
            )
        if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
            raise NotImplementedError(
                "Elastic EP is not compatible with data_parallel_external_lb "
                "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                "server and core client to coordinate scale up/down."
            )
        # ### PATCH START: no async EPLB/NIXL dependency for AFD attention.
        if self.eplb_config.use_async and not is_elastic_attention_worker(self):
            from vllm.distributed.nixl_utils import is_nixl_available

            if not is_nixl_available():
                raise ValueError(
                    "Elastic EP with async EPLB requires the NIXL "
                    "package. Either install NIXL or set "
                    "--eplb-config.use_async=false."
                )
            # ### PATCH END: no async EPLB/NIXL dependency for AFD attention.

    if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
        # Data parallel was specified in the engine args.
        if self.distributed_executor_backend == "external_launcher":
            # For external launcher,
            # we need to set the data parallel rank automatically
            self.data_parallel_rank = int(os.environ["RANK"]) // (
                self.world_size // self.data_parallel_size
            )
            logger.info(
                "Set data_parallel_rank to %d automatically.",
                self.data_parallel_rank,
            )
        if not self.enable_elastic_ep:
            if not self._data_parallel_master_port_list:
                self._data_parallel_master_port_list = get_open_ports_list(5)
            self.data_parallel_master_port = self._data_parallel_master_port_list.pop()

        if not (0 <= self.data_parallel_rank < self.data_parallel_size):
            raise ValueError(
                f"data_parallel_rank ({self.data_parallel_rank})"
                f" must be in the range [0, {self.data_parallel_size})"
            )
    else:
        # Otherwise fall back to env vars (e.g. for offline SPMD case).
        self.data_parallel_size = envs.VLLM_DP_SIZE
        self.data_parallel_rank = envs.VLLM_DP_RANK
        self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
        self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
        self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

        if self.data_parallel_size > 1 and self.is_moe_model is False:
            raise ValueError(
                "Offline data parallel mode is not supported/useful for dense models."
            )

    max_dp_size = self.elastic_ep_max_dp_size
    self.elastic_ep_max_dp_size = (
        max_dp_size
        if self.enable_elastic_ep and max_dp_size is not None
        else self.data_parallel_size
    )
    if self.elastic_ep_max_dp_size < self.data_parallel_size:
        raise ValueError(
            "--elastic-ep-max-dp-size must be greater than or equal to "
            f"the initial data_parallel_size ({self.data_parallel_size})."
        )

    self.data_parallel_index = self.data_parallel_rank

    if self.distributed_executor_backend == "external_launcher":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        logger.info("Disabling V1 multiprocessing for external launcher.")

    if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
        # We use multiprocessing by default if world_size fits on the
        # current node and we aren't in a ray placement group.

        from vllm.v1.executor import ray_utils

        backend: DistributedExecutorBackend = "mp"
        ray_found = ray_utils.ray_is_available()
        if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
            backend = "uni"
        elif current_platform.is_cuda() and self.nnodes > 1:
            backend = "mp"
        elif (
            current_platform.is_cuda()
            and current_platform.device_count() < self.world_size
        ):
            gpu_count = current_platform.device_count()
            raise ValueError(
                f"World size ({self.world_size}) is larger than the number of "
                f"available GPUs ({gpu_count}) in this node. If this is "
                "intentional and you are using:\n"
                "- ray, set '--distributed-executor-backend ray'.\n"
                "- multiprocessing, set '--nnodes' appropriately."
            )
        elif self.data_parallel_backend == "ray":
            logger.info(
                "Using ray distributed inference because data_parallel_backend is ray"
            )
            backend = "ray"
        elif ray_found:
            if self.placement_group:
                backend = "ray"
            else:
                from ray import is_initialized as ray_is_initialized

                if ray_is_initialized():
                    from ray.util import get_current_placement_group

                    if get_current_placement_group():
                        backend = "ray"
        self.distributed_executor_backend = backend
        logger.debug("Defaulting to use %s for distributed inference", backend)

    if self.distributed_executor_backend is None and self.world_size == 1:
        self.distributed_executor_backend = "uni"

    if self.max_parallel_loading_workers is not None:
        logger.warning(
            "max_parallel_loading_workers is currently "
            "not supported and will be ignored."
        )
    allowed_backends = ("mp", "uni", "external_launcher")
    if self.distributed_executor_backend not in allowed_backends and self.nnodes > 1:
        raise ValueError(
            "nnodes > 1 can only be set when distributed executor "
            "backend is mp, uni or external_launcher."
        )

    if self.enable_eplb and self.eplb_config.communicator is None:
        # Prefer NIXL when available: zero-copy RDMA reads, compatible
        # with both async EPLB and elastic EP.
        # Fallbacks: pynccl for elastic EP (stateless groups need it),
        # torch_gloo for static EP.  torch_nccl is avoided because NCCL
        # is incompatible with async EPLB (multi-stream conflicts) and
        # batched isend/irecv hangs under high load.
        # See https://github.com/pytorch/pytorch/issues/174288
        from vllm.distributed.nixl_utils import is_nixl_available

        if is_nixl_available():
            self.eplb_config.communicator = "nixl"
        elif self.enable_elastic_ep:
            self.eplb_config.communicator = "pynccl"
        else:
            self.eplb_config.communicator = "torch_gloo"


ParallelConfig.__post_init__ = __post_init__
