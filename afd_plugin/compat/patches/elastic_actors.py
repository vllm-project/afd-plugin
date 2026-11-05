# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actor factory seams, copied from vLLM 0.30.0 (ced6857afa0e) and verified
line-for-line against main@544f182d9; no new A manager."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import vllm.v1.engine.utils as upstream
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.engine.utils import CoreEngineActorManager, EngineZmqAddresses
from vllm.v1.executor import Executor

from afd_plugin.elastic.config import is_elastic_attention

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)


# Patch reason: EEP's used-device local-rank arithmetic includes colocated F.
# Change: choose the native-derived AFD actor only for elastic attention.
# Signature: matches vLLM 0.30.0 CoreEngineActorManager.__init__.
# Upstream target/removal: an actor-class factory in the native manager.
def __init__(
    self,
    vllm_config: VllmConfig,
    addresses: EngineZmqAddresses,
    executor_class: type[Executor],
    log_stats: bool,
    placement_groups: list[PlacementGroup] | None = None,
    local_dp_ranks: list[int] | None = None,
):
    import copy

    import ray
    from ray.runtime_env import RuntimeEnv
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

    dp_size = vllm_config.parallel_config.data_parallel_size
    actor_class = (
        DPMoEEngineCoreActor
        if dp_size > 1 and vllm_config.model_config.is_moe
        else EngineCoreActor
    )
    # ### PATCH START: AFD actor uses its actual Ray GPU bundle assignment.
    if is_elastic_attention(vllm_config):
        from afd_plugin.elastic.ray_actor import AFDEngineCoreActor

        actor_class = AFDEngineCoreActor
    # ### PATCH END: AFD actor uses its actual Ray GPU bundle assignment.

    self.local_engine_actors: list[ray.ActorHandle] = []
    self.remote_engine_actors: list[ray.ActorHandle] = []

    env_vars_list = upstream.get_env_vars_to_copy(
        destination=actor_class.__name__,
        exclude_vars=upstream.WORKER_SPECIFIC_ENV_VARS,
    )
    self.env_vars_dict = {
        name: os.environ[name] for name in env_vars_list if name in os.environ
    }
    runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

    self.addresses = addresses
    self.executor_class = executor_class
    self.log_stats = log_stats
    local_engine_count = vllm_config.parallel_config.data_parallel_size_local
    world_size = vllm_config.parallel_config.world_size
    self.manager_stopped = threading.Event()
    self.failed_proc_name: str | None = None

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    else:
        ray.init()

    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_elastic_ep:
        from vllm.distributed.utils import create_tcp_store

        ip = parallel_config.data_parallel_master_ip
        store = create_tcp_store(
            ip,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = store.port
        self._coord_store = store

    if placement_groups is not None:
        assert local_dp_ranks is not None, (
            "local_dp_ranks must be provided if placement_groups is provided"
        )
        assert len(placement_groups) == len(local_dp_ranks), (
            "placement_groups and local_dp_ranks must have the same length"
        )
        logger.info("Using provided placement groups")
        # TODO(rui): validate passed-in placement groups
        self.created_placement_groups = []
    else:
        placement_groups, local_dp_ranks = (
            CoreEngineActorManager.create_dp_placement_groups(vllm_config)
        )
        self.created_placement_groups = placement_groups
    assert len(placement_groups) == dp_size, (
        "Number of placement groups must match data parallel size"
    )

    self.placement_group_is_local = []
    refs = []
    for index, local_index, pg in zip(
        range(dp_size), local_dp_ranks, placement_groups, strict=False
    ):
        dp_vllm_config = copy.deepcopy(vllm_config)
        if dp_size > 1:
            upstream._apply_dp_identity_suffix(dp_vllm_config, index)
        dp_vllm_config.parallel_config.placement_group = pg
        local_client = index < local_engine_count

        # Ray XPU known issue: dpctl initializes the GPU runtime early, so
        # setting device env vars in Ray actor's initialization method
        # will not affect device selection. See:
        # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
        if current_platform.is_xpu():
            device_evar = current_platform.device_control_env_var
            physical_gpu_ids = upstream.get_physical_gpu_ids_for_local_dp_rank(
                device_evar, local_index, world_size
            )
            actor_env_vars = self.env_vars_dict.copy()
            actor_env_vars[device_evar] = ",".join(str(d) for d in physical_gpu_ids)
            runtime_env = RuntimeEnv(env_vars=actor_env_vars)

        actor = (
            ray.remote(actor_class)
            .options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=world_size,
                ),
                runtime_env=runtime_env,
            )
            .remote(
                vllm_config=dp_vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                local_client=local_client,
                addresses=addresses,
                dp_rank=index,
                local_dp_rank=local_index,
            )
        )
        if local_client:
            self.local_engine_actors.append(actor)
        else:
            self.remote_engine_actors.append(actor)
        self.placement_group_is_local.append(local_client)
        refs.append(actor.wait_for_init.remote())

    ray.get(refs)
    self.run_refs = []
    self.actor_run_ref_dict = dict()
    for actor in self.local_engine_actors + self.remote_engine_actors:
        ref = actor.run.remote()
        self.run_refs.append(ref)
        self.actor_run_ref_dict[actor] = ref


def scale_up_elastic_ep(
    self,
    cur_vllm_config: VllmConfig,
    new_data_parallel_size: int,
    num_redundant_experts: int,
) -> None:
    import copy

    import ray
    from ray.runtime_env import RuntimeEnv
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

    actor_class = (
        DPMoEEngineCoreActor if cur_vllm_config.model_config.is_moe else EngineCoreActor
    )
    # ### PATCH START: AFD actor uses its actual Ray GPU bundle assignment.
    if is_elastic_attention(cur_vllm_config):
        from afd_plugin.elastic.ray_actor import AFDEngineCoreActor

        actor_class = AFDEngineCoreActor
    # ### PATCH END: AFD actor uses its actual Ray GPU bundle assignment.

    cur_data_parallel_size = len(self.local_engine_actors) + len(
        self.remote_engine_actors
    )

    assert new_data_parallel_size > cur_data_parallel_size, (
        f"New data parallel size {new_data_parallel_size} must be greater "
        f"than current data parallel size {cur_data_parallel_size} "
        "for scale up"
    )

    placement_groups, local_dp_ranks = self.add_dp_placement_groups(
        cur_vllm_config, new_data_parallel_size
    )

    world_size = cur_vllm_config.parallel_config.world_size
    dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
    new_local_engines = 0

    runtime_env = RuntimeEnv(
        env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
    )
    for i, (pg, local_rank) in enumerate(
        zip(placement_groups, local_dp_ranks, strict=False)
    ):
        rank = cur_data_parallel_size + i
        dp_vllm_config = copy.deepcopy(cur_vllm_config)
        if new_data_parallel_size > 1:
            upstream._apply_dp_identity_suffix(dp_vllm_config, rank)
        dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        dp_vllm_config.parallel_config.eplb_config.num_redundant_experts = (
            num_redundant_experts
        )
        dp_vllm_config.parallel_config.placement_group = pg

        # Check if this placement group is on the head node
        local_client = any(
            bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
        )

        if local_client:
            new_local_engines += 1
            # Update data_parallel_size_local
            dp_vllm_config.parallel_config.data_parallel_size_local = (
                cur_vllm_config.parallel_config.data_parallel_size_local
                + new_local_engines
            )

        actor = (
            ray.remote(actor_class)
            .options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=world_size,
                ),
                runtime_env=runtime_env,
            )
            .remote(
                vllm_config=dp_vllm_config,
                executor_class=self.executor_class,
                log_stats=self.log_stats,
                local_client=local_client,
                addresses=self.addresses,
                dp_rank=rank,
                local_dp_rank=local_rank,
            )
        )

        if local_client:
            self.local_engine_actors.append(actor)
        else:
            self.remote_engine_actors.append(actor)
        self.created_placement_groups.append(pg)
        self.placement_group_is_local.append(local_client)

    # ### PATCH START: zero remote additions must not select old running actors.
    new_remote_engines = len(placement_groups) - new_local_engines
    actors = (
        self.local_engine_actors[-new_local_engines:] if new_local_engines > 0 else []
    ) + (
        self.remote_engine_actors[-new_remote_engines:]
        if new_remote_engines > 0
        else []
    )
    ray.get([actor.wait_for_init.remote() for actor in actors])
    # ### PATCH END: zero remote additions must not select old running actors.
    for actor in actors:
        ref = actor.run.remote()
        self.run_refs.append(ref)
        self.actor_run_ref_dict[actor] = ref


CoreEngineActorManager.__init__ = __init__
CoreEngineActorManager.scale_up_elastic_ep = scale_up_elastic_ep
