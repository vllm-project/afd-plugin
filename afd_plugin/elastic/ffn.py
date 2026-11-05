# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A separate Ray FFN pool; no EngineCore, scheduler, or A routing identities."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import sys
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig
from afd_plugin.elastic.config import ElasticTopology, set_afd_config

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor


def _device_attribution() -> tuple[str, str]:
    """(Ray resource key, device visibility env var) for this platform.

    CUDA reports accelerators under the "GPU" resource; Ray 2.48 detects
    Ascend NPUs natively as "NPU" with ASCEND_RT_VISIBLE_DEVICES. Both come
    from the vLLM platform, which vllm-ascend already defines.
    """
    from vllm.platforms import current_platform

    return current_platform.ray_device_key, current_platform.device_control_env_var


class FFNActor:
    """Short control RPCs; model computation runs in the worker's existing thread.

    Runtime imports deliberately follow Ray device assignment in the actor,
    so importing this module on the API process does not initialize the device.
    """

    def __init__(self) -> None:
        import ray

        device_key, env_var = _device_attribution()
        self.device_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
        os.environ[env_var] = ",".join(self.device_ids)
        os.environ["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "0"
        self.executor: Executor | None = None

    def rendezvous(self) -> tuple[str, list[int]]:
        from vllm.utils.network_utils import get_ip, get_open_ports_list

        # Both AFD world rank 0 and F role rank 0 reside here. Allocate on the
        # host that actually binds the stores, not on the API host.
        return get_ip(), get_open_ports_list(2)

    def initialize_role(self, config: VllmConfig) -> None:
        from vllm.platforms.interface import set_assigned_physical_gpu_ids
        from vllm.v1.executor import Executor

        assert self.executor is None
        physical_ids = [int(device) for device in self.device_ids]
        config.parallel_config.assigned_physical_gpu_ids = physical_ids
        set_assigned_physical_gpu_ids(physical_ids)
        self.executor = Executor.get_class(config)(config)

    def collective_rpc(self, method: str, args: tuple = ()):
        assert self.executor is not None
        return self.executor.collective_rpc(method, args=args)

    def release_role(self) -> None:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
            cleanup_dist_env_and_memory()


class FFNActorManager:
    def __init__(self, startup_config: VllmConfig):
        # API-side startup config still has the unaliased model identity.
        self.startup_config = copy.deepcopy(startup_config)
        self.actors: list[ActorHandle] = []
        self.placement_groups: list[PlacementGroup] = []
        # Reserved but not yet populated groups, held between reserve() and
        # spawn() so a resize can secure devices before anything is torn down.
        self._reserved: list[PlacementGroup] = []

    def _replaces_all(self, size: int) -> bool:
        from vllm.platforms import current_platform

        # Existing roles have already been released collectively before removal.
        # On Ascend, a reused FFN actor process cannot re-establish its HCCL
        # communicators across a world-size change: the rootInfo rendezvous of
        # a kept rank never meets the fresh rank's (hcclCommInitRootInfoConfig
        # stalls until the 180 s timeout), so every NPU resize replaces the
        # whole actor set with fresh processes (weight reload ~15 s per rank).
        return bool(current_platform.device_type == "npu" and len(self.actors) != size)

    async def reserve(self, size: int, timeout: float | None = None) -> None:
        """Create the placement groups a resize to ``size`` still needs.

        ``timeout`` bounds each group's readiness. The device-capacity preflight
        can only see what Ray reports as *present*, not as *free*, so a device
        held by another job surfaces exactly here -- and if this call sits behind
        ``prepare()`` the schedulers are already paused and the 503 gate already
        raised, so an unsatisfiable request would end the deployment instead of
        being refused. Callers that have not torn anything down pass a deadline.
        With ``timeout`` the reservations made by this call are removed on
        failure; nothing leaks and the old topology is untouched.
        """
        import ray

        # Leftovers from a request that never reached spawn() are by definition
        # unconsumed, so drop them before reserving again -- otherwise a retry
        # after a refused or timed-out request would stack groups.
        for pg in self._reserved:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
        self._reserved.clear()

        device_key, _ = _device_attribution()
        # Only the net increase: an NPU resize replaces the actor processes but
        # reuses their bundles, so requesting a full second set would stack the
        # old and new devices on top of the final size and refuse a resize that
        # fits (observed: 4Ax1F -> 4Ax2F on six NPUs demanded 7).
        needed = max(0, size - len(self.actors))
        fresh: list[PlacementGroup] = []
        try:
            for _ in range(needed):
                pg = ray.util.placement_group(
                    [{device_key: 1, "CPU": 1}], strategy="STRICT_PACK"
                )
                fresh.append(pg)
                if timeout is None:
                    await pg.ready()
                else:
                    await asyncio.wait_for(pg.ready(), timeout)
        except BaseException as exc:
            for pg in fresh:
                with contextlib.suppress(Exception):
                    ray.util.remove_placement_group(pg)
            if isinstance(exc, TimeoutError):
                # A bounded wait that expired is a refusal, not a failure of a
                # running switch: the raw asyncio timeout carries no message and
                # the route would answer an empty 408. State what was attempted,
                # the deadline and that nothing was torn down.
                raise TimeoutError(
                    f"Cannot place the FFN ranks for a resize to {size}: no "
                    f"placement group became schedulable within {timeout}s. "
                    "Nothing was torn down and the deployment keeps serving."
                ) from exc
            raise
        self._reserved.extend(fresh)

    async def spawn(self, size: int) -> None:
        """Bring the actor set to ``size``, consuming reserved groups.

        Actors replaced on NPU are killed, but their placement bundles are kept
        and reused for the fresh processes: the process must change (HCCL), the
        device slot does not. Releasing those bundles and reserving a full
        second set would demand old + new devices at once, which refuses a
        resize whose final size fits the host.
        """
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from vllm.ray.ray_env import get_env_vars_to_copy

        device_key, _ = _device_attribution()
        reusable: list[PlacementGroup] = []
        while self._replaces_all(size) and self.actors:
            ray.kill(self.actors.pop())
            reusable.append(self.placement_groups.pop())
        while len(self.actors) > size:
            ray.kill(self.actors.pop())
            reusable.append(self.placement_groups.pop())

        env_names = get_env_vars_to_copy(destination="AFD FFNActor")
        env = {name: os.environ[name] for name in env_names if name in os.environ}
        env["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "0"
        # Ray maps num_gpus to the "GPU" resource only; non-CUDA accelerators
        # (Ray detects Ascend NPUs as "NPU") are requested through resources.
        # Same pattern as vllm.v1.executor.ray_executor.
        device_options: dict[str, object] = (
            {"num_gpus": 1}
            if device_key == "GPU"
            else {"num_gpus": 0, "resources": {device_key: 1}}
        )
        while len(self.actors) < size:
            if reusable:
                pg = reusable.pop(0)
            elif self._reserved:
                pg = self._reserved.pop(0)
            else:
                raise RuntimeError(
                    "FFN actor spawn has no placement group; call reserve() "
                    "for the net increase first"
                )
            self.placement_groups.append(pg)
            actor = (
                ray.remote(FFNActor)
                .options(
                    num_cpus=1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0
                    ),
                    runtime_env={"env_vars": env},
                    **device_options,
                )
                .remote()
            )
            self.actors.append(actor)
        # Surplus bundles from a shrink, or from a replace-all that kept more
        # than the new set needs: release them so the devices return to the
        # cluster instead of leaking behind the manager.
        for pg in reusable:
            ray.util.remove_placement_group(pg)

    def release_reservations(self) -> None:
        """Drop bundles reserved for a resize that never reached spawn().

        A drain timeout or a pre-switch failure leaves them holding NPU and CPU
        resources that a later Attention scale-up cannot use, so the reversible
        exit path calls this before releasing the resize lock. The next
        reserve() would clear them too; releasing here closes the window.
        """
        import ray

        for pg in self._reserved:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
        self._reserved.clear()

    async def resize_actors(self, size: int, timeout: float | None = None) -> None:
        await self.reserve(size, timeout)
        await self.spawn(size)

    async def rendezvous(self) -> tuple[str, list[int]]:
        return await self.actors[0].rendezvous.remote()

    async def initialize_roles(
        self, topology: ElasticTopology, afd: AFDConfig, role_port: int
    ) -> None:
        from vllm.platforms import current_platform

        from afd_plugin.validation import NPU_FFN_WORKER_FQCN

        worker_cls = (
            NPU_FFN_WORKER_FQCN
            if current_platform.device_type == "npu"
            else "afd_plugin.v1.worker.AFDFFNWorker"
        )
        configs = []
        for rank in range(topology.ffn_dp):
            config = copy.deepcopy(self.startup_config)
            set_afd_config(config, afd)
            parallel = config.parallel_config
            parallel.enable_elastic_ep = False
            parallel.enable_eplb = False
            parallel.data_parallel_size = topology.ffn_dp
            parallel.data_parallel_size_local = 1
            parallel.data_parallel_rank = parallel.data_parallel_index = rank
            parallel.data_parallel_rank_local = 0
            parallel.tensor_parallel_size = topology.ffn_tp
            parallel.world_size = topology.ffn_tp
            parallel.data_parallel_master_ip = afd.host
            parallel.data_parallel_master_port = role_port
            parallel._data_parallel_master_port_list = [role_port]
            parallel._coord_store_port = None
            parallel.placement_group = None
            parallel.assigned_physical_gpu_ids = None
            parallel.worker_cls = worker_cls
            parallel.distributed_executor_backend = "uni"
            configs.append(config)
        # Submit every rank before awaiting any collective initialization.
        await asyncio.gather(
            *(
                actor.initialize_role.remote(cfg)
                for actor, cfg in zip(self.actors, configs, strict=True)
            )
        )

    async def collective_rpc(self, method: str, args: tuple = ()) -> None:
        await asyncio.gather(
            *(actor.collective_rpc.remote(method, args) for actor in self.actors)
        )

    async def release_roles(self) -> None:
        await asyncio.gather(*(actor.release_role.remote() for actor in self.actors))

    def shutdown(self) -> None:
        """Force-stop owned actors on process teardown, including blocked recv.

        Normal resizes use STOP/join and release_roles. Shutdown may follow an
        A-side crash, so it must not rely on another A control message arriving.
        """
        # This also runs from the weakref finalizer while the interpreter is
        # shutting down (AsyncLLM.__del__ -> shutdown), where the import
        # machinery is already gone and `import ray` raises
        # "ImportError: sys.meta_path is None". Nothing can be reaped there:
        # the process is exiting and Ray removes non-detached actors with it.
        if sys.is_finalizing():
            return

        import ray

        for actor in self.actors:
            ray.kill(actor)
        self.actors.clear()
        for pg in [*self.placement_groups, *self._reserved]:
            ray.util.remove_placement_group(pg)
        self.placement_groups.clear()
        self._reserved.clear()
