# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import copy
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import ElasticTopology
from afd_plugin.elastic.ffn import FFNActorManager


def test_shutdown_does_not_import_ray_during_interpreter_finalization(
    config, monkeypatch
):
    # The client registers a weakref finalizer on FFNActorManager, so shutdown
    # can run while the interpreter is tearing itself down (observed as
    # AsyncLLM.__del__ -> shutdown -> ImportError: sys.meta_path is None).
    # Nothing can be reaped at that point, and failing there hides the real
    # exit reason, so the guard must hold before the ray import.
    monkeypatch.setattr(sys, "is_finalizing", lambda: True, raising=False)
    monkeypatch.setitem(sys.modules, "ray", None)
    manager = FFNActorManager(config)
    actor = object()
    manager.actors = [actor]

    manager.shutdown()

    assert manager.actors == [actor], "finalizing must not touch the actor handles"


def test_ffn_configs_are_independent_and_initialize_collectively(config, monkeypatch):
    # initialize_roles picks the FFN worker class from the running platform, so
    # the seam needs a platform double, not a vLLM install: tests/unit must also
    # run on a host that has neither vLLM nor torch.
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        device_type="cpu"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    original = copy.deepcopy(config)
    manager = FFNActorManager(config)
    seen = []

    async def initialize(cfg):
        seen.append(cfg)
        # Collective setup must enter all ranks before any awaits completion.
        await asyncio.sleep(0)
        assert len(seen) == 2

    manager.actors = [
        SimpleNamespace(initialize_role=SimpleNamespace(remote=initialize))
        for _ in range(2)
    ]
    topology = ElasticTopology(4, 2)
    afd = topology.afd_config(parse_afd_config(config), port=32100, role="ffn")
    asyncio.run(manager.initialize_roles(topology, afd, 32101))
    assert config == original
    for rank, cfg in enumerate(seen):
        parallel = cfg.parallel_config
        assert parallel.data_parallel_size == 2
        assert parallel.data_parallel_rank == rank
        assert parallel.data_parallel_rank_local == 0
        assert not parallel.enable_elastic_ep
        assert parallel.placement_group is None
        assert parallel.assigned_physical_gpu_ids is None
        assert parallel._data_parallel_master_port_list == [32101]
        assert cfg.additional_config["afd"]["role"] == "ffn"
    seen[0].parallel_config._data_parallel_master_port_list.pop()
    assert seen[1].parallel_config._data_parallel_master_port_list == [32101]


def _stub_ray_and_platform(monkeypatch, *, never_ready: bool):
    """A minimal Ray double: placement groups that are created and removed."""
    created: list = []
    removed: list = []

    class _NeverReady:
        async def ready(self):
            await asyncio.sleep(30)

    class _Ready:
        async def ready(self):
            await asyncio.sleep(0)

    kind = _NeverReady if never_ready else _Ready

    def placement_group(bundles, strategy=None):
        pg = kind()
        created.append(pg)
        return pg

    util = types.ModuleType("ray.util")
    util.placement_group = placement_group  # type: ignore[attr-defined]
    util.remove_placement_group = removed.append  # type: ignore[attr-defined]
    ray = types.ModuleType("ray")
    ray.util = util  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", util)

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        device_type="npu",
        ray_device_key="NPU",
        device_control_env_var="ASCEND_RT_VISIBLE_DEVICES",
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    strategies = types.ModuleType("ray.util.scheduling_strategies")
    strategies.PlacementGroupSchedulingStrategy = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    monkeypatch.setitem(sys.modules, "vllm.ray", types.ModuleType("vllm.ray"))
    ray_env = types.ModuleType("vllm.ray.ray_env")
    ray_env.get_env_vars_to_copy = lambda destination=None: ()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.ray.ray_env", ray_env)
    return created, removed


def test_reserve_gives_up_on_a_deadline_and_removes_what_it_created(
    config, monkeypatch
):
    # An unplaceable bundle must come back as a refusal. The device-capacity
    # preflight only sees what Ray reports as present, so a device held by
    # another job surfaces here -- and if this wait sat behind prepare() the
    # schedulers would already be paused and the gate raised, ending the
    # deployment instead of refusing the request. Removing the cleanup makes
    # `removed` empty; removing the deadline makes this hang for 30s.
    created, removed = _stub_ray_and_platform(monkeypatch, never_ready=True)
    manager = FFNActorManager(config)

    with pytest.raises(TimeoutError, match="no placement group became schedulable"):
        asyncio.run(manager.reserve(2, 0.05))

    # Groups are reserved one at a time, so the first timeout aborts the rest;
    # what matters is that everything created by this call is removed.
    assert created, "reserve() must actually create the bundles it waits on"
    assert removed == created, "a failed reservation must not leak bundles"
    assert manager._reserved == []
    assert manager.actors == []

    # A leftover from an earlier request that never reached spawn() is dropped
    # by the next reserve(), so retries do not stack groups.
    stale = object()
    manager._reserved = [stale]
    with pytest.raises(TimeoutError, match="no placement group"):
        asyncio.run(manager.reserve(2, 0.05))
    assert stale in removed
    assert manager._reserved == []


def test_spawn_never_waits_on_an_unreserved_placement(config, monkeypatch):
    # spawn() consumes reservations; it must not fall back to creating its own
    # and waiting indefinitely, or the deadline above could be bypassed by any
    # future caller.
    created, removed = _stub_ray_and_platform(monkeypatch, never_ready=True)
    manager = FFNActorManager(config)

    with pytest.raises(RuntimeError, match="no reserved placement group"):
        asyncio.run(manager.spawn(1))
    assert created == []
    assert manager.actors == []
