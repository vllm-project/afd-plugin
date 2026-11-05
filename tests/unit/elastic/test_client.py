# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.elastic import client as module


class FakeFFN:
    def __init__(self, config):
        self.events = []
        self.connect_barrier = None
        self._prepared = None
        self.generation = 0

    # A replacement rank zero binds its own endpoint: asking for it before the
    # actors exist returns a retired address, which is what `spawn` counts here.
    async def rendezvous(self):
        return f"10.0.0.{2 + self.generation}", [30000 + self.generation, 30001]

    async def reserve(self, size, timeout=None):
        self.events.append(("f_reserve", size, timeout))

    async def spawn(self, size):
        self.generation += 1
        self.events.append(("f_actors", size))

    async def resize_actors(self, size, timeout=None):
        await self.reserve(size, timeout)
        await self.spawn(size)

    async def initialize_roles(self, topology, afd, port):
        self.events.append(("f_init", topology.ffn_dp, afd.role, port))

    async def collective_rpc(self, method, args=()):
        if method == "afd_update_topology":
            assert isinstance(args[0], dict)
            assert args[0]["role"] == "ffn"
        self.events.append(("f", method))
        if method == "afd_connect" and self.connect_barrier is not None:
            await self.connect_barrier()

    async def release_roles(self):
        self.events.append(("f_release_roles",))

    def shutdown(self):
        pass


class FakeClient:
    def __init__(self, config):
        self.vllm_config = config
        self.core_engines = [0, 1]
        self.events = []
        self.polls = {}
        self.connect_barrier = None

    async def collective_rpc_async(self, method, args=()):
        if method == "afd_update_topology":
            assert isinstance(args[0], dict)
            assert args[0]["role"] == "attention"
        if method == "afd_update_topology":
            self.vllm_config.additional_config["afd"] = dict(args[0])
        self.events.append(("a", method))
        if method == "afd_connect" and self.connect_barrier is not None:
            await self.connect_barrier()

    async def pause_scheduler_async(self, *, mode, clear_cache):
        self.events.append(("pause", mode, clear_cache))

    async def resume_scheduler_async(self):
        self.events.append(("resume",))

    async def call_utility_async(self, method):
        self.events.append((method,))

    async def _call_utility_async(self, method, *, engine):
        assert method == "afd_eep_complete"
        self.polls[engine] = self.polls.get(engine, 0) + 1
        done = self.polls[engine] > engine + 1
        self.events.append(("eep_complete", engine, done))
        return done

    async def prepare_elastic_ep(self, new_data_parallel_size):
        assert (
            self.vllm_config.additional_config["afd"]["num_attention_ranks"]
            == new_data_parallel_size
        )
        self.events.append(("native_eep_prepare", new_data_parallel_size))
        # Mirror native bookkeeping (core_client prepare/commit): physical
        # experts per engine stay fixed across a scale, so the redundancy
        # recomputed at prepare is what commit stores on eplb_config. The
        # min-bound preflight reads that same state.
        current = len(self.core_engines)
        pc = self.vllm_config.parallel_config
        num_experts = self.vllm_config.model_config.get_num_experts()
        num_physical = num_experts + pc.eplb_config.num_redundant_experts
        self._prepared = (
            new_data_parallel_size,
            num_physical * new_data_parallel_size // current - num_experts,
        )

    async def commit_elastic_ep(self):
        assert self._prepared is not None
        new, redundant = self._prepared
        self.events.append(("native_eep_commit", new))
        self.core_engines = list(range(new))
        pc = self.vllm_config.parallel_config
        pc.data_parallel_size = new
        pc.eplb_config.num_redundant_experts = redundant


@pytest.fixture
def runtime(config, monkeypatch):
    monkeypatch.setattr(module, "FFNActorManager", FakeFFN)
    client = FakeClient(config)
    controller = module.ElasticAFDClient(client)
    controller.ffn.events = client.events
    return SimpleNamespace(client=client, controller=controller, events=client.events)


def test_startup_connects_both_roles_before_forward(runtime):
    async def run():
        waiting = 0
        connected = asyncio.Event()

        async def connect_barrier():
            nonlocal waiting
            waiting += 1
            if waiting == 2:
                connected.set()
            await connected.wait()

        runtime.client.connect_barrier = connect_barrier
        runtime.controller.ffn.connect_barrier = connect_barrier
        await asyncio.wait_for(runtime.controller.initialize(), 1)

    asyncio.run(run())
    events = runtime.events
    assert events.index(("f", "start_ffn_server_loop")) < events.index(
        ("a", "afd_finish")
    )
    assert events[-2:] == [("afd_reset_after_warmup",), ("resume",)]


def test_a_resize_reuses_native_eep_and_waits_every_target(runtime):
    asyncio.run(runtime.controller.resize(4, "attention"))
    events = runtime.events
    assert events.index(("pause", "keep", True)) < events.index(("a", "afd_send_stop"))
    assert events.index(("a", "afd_send_stop")) < events.index(
        ("f", "join_ffn_server_loop")
    )
    assert events.index(("f", "join_ffn_server_loop")) < events.index(
        ("a", "afd_release_link")
    )
    assert events.index(("a", "afd_release_link")) < events.index(
        ("native_eep_prepare", 4)
    )
    for engine in range(4):
        assert events.index(("eep_complete", engine, True)) < events.index(
            ("a", "afd_connect")
        )
    assert not any(event[0] == "f_release_roles" for event in events)
    assert runtime.controller.topology.attention_dp == 4


def test_f_target_equal_to_a_size_does_not_short_circuit(runtime):
    before = runtime.client.core_engines.copy()
    asyncio.run(runtime.controller.reserve(2, "ffn"))
    asyncio.run(runtime.controller.resize(2, "ffn"))
    assert runtime.client.core_engines == before
    assert runtime.client.vllm_config.parallel_config.data_parallel_size == 2
    assert ("f_actors", 2) in runtime.events
    assert not any(event[0].startswith("native_eep") for event in runtime.events)
    assert runtime.events.index(("f_release_roles",)) < runtime.events.index(
        ("f_actors", 2)
    )
    assert runtime.controller.topology.ffn_dp == 2


def test_invalid_target_does_not_pause_or_touch_actors(runtime):
    with pytest.raises(ValueError):
        asyncio.run(runtime.controller.resize(3, "ffn"))
    assert not runtime.events


def test_complete_resize_chain(runtime):
    async def run():
        for role, size in [("attention", 4), ("ffn", 2), ("attention", 2), ("ffn", 1)]:
            await runtime.controller.reserve(size, role)
            await runtime.controller.resize(size, role)

    asyncio.run(run())
    assert runtime.controller.topology == module.validate_elastic_config(
        runtime.client.vllm_config
    )
    assert runtime.events.count(("resume",)) == 4
    assert ("native_eep_prepare", 4) in runtime.events
    assert ("native_eep_commit", 4) in runtime.events
    assert ("native_eep_prepare", 2) in runtime.events
    assert ("native_eep_commit", 2) in runtime.events


def test_ffn_scale_up_reserves_devices_before_the_teardown(runtime):
    # Ray reports what is present, not what is free: a device held by another job
    # only shows up when the placement group is waited on. That wait must not
    # pause anything -- the reservation happens with the deployment untouched,
    # before the seam raises the gate or drains.
    asyncio.run(runtime.controller.reserve(2, "ffn"))
    assert ("f_reserve", 2, module.PLACEMENT_RESERVE_TIMEOUT_S) in runtime.events
    assert ("pause", "keep", True) not in runtime.events
    assert ("f", "afd_send_stop") not in runtime.events


def test_ffn_resize_publishes_the_endpoint_of_the_new_rank_zero(runtime):
    # On Ascend the whole F set is replaced, and Ray may schedule the new rank 0
    # elsewhere. Publishing the retired F0 host/ports would leave every rank
    # rendezvousing with a store nobody binds. Mutating: rendezvous before spawn
    # and the published host stays 10.0.0.2.
    asyncio.run(runtime.controller.reserve(2, "ffn"))
    asyncio.run(runtime.controller.resize(2, "ffn"))
    published = runtime.client.vllm_config.additional_config["afd"]
    assert published["host"] == "10.0.0.3"
    assert runtime.events.index(("f_actors", 2)) < runtime.events.index(
        ("a", "afd_update_topology")
    )


def test_attention_resize_keeps_the_existing_rank_zero(runtime):
    # The F pool is untouched by an A resize, so its original endpoint is still
    # the right one -- the guard rail on the two tests above.
    asyncio.run(runtime.controller.resize(4, "attention"))
    published = runtime.client.vllm_config.additional_config["afd"]
    assert published["host"] == "10.0.0.2"


def test_native_max_bound_rejected_before_teardown(runtime):
    # A size that passes topology validation but exceeds the native EEP
    # maximum (--elastic-ep-max-dp-size) must be refused before prepare():
    # pausing, STOP, connector release and config rewrite happen later.
    runtime.client.vllm_config.parallel_config.elastic_ep_max_dp_size = 4
    with pytest.raises(ValueError, match="elastic-ep-max-dp-size"):
        asyncio.run(runtime.controller.resize(6, "attention"))
    assert not runtime.events
    assert runtime.client.vllm_config.parallel_config.data_parallel_size == 2


def _fake_ray(monkeypatch, count, initialized=True):
    """Give the client a Ray cluster resource report without importing Ray."""
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: initialized  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: {} if count is None else {"NPU": count}  # type: ignore[attr-defined]
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        ray_device_key="NPU"
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)


def test_capacity_preflight_rejects_more_ranks_than_devices(runtime, monkeypatch):
    # Without this, the request reaches the native store handshake and waits
    # for engines that can never be scheduled: the observed sequence on Ascend
    # was a timeout after the drain, a healthy-looking service for thirty
    # minutes, then the abandoned new-DP-group future inside EngineCore timed
    # out and took an engine with it.
    _fake_ray(monkeypatch, count=4)
    with pytest.raises(ValueError, match=r"needs 5 devices but only 4 can be"):
        asyncio.run(runtime.controller.resize(4, "attention"))
    assert not runtime.events, "rejected without pausing or tearing anything down"
    assert runtime.controller.topology.attention_dp == 2


def test_capacity_preflight_uses_the_visible_device_ceiling(runtime, monkeypatch):
    # Ray counts every accelerator on the node; the launch script restricts the
    # deployment to a subset. Taking only the cluster report would wave an
    # impossible scale-up straight through to the store handshake, which is
    # exactly how the thirty-minute delayed engine death was produced.
    _fake_ray(monkeypatch, count=16)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match=r"needs 5 devices but only 4 can be"):
        asyncio.run(runtime.controller.resize(4, "attention"))
    assert not runtime.events

    # With enough visible devices the same request is satisfiable.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5")
    asyncio.run(runtime.controller.resize(4, "attention"))
    assert ("native_eep_commit", 4) in runtime.events


def test_capacity_preflight_fails_open_when_devices_are_unknown(runtime, monkeypatch):
    # A missing or uninitialised resource report must not refuse a request that
    # would otherwise have worked; only a known shortfall is actionable.
    _fake_ray(monkeypatch, count=4, initialized=False)
    asyncio.run(runtime.controller.resize(4, "attention"))
    assert ("native_eep_commit", 4) in runtime.events


def test_native_min_bound_rejected_before_teardown(runtime):
    # Reviewer follow-up case: an initial 4A/2F deployment with the default
    # num_redundant_experts=0 cannot shrink to 2A -- physical experts per
    # engine stay fixed, so the redundancy recomputes NEGATIVE and native
    # prepare_elastic_ep rejects with "minimum 4", only after the same
    # pause/STOP/release/rewrite. The 2->4->2 chain does not cover this
    # because the preceding scale-up created the redundancy needed to
    # return; refuse before prepare() with no teardown and no mutation.
    config = runtime.client.vllm_config
    config.additional_config["afd"].update(num_attention_ranks=4, num_ffn_ranks=2)
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.eplb_config.num_redundant_experts = 0
    runtime.client.core_engines = [0, 1, 2, 3]
    runtime.controller.topology = module.validate_elastic_config(config)
    with pytest.raises(ValueError, match="minimum is 4"):
        asyncio.run(runtime.controller.resize(2, "attention"))
    assert not runtime.events
    assert config.parallel_config.data_parallel_size == 4
