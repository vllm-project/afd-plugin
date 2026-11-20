# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD pre/post steps around the native EEP A client, and independent F resize."""

from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.elastic.config import (
    ElasticTopology,
    Role,
    set_afd_config,
    validate_elastic_config,
)
from afd_plugin.elastic.ffn import FFNActorManager

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

# Deadline for reserving FFN placement groups before any teardown. It is
# deliberately finite: an unsatisfiable scale-up must come back as a refusal
# while the deployment is still serving, not as an unbounded wait behind a
# closed gate. Ray brings up a one-device bundle in seconds when the device is
# there, so this only bounds the case where it is not.
PLACEMENT_RESERVE_TIMEOUT_S = 120.0


class ElasticAFDClient:
    """Own only F and the cross-role lifecycle; native EEP still owns every A."""

    def __init__(self, client: DPLBAsyncMPClient):
        self.client = client
        self.topology = validate_elastic_config(client.vllm_config)
        self.ffn = FFNActorManager(client.vllm_config)
        self._finalizer = weakref.finalize(client, self.ffn.shutdown)
        # Set by the scaling seam when a switch fails after it became
        # destructive; while it is non-None admission stays closed and every
        # further resize is refused with this reason. The rule itself lives in
        # the scaling seam: compat/patches/elastic.py, in scale().
        self.degraded: str | None = None

    async def initialize(self) -> None:
        await self.ffn.resize_actors(self.topology.ffn_dp)
        host, (link_port, role_port) = await self.ffn.rendezvous()
        base = replace(parse_afd_config(self.client.vllm_config), host=host)
        attention = self.topology.afd_config(base, port=link_port, role="attention")
        ffn = replace(attention, role="ffn")
        set_afd_config(self.client.vllm_config, attention)
        await asyncio.gather(
            self.client.collective_rpc_async(
                "afd_update_topology", args=(asdict(attention),)
            ),
            self.ffn.initialize_roles(self.topology, ffn, role_port),
        )
        await self.finish()

    async def prepare(self) -> None:
        # The outer AsyncLLM gate has drained requests, but only native DP pause
        # establishes consensus and prevents a late START_DP_WAVE/dummy forward.
        await self.client.pause_scheduler_async(mode="keep", clear_cache=True)
        await self.client.collective_rpc_async("afd_send_stop")
        await self.ffn.collective_rpc("join_ffn_server_loop")
        await asyncio.gather(
            self.client.collective_rpc_async("afd_release_link"),
            self.ffn.collective_rpc("afd_release_link"),
        )

    def preflight(self, size: int, role: Role) -> None:
        """Reject a request that cannot be satisfied, before touching anything.

        Native elastic EP enforces its own DP bound inside
        prepare_elastic_ep. After prepare() the schedulers are paused, the FFN
        serve loop is stopped, both connectors are released and the AFD config
        describes the new topology, so a rejection raised there leaves an
        otherwise healthy deployment stopped -- which is precisely what the PR
        review asked to prevent. scale() calls this before raising the scaling
        gate so a refused request does not even open a 503 window, and resize()
        calls it again so any other entry point stays covered. Everything here
        is pure reads of live state.
        """
        target = self.topology.resize(role, size)
        if target == self.topology:
            return
        if role == "attention":
            parallel_config = self.client.vllm_config.parallel_config
            maximum = getattr(parallel_config, "elastic_ep_max_dp_size", None)
            if maximum is not None and size > maximum:
                raise ValueError(
                    f"Cannot scale attention to data_parallel_size {size}; "
                    f"--elastic-ep-max-dp-size is {maximum}."
                )
            # Mirror the native MINIMUM of prepare_elastic_ep too: scaling keeps
            # physical experts per engine fixed, so shrinking below
            # ceil(E * current / physical) recomputes a negative redundancy and
            # native rejects only after the same pause/STOP/release/rewrite.
            # Same arithmetic, same wording.
            num_experts = self.client.vllm_config.model_config.get_num_experts()
            num_physical = (
                num_experts + parallel_config.eplb_config.num_redundant_experts
            )
            current = len(self.client.core_engines)
            if num_physical and num_physical * size // current - num_experts < 0:
                minimum = -(-num_experts * current // num_physical)
                raise ValueError(
                    f"Cannot scale attention to data_parallel_size {size}, "
                    f"minimum is {minimum}"
                )
        self._preflight_devices(target)

    def _preflight_devices(self, target: ElasticTopology) -> None:
        """Refuse a topology with more ranks than devices can host.

        The scale-up handshake waits for every new engine to signal ready on
        the coordination store, and a rank with no device to land on never
        does. Observed on Ascend: a 2A2F deployment on four devices was asked
        for attention 4, the request timed out after the drain, the deployment
        looked healthy for thirty minutes, and then the abandoned new-DP-group
        future inside EngineCore hit its own store timeout and took an engine
        down. One arithmetic check here is cheap; discovering it live is not.
        """
        need = target.attention_dp * target.attention_tp + target.ffn_dp * target.ffn_tp
        have = self._schedulable_devices()
        if have is not None and need > have:
            raise ValueError(
                f"Cannot scale to {target.attention_dp}A x {target.ffn_dp}F: "
                f"that topology needs {need} devices but only {have} can be "
                "scheduled here"
            )

    def _schedulable_devices(self) -> int | None:
        """Devices this deployment can place ranks on, or None when unknown.

        Two independent ceilings, taken as their minimum. Ray's cluster report
        counts every accelerator on the node, which overstates what one
        deployment may use once the launch script pins a visible-device list
        (observed: Ray advertised 16 NPUs while the server could use 4, so the
        cluster number alone would wave an impossible scale-up through). It
        deliberately fails open: a missing or unparsable report must not
        refuse a request that would otherwise have worked, so an unknown
        ceiling never rejects anything.
        """
        limits: list[int] = []
        try:
            import ray
            from vllm.platforms import current_platform

            if ray.is_initialized():
                total = ray.cluster_resources().get(current_platform.ray_device_key)
                if total is not None:
                    limits.append(int(total))
        except Exception:
            pass
        for var in ("ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            value = os.environ.get(var)
            if value:
                # An empty value asks for "no devices" in some runtimes and
                # "inherit" in others; only an explicit list is a ceiling.
                limits.append(len([d for d in value.split(",") if d.strip()]))
        return min(limits) if limits else None

    async def reserve(self, size: int, role: Role) -> None:
        """Secure the devices an F-role resize needs, with nothing torn down.

        Called with the admission gate still down, right after the preflight: a
        placement that never becomes schedulable must refuse the request the way
        the statically-knowable rejections do -- no pause, no drain, no 503
        window. The device-capacity preflight can only compare against what Ray
        reports as *present*, so a device held by another job surfaces exactly
        here. Attention-role resizes need nothing (native EEP owns their
        membership).
        """
        target = self.topology.resize(role, size)
        if role != "ffn" or target == self.topology:
            return
        await self.ffn.reserve(target.ffn_dp, PLACEMENT_RESERVE_TIMEOUT_S)

    async def resize(self, size: int, role: Role) -> None:
        target = self.topology.resize(role, size)
        if target == self.topology:
            return
        self.preflight(size, role)
        await self.prepare()
        if role == "attention":
            # The F pool keeps its identity, so the rank that owns the store is
            # still the one every new A rank must meet.
            ffn, _ = await self._apply_topology(target)
            await self.ffn.collective_rpc("afd_update_topology", args=(asdict(ffn),))
            # Native v0.30 two-phase RPC; prepare computes redundancy bounds
            # and routes up/down internally, commit performs the switch.
            await self.client.prepare_elastic_ep(size)
            await self.client.commit_elastic_ep()
            await self._await_eep_complete()
        else:
            await self.ffn.release_roles()
            await self.ffn.spawn(target.ffn_dp)
            # Ask the *replacement* rank 0 for the endpoint. On Ascend the whole
            # F set is new processes in new placement groups, and Ray may schedule
            # them elsewhere: publishing the retired F0's host and ports would
            # leave every rank rendezvousing with a store nobody binds.
            ffn, role_port = await self._apply_topology(target)
            await self.ffn.initialize_roles(target, ffn, role_port)
        await self.finish()
        self.topology = target

    async def _apply_topology(self, target: ElasticTopology) -> tuple[AFDConfig, int]:
        """Negotiate the rendezvous endpoint, then publish the target topology.

        Returns the FFN config and the role-store port, both taken from the FFN
        rank zero that is live at this moment.
        """
        host, (link_port, role_port) = await self.ffn.rendezvous()
        base = replace(parse_afd_config(self.client.vllm_config), host=host)
        attention = target.afd_config(base, port=link_port, role="attention")
        ffn = replace(attention, role="ffn")
        # The API config is the source copied by the *native* A ActorManager.
        set_afd_config(self.client.vllm_config, attention)
        await self.client.collective_rpc_async(
            "afd_update_topology", args=(asdict(attention),)
        )
        return ffn, role_port

    async def _await_eep_complete(self) -> None:
        # RECONFIGURE_FINISHED precedes COMPLETE. Query EngineCore itself,
        # yielding so its busy loop can continue state.progress().
        while not all(
            await asyncio.gather(
                *(
                    self.client._call_utility_async("afd_eep_complete", engine=engine)
                    for engine in self.client.core_engines
                )
            )
        ):
            await asyncio.sleep(0.01)

    async def finish(self) -> None:
        # Initiate both sides concurrently; connector construction is collective.
        await asyncio.gather(
            self.client.collective_rpc_async("afd_connect"),
            self.ffn.collective_rpc("afd_connect"),
        )
        await self.ffn.collective_rpc("start_ffn_server_loop")
        await self.client.collective_rpc_async("afd_finish")
        await self.client.call_utility_async("afd_reset_after_warmup")
        await self.client.resume_scheduler_async()
