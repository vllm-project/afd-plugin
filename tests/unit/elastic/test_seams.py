# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Run the installed patch methods with small upstream doubles, without CUDA."""

import asyncio
import importlib.util
import logging
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from afd_plugin.elastic.config import ElasticTopology


@pytest.fixture
def seams(monkeypatch, config):
    events = []
    scaling = False

    def set_scaling(value):
        nonlocal scaling
        scaling = value

    class PauseState(Enum):
        PAUSED_ALL = 1

    class NativeCore:
        def __init__(self, *args):
            self.scheduler = SimpleNamespace(
                set_pause_state=lambda state: events.append(state)
            )

    class NativeClient:
        vllm_config: Any
        reqs_in_flight: dict[Any, Any]
        _afd_elastic: Any

        async def scale_elastic_ep(self, size):
            events.append(("native_client", size))

        def shutdown(self, timeout=None):
            events.append("shutdown")

        def dp_engines_running(self):
            return False

    class NativeAsyncLLM:
        engine_core: NativeClient
        vllm_config: Any
        log_stats: bool
        output_processor: Any

        async def scale_elastic_ep(self, size, drain_timeout=300):
            events.append(("native_async", size))
            await self.engine_core.scale_elastic_ep(size)
            self.vllm_config.parallel_config.data_parallel_size = size
            set_scaling(False)

    stubs: dict[str, dict[str, Any]] = {
        "vllm.config": {"VllmConfig": SimpleNamespace},
        "vllm.entrypoints.serve.elastic_ep.middleware": {
            "get_scaling_elastic_ep": lambda: scaling,
            "set_scaling_elastic_ep": set_scaling,
        },
        # The patched module logs at import time; stubbing the logger keeps this
        # seam covered on hosts without vLLM installed instead of skipping it.
        "vllm.logger": {"init_logger": lambda _name: logging.getLogger("afd-test")},
        "vllm.v1.core.sched.scheduler": {"PauseState": PauseState},
        "vllm.v1.engine.async_llm": {"AsyncLLM": NativeAsyncLLM},
        "vllm.v1.engine.core": {"DPEngineCoreProc": NativeCore},
        "vllm.v1.engine.core_client": {"DPLBAsyncMPClient": NativeClient},
        "vllm.v1.executor": {"Executor": type("Executor", (), {})},
        "vllm.v1.metrics.loggers": {
            "StatLoggerManager": type("StatLoggerManager", (), {})
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[3] / "afd_plugin/compat/patches/elastic.py"
    spec = importlib.util.spec_from_file_location("_afd_seams_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def resize(size, role):
        assert scaling
        events.append((role, size))

    def preflight(size, role):
        # A request that cannot be satisfied is refused while the gate is still
        # down: no drain, no pause, and no 503 window for a rejected request.
        assert not scaling, "preflight must run before the scaling gate is set"
        events.append(("preflight", role, size))

    engine = NativeClient()
    engine.vllm_config = config
    engine.reqs_in_flight = {}

    async def reserve(size, role):
        # The reservation must run with the gate still down: if it ever moved
        # behind set_scaling_elastic_ep(True), a refused placement would first
        # open the 503 window it is supposed to avoid.
        assert not scaling, "reserve() must run before the scaling gate is set"
        events.append(("reserve", role, size))

    engine._afd_elastic = SimpleNamespace(
        topology=ElasticTopology(2, 1),
        resize=resize,
        preflight=preflight,
        reserve=reserve,
    )
    llm = NativeAsyncLLM()
    llm.vllm_config = config
    llm.log_stats = False
    llm.engine_core = engine
    llm.output_processor = SimpleNamespace(has_unfinished_requests=lambda: False)
    return SimpleNamespace(
        module=module,
        llm=llm,
        events=events,
        core_class=NativeCore,
        set_scaling=set_scaling,
        get_scaling=lambda: scaling,
    )


def test_a_and_f_resize_route_through_the_controller(seams):
    # v0.30 re-anchor: A no longer re-enters the native client entry; the
    # controller drives native prepare/commit internally (see test_client).
    asyncio.run(seams.llm.scale_elastic_ep(4))
    assert seams.events == [
        ("preflight", "attention", 4),
        ("reserve", "attention", 4),
        ("attention", 4),
    ]
    assert seams.llm.vllm_config.parallel_config.data_parallel_size == 4
    seams.events.clear()
    asyncio.run(seams.llm.scale_elastic_ep(2, role="ffn"))
    assert seams.events == [
        ("preflight", "ffn", 2),
        ("reserve", "ffn", 2),
        ("ffn", 2),
    ]
    assert seams.llm.vllm_config.parallel_config.data_parallel_size == 4
    assert not seams.get_scaling()


def test_drain_waits_for_frontend_even_if_coordinator_is_idle(seams):
    async def run():
        active = True
        seams.llm.output_processor.has_unfinished_requests = lambda: active
        task = asyncio.create_task(seams.llm.scale_elastic_ep(4))
        await asyncio.sleep(0)
        assert seams.get_scaling()
        assert seams.events == [
            ("preflight", "attention", 4),
            ("reserve", "attention", 4),
        ]
        active = False
        await asyncio.wait_for(task, 1)

    asyncio.run(run())
    assert ("attention", 4) in seams.events


def test_parallel_resize_rejected_with_existing_gate(seams):
    seams.set_scaling(True)
    with pytest.raises(RuntimeError, match="already in progress"):
        asyncio.run(seams.llm.scale_elastic_ep(4))
    assert not seams.events
    assert seams.get_scaling()


def test_preflight_rejection_never_opens_the_gate(seams):
    # The device-capacity and DP-bound refusals happen here, before the gate:
    # an unsatisfiable request must leave scaling down and touch nothing.
    def reject(size, role):
        raise ValueError("that topology needs more devices than the cluster has")

    seams.llm.engine_core._afd_elastic.preflight = reject
    with pytest.raises(ValueError, match="more devices"):
        asyncio.run(seams.llm.scale_elastic_ep(4))
    assert not seams.get_scaling()
    assert not seams.events


def test_failure_after_the_switch_begins_fails_closed(seams):
    # A half-applied topology cannot be restored from the plugin (a rollback was
    # built, measured and removed), so the only honest state is 503 until
    # restart. Mutating: reopen the gate in the finally and this fails.
    controller = seams.llm.engine_core._afd_elastic

    async def broken_resize(size, role):
        raise RuntimeError("new DP group never became ready")

    controller.resize = broken_resize
    with pytest.raises(RuntimeError, match="never became ready"):
        asyncio.run(seams.llm.scale_elastic_ep(4))
    assert seams.get_scaling(), "admission must stay closed after a failed switch"
    assert controller.degraded.startswith("RuntimeError: new DP group")

    # And the next request says what is wrong, rather than claiming a resize is
    # still running for as long as the deployment is degraded.
    seams.events.clear()
    with pytest.raises(RuntimeError, match="restart the deployment"):
        asyncio.run(seams.llm.scale_elastic_ep(2, role="ffn"))
    assert not seams.events


def test_drain_timeout_reopens_admission_and_is_not_a_failure(seams):
    # Draining mutates nothing, so timing out there must leave the deployment
    # serving: gate down, no degraded marker, no retry refusal.
    seams.llm.output_processor.has_unfinished_requests = lambda: True
    with pytest.raises(TimeoutError, match="draining elastic AFD requests"):
        asyncio.run(seams.llm.scale_elastic_ep(4, drain_timeout=0))
    assert not seams.get_scaling()
    assert getattr(seams.llm.engine_core._afd_elastic, "degraded", None) is None
    # Once the requests are gone the deployment resizes normally again, with the
    # gate closing as it always has.
    seams.llm.output_processor.has_unfinished_requests = lambda: False
    asyncio.run(seams.llm.scale_elastic_ep(4))
    assert ("attention", 4) in seams.events
    assert not seams.get_scaling()


def test_reservation_failure_refuses_before_the_gate_opens(seams):
    # A placement that never becomes schedulable must behave like the preflight
    # rejections: the request fails, the gate never rises, nothing is paused,
    # and no degraded marker appears -- the deployment was never touched.
    async def refuse(size, role):
        raise TimeoutError(
            "Cannot place the FFN ranks for a resize to 2: no placement group "
            "became schedulable within 120.0s."
        )

    seams.llm.engine_core._afd_elastic.reserve = refuse
    with pytest.raises(TimeoutError, match="no placement group"):
        asyncio.run(seams.llm.scale_elastic_ep(2, role="ffn"))
    assert not seams.get_scaling()
    assert not seams.events[:-1] or all(
        e[0] in ("preflight",) for e in seams.events[:-1]
    ), "only the preflight ran before the refusal"
    assert getattr(seams.llm.engine_core._afd_elastic, "degraded", None) is None


def test_non_afd_uses_native_client_unchanged(seams):
    seams.llm.vllm_config.additional_config = {}
    asyncio.run(seams.llm.scale_elastic_ep(4))
    assert seams.events == [("native_async", 4), ("native_client", 4)]


def test_engine_startup_pauses_and_complete_query_is_nonblocking(seams):
    engine = seams.core_class(seams.llm.vllm_config, True, "", None, False)
    assert not engine.engines_running
    assert engine.ignore_start_dp_wave
    engine.eep_scaling_state = SimpleNamespace(state="EPLB_RESHUFFLE")
    assert not engine.afd_eep_complete()
    engine.eep_scaling_state = None
    assert engine.afd_eep_complete()
