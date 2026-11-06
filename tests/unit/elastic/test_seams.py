# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Run the installed patch methods with small upstream doubles, without CUDA."""

import asyncio
import importlib.util
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

    engine = NativeClient()
    engine.vllm_config = config
    engine.reqs_in_flight = {}
    engine._afd_elastic = SimpleNamespace(topology=ElasticTopology(2, 1), resize=resize)
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
    assert seams.events == [("attention", 4)]
    assert seams.llm.vllm_config.parallel_config.data_parallel_size == 4
    seams.events.clear()
    asyncio.run(seams.llm.scale_elastic_ep(2, role="ffn"))
    assert seams.events == [("ffn", 2)]
    assert seams.llm.vllm_config.parallel_config.data_parallel_size == 4
    assert not seams.get_scaling()


def test_drain_waits_for_frontend_even_if_coordinator_is_idle(seams):
    async def run():
        active = True
        seams.llm.output_processor.has_unfinished_requests = lambda: active
        task = asyncio.create_task(seams.llm.scale_elastic_ep(4))
        await asyncio.sleep(0)
        assert seams.get_scaling()
        assert not seams.events
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
