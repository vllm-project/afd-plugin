# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import inspect
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runtime(monkeypatch):
    class KVCacheManager:
        def __init__(self):
            self.calls = []
            self.groups = {}
            self.evicted = set()

        def allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=None,
            num_lookahead_tokens=0,
            num_external_computed_tokens=0,
            delay_cache_blocks=False,
            num_encoder_tokens=0,
            full_sequence_must_fit=False,
            reserved_blocks=0,
        ):
            self.calls.append(
                dict(
                    request=request,
                    num_new_tokens=num_new_tokens,
                    num_new_computed_tokens=num_new_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=num_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=delay_cache_blocks,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=full_sequence_must_fit,
                    reserved_blocks=reserved_blocks,
                )
            )
            return new_computed_blocks

        def get_block_ids(self, request_id):
            return self.groups[request_id]

        def evict_blocks(self, blocks):
            self.evicted.update(blocks)

    class Scheduler:
        def __init__(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=None,
            mm_registry=None,
            include_finished_set=False,
            log_stats=False,
        ):
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config
            self.kv_cache_manager = KVCacheManager()
            self.recompute_kv_load_failures = vllm_config.recompute
            self.running = []
            self.skipped_waiting = []
            self.failed_recving_kv_req_ids = set()
            self.recovery_calls = []

        def _handle_invalid_blocks(self, invalid_block_ids, num_scheduled_tokens):
            raise AssertionError("unpatched hybrid recovery")

        def _update_requests_with_invalid_blocks(
            self,
            requests,
            invalid_block_ids,
            num_scheduled_tokens,
            evict_blocks=True,
        ):
            self.recovery_calls.append(evict_blocks)
            failed = set()
            for request in requests:
                (blocks,) = self.kv_cache_manager.get_block_ids(request.request_id)
                if invalid_block_ids.intersection(blocks):
                    failed.add(request.request_id)
            return failed, len(failed), set()

    paths = (
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.kv_cache_manager",
        "vllm_ascend",
        "vllm_ascend.utils",
    )
    modules = {}
    for path in paths:
        module = ModuleType(path)
        module.__path__ = []
        modules[path] = module
        monkeypatch.setitem(sys.modules, path, module)
        if "." in path:
            parent, name = path.rsplit(".", 1)
            setattr(modules[parent], name, module)
    upstream = modules["vllm.v1.core.sched.scheduler"]
    upstream.Scheduler = Scheduler
    upstream.MULTIMODAL_REGISTRY = None
    upstream.logger = logging.getLogger("test-dspark-pd")
    upstream.RequestStatus = SimpleNamespace(WAITING_FOR_REMOTE_KVS="remote")
    modules["vllm.v1.core.kv_cache_manager"].KVCacheManager = KVCacheManager
    modules["vllm_ascend.utils"].is_dspark_config = lambda c: c.dspark
    patch = importlib.import_module("afd_plugin.compat.patches.npu.dspark_pd")
    monkeypatch.setattr(patch, "version", lambda package: "0.23.0+empty")
    return SimpleNamespace(patch=patch, scheduler=Scheduler, manager=KVCacheManager)


def config(**changes):
    values = dict(
        additional_config={
            "afd": {
                "role": "attention",
                "connector": "P2pHcclAFDConnector",
            }
        },
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeHybridConnector",
            is_kv_consumer=True,
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        dspark=True,
        recompute=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def scheduler(runtime, **changes):
    assert runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    return runtime.scheduler(
        config(**changes),
        SimpleNamespace(kv_cache_groups=[None, None]),
        None,
        32,
    )


def test_async_load_defers_lookahead_and_decode_keeps_it(runtime):
    instance = scheduler(runtime)
    manager = instance.kv_cache_manager
    request = SimpleNamespace(request_id="eos")
    cached = object()
    assert (
        manager.allocate_slots(
            request,
            0,
            num_new_computed_tokens=3,
            new_computed_blocks=cached,
            num_lookahead_tokens=5,
            num_external_computed_tokens=19,
            delay_cache_blocks=True,
            num_encoder_tokens=7,
            full_sequence_must_fit=True,
            reserved_blocks=11,
        )
        is cached
    )
    call = manager.calls[-1]
    assert call == dict(
        request=request,
        num_new_tokens=0,
        num_new_computed_tokens=3,
        new_computed_blocks=cached,
        num_lookahead_tokens=0,
        num_external_computed_tokens=19,
        delay_cache_blocks=True,
        num_encoder_tokens=7,
        full_sequence_must_fit=True,
        reserved_blocks=11,
    )
    manager.allocate_slots(request, 1, num_lookahead_tokens=5)
    assert manager.calls[-1]["num_lookahead_tokens"] == 5


@pytest.mark.parametrize("delay,external", [(False, 19), (True, 0)])
def test_only_async_external_load_changes_allocation(runtime, delay, external):
    manager = scheduler(runtime).kv_cache_manager
    manager.allocate_slots(
        None,
        1,
        num_lookahead_tokens=5,
        num_external_computed_tokens=external,
        delay_cache_blocks=delay,
    )
    assert manager.calls[-1]["num_lookahead_tokens"] == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"additional_config": {}},
        {
            "additional_config": {
                "afd": {"role": "ffn", "connector": "P2pHcclAFDConnector"}
            }
        },
        {
            "additional_config": {
                "afd": {"role": "attention", "connector": "P2pNcclAFDConnector"}
            }
        },
        {"kv_transfer_config": None},
        {
            "kv_transfer_config": SimpleNamespace(
                kv_connector="Other", is_kv_consumer=True
            )
        },
        {
            "kv_transfer_config": SimpleNamespace(
                kv_connector="MooncakeHybridConnector", is_kv_consumer=False
            )
        },
        {"speculative_config": None},
        {"dspark": False},
    ],
)
def test_allocation_scope_leaves_other_configurations_unchanged(runtime, changes):
    manager = scheduler(runtime, **changes).kv_cache_manager
    manager.allocate_slots(
        None,
        0,
        num_lookahead_tokens=5,
        num_external_computed_tokens=19,
        delay_cache_blocks=True,
    )
    assert manager.calls[-1]["num_lookahead_tokens"] == 5


def test_hybrid_fail_markers_identify_requests_without_cross_group_collisions(runtime):
    instance = scheduler(runtime)
    waiting = SimpleNamespace(request_id="waiting", status="remote")
    running = SimpleNamespace(request_id="running", status="running")
    healthy = SimpleNamespace(request_id="healthy", status="running")
    unscheduled = SimpleNamespace(request_id="other-wait", status="grammar")
    instance.skipped_waiting = [waiting, unscheduled]
    instance.running = [running, healthy]
    instance.kv_cache_manager.groups = {
        "waiting": ([10, 11], [20, 21]),
        "running": ([30, 31], [40, 41]),
        # An ID present only in a later group must not mark this request failed.
        "healthy": ([50, 51], [11, 31]),
    }
    assert instance._handle_invalid_blocks({11, 31}, {}) == {"waiting", "running"}
    assert instance.recovery_calls == []
    assert instance.kv_cache_manager.evicted == {30, 31, 40, 41}
    assert instance.failed_recving_kv_req_ids == set()
    assert instance._handle_invalid_blocks({999}, {}) == set()
    # A subsequent decode allocation still works and keeps its draft slots.
    instance.kv_cache_manager.allocate_slots(healthy, 1, num_lookahead_tokens=5)
    assert instance.kv_cache_manager.calls[-1]["num_lookahead_tokens"] == 5


@pytest.mark.parametrize("recompute", [False, True])
def test_single_group_preserves_upstream_policy(runtime, recompute):
    instance = scheduler(runtime, recompute=recompute)
    instance.kv_cache_config.kv_cache_groups = [None]
    instance.skipped_waiting = [SimpleNamespace(request_id="a", status="remote")]
    instance.kv_cache_manager.groups = {"a": ([10],)}
    result = instance._handle_invalid_blocks({10}, {})
    assert instance.recovery_calls == [False, True]
    if recompute:
        assert result == set()
        assert instance.failed_recving_kv_req_ids == {"a"}
    else:
        assert result == {"a"}


def test_installer_is_idempotent_and_preserves_signatures(runtime):
    init_signature = inspect.signature(runtime.scheduler.__init__)
    assert runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    patched_init = runtime.scheduler.__init__
    assert runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    assert runtime.scheduler.__init__ is patched_init
    assert tuple(inspect.signature(patched_init).parameters) == tuple(
        init_signature.parameters
    )
    manager = scheduler(runtime).kv_cache_manager
    assert (
        tuple(inspect.signature(manager.allocate_slots).parameters)
        == tuple(inspect.signature(runtime.manager.allocate_slots).parameters)[1:]
    )


def test_other_version_and_changed_api_do_not_patch(runtime, monkeypatch):
    init = runtime.scheduler.__init__
    monkeypatch.setattr(runtime.patch, "version", lambda package: "0.26.0")
    assert not runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    assert runtime.scheduler.__init__ is init
    monkeypatch.setattr(runtime.patch, "version", lambda package: "0.23.0")
    monkeypatch.setattr(runtime.manager, "allocate_slots", lambda self, request: None)
    assert not runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    assert runtime.scheduler.__init__ is init


def test_changed_failure_handler_api_does_not_patch(runtime, monkeypatch):
    init = runtime.scheduler.__init__
    monkeypatch.setattr(
        runtime.scheduler,
        "_handle_invalid_blocks",
        lambda self, invalid_blocks: set(),
    )
    assert not runtime.patch.apply_afd_dspark_pd_scheduler_patch()
    assert runtime.scheduler.__init__ is init
