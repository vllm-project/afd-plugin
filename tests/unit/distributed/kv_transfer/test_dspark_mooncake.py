# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-side allocation completion; these tests do not simulate device DMA."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.distributed.kv_transfer.dspark_mooncake import (  # noqa: E402
    AFDDSparkMooncakeConnector,
    _Allocation,
    _AuxPool,
    _Destination,
    _Metadata,
    _Source,
    _Worker,
    native,
)
from tests.unit.fakes.kv_cache import KVTransferEngine  # noqa: E402


class Completion:
    def __init__(self, complete: bool = True):
        self.complete = complete

    def query(self):
        return self.complete


@pytest.fixture
def auxiliary_pool():
    pool = object.__new__(_AuxPool)
    pool.condition = threading.Condition()
    pool.available = []
    pool.closed = False
    pool.timeout = 0
    return pool


@pytest.fixture
def source_worker(auxiliary_pool):
    worker = object.__new__(_Worker)
    worker._lock = threading.RLock()
    worker._sources = {"request": _Source("source", 0, leased="destination")}
    worker._source_generations = {"request": "source"}
    worker._pending_send_completion = set()
    worker._retiring_sources = set()
    worker._done_sources = set()
    worker._acknowledged = OrderedDict()
    worker._completed_sources = OrderedDict()
    worker._pool = auxiliary_pool
    worker.max_requests = 1
    return worker


def test_expired_native_send_waits_for_source_read_ack(source_worker):
    assert source_worker.finish_sending(set(), {"request"}) == set()
    source_worker.acknowledge_source("request", "source", "destination")
    assert source_worker.finish_sending(set(), set()) == {"request"}
    assert source_worker.finish_sending(set(), set()) == set()


def test_source_copy_completion_precedes_native_block_release(source_worker):
    completion = Completion(False)
    source_worker._sources["request"].leased = None
    source_worker._sources["request"].event = completion
    assert source_worker.finish_sending(set(), {"request"}) == set()
    completion.complete = True
    assert source_worker.finish_sending(set(), set()) == {"request"}


@pytest.mark.parametrize(
    "source,destination", [("old", "destination"), ("source", "old")]
)
def test_stale_ack_preserves_the_active_read(source_worker, source, destination):
    with pytest.raises(ValueError, match="Stale DSpark source acknowledgement"):
        source_worker.acknowledge_source("request", source, destination)
    assert source_worker.finish_sending(set(), {"request"}) == set()


def test_stale_completion_cannot_finish_a_reused_source(source_worker):
    assert not source_worker.complete_source("request", "old-source", "old-destination")
    assert source_worker.finish_sending(set(), {"request"}) == set()


@pytest.fixture
def active_receive(auxiliary_pool):
    worker = object.__new__(_Worker)
    allocation = _Allocation("destination", "source", 1, ((1,),), ((2,),))
    completion = Completion(False)
    worker._lock = threading.RLock()
    worker.draft = SimpleNamespace()
    worker._destinations = {
        "request": _Destination(
            allocation, slot=0, target_done=True, cancelled=True, event=completion
        )
    }
    worker._pool = auxiliary_pool
    connector = object.__new__(AFDDSparkMooncakeConnector)
    connector.connector_worker = worker
    connector._connector_metadata = _Metadata(
        SimpleNamespace(), {}, {"request": allocation}
    )
    return connector, worker, completion, auxiliary_pool


def test_duplicate_metadata_preserves_pending_receive_completion(active_receive):
    connector, worker, completion, pool = active_receive
    with pytest.raises(RuntimeError, match="already has an active receive"):
        connector.start_load_kv(None)
    assert worker.advance(set(), set()) == set()
    completion.complete = True
    assert worker.advance(set(), set()) == {"request"}
    assert pool.acquire() == 0


@pytest.fixture(params=[False, True], ids=["legacy-engine", "disconnect-api"])
def shutdown_worker(request, monkeypatch, auxiliary_pool):
    events = []
    target_pointer = 1

    def unregister_auxiliary(ptr: int) -> int:
        events.append(
            "target-unregistered" if ptr == target_pointer else "auxiliary-unregistered"
        )
        return 0

    worker = object.__new__(_Worker)
    worker._lock = threading.RLock()
    worker._shutdown_complete = worker._draining = False
    worker._pool = auxiliary_pool
    worker.kv_recv_thread = worker.kv_send_thread = None
    auxiliary_pool.tensor = torch.empty(1)
    auxiliary_pool.registered = True
    auxiliary_pool.engine = KVTransferEngine(unregister_memory=unregister_auxiliary)
    worker.engine = auxiliary_pool.engine
    weakref.finalize(worker.engine, events.append, "engine-released")
    if request.param:
        worker.engine.disconnect_all_peers = lambda: 0

    def unregister_target():
        events.append("target-unregistered")
        native.global_te.is_register_buffer = False

    monkeypatch.setattr(
        native,
        "global_te",
        SimpleNamespace(
            is_register_buffer=True,
            registered_buffers=[(target_pointer, 8)],
            register_buffer_lock=threading.Lock(),
            transfer_engine_lock=threading.Lock(),
            transfer_engine=worker.engine,
            unregister_buffer=unregister_target,
        ),
    )
    monkeypatch.setattr(
        torch.npu,
        "current_stream",
        lambda device: SimpleNamespace(
            synchronize=lambda: events.append("device-complete")
        ),
    )
    return worker, events


def test_shutdown_retires_registered_regions_once_after_device_completion(
    shutdown_worker,
):
    worker, events = shutdown_worker
    worker.shutdown()
    worker.shutdown()
    assert events == [
        "device-complete",
        "target-unregistered",
        "auxiliary-unregistered",
        "engine-released",
    ]
