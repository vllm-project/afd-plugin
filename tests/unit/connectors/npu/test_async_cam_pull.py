# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host Pull shutdown protocol; these tests do not qualify device execution."""

from types import SimpleNamespace

import pytest

from afd_plugin.connectors.npu.async_cam_runtime import (
    async_cam_startup as startup_module,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_pull import PullFFNSession
from tests.unit.fakes.async_cam_startup import MemoryStore, startup_world


@pytest.fixture
def pull_world(monkeypatch):
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    store = MemoryStore()
    connectors = [
        SimpleNamespace(
            graph_startup=startup,
            max_num_batched_tokens=8,
            topk=2,
            _pull_has_active_source=lambda: False,
            _check_pull_status=lambda: None,
        )
        for startup in startup_world(store=store)
    ]
    return SimpleNamespace(
        sessions=[PullFFNSession(connector) for connector in connectors],
        connectors=connectors,
        store=store,
    )


@pytest.mark.parametrize("shutdown_requested", [False, True])
@pytest.mark.parametrize("quiet_ranks", [(), (0,), (1,), (0, 1)])
def test_keep_serving_requires_every_attention_quiet(
    pull_world, quiet_ranks, shutdown_requested
):
    for rank in quiet_ranks:
        pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
    for receiver in pull_world.sessions[2:]:
        assert receiver.keep_serving(shutdown_requested) is (len(quiet_ranks) != 2)


def test_peer_failure_wins_over_quiet_shutdown(pull_world):
    for rank in range(2):
        pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
    pull_world.connectors[0].graph_startup.fail(RuntimeError("attention failed"))
    with pytest.raises(RuntimeError, match="attention failed"):
        pull_world.sessions[2].keep_serving(True)


@pytest.mark.parametrize("transactions_to_drain", [0, 1, 4, 5, None])
def test_drain_requires_source_completion_within_capacity_bound(
    pull_world, monkeypatch, transactions_to_drain
):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", str(4 / 262144))
    session, connector = pull_world.sessions[2], pull_world.connectors[2]
    remaining = [transactions_to_drain]
    connector._pull_has_active_source = lambda: remaining[0] != 0

    def run_one():
        if remaining[0] is not None:
            remaining[0] -= 1

    if transactions_to_drain in (5, None):
        with pytest.raises(RuntimeError, match="retain resources"):
            session.drain(run_one)
        if transactions_to_drain == 5:
            assert remaining[0] == 1
        with pytest.raises(RuntimeError, match="not drained.*retain resources"):
            session.finish_receiver()
        assert not pull_world.store.check(["ffn_graph/test/pull_stopped/2"])
    else:
        session.drain(run_one)
        assert remaining[0] == 0
        for rank in range(2):
            pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
        session.finish_receiver()
        assert pull_world.store.check(["ffn_graph/test/pull_stopped/2"])


def test_failed_drain_does_not_publish_receiver_stopped(pull_world):
    session, connector = pull_world.sessions[2], pull_world.connectors[2]
    connector._pull_has_active_source = lambda: True

    def fail_transaction():
        raise RuntimeError("transaction failed")

    with pytest.raises(RuntimeError, match="transaction failed"):
        session.drain(fail_transaction)
    with pytest.raises(RuntimeError, match="retain resources"):
        session.finish_receiver()
    assert not pull_world.store.check(["ffn_graph/test/pull_stopped/2"])


def test_close_waits_for_all_attention_then_all_drained_receivers(pull_world):
    first_a, last_a, first_f, last_f = pull_world.sessions
    with pytest.raises(TimeoutError):
        first_a.close_attention()
    first_f.drain(lambda: pytest.fail("an idle receiver needs no transaction"))
    with pytest.raises(TimeoutError):
        first_f.finish_receiver()
    assert not pull_world.store.check(["ffn_graph/test/pull_stopped/2"])
    with pytest.raises(TimeoutError):
        last_a.close_attention()
    first_f.finish_receiver()
    with pytest.raises(TimeoutError):
        first_a.close_attention()
    last_f.drain(lambda: pytest.fail("an idle receiver needs no transaction"))
    last_f.finish_receiver()
    first_a.close_attention()
    last_a.close_attention()
    assert pull_world.store.check(
        [f"ffn_graph/test/pull_stopped/{rank}" for rank in (2, 3)]
    )
    pull_world.connectors[3].graph_startup.fail(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        first_a.close_attention()


def test_finish_receiver_is_idempotent_after_store_release(pull_world):
    session = pull_world.sessions[2]
    for rank in range(2):
        pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
    session.drain(lambda: pytest.fail("an idle receiver needs no transaction"))
    session.finish_receiver()
    pull_world.store.closed = True
    session.finish_receiver()


@pytest.mark.parametrize("state", ["unfinished", "active", "failed", "drained"])
def test_shutdown_requires_finished_idle_error_free_receiver(pull_world, state):
    session, connector = pull_world.sessions[2], pull_world.connectors[2]
    if state != "unfinished":
        for rank in range(2):
            pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
        session.drain(lambda: pytest.fail("an idle receiver needs no transaction"))
        session.finish_receiver()
    if state == "active":
        connector._pull_has_active_source = lambda: True
    if state == "failed":

        def status_failure():
            raise RuntimeError("status -5")

        connector._check_pull_status = status_failure
    if state == "drained":
        session.validate_shutdown()
    else:
        with pytest.raises(RuntimeError, match="retain resources|status -5"):
            session.validate_shutdown()
