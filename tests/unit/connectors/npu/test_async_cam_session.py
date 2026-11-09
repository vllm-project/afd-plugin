# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host session lifecycle contracts; not device execution qualification."""

import sys
from types import SimpleNamespace

import pytest

from afd_plugin.connectors.npu.async_cam_runtime import (
    async_cam_startup as startup_module,
)
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_session import (
    PullCAMSession,
)
from tests.unit.fakes.async_cam import FakePullState, _FakeTorch
from tests.unit.fakes.async_cam_startup import MemoryStore, startup_world


@pytest.fixture
def pull_world(monkeypatch):
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    store = MemoryStore()
    startups = startup_world(store=store)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    states = [FakePullState() for _ in startups]
    sessions = []
    for startup, state in zip(startups, states, strict=True):
        session = PullCAMSession(
            startup,
            SimpleNamespace(
                ffn_size=2,
                attn_size=2,
                expert_per_rank=2,
                world_rank=startup.world_rank,
            ),
            hidden_size=16,
            topk=2,
            max_num_batched_tokens=8,
            tp_size=1,
            receive_capacity=4,
        )
        session.comm_args = state
        sessions.append(session)
    return SimpleNamespace(
        sessions=sessions, states=states, startups=startups, store=store
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
    pull_world.startups[0].fail(RuntimeError("attention failed"))
    with pytest.raises(RuntimeError, match="attention failed"):
        pull_world.sessions[2].keep_serving(True)


@pytest.mark.parametrize("transactions_to_drain", [0, 1, 4, 5, None])
def test_drain_requires_source_completion_within_capacity_bound(
    pull_world, transactions_to_drain
):
    session, words = pull_world.sessions[2], pull_world.states[2].words
    remaining = [transactions_to_drain]
    words[32] = -1 if remaining[0] == 0 else 0

    def run_one():
        if remaining[0] is not None:
            remaining[0] -= 1
        words[32] = -1 if remaining[0] == 0 else 0

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
    session, words = pull_world.sessions[2], pull_world.states[2].words
    words[32] = 0

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
    pull_world.startups[3].fail(RuntimeError("drain failed"))
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
    session, words = pull_world.sessions[2], pull_world.states[2].words
    if state != "unfinished":
        for rank in range(2):
            pull_world.store.set(f"ffn_graph/test/pull_quiet/{rank}", "ready")
        session.drain(lambda: pytest.fail("an idle receiver needs no transaction"))
        session.finish_receiver()
    if state == "active":
        words[32] = 0
    if state == "failed":
        words[24] = -5
    if state == "drained":
        session.validate_shutdown()
    else:
        with pytest.raises(RuntimeError, match="retain resources|status -5"):
            session.validate_shutdown()
