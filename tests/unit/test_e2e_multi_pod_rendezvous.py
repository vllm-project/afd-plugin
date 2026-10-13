# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Rendezvous semantics, exercised against a real in-process TCPStore."""

from __future__ import annotations

import socket
import threading

import pytest

pytest.importorskip("torch")

from tests.e2e.multi_pod.rendezvous import (  # noqa: E402
    PASS_VERDICT,
    VERDICT_KEY,
    BarrierTimeoutError,
    PeerFailureError,
    Rendezvous,
)

POLL_INTERVAL_S = 0.05
SHORT_TIMEOUT_S = 2.0
EXPIRED_TIMEOUT_S = 0.3


@pytest.fixture
def store_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def pods(store_port: int):
    """A two-pod rendezvous; pod 0 masters the store."""
    members = [
        Rendezvous(
            host="127.0.0.1",
            port=store_port,
            pod_index=index,
            num_pods=2,
            poll_interval_s=POLL_INTERVAL_S,
        )
        for index in (0, 1)
    ]
    yield members
    members.clear()


def test_rendezvous_rejects_a_pod_index_outside_the_layout(store_port):
    """A pod cannot join a rendezvous it has no place in."""
    with pytest.raises(ValueError, match="outside 0..1"):
        Rendezvous(host="127.0.0.1", port=store_port, pod_index=2, num_pods=2)


def test_barrier_releases_when_every_pod_arrives(pods):
    """A barrier releases once every pod has arrived, and not before."""
    zero, one = pods
    errors: list[BaseException] = []

    def join_late() -> None:
        try:
            one.barrier("launched", SHORT_TIMEOUT_S)
        except BaseException as exc:  # noqa: BLE001 - reported to the assertion
            errors.append(exc)

    peer = threading.Thread(target=join_late)
    peer.start()
    zero.barrier("launched", SHORT_TIMEOUT_S)
    peer.join(timeout=SHORT_TIMEOUT_S * 2)

    assert errors == []


def test_barrier_timeout_names_the_pods_that_did_not_arrive(pods):
    """A barrier timeout identifies which pods are missing."""
    zero, _one = pods

    with pytest.raises(BarrierTimeoutError) as error:
        zero.barrier("serving", EXPIRED_TIMEOUT_S)

    assert "barrier serving: 1/2" in str(error.value)
    assert "missing=['pod-1']" in str(error.value)


def test_a_peer_abort_unwinds_every_barrier_before_its_deadline(pods):
    """One pod's failure unwinds its peers in seconds instead of at the deadline."""
    zero, one = pods
    one.publish_abort("FFN exited (rc=1) during launch")

    with pytest.raises(PeerFailureError, match="FFN exited"):
        zero.barrier("launched", SHORT_TIMEOUT_S)


def test_abort_is_published_once_per_pod(pods):
    """A pod's first failure is the one reported, not whatever followed it."""
    _zero, one = pods
    one.publish_abort("first")
    one.publish_abort("second")

    assert one.get("abort/1") == "first"


def test_a_pod_does_not_abort_on_its_own_message(pods):
    """A pod does not mistake its own abort for a peer failure."""
    zero, _one = pods
    zero.publish_abort("local failure")

    assert zero.poll_abort() is None


def test_verify_agreement_publishes_from_pod_zero(pods):
    """Pods launched with matching arguments agree and proceed."""
    zero, one = pods
    zero.verify_agreement("2A0F,0A2F", "afd-graph-2a2f")

    one.verify_agreement("2A0F,0A2F", "afd-graph-2a2f")


def test_verify_agreement_rejects_a_mismatched_layout(pods):
    """Pods launched with different layouts fail fast instead of hanging."""
    zero, one = pods
    zero.verify_agreement("2A0F,0A2F", "afd-graph-2a2f")

    with pytest.raises(PeerFailureError, match="run/layout"):
        one.verify_agreement("1A1F,1A1F", "afd-graph-2a2f")


def test_verify_agreement_rejects_a_mismatched_scenario(pods):
    """Pods launched for different scenarios fail fast instead of hanging."""
    zero, one = pods
    zero.verify_agreement("2A0F,0A2F", "afd-graph-2a2f")

    with pytest.raises(PeerFailureError, match="run/scenario"):
        one.verify_agreement("2A0F,0A2F", "afd-eager-2a2f")


def test_wait_for_key_returns_the_published_verdict(pods):
    """A pod that does not evaluate still learns the run's verdict."""
    zero, one = pods
    zero.set(VERDICT_KEY, PASS_VERDICT)

    assert one.wait_for_key(VERDICT_KEY, SHORT_TIMEOUT_S) == PASS_VERDICT


def test_wait_for_key_times_out_when_nothing_is_published(pods):
    """Waiting for a verdict that never comes ends at the deadline, naming the key."""
    _zero, one = pods

    with pytest.raises(BarrierTimeoutError, match="verdict was not published"):
        one.wait_for_key(VERDICT_KEY, EXPIRED_TIMEOUT_S)


def test_a_barrier_reports_a_local_child_failure_through_on_poll(pods):
    """A pod waiting at a barrier still notices its own children dying."""
    zero, _one = pods

    def local_child_died() -> None:
        raise RuntimeError("pod 0 FFN exited (rc=1) during launch")

    with pytest.raises(RuntimeError, match="pod 0 FFN exited"):
        zero.barrier("launched", SHORT_TIMEOUT_S, on_poll=local_child_died)


def test_get_returns_none_for_a_key_that_was_never_set(pods):
    """Reading an unset key answers at once rather than blocking."""
    zero, _one = pods

    assert zero.get("phase/1/serving") is None
