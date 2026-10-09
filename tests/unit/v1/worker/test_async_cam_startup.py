# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-side Async CAM startup ordering and failure tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch_npu")

from afd_plugin.connectors.npu.async_cam import AFDAsyncTopology  # noqa: E402
from afd_plugin.distributed.afd_process_group import (  # noqa: E402
    ProcessGroupRendezvousContext,
)
from afd_plugin.v1.worker.npu import async_cam_startup  # noqa: E402
from afd_plugin.v1.worker.npu.async_cam_startup import (  # noqa: E402
    AsyncCamStartupCoordinator,
    AsyncCamStartupSpec,
    FFNStartupPlan,
)


class FakeStore:
    def __init__(self, values: dict[str, bytes] | None = None):
        self.values = dict(values or {})

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode()

    def check(self, keys: list[str]) -> bool:
        return all(key in self.values for key in keys)

    def get(self, key: str) -> bytes:
        return self.values[key]

    def compare_set(self, key: str, expected: str, desired: str) -> bytes:
        current = self.values.get(key, b"")
        if current == expected.encode():
            current = desired.encode()
            self.values[key] = current
        return current


def make_coordinator(
    *, role: str, attn_size: int = 1, ffn_size: int = 1, store: FakeStore
) -> tuple[AsyncCamStartupCoordinator, SimpleNamespace]:
    world_rank = 0 if role == "attention" else attn_size
    topology = AFDAsyncTopology(
        role=role,
        role_rank=0,
        world_rank=world_rank,
        attn_size=attn_size,
        ffn_size=ffn_size,
        expert_per_rank=1,
    )
    connector = SimpleNamespace(is_initialized=True)
    context = ProcessGroupRendezvousContext()
    context.retain_store(store)
    context.bind(SimpleNamespace())
    coordinator = AsyncCamStartupCoordinator(
        connector,
        context,
        AsyncCamStartupSpec(
            topology=topology,
            local_rank=0,
            tp_size=1,
            hidden_size=16,
            topk=1,
            activation_dtype=async_cam_startup.torch.bfloat16,
        ),
    )
    coordinator._store = store
    return coordinator, connector


def test_mode_disagreement_prevents_receiver_and_capture():
    store = FakeStore({"ffn/mode/2": b"eager"})
    coordinator, _ = make_coordinator(role="ffn", attn_size=1, ffn_size=2, store=store)
    calls = []
    with pytest.raises(RuntimeError, match="modes disagree"):
        coordinator.start_ffn(
            prepare=lambda: FFNStartupPlan(use_graph=True, first_layer_idx=3),
            consume_warmup=lambda count: calls.append("warmup"),
            capture=lambda: calls.append("capture"),
            start_receiver=lambda: calls.append("receiver"),
        )
    assert calls == []
    assert store.values["ffn/1"].startswith(b"failed:")


def test_all_attention_and_ffn_warmup_must_finish_before_capture(monkeypatch):
    store = FakeStore(
        {
            "attn/prepared/0": b"ready",
            "attn/done/0": b"ready",
            "ffn/mode/2": b"graph:3",
        }
    )
    coordinator, _ = make_coordinator(role="ffn", attn_size=1, ffn_size=2, store=store)
    events = []

    def finish_peer(_: float) -> None:
        assert events == ["prepare", "warmup:1"]
        store.set("ffn/warmup/done/2", "ready")

    monkeypatch.setattr(async_cam_startup.time, "sleep", finish_peer)
    coordinator.start_ffn(
        prepare=lambda: (
            events.append("prepare")
            or FFNStartupPlan(use_graph=True, first_layer_idx=3)
        ),
        consume_warmup=lambda count: events.append(f"warmup:{count}"),
        capture=lambda: events.append("capture"),
        start_receiver=lambda: events.append("receiver"),
    )
    assert events == ["prepare", "warmup:1", "capture", "receiver"]
    assert store.values["ffn/1"] == b"ready"
    assert coordinator.started


@pytest.mark.parametrize("failure_phase", ["capture", "receiver"])
def test_capture_or_receiver_failure_is_sticky(failure_phase):
    store = FakeStore({"attn/prepared/0": b"ready", "attn/done/0": b"ready"})
    coordinator, _ = make_coordinator(role="ffn", store=store)

    def fail_if_selected(phase: str) -> None:
        if phase == failure_phase:
            raise RuntimeError(f"{phase} failed")

    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        coordinator.start_ffn(
            prepare=lambda: FFNStartupPlan(use_graph=True, first_layer_idx=3),
            consume_warmup=lambda count: None,
            capture=lambda: fail_if_selected("capture"),
            start_receiver=lambda: fail_if_selected("receiver"),
        )
    assert store.values["ffn/1"].startswith(b"failed:")
    assert not coordinator.started
    with pytest.raises(RuntimeError, match="previously failed"):
        coordinator.start_ffn(
            prepare=lambda: FFNStartupPlan(use_graph=True, first_layer_idx=3),
            consume_warmup=lambda count: None,
            capture=lambda: None,
            start_receiver=lambda: None,
        )


def test_ready_cannot_replace_failed_status():
    class RacingStore(FakeStore):
        def compare_set(self, key: str, expected: str, desired: str) -> bytes:
            self.set(key, "failed:loop stopped")
            return self.values[key]

    store = RacingStore()
    coordinator, _ = make_coordinator(role="ffn", store=store)
    with pytest.raises(RuntimeError, match="loop stopped"):
        coordinator.start_ffn(
            prepare=lambda: FFNStartupPlan(use_graph=False),
            consume_warmup=lambda count: None,
            capture=lambda: None,
            start_receiver=lambda: None,
        )
    assert store.values["ffn/1"].startswith(b"failed:")


def test_attention_rereads_ready_ranks_for_later_failure(monkeypatch):
    store = FakeStore({"ffn/1": b"ready"})
    coordinator, _ = make_coordinator(
        role="attention", attn_size=1, ffn_size=2, store=store
    )

    def fail_after_first_poll(_: float) -> None:
        store.set("ffn/1", "failed:receiver stopped")
        store.set("ffn/2", "ready")

    monkeypatch.setattr(async_cam_startup.time, "sleep", fail_after_first_poll)
    with pytest.raises(RuntimeError, match="receiver stopped"):
        coordinator._wait_for_ffn_ready()


def test_attention_start_does_not_return_before_ffn_ready(monkeypatch):
    store = FakeStore({"ffn/mode/1": b"eager"})
    coordinator, _ = make_coordinator(role="attention", store=store)
    polls = []

    def publish_after_wait(_: float) -> None:
        assert not coordinator.started
        polls.append(True)
        store.set("ffn/1", "ready")

    monkeypatch.setattr(async_cam_startup.time, "sleep", publish_after_wait)
    coordinator.start_attention()
    assert polls == [True]
    assert coordinator.started


def test_prepare_failure_before_store_does_not_start_nonce_collective(monkeypatch):
    store = FakeStore()
    coordinator, connector = make_coordinator(role="ffn", store=store)
    connector.is_initialized = False
    connector.init_afd_connector = lambda: pytest.fail("group was initialized")
    coordinator._store = None
    monkeypatch.setattr(
        coordinator,
        "_get_store",
        lambda: pytest.fail("nonce collective was started"),
    )
    with pytest.raises(RuntimeError, match="weights unavailable"):
        coordinator.start_ffn(
            prepare=lambda: (_ for _ in ()).throw(RuntimeError("weights unavailable")),
            consume_warmup=lambda count: None,
            capture=lambda: None,
            start_receiver=lambda: None,
        )
    assert coordinator.failed
    assert store.values == {}


def test_repeated_eager_start_is_idempotent_and_closed_context_is_rejected():
    store = FakeStore()
    coordinator, connector = make_coordinator(role="ffn", store=store)
    events = []
    connector.init_afd_connector = lambda: events.append("init")
    callbacks = dict(
        prepare=lambda: events.append("prepare") or FFNStartupPlan(False),
        consume_warmup=lambda count: events.append("warmup"),
        capture=lambda: events.append("capture"),
        start_receiver=lambda: events.append("receiver"),
    )
    coordinator.start_ffn(**callbacks)
    coordinator.start_ffn(**callbacks)
    assert events == ["prepare", "receiver"]
    coordinator._rendezvous_context.invalidate()
    with pytest.raises(RuntimeError, match="not bound"):
        coordinator.start_ffn(**callbacks)
