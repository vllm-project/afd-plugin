# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure host negotiation tests; no device or model runner initialization."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Condition
from types import SimpleNamespace

import pytest

from afd_plugin import async_cam_startup as startup_module
from afd_plugin.async_cam_startup import (
    AsyncCAMGraphStartup,
    FFNGraphPlan,
    resolve_ffn_mode,
)
from afd_plugin.validation import (
    cam_receive_capacity,
    requested_ffn_mode,
    validate_graph_capacity,
)


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.condition = Condition()
        self.closed = False

    def set(self, key, value):
        assert not self.closed, "Store used after release"
        with self.condition:
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def get(self, key):
        assert not self.closed, "Store used after release"
        with self.condition:
            assert self.condition.wait_for(lambda: key in self.values, timeout=3)
            return self.values[key]

    def check(self, keys):
        assert not self.closed, "Store used after release"
        with self.condition:
            return all(key in self.values for key in keys)


def startup_world(attn=2, ffn=2, store=None):
    store = store or MemoryStore()
    owners = []
    for rank in range(attn + ffn):
        owner = AsyncCAMGraphStartup(rank, attn, attn + ffn)
        owner._store = store
        owner._prefix = "ffn_graph/test/"
        owners.append(owner)
    return owners


def full_plan(**changes):
    return replace(
        FFNGraphPlan(requested="full", signature="weights/capacity", layer_ids=(2, 5)),
        **changes,
    )


@pytest.mark.parametrize("mode", ["FULL_DECODE_ONLY", "PIECEWISE", "NONE"])
def test_full_requires_explicit_complete_transaction_mode(mode):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name=mode)),
    )
    with pytest.raises(ValueError, match="explicit"):
        requested_ffn_mode(config)
    config.model_config.enforce_eager = True
    assert requested_ffn_mode(config) == "eager"
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode.name = "FULL"
    assert requested_ffn_mode(config) == "full"


def test_mixed_requested_modes_are_not_silently_downgraded():
    with pytest.raises(ValueError, match="same execution mode"):
        resolve_ffn_mode([full_plan(), FFNGraphPlan()])


def test_unsupported_full_rejects_without_automatic_fallback():
    with pytest.raises(ValueError, match="unsupported executor"):
        resolve_ffn_mode([full_plan(), full_plan(graph_error="unsupported executor")])


def test_layout_mismatch_is_a_static_full_rejection():
    with pytest.raises(ValueError, match="signatures differ"):
        resolve_ffn_mode([full_plan(), full_plan(signature="other-capacity")])
    with pytest.raises(ValueError, match="layer tables"):
        resolve_ffn_mode([full_plan(), full_plan(layer_ids=(2, 6))])
    assert resolve_ffn_mode([FFNGraphPlan(graph_error="not graph capable")]) == "eager"


def test_empty_graph_signature_cannot_advertise_support():
    with pytest.raises(ValueError, match="loaded executor"):
        resolve_ffn_mode([full_plan(signature="")])


def test_whole_world_agrees_before_service_readiness():
    owners = startup_world()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(owner._negotiate, None if i < 2 else full_plan(), "same-wire")
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(owner.mode == "full" and owner.warmup_layer_id == 2 for owner in owners)
    for owner in owners[2:]:
        owner.service_ready()
    for owner in owners[:2]:
        owner.wait_service_ready()


def test_transport_mismatch_rejects_before_payload_initialization():
    owners = startup_world(1, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._negotiate, None if i == 0 else full_plan(), str(i))
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            with pytest.raises(
                (ValueError, RuntimeError), match="transport signatures"
            ):
                future.result(timeout=3)


def test_explicit_eager_preserves_existing_transport_check_scope():
    owners = startup_world(1, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._negotiate, None if i == 0 else FFNGraphPlan(), str(i))
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(owner.mode == "eager" for owner in owners)


@pytest.mark.parametrize("failure", ["timeout", "publication"])
def test_nonowner_attention_failure_is_visible_to_late_ffn(monkeypatch, failure):
    owners = startup_world()
    attention, ffn = owners[1], owners[2]
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    if failure == "publication":
        original = attention._store.set

        def fail_plan(key, value):
            if key.endswith("plan/1"):
                raise RuntimeError("plan publication failed")
            original(key, value)

        monkeypatch.setattr(attention._store, "set", fail_plan)
    with pytest.raises((TimeoutError, RuntimeError)):
        attention._negotiate(None, "same-wire")
    assert attention._store.check(attention._keys("error", range(1, 2)))
    # Late plan and ready publications must not conceal the first failure.
    with pytest.raises(RuntimeError):
        ffn._negotiate(full_plan(), "same-wire")
    assert ffn.mode == "eager"
    with pytest.raises(RuntimeError):
        ffn.service_ready()
    assert not ffn._store.check(ffn._keys("service", range(2, 3)))


def test_capacity_matches_native_float32_factor(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", "0.000061035156249")
    # Python float truncates to 15; native std::stof rounds this to 16 rows.
    assert cam_receive_capacity() == 16


@pytest.mark.parametrize(
    "tokens,tp,capacity", [(0, 1, 16), (7, 2, 16), (32, 1, 16), (65536, 1, 65536)]
)
def test_graph_rejects_unsafe_static_envelope(tokens, tp, capacity):
    with pytest.raises(ValueError):
        validate_graph_capacity(tokens, tp, capacity)


def test_graph_accepts_conservative_uint16_boundary():
    validate_graph_capacity(65535, 1, 65535)
    validate_graph_capacity(65536, 2, 65536)


@pytest.mark.parametrize("stage", ["validated", "warmup", "captured"])
def test_phase_requires_all_ranks_and_failure_wins_over_ready(monkeypatch, stage):
    owners = startup_world()
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    # Every earlier participant waits until the final participant publishes.
    for owner in owners[:-1]:
        with pytest.raises(TimeoutError):
            owner.phase(stage)
    owners[-1].phase(stage)
    owners[0].fail(RuntimeError("peer failed"))
    with pytest.raises(RuntimeError, match="peer failed"):
        owners[-1].phase(stage)
    with pytest.raises(RuntimeError, match="peer failed"):
        owners[-1].service_ready()


def test_attention_waits_for_every_ffn_service(monkeypatch):
    attention, _, first_ffn, last_ffn = startup_world()
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    first_ffn.service_ready()
    with pytest.raises(TimeoutError):
        attention.wait_service_ready()
    last_ffn.service_ready()
    attention.wait_service_ready()
    last_ffn.fail(RuntimeError("receiver failed"))
    with pytest.raises(RuntimeError, match="receiver failed"):
        attention.wait_service_ready()


def test_initialize_owns_nonce_and_plan_negotiation():
    store = MemoryStore()
    owners = [AsyncCAMGraphStartup(rank, 1, 2) for rank in range(2)]
    owners[1].plan = full_plan()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(owner.initialize, store, "wire") for owner in owners]
        for future in futures:
            future.result(timeout=3)
    assert all(owner.mode == "full" and owner.warmup_layer_id == 2 for owner in owners)
    nonce = store.get("ffn_graph/nonce").decode()
    assert all(
        key.startswith(f"ffn_graph/{nonce}/")
        for key in store.values
        if key != "ffn_graph/nonce"
    )
