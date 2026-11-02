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


def negotiate(owners, plans, signatures):
    with ThreadPoolExecutor(max_workers=len(owners)) as pool:
        futures = [
            pool.submit(owner._negotiate, plan, signature)
            for owner, plan, signature in zip(owners, plans, signatures, strict=True)
        ]
        return [future.exception(timeout=3) for future in futures]


def test_mixed_requested_modes_are_not_silently_downgraded():
    with pytest.raises(ValueError, match="same execution mode"):
        resolve_ffn_mode([full_plan(), FFNGraphPlan()])


def test_unsupported_full_rejects_without_automatic_fallback():
    with pytest.raises(ValueError, match="unsupported executor"):
        resolve_ffn_mode([full_plan(), full_plan(graph_error="unsupported executor")])


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"signature": "other"}, "signatures differ"),
        ({"layer_ids": (2, 6)}, "layer tables"),
    ],
)
def test_full_rejects_incompatible_layout(changes, message):
    with pytest.raises(ValueError, match=message):
        resolve_ffn_mode([full_plan(), full_plan(**changes)])
    assert resolve_ffn_mode([FFNGraphPlan(graph_error="not graph capable")]) == "eager"


def test_empty_graph_signature_cannot_advertise_support():
    with pytest.raises(ValueError, match="loaded executor"):
        resolve_ffn_mode([full_plan(signature="")])


@pytest.mark.parametrize("mode", ["eager", "full"])
def test_two_stream_signature_and_failure_are_checked_in_both_modes(mode):
    plan = full_plan(requested=mode, stream_signature="2/12C/24V/24V")
    assert resolve_ffn_mode([plan, plan]) == mode
    with pytest.raises(ValueError, match="two-stream execution signatures"):
        resolve_ffn_mode([plan, replace(plan, stream_signature="")])
    with pytest.raises(ValueError, match="core budget"):
        resolve_ffn_mode([plan, replace(plan, stream_error="core budget")])
    with pytest.raises(ValueError, match="signatures differ"):
        resolve_ffn_mode([plan, replace(plan, signature="other-capacity")])
    with pytest.raises(ValueError, match="layer tables"):
        resolve_ffn_mode([plan, replace(plan, layer_ids=(2, 6))])
    with pytest.raises(ValueError, match="loaded executor"):
        resolve_ffn_mode([replace(plan, signature="")])


@pytest.mark.parametrize("peer_failure", [False, True])
def test_two_stream_waits_for_attention_capability_check(peer_failure):
    owners = startup_world(1, 1)
    checked = []

    def validate():
        checked.append("attention")
        if peer_failure:
            raise ValueError("DBO is not qualified")

    owners[0].validate_attention_streams = validate
    plan = full_plan(requested="eager", stream_signature="2/12C/24V/24V")
    errors = negotiate(owners, [None, plan], ["wire"] * 2)
    if peer_failure:
        assert all(
            isinstance(error, (ValueError, RuntimeError)) and "DBO" in str(error)
            for error in errors
        ), errors
    else:
        assert errors == [None, None]
    assert checked == ["attention"]


def test_whole_world_agrees_before_service_readiness():
    owners = startup_world()
    assert (
        negotiate(owners, [None, None, full_plan(), full_plan()], ["same-wire"] * 4)
        == [None] * 4
    )
    assert all(owner.mode == "full" and owner.warmup_layer_id == 2 for owner in owners)
    for owner in owners[2:]:
        owner.service_ready()
    for owner in owners[:2]:
        owner.wait_service_ready()


@pytest.mark.parametrize("two_stream", [False, True])
@pytest.mark.parametrize("mode", ["eager", "full"])
def test_transport_check_preserves_single_stream_eager_scope(two_stream, mode):
    owners = startup_world(1, 1)
    plan = full_plan(requested=mode, stream_signature="two" if two_stream else "")
    errors = negotiate(owners, [None, plan], ["a", "b"])
    if two_stream or mode == "full":
        assert all(
            isinstance(error, (ValueError, RuntimeError))
            and "transport signatures" in str(error)
            for error in errors
        ), errors
    else:
        assert errors == [None, None]
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
