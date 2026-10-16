# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure host negotiation tests; no device or model runner initialization."""

import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Condition
from types import ModuleType, SimpleNamespace
from typing import Literal

import pytest

# Compile the real connector's host-only methods without importing its NPU
# dependencies. This keeps the existing CPU negotiation tests runnable;
# full connector construction is covered by test_async_ffn_runtime.py.
source = Path(__file__).resolve().parents[3] / "afd_plugin/connectors/npu/async_cam.py"
module_ast = ast.parse(source.read_text())
host_methods = {
    "_graph_keys",
    "_graph_publish",
    "check_graph_failure",
    "fail_graph_startup",
    "_graph_wait",
    "wait_graph_device",
    "graph_phase",
    "_negotiate_graph",
    "graph_service_ready",
    "_wait_graph_service_ready",
}
body: list[ast.stmt] = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
]
for node in module_ast.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
        "FFNGraphPlan",
        "resolve_ffn_mode",
        "CAMAsyncAFDConnector",
    ):
        if isinstance(node, ast.ClassDef) and node.name == "CAMAsyncAFDConnector":
            node.bases = []
            node.body = [
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name in host_methods
            ]
            assert {
                m.name for m in node.body if isinstance(m, ast.FunctionDef)
            } == host_methods
        body.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name)
        and t.id in {"GRAPH_STARTUP_TIMEOUT_SECONDS", "POLL_SECONDS"}
        for t in node.targets
    ):
        body.append(node)
startup_module = ModuleType("afd_graph_startup_contract")
sys.modules[startup_module.__name__] = startup_module
vars(startup_module).update(
    json=json, time=time, asdict=asdict, dataclass=dataclass, Literal=Literal
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        str(source),
        "exec",
    ),
    vars(startup_module),
)
CAMAsyncAFDConnector = startup_module.CAMAsyncAFDConnector
FFNGraphPlan = startup_module.FFNGraphPlan
resolve_ffn_mode = startup_module.resolve_ffn_mode

from afd_plugin.validation import (  # noqa: E402
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
        owner = CAMAsyncAFDConnector()
        owner._graph_store = store
        owner.world_rank, owner.attn_size = rank, attn
        owner.topology = SimpleNamespace(world_size=attn + ffn)
        owner._graph_prefix = "ffn_graph/test/"
        owner.ffn_graph_mode = "eager"
        owner._graph_warmup_layer_id = None
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
            pool.submit(
                owner._negotiate_graph, None if i < 2 else full_plan(), "same-wire"
            )
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(
        owner.ffn_graph_mode == "full" and owner._graph_warmup_layer_id == 2
        for owner in owners
    )
    for owner in owners[2:]:
        owner.graph_service_ready()
    for owner in owners[:2]:
        owner._wait_graph_service_ready()


def test_transport_mismatch_rejects_before_payload_initialization():
    owners = startup_world(1, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._negotiate_graph, None if i == 0 else full_plan(), str(i))
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
            pool.submit(
                owner._negotiate_graph, None if i == 0 else FFNGraphPlan(), str(i)
            )
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(owner.ffn_graph_mode == "eager" for owner in owners)


@pytest.mark.parametrize("failure", ["timeout", "publication"])
def test_nonowner_attention_failure_is_visible_to_late_ffn(monkeypatch, failure):
    owners = startup_world()
    attention, ffn = owners[1], owners[2]
    monkeypatch.setattr(startup_module, "GRAPH_STARTUP_TIMEOUT_SECONDS", 0)
    if failure == "publication":
        original = attention._graph_store.set

        def fail_plan(key, value):
            if key.endswith("plan/1"):
                raise RuntimeError("plan publication failed")
            original(key, value)

        monkeypatch.setattr(attention._graph_store, "set", fail_plan)
    with pytest.raises((TimeoutError, RuntimeError)):
        attention._negotiate_graph(None, "same-wire")
    assert attention._graph_store.check(attention._graph_keys("error", range(1, 2)))
    # Late plan and ready publications must not conceal the first failure.
    with pytest.raises(RuntimeError):
        ffn._negotiate_graph(full_plan(), "same-wire")
    assert ffn.ffn_graph_mode == "eager"
    with pytest.raises(RuntimeError):
        ffn.graph_service_ready()
    assert not ffn._graph_store.check(ffn._graph_keys("service", range(2, 3)))


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
