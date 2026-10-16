# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import builtins

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.v1.worker import dbo
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield


def test_maybe_apply_dbo_yield_uses_custom_op(monkeypatch):
    calls: list[str | tuple[str, object]] = []
    tensor = object()

    monkeypatch.setattr(
        dbo,
        "register_dbo_yield_custom_op",
        lambda: calls.append("register"),
    )
    monkeypatch.setattr(
        dbo.torch.ops.vllm,
        "manual_dbo_yield",
        lambda x: calls.append(("yield", x)),
        raising=False,
    )

    assert maybe_apply_dbo_yield(tensor, role="attention") is tensor
    assert calls == ["register", ("yield", tensor)]


def test_register_dbo_yield_custom_op_declares_input_mutation(monkeypatch):
    registrations = []
    yield_calls = []
    tensor = object()

    monkeypatch.setattr(dbo, "_AFD_DBO_YIELD_OP_REGISTERED", False)
    monkeypatch.setattr(
        dbo,
        "direct_register_custom_op",
        lambda **kwargs: registrations.append(kwargs),
    )
    monkeypatch.setattr(
        dbo,
        "_yield_if_dbo_enabled",
        lambda: yield_calls.append("yield"),
    )

    dbo.register_dbo_yield_custom_op()

    assert len(registrations) == 1
    registration = registrations[0]
    assert registration["op_name"] == "manual_dbo_yield"
    assert registration["mutates_args"] == ["x"]
    assert registration["op_func"](tensor) is None
    assert yield_calls == ["yield"]
    assert registration["fake_impl"](tensor) is None


def test_maybe_apply_dbo_yield_does_not_probe_ascend(monkeypatch):
    tensor = object()

    real_import = builtins.__import__

    def fail_on_ascend_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("afd_plugin.v1.worker.npu"):
            pytest.fail(f"unexpected Ascend import from DBO yield helper: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_on_ascend_import)
    monkeypatch.setattr(
        dbo,
        "register_dbo_yield_custom_op",
        lambda: (_ for _ in ()).throw(ImportError),
    )

    assert maybe_apply_dbo_yield(tensor, role="attention") is tensor


def test_dbo_yield_without_native_context_does_not_import_npu_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(dbo, "dbo_enabled", lambda: False)
    monkeypatch.setattr(dbo, "dbo_yield", lambda: calls.append("vllm"))
    real_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        assert not name.startswith("afd_plugin.v1.worker.npu")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    dbo._yield_if_dbo_enabled()
    assert calls == []


def test_dbo_yield_prefers_native_context_without_importing_legacy(monkeypatch):
    calls = []

    monkeypatch.setattr(dbo, "dbo_enabled", lambda: True)
    monkeypatch.setattr(dbo, "dbo_yield", lambda: calls.append("vllm"))
    real_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        assert name != "afd_plugin.v1.worker.npu.ubatching"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    dbo._yield_if_dbo_enabled()
    assert calls == ["vllm"]
