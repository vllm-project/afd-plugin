# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage for the A2E operator capability probe.

The probe answers a question about an interface, so it can be exercised without an Ascend
device: the module is loaded from its file (importing the package pulls vLLM in through its
``__init__``) and torch is stood in for by an object carrying the schema the probe reads.

The three answers matter separately. ``False`` means this build cannot run the quantized
transport, ``None`` means the probe could not tell, and ``True`` only says the arguments are
declared -- whether the kernel executes a pre-quantized transport is a device question.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _module_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "afd_plugin"
        / "compat"
        / "npu"
        / "ops.py"
    )


@pytest.fixture(scope="module")
def ops_module():
    spec = importlib.util.spec_from_file_location("afd_ops_under_test", _module_path())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_torch(monkeypatch, ops_module):
    """Install a stand-in torch whose a2e schema the probe can read."""

    monkeypatch.setattr(ops_module, "ensure_cam_p2p_ops_available", lambda: None)

    def install(arguments: tuple[str, ...] = ("x", "x_scale", "quant_mode"), schema: bool = True):
        a2e = types.SimpleNamespace()
        a2e.default = types.SimpleNamespace()
        if schema:
            a2e.default._schema = types.SimpleNamespace(
                arguments=[types.SimpleNamespace(name=name) for name in arguments],
            )
        module = types.SimpleNamespace(
            ops=types.SimpleNamespace(afd_ascend=types.SimpleNamespace(a2e=a2e)),
        )
        monkeypatch.setitem(sys.modules, "torch", module)
        return module

    return install


def test_the_arguments_are_reported_when_the_schema_lists_them(ops_module, fake_torch):
    fake_torch()

    assert ops_module.a2e_accepts_mxfp8_arguments() is True


@pytest.mark.parametrize("arguments", [("x",), ("x", "x_scale")])
def test_a_schema_without_both_arguments_is_reported_as_missing(
    ops_module, fake_torch, arguments
):
    fake_torch(arguments=arguments)

    assert ops_module.a2e_accepts_mxfp8_arguments() is False


def test_an_unreadable_schema_is_reported_as_unknown(ops_module, fake_torch):
    """``None`` rather than ``False``: the caller must say it could not confirm."""

    fake_torch(schema=False)

    assert ops_module.a2e_accepts_mxfp8_arguments() is None


def test_a_build_without_the_extension_is_reported_as_missing(ops_module, monkeypatch):
    def explode():
        raise RuntimeError("no extension")

    monkeypatch.setattr(ops_module, "ensure_cam_p2p_ops_available", explode)

    assert ops_module.a2e_accepts_mxfp8_arguments() is False
