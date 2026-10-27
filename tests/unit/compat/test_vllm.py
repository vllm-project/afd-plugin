# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat import vllm as compat


@pytest.mark.parametrize(
    ("device_type", "version", "supported"),
    [
        ("cuda", "0.32.0", True),
        ("cuda", "0.32.0.dev42+g5df63a7", True),
        ("cuda", "0.30.0", False),
        ("cuda", "0.30.1", False),
        ("cuda", "0.29.0", False),
        ("cuda", "0.30.0.dev42+gced6857", False),
        ("npu", "0.30.0", True),
        ("npu", "0.30.0.dev42+gced6857", True),
        ("npu", "0.30.1", False),
        ("npu", "0.32.0", False),
        ("npu", "0.32.0.dev42+g5df63a7", False),
    ],
)
def test_version_support_matches_pinned_backend(device_type, version, supported):
    assert (
        compat.is_vllm_version_supported(version, device_type=device_type) is supported
    )


@pytest.mark.parametrize(
    ("device_type", "version", "compatible"),
    [
        ("cuda", "0.32.0", True),
        ("cuda", "0.30.0", False),
        ("npu", "0.30.0", True),
        ("npu", "0.32.0", False),
        ("npu", "0.32.0.dev42+g5df63a7", False),
    ],
)
def test_patch_gate_uses_active_platform(monkeypatch, device_type, version, compatible):
    runtime = ModuleType("vllm")
    runtime.__version__ = version
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type=device_type)
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    assert compat.is_target_vllm_compatible() is compatible


@pytest.mark.parametrize("device_type, target", [("cuda", "0.32.0"), ("npu", "0.30.0")])
def test_version_error_reports_backend_pin(monkeypatch, device_type, target):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: "0.29.0")

    with pytest.raises(RuntimeError, match=f"exactly vLLM {target}"):
        compat.assert_vllm_version_supported(device_type=device_type)


def test_missing_runtime_keeps_cpu_safe_patch_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)

    assert compat.is_target_vllm_compatible()
    assert compat.get_target_vllm_version() == "0.32.0"
