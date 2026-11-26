# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

import sys
from types import ModuleType

import pytest

from afd_plugin.compat import vllm as compat


@pytest.mark.parametrize(
    ("version", "compatible"),
    [
        ("0.34.0", True),
        ("0.34.0.dev42+g62d4bc3c", True),
        ("0.32.0", False),
        ("0.32.0.dev42+g5df63a7", False),
        ("0.34.1", False),
        ("0.35.0", False),
        ("0.31.0", False),
        ("unknown", False),
    ],
)
def test_patch_gate_accepts_every_supported_pin(monkeypatch, version, compatible):
    runtime = ModuleType("vllm")
    runtime.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", runtime)

    assert compat.is_target_vllm_compatible() is compatible


@pytest.mark.parametrize(
    ("device_type", "version", "compatible"),
    [
        ("cuda", "0.34.0", True),
        ("cuda", "0.32.0", False),
        ("npu", "0.32.0", False),
        ("npu", "0.34.0", True),
    ],
)
def test_patch_gate_with_device_type_requires_that_backend_pin(
    monkeypatch, device_type, version, compatible
):
    runtime = ModuleType("vllm")
    runtime.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", runtime)

    assert compat.is_target_vllm_compatible(device_type) is compatible


@pytest.mark.parametrize("installed_version", ["0.29.0", None])
@pytest.mark.parametrize(
    ("device_type", "target"), [("cuda", "0.34.0"), ("npu", "0.34.0")]
)
def test_version_error_reports_target_pin(
    monkeypatch, installed_version, device_type, target
):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.raises(RuntimeError) as error:
        compat.assert_vllm_version_supported(device_type=device_type)
    assert str(error.value) == (
        f"AFD plugin currently supports exactly vLLM {target}; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


@pytest.mark.parametrize("installed_version", ["0.29.0", None])
@pytest.mark.parametrize(
    ("device_type", "target"), [("cuda", "0.34.0"), ("npu", "0.34.0")]
)
def test_version_warning_reports_target_pin(
    monkeypatch, installed_version, device_type, target
):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.warns(RuntimeWarning) as recorded:
        compat.assert_vllm_version_supported(strict=False, device_type=device_type)
    assert str(recorded[0].message) == (
        f"AFD plugin currently supports exactly vLLM {target}; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


def test_missing_runtime_keeps_cpu_safe_patch_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)

    assert compat.is_target_vllm_compatible()
    assert not compat.is_target_vllm_compatible("npu")
    assert compat.get_target_vllm_version() == "0.34.0"
