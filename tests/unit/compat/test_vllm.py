# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

import sys
from types import ModuleType

import pytest

from afd_plugin.compat import vllm as compat


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.32.0", True),
        ("0.32.0.dev42+g5df63a7", True),
        ("0.32.1", False),
        ("0.33.0", False),
        ("0.30.0", False),
        ("0.30.0.dev42+gced6857", False),
    ],
)
def test_version_support_matches_target_release(version, supported):
    assert compat.is_vllm_version_supported(version) is supported


@pytest.mark.parametrize(
    ("version", "compatible"),
    [
        ("0.32.0", True),
        ("0.32.0.dev42+g5df63a7", True),
        ("0.32.1", False),
        ("0.30.0", False),
        ("0.30.0.dev42+gced6857", False),
        ("unknown", False),
    ],
)
def test_patch_gate_matches_target_release(monkeypatch, version, compatible):
    runtime = ModuleType("vllm")
    runtime.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", runtime)

    assert compat.is_target_vllm_compatible() is compatible


@pytest.mark.parametrize("installed_version", ["0.29.0", None])
def test_version_error_reports_target_pin(monkeypatch, installed_version):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.raises(RuntimeError) as error:
        compat.assert_vllm_version_supported()
    assert str(error.value) == (
        "AFD plugin currently supports exactly vLLM 0.32.0; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


@pytest.mark.parametrize("installed_version", ["0.29.0", None])
def test_version_warning_reports_target_pin(monkeypatch, installed_version):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.warns(RuntimeWarning) as recorded:
        compat.assert_vllm_version_supported(strict=False)
    assert str(recorded[0].message) == (
        "AFD plugin currently supports exactly vLLM 0.32.0; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


def test_missing_runtime_keeps_cpu_safe_patch_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)

    assert compat.is_target_vllm_compatible()
    assert compat.get_target_vllm_version() == "0.32.0"
