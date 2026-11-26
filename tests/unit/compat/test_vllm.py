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
def test_patch_gate_accepts_only_the_pinned_release(monkeypatch, version, compatible):
    runtime = ModuleType("vllm")
    runtime.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", runtime)

    assert compat.is_target_vllm_compatible() is compatible


@pytest.mark.parametrize("installed_version", ["0.32.0", None])
def test_version_error_reports_target_pin(monkeypatch, installed_version):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.raises(RuntimeError) as error:
        compat.assert_vllm_version_supported()
    assert str(error.value) == (
        "AFD plugin currently supports exactly vLLM 0.34.0; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


@pytest.mark.parametrize("installed_version", ["0.32.0", None])
def test_version_warning_reports_target_pin(monkeypatch, installed_version):
    monkeypatch.setattr(compat, "get_installed_vllm_version", lambda: installed_version)

    with pytest.warns(RuntimeWarning) as recorded:
        compat.assert_vllm_version_supported(strict=False)
    assert str(recorded[0].message) == (
        "AFD plugin currently supports exactly vLLM 0.34.0; "
        f"installed vLLM version is {installed_version or 'not installed'}"
    )


def test_missing_runtime_keeps_cpu_safe_patch_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)

    assert compat.is_target_vllm_compatible()
    assert not compat.is_target_vllm_compatible(allow_missing=False)
