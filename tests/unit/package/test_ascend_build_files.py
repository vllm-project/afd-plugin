# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
import setuptools

_ASCEND_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "TORCH_NPU_PATH",
)


def _run_setup_py(
    monkeypatch: pytest.MonkeyPatch,
    *,
    afd_build_ascend_ops: str | None = None,
    has_torch_npu: bool = False,
    has_ascend_toolkit: bool = False,
    ascend_env_var: str | None = None,
) -> list[str]:
    root = Path(__file__).resolve().parents[3]
    captured: dict[str, object] = {}

    def fake_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "torch_npu":
            return ModuleSpec(name, loader=None) if has_torch_npu else None
        return real_find_spec(name, package)

    real_path_exists = Path.exists

    def fake_path_exists(path: Path) -> bool:
        if path.as_posix() == "/usr/local/Ascend/ascend-toolkit/latest":
            return has_ascend_toolkit
        return real_path_exists(path)

    monkeypatch.setattr(setuptools, "setup", fake_setup)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(Path, "exists", fake_path_exists)
    monkeypatch.delenv("AFD_BUILD_ASCEND_OPS", raising=False)
    for name in _ASCEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    if afd_build_ascend_ops is not None:
        monkeypatch.setenv("AFD_BUILD_ASCEND_OPS", afd_build_ascend_ops)
    if ascend_env_var is not None:
        monkeypatch.setenv(ascend_env_var, "/opt/ascend")

    runpy.run_path(str(root / "setup.py"))

    ext_modules = captured["ext_modules"]
    return [ext.name for ext in ext_modules]  # type: ignore[attr-defined]


def test_ascend_ops_build_is_disabled_by_default_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _run_setup_py(monkeypatch) == []


@pytest.mark.parametrize(
    ("has_torch_npu", "has_ascend_toolkit", "ascend_env_var"),
    [
        (True, False, None),
        (False, True, None),
        (False, False, "ASCEND_HOME_PATH"),
    ],
)
def test_ascend_ops_build_is_enabled_by_default_on_npu(
    monkeypatch: pytest.MonkeyPatch,
    has_torch_npu: bool,
    has_ascend_toolkit: bool,
    ascend_env_var: str | None,
):
    assert _run_setup_py(
        monkeypatch,
        has_torch_npu=has_torch_npu,
        has_ascend_toolkit=has_ascend_toolkit,
        ascend_env_var=ascend_env_var,
    ) == ["afd_plugin._C_ascend"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", ["afd_plugin._C_ascend"]),
        ("true", ["afd_plugin._C_ascend"]),
        (" yes ", ["afd_plugin._C_ascend"]),
        ("yes", ["afd_plugin._C_ascend"]),
        ("on", ["afd_plugin._C_ascend"]),
        ("0", []),
        ("false", []),
        (" no ", []),
        ("no", []),
        ("off", []),
    ],
)
def test_ascend_ops_build_env_overrides_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: list[str],
):
    assert (
        _run_setup_py(
            monkeypatch,
            afd_build_ascend_ops=value,
            has_torch_npu=value in {"0", "false", "no", "off"},
        )
        == expected
    )


def test_ascend_ops_build_env_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(RuntimeError, match="AFD_BUILD_ASCEND_OPS"):
        _run_setup_py(monkeypatch, afd_build_ascend_ops="maybe")


def test_empty_ascend_ops_build_env_uses_platform_default(
    monkeypatch: pytest.MonkeyPatch,
):
    assert _run_setup_py(monkeypatch, afd_build_ascend_ops="") == []
    assert _run_setup_py(
        monkeypatch,
        afd_build_ascend_ops=" ",
        has_torch_npu=True,
    ) == ["afd_plugin._C_ascend"]


def _load_select_ops():
    """Import csrc/npu/scripts/select_ops.py, which is not an importable package."""
    root = Path(__file__).resolve().parents[3]
    path = root / "csrc/npu/scripts/select_ops.py"
    spec = importlib.util.spec_from_file_location("afd_select_ops", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_select_ops(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable,
            str(root / "csrc/npu/scripts/select_ops.py"),
            "--registry",
            str(root / "csrc/npu/ascend_kernels/operator_registry.json"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("soc", "expected"),
    [
        (
            "ascend910_93",
            [
                "a2e",
                "e2a",
                "afd_async_dispatch_send",
                "afd_async_dispatch_recv",
                "afd_async_combine_send",
                "afd_async_combine_recv",
                "grouped_matmul_swiglu_quant_v2_layered",
                "grouped_matmul_layered",
            ],
        ),
        ("ascend950", ["a2e", "e2a"]),
    ],
)
@pytest.mark.parametrize("shmem", ["0", "1"])
def test_select_ops_resolves_registry_order(soc: str, expected: list[str], shmem: str):
    result = _run_select_ops("--soc", soc, "--shmem", shmem)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected


def test_select_ops_accepts_explicit_operator_subset():
    result = _run_select_ops("--soc", "ascend950", "--shmem", "0", "--ops", "e2a")

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["e2a"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--soc", "ascend310p", "--shmem", "0"), "not registered"),
        (("--soc", "ascend910_93", "--shmem", "0", "--ops", "nope"), "not in SOC"),
        (("--soc", "ascend910_93", "--shmem", "0", "--ops", "a2e;"), "empty entry"),
    ],
)
def test_select_ops_rejects_invalid_selection(args: tuple[str, ...], expected: str):
    """Registry validation replaces the previous unvalidated shell selection."""
    result = _run_select_ops(*args)

    assert result.returncode != 0
    assert expected in result.stderr


def test_select_ops_drops_shmem_operators_when_shmem_missing():
    select_ops = _load_select_ops()
    registry = {
        "soc_versions": {"ascend910_93": ["a2e", "e2a"]},
        "operator_meta": {"a2e": {"requires_shmem": True}, "e2a": {}},
    }

    assert select_ops.resolve(registry, ["a2e", "e2a"], False, None) == ["e2a"]
    assert select_ops.resolve(registry, ["a2e", "e2a"], True, None) == ["a2e", "e2a"]
