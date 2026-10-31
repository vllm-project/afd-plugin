# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real child-process exit checks; these never construct a device backend."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli_case(checkpoint_dir, tmp_path):
    def run(fault, released=False):
        marker = tmp_path / "normal-shutdown"
        script = (
            "import sys; from tests.unit.fakes.ffn_faas import run_cli_host_case; "
            "raise SystemExit(run_cli_host_case(*sys.argv[1:4], "
            "released=sys.argv[4]=='1'))"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(checkpoint_dir),
                str(marker),
                fault,
                "1" if released else "0",
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result, marker

    return run


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "status_file",
        "stdout_write",
        "stdout_flush",
        "stderr_write",
        "stderr_flush",
        "close_http",
        "snapshot",
        "returned_unreleased",
    ],
)
def test_unreleased_cli_isolates_despite_reporting_failures(cli_case, fault):
    result, marker = cli_case(fault)
    assert result.returncode == 1, result.stderr
    assert not marker.exists(), "unsafe session used normal interpreter teardown"


@pytest.mark.parametrize("fault", ["none", "status_file", "close_http"])
def test_released_cli_keeps_normal_shutdown(cli_case, fault):
    result, marker = cli_case(fault, released=True)
    assert result.returncode == (0 if fault == "none" else 1), result.stderr
    assert marker.read_text() == "ordinary shutdown"


@pytest.fixture
def cpu_platform_case(tmp_path):
    status = tmp_path / "status.json"
    model = tmp_path / "no-checkpoint-required"
    # Install before importing even the fake: its module imports interfaces/service.
    script = """
import builtins
import importlib
import importlib.util
import runpy
import sys

BLOCKED = ("torch", "torch_npu", "vllm", "vllm_ascend")
assert not any(name.split(".", 1)[0] in BLOCKED for name in sys.modules)

ORIGINAL_IMPORT = builtins.__import__
ORIGINAL_IMPORT_MODULE = importlib.import_module

def require_cpu_import(name):
    if name.split(".", 1)[0] in BLOCKED:
        raise AssertionError(f"CPU host contract imported {name}")

def cpu_import(name, *args, **kwargs):
    require_cpu_import(name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)

def cpu_import_module(name, package=None):
    require_cpu_import(importlib.util.resolve_name(name, package))
    return ORIGINAL_IMPORT_MODULE(name, package)

# Allow find_spec capability queries; block ordinary and dynamic module imports.
builtins.__import__ = cpu_import
importlib.import_module = cpu_import_module
if sys.argv[3]:
    runpy.run_path(sys.argv[3])
from tests.unit.fakes.ffn_faas import run_cpu_platform
raise SystemExit(run_cpu_platform(*sys.argv[1:3]))
"""

    def run(probe=None):
        result = subprocess.run(
            [sys.executable, "-c", script, str(model), str(status), str(probe or "")],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result, model, status

    return run


def test_real_cli_accepts_an_independent_cpu_adapter(cpu_platform_case):
    result, model, status = cpu_platform_case()
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(status.with_name("status-rank-3.json").read_text())
    assert snapshot["backend"] == "host-contract"
    assert "cam_paired" not in snapshot
    assert snapshot["model_path"] == str(model)
    assert snapshot["phase"] == "stopped"
    assert snapshot["resources_released"] is True


def test_supported_model_selection_keeps_cpu_import_boundary(
    cpu_platform_case, checkpoint_dir, tmp_path
):
    probe = tmp_path / "select_model.py"
    probe.write_text(
        "from pathlib import Path\n"
        "from afd_plugin.ffn_faas.platforms.npu import get_model\n"
        f"model = get_model(Path({str(checkpoint_dir)!r}), 2, 1)\n"
        "assert model.get_experts_layer_indices() == (0, 1)\n"
        "assert model.parameter_bytes > 0\n"
        "model.close()\n"
    )
    result, _, _ = cpu_platform_case(probe)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("package", "statement"),
    [
        ("torch", "import torch"),
        ("torch_npu", "import torch_npu"),
        ("vllm", "from vllm import config"),
        (
            "vllm_ascend",
            "from importlib import import_module; import_module('vllm_ascend.config')",
        ),
    ],
)
def test_cpu_entry_rejects_platform_imports(
    cpu_platform_case, tmp_path, package, statement
):
    # An importable stub ensures missing packages cannot make a broken gate pass.
    stub = tmp_path / package
    stub.mkdir()
    (stub / "__init__.py").write_text("raise RuntimeError('platform stub ran')\n")
    probe = tmp_path / "import_probe.py"
    probe.write_text(
        f"import importlib.util, sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
        f"assert importlib.util.find_spec({package!r}) is not None\n{statement}\n"
    )
    result, _, _ = cpu_platform_case(probe)
    assert result.returncode == 1
    assert f"CPU host contract imported {package}" in result.stderr
    assert "platform stub ran" not in result.stderr
