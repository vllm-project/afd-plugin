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


def test_real_cli_accepts_an_independent_cpu_adapter(tmp_path):
    status = tmp_path / "status.json"
    model = tmp_path / "no-checkpoint-required"
    script = (
        "import sys; from tests.unit.fakes.ffn_faas import run_cpu_platform; "
        "raise SystemExit(run_cpu_platform(*sys.argv[1:3]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(model), str(status)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(status.with_name("status-rank-3.json").read_text())
    assert snapshot["backend"] == "host-contract"
    assert "cam_paired" not in snapshot
    assert snapshot["model_path"] == str(model)
    assert snapshot["phase"] == "stopped"
    assert snapshot["resources_released"] is True
