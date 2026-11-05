# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU DeepEP M2N launch validation happens on the CPU, before any device."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest

from afd_plugin.ffn_faas.platforms import get_platform_adapter
from tests.unit.fakes.ffn_faas import write_qwen3_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_MODULES = ("torch", "vllm", "afdep", "deep_gemm")


@pytest.fixture
def launch_env(monkeypatch):
    """A two-rank torchrun environment for F rank 1."""
    for key, value in {
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def configure(tmp_path):
    """Parse a ``serve`` command line for the GPU adapter and configure it."""
    checkpoint = write_qwen3_checkpoint(tmp_path / "model")

    def run(*options, model=checkpoint):
        adapter = get_platform_adapter("gpu-deepep-m2n")
        parser = ArgumentParser()
        parser.add_argument("model", type=Path)
        parser.add_argument("--shutdown-timeout", type=float, default=5.0)
        adapter.add_arguments(parser)
        args = parser.parse_args(
            [str(model), "--attention-ranks", "2", "--ffn-ranks", "2", *options]
        )
        return adapter.configure(args)

    return run


def test_backend_selection_imports_no_native_stack():
    script = (
        "import json, sys; "
        "from afd_plugin.ffn_faas.platforms import get_platform_adapter; "
        "get_platform_adapter('gpu-deepep-m2n'); "
        f"print(json.dumps([m for m in {NATIVE_MODULES!r} if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_valid_launch_describes_the_rank_and_its_lanes(launch_env, configure, tmp_path):
    launch = configure("--graph", "--dbo", "--port", "6239")

    assert (launch.rank, launch.local_rank) == (1, 1)
    assert launch.model_path == str((tmp_path / "model").resolve())
    assert launch.backend_name == "DeepEPM2NAFDConnector"
    assert launch.initial_status == {
        "payload_format": "bf16",
        "num_lanes": 2,
        "graph_mode": "FULL_DECODE_ONLY",
        "dbo": True,
        "m2n_paired": False,
    }


@pytest.mark.parametrize(
    "env, options, message",
    [
        ({"WORLD_SIZE": "4"}, (), "WORLD_SIZE must equal --ffn-ranks"),
        ({"VLLM_USE_V2_MODEL_RUNNER": "0"}, (), "VLLM_USE_V2_MODEL_RUNNER=1"),
        ({}, ("--connector-extra-config", "{bad"), "not valid JSON"),
        ({}, ("--connector-extra-config", '{"num_lanes": 2}'), "num_lanes"),
    ],
    ids=["world-size", "model-runner-v2", "malformed-json", "lanes-vs-dbo"],
)
def test_invalid_launch_is_rejected(
    launch_env, configure, monkeypatch, env, options, message
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        configure(*options)


def test_unsupported_checkpoint_is_rejected(launch_env, configure, tmp_path):
    fp8 = write_qwen3_checkpoint(
        tmp_path / "fp8", quantization_config={"quant_method": "fp8"}
    )
    with pytest.raises(ValueError, match="quantized"):
        configure(model=fp8)


def test_cli_reports_a_bad_launch_as_a_usage_error(tmp_path):
    checkpoint = write_qwen3_checkpoint(tmp_path / "model")
    env = {**os.environ, "WORLD_SIZE": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "afd_plugin.ffn_faas",
            "serve",
            str(checkpoint),
            "--backend",
            "gpu-deepep-m2n",
            "--attention-ranks",
            "2",
            "--ffn-ranks",
            "2",
            "--status-port",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "WORLD_SIZE must equal --ffn-ranks" in result.stderr
