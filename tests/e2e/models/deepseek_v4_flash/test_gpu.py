# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA DeepSeek-V4-Flash correctness and separate DBO qualification cases."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import download_dataset, run_runner
from tests.e2e.environment import devices_from_env, required_env
from tests.e2e.models.deepseek_v4_flash.gpu_config import (
    DSV4_GPU_BASIC_SCENARIOS,
    DSV4_GPU_DBO_SCENARIOS,
    DSV4_GPU_DEVICE_COUNT,
    DSV4_GPU_ROLE_RANKS,
    DSV4_GPU_SCENARIO_SETTINGS,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"


@pytest.fixture(scope="module", autouse=True)
def prepare_e2e_assets() -> None:
    if required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("CUDA DSV4 E2E requires AFD_E2E_BACKEND=gpu")
    devices_from_env("AFD_E2E_DEVICES", DSV4_GPU_DEVICE_COUNT)
    # The Flash checkpoint is large: provision it explicitly instead of
    # starting an implicit model download after obtaining a GPU reservation.
    required_env("AFD_GPU_E2E_MODEL")
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)


def build_runner_command(scenario: str, output_dir: Path) -> list[str]:
    devices = devices_from_env("AFD_E2E_DEVICES", DSV4_GPU_DEVICE_COUNT)
    baseline = DSV4_GPU_SCENARIO_SETTINGS[scenario][0]
    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "gpu",
        "--attention-devices",
        ",".join(devices[:DSV4_GPU_ROLE_RANKS]),
        "--scenario",
        scenario,
        "--served-model-name-prefix",
        "dsv4-flash-gpu",
        "--completion-output-path",
        str(output_dir / "completions.json"),
        "--gsm8k-output-path",
        str(output_dir / "gsm8k"),
        "--startup-timeout",
        os.environ.get("AFD_GPU_E2E_STARTUP_TIMEOUT", "1800"),
        "--afd-port",
        os.environ.get("AFD_GPU_DSV4_E2E_AFD_PORT", "6457"),
        "--api-port-base",
        os.environ.get("AFD_GPU_DSV4_E2E_API_PORT", "19480"),
    ]
    if not baseline:
        command.extend(["--ffn-devices", ",".join(devices[DSV4_GPU_ROLE_RANKS:])])
    return command


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", DSV4_GPU_BASIC_SCENARIOS)
def test_deepseek_v4_flash_gpu(scenario: str, tmp_path: Path) -> None:
    run_runner(build_runner_command(scenario, tmp_path))


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", DSV4_GPU_DBO_SCENARIOS)
def test_deepseek_v4_flash_gpu_dbo(scenario: str, tmp_path: Path) -> None:
    """Qualify DBO separately with the shared nonempty-stage execution oracle."""
    run_runner(build_runner_command(scenario, tmp_path))
