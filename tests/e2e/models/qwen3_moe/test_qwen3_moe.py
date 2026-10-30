# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA Qwen3 MoE E2E scenarios."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_runner,
)
from tests.e2e.runner import DEEPEP_M2N_SCENARIOS

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
QWEN3_MOE_REPO_ID = "Qwen/Qwen3-30B-A3B"
QWEN3_MOE_MAX_MODEL_LEN = 4096
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3")
ATTENTION_DEVICE_COUNT = 2
AFD_FFN_DEVICE_COUNT = 1
DEEPEP_M2N_FFN_DEVICE_COUNT = 2
BASELINE_DEVICE_COUNT = 4
SCENARIOS = (
    "baseline-graph",
    "afd-eager-2a1f",
    "afd-graph-2a1f",
    "afd-graph-dbo-2a1f",
)
# DeepEP M2N (routing on Attention) needs EP over at least two FFN ranks, so
# its scenarios (DEEPEP_M2N_SCENARIOS) run 2A2F. CI selects test ids
# explicitly and does not run them; running the whole module (pytest -m e2e
# tests/e2e/models/qwen3_moe) does, and they need DeepEP and 4 GPUs.


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and Qwen3-30B-A3B are available for the runner."""
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("Qwen3 MoE E2E supports only the 'gpu' backend")

    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(QWEN3_MOE_REPO_ID))


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "gpu":
        raise RuntimeError("Qwen3 MoE E2E supports only the 'gpu' backend")

    env_devices = os.environ.get("AFD_E2E_DEVICES")
    devices = (
        [item.strip() for item in env_devices.split(",") if item.strip()]
        if env_devices
        else list(DEFAULT_DEVICE_IDS)
    )
    if scenario == "baseline-graph":
        attention_devices = devices[:BASELINE_DEVICE_COUNT]
        ffn_devices = []
    else:
        ffn_device_count = (
            DEEPEP_M2N_FFN_DEVICE_COUNT
            if scenario in DEEPEP_M2N_SCENARIOS
            else AFD_FFN_DEVICE_COUNT
        )
        attention_devices = devices[:ATTENTION_DEVICE_COUNT]
        ffn_devices = devices[
            ATTENTION_DEVICE_COUNT : ATTENTION_DEVICE_COUNT + ffn_device_count
        ]

    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        backend,
        f"--common-vllm-arg=--max-model-len={QWEN3_MOE_MAX_MODEL_LEN}",
        "--attention-devices",
        ",".join(attention_devices),
    ]
    if scenario != "baseline-graph":
        command.extend(["--ffn-devices", ",".join(ffn_devices)])
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
            "--served-model-name-prefix",
            "qwen3-moe-afd",
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_qwen3_moe(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", DEEPEP_M2N_SCENARIOS, ids=DEEPEP_M2N_SCENARIOS)
def test_qwen3_moe_deepep_m2n(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
