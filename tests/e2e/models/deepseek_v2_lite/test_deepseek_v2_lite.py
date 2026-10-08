# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-neutral DeepSeek-V2-Lite E2E scenarios."""

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
from tests.e2e.runner import (
    DEFAULT_GSM8K_SAMPLE_LIMIT,
    GSM8K_LIMIT_ENV,
    GSM8K_THRESHOLD_ENV,
    V2_DBO_COMPARISON_SCENARIOS,
    V2_DBO_SAMPLE_LIMIT,
    V2_SCENARIOS,
    V2_SINGLE_RANK_SCENARIOS,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
DEEPSEEK_V2_LITE_REPO_ID = "deepseek-ai/DeepSeek-V2-Lite"
DEEPSEEK_V2_LITE_MAX_MODEL_LEN = 4096
SMOKE_MIN_ACCURACY = 0.25
SMOKE_MAX_SAMPLES = 24
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3")
ATTENTION_DEVICE_COUNT = 2
AFD_FFN_DEVICE_COUNT = 1
AFD_TWO_FFN_DEVICE_COUNT = 2
BASELINE_DEVICE_COUNT = 4
TWO_FFN_SCENARIO_SUFFIX = "-2a2f"
# CI gates select the gate scenarios by pytest node ID (E2E-INV-007); the
# 2A1F scenarios below are local-only cases.
SCENARIOS = (
    # Gate scenarios: baseline plus the 2A2F AFD cases.
    "baseline-graph",
    "afd-eager-2a2f",
    "afd-graph-2a2f",
    "afd-graph-dbo-2a2f",
    # Local scenarios: 2 Attention ranks + 1 FFN rank.
    "afd-eager-2a1f",
    "afd-graph-2a1f",
    "afd-graph-dbo-2a1f",
    *V2_SCENARIOS,
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and DeepSeek-V2-Lite are available for the runner."""
    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    backend = _required_env("AFD_E2E_BACKEND")
    if backend == "gpu":
        env_name = "AFD_GPU_E2E_MODEL"
    elif backend == "npu":
        env_name = "AFD_NPU_E2E_MODEL"
    else:
        raise RuntimeError("AFD_E2E_BACKEND must be 'gpu' or 'npu'")

    existing = os.environ.get(env_name)
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    model_path = download_model(DEEPSEEK_V2_LITE_REPO_ID)
    os.environ[env_name] = str(model_path)


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend == "gpu":
        model = _required_env("AFD_GPU_E2E_MODEL")
        vllm_bin = os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm")
    elif backend == "npu":
        model = _required_env("AFD_NPU_E2E_MODEL")
        vllm_bin = os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm")
    else:
        raise RuntimeError("AFD_E2E_BACKEND must be 'gpu' or 'npu'")

    env_devices = os.environ.get("AFD_E2E_DEVICES")
    if env_devices:
        devices = [item.strip() for item in env_devices.split(",") if item.strip()]
    elif scenario in V2_SINGLE_RANK_SCENARIOS:
        devices = ["0", "1"]
    else:
        devices = list(DEFAULT_DEVICE_IDS)
    if scenario == "baseline-graph":
        attention_devices = devices[:BASELINE_DEVICE_COUNT]
        ffn_devices = []
    elif scenario in V2_SINGLE_RANK_SCENARIOS:
        attention_devices = devices[:1]
        ffn_devices = devices[1:2]
    else:
        attention_devices = devices[:ATTENTION_DEVICE_COUNT]
        ffn_device_count = (
            AFD_TWO_FFN_DEVICE_COUNT
            if scenario.endswith(TWO_FFN_SCENARIO_SUFFIX) or scenario in V2_SCENARIOS
            else AFD_FFN_DEVICE_COUNT
        )
        ffn_devices = devices[
            ATTENTION_DEVICE_COUNT : ATTENTION_DEVICE_COUNT + ffn_device_count
        ]

    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        model,
        "--vllm-bin",
        vllm_bin,
        "--device-backend",
        backend,
        f"--common-vllm-arg=--max-model-len={DEEPSEEK_V2_LITE_MAX_MODEL_LEN}",
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
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    backend = _required_env("AFD_E2E_BACKEND")
    if backend == "gpu":
        model_env_name = "AFD_GPU_E2E_MODEL"
    elif backend == "npu":
        model_env_name = "AFD_NPU_E2E_MODEL"
    else:
        raise RuntimeError("AFD_E2E_BACKEND must be 'gpu' or 'npu'")

    with preserve_environment_variable(model_env_name):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_deepseek_v2_lite(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    env = os.environ.copy()
    default_limit = (
        V2_DBO_SAMPLE_LIMIT
        if scenario in V2_DBO_COMPARISON_SCENARIOS
        else DEFAULT_GSM8K_SAMPLE_LIMIT
    )
    sample_limit = env.get(GSM8K_LIMIT_ENV, str(default_limit))
    if sample_limit != "all" and int(sample_limit) <= SMOKE_MAX_SAMPLES:
        # Model smoke gates require 2/7 or 6/24 correct; larger runs keep 0.27.
        env.setdefault(GSM8K_THRESHOLD_ENV, str(SMOKE_MIN_ACCURACY))
    run_runner(command, env=env)
