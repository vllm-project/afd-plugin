# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backend-neutral DeepSeek-V2-Lite E2E scenarios."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_case_entry,
)
from tests.e2e.environment import model_case_command
from tests.e2e.models.deepseek_v2_lite.cases import (
    DEEPEP_M2N_SCENARIOS,
    V2_SINGLE_RANK_SCENARIOS,
    stable_ids,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
DEEPSEEK_V2_LITE_REPO_ID = "deepseek-ai/DeepSeek-V2-Lite"
# CI selects exact node IDs (E2E-INV-007). GPU uses the V2 cases for the
# shared 2A2F coverage; NPU retains its existing scenario IDs and now runs
# every case on native Ascend MRV2 through the shared harness.
SCENARIOS = stable_ids(os.environ.get("AFD_E2E_BACKEND", "gpu"))


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
    return model_case_command(
        "tests.e2e.models.deepseek_v2_lite.cases",
        scenario,
        gsm8k_output_path,
        backend=_required_env("AFD_E2E_BACKEND"),
        default_devices=("0", "1")
        if scenario in V2_SINGLE_RANK_SCENARIOS
        else ("0", "1", "2", "3"),
    )


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
    run_case_entry(command, tmp_path / scenario)


@pytest.mark.e2e
# DeepEP M2N (routing on Attention) runs 2A2F, including Attention TP2. The
# cases need DeepEP and 4 GPUs; CI selects test ids explicitly and does not
# run them.
@pytest.mark.parametrize("scenario", DEEPEP_M2N_SCENARIOS, ids=DEEPEP_M2N_SCENARIOS)
def test_deepseek_v2_lite_deepep_m2n(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_case_entry(command, tmp_path / scenario)
