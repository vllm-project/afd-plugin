# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in A3 DSpark acceptance cases."""

import os
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, run_case_entry
from tests.e2e.environment import model_case_command, required_env
from tests.e2e.models.deepseek_v4_flash.dspark_cases import CASE_IDS, GRAPH


def build_runner_command(output_path: Path, scenario: str) -> list[str]:
    if required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSpark E2E requires AFD_E2E_BACKEND=npu")
    command = model_case_command(
        "tests.e2e.models.deepseek_v4_flash.dspark_cases",
        scenario,
        output_path,
        backend="npu",
    )
    command.extend(
        [
            "--afd-host",
            required_env("HCCL_IF_IP"),
            "--api-port-base",
            os.environ.get("AFD_NPU_DSPARK_E2E_API_PORT", "19280"),
            "--afd-port",
            os.environ.get("AFD_NPU_DSPARK_E2E_AFD_PORT", "6455"),
            "--startup-timeout",
            os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        ]
    )
    if scenario == GRAPH and os.environ.get("AFD_DSPARK_EAGER_REFERENCE"):
        command.extend(["--eager-reference", os.environ["AFD_DSPARK_EAGER_REFERENCE"]])
    return command


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    interface = required_env("HCCL_SOCKET_IFNAME")
    vendor = REPO_ROOT / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    env.update(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            "HCCL_BUFFSIZE": "256",
            "LCCL_BUFFER_SIZE": "408",
            "HCCL_CONNECT_TIMEOUT": "600",
            "HCCL_EXEC_TIMEOUT": "600",
            "GLOO_SOCKET_IFNAME": interface,
            "TP_SOCKET_IFNAME": interface,
            "OMP_NUM_THREADS": "4",
            "VLLM_HOST_IP": required_env("HCCL_IF_IP"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ASCEND_CUSTOM_OPP_PATH": str(vendor),
            "AFD_CUST_OPAPI_LIB_PATH": str(vendor / "op_api/lib/libcust_opapi.so"),
            "LD_LIBRARY_PATH": str(vendor / "op_api/lib")
            + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""),
        }
    )
    return env


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", CASE_IDS)
def test_dspark_core(scenario: str, tmp_path: Path) -> None:
    output = tmp_path / scenario
    run_case_entry(
        build_runner_command(output, scenario), output, env=build_environment()
    )
