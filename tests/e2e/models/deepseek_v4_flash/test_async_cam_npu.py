# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local 16-NPU DSV4 Flash async CAM concurrent-request acceptance case."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import run_case_entry
from tests.e2e.environment import devices_from_env, model_case_command, required_env
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_ASYNC_CAM_SCENARIO,
    DSV4_ATTENTION_RANKS,
    DSV4_FFN_RANKS,
)


def build_runner_command(output_path: Path) -> list[str]:
    if required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSV4 async CAM E2E requires AFD_E2E_BACKEND=npu")
    devices_from_env("AFD_E2E_DEVICES", DSV4_ATTENTION_RANKS + DSV4_FFN_RANKS)
    command = model_case_command(
        "tests.e2e.models.deepseek_v4_flash.cases",
        DSV4_ASYNC_CAM_SCENARIO,
        output_path,
        backend="npu",
    )
    command.extend(
        [
            "--afd-host",
            required_env("HCCL_IF_IP"),
            "--api-port-base",
            os.environ.get("AFD_NPU_DSV4_E2E_API_PORT", "19280"),
            "--afd-port",
            os.environ.get("AFD_NPU_DSV4_E2E_AFD_PORT", "6455"),
            "--startup-timeout",
            os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        ]
    )
    return command


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    interface = required_env("HCCL_SOCKET_IFNAME")
    required_env("HCCL_IF_IP")
    env.update(
        {
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            "HCCL_BUFFSIZE": "4096",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_CONNECT_TIMEOUT": "1200",
            "HCCL_EXEC_TIMEOUT": "2000",
            "VLLM_RPC_TIMEOUT": "3600000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "TASK_QUEUE_ENABLE": "1",
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
            "AFD_CAM_OP_IO_LOG": "0",
            "AFD_ASYNC_MOE_LAYOUT_LOG": "0",
            "GLOO_SOCKET_IFNAME": interface,
            "TP_SOCKET_IFNAME": interface,
        }
    )
    return env


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", [DSV4_ASYNC_CAM_SCENARIO])
def test_deepseek_v4_flash_async_cam(scenario: str, tmp_path: Path) -> None:
    output_path = tmp_path / scenario
    run_case_entry(
        build_runner_command(output_path),
        output_path,
        env=build_environment(),
    )
