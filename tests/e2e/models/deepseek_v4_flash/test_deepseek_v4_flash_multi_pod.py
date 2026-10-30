# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4 Flash async CAM multi-pod acceptance cases on Ascend NPU.

Like the DeepSeek-V2-Lite multi-pod cases, this runs as *one pod's slice* of
an already-provisioned deployment, invoked once inside every pod. The
deployment is the fixed 16-NPU DSV4 case split across two 8-NPU pods; every
launch, readiness, evaluation, and teardown decision is made by the in-pod
runner.

Each pod must export its own HCCL_IF_IP. The pods exchange addresses through
the rendezvous store, and a pod advertises HCCL_IF_IP there, so the async CAM
rendezvous and every DP placement flag name the interface HCCL binds to.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests.conftest import run_runner
from tests.e2e.environment import required_env
from tests.e2e.models.deepseek_v4_flash.config import DSV4_ASYNC_CAM_SCENARIO
from tests.e2e.models.deepseek_v4_flash.test_async_cam_npu import build_environment

# Attention TP groups of four stay within one pod, which validate_layout
# enforces; the layout only varies which pod leads each role.
POD_LAYOUTS = {
    "2pod-role-split": "8A0F,0A8F",
    "2pod-interleaved": "4A4F,4A4F",
}


def build_runner_command(layout_name: str) -> list[str]:
    """Build this pod's in-pod runner argv for one DSV4 layout."""
    if required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSV4 async CAM E2E requires AFD_E2E_BACKEND=npu")
    command = [
        sys.executable,
        "-m",
        "tests.e2e.multi_pod.runner",
        "--scenario",
        DSV4_ASYNC_CAM_SCENARIO,
        "--pod-layout",
        POD_LAYOUTS[layout_name],
        "--run-id",
        required_env("AFD_E2E_RUN_ID"),
        "--model",
        required_env("AFD_NPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--served-model-name-prefix",
        "dsv4-flash",
        "--api-port-base",
        os.environ.get("AFD_NPU_DSV4_E2E_API_PORT", "19280"),
        "--afd-port",
        os.environ.get("AFD_NPU_DSV4_E2E_AFD_PORT", "6455"),
        "--serving-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        "--completion-output-path",
        required_env("AFD_E2E_COMPLETION_OUTPUT"),
        "--store-host",
        required_env("AFD_E2E_STORE_HOST"),
    ]
    store_port = os.environ.get("AFD_E2E_STORE_PORT")
    if store_port:
        command.extend(["--store-port", store_port])
    for value in os.environ.get("AFD_E2E_POD_ENV", "").split(";"):
        if value.strip():
            command.extend(["--pod-env", value.strip()])
    return command


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("layout_name", list(POD_LAYOUTS))
def test_deepseek_v4_flash_multi_pod(layout_name: str) -> None:
    run_runner(build_runner_command(layout_name), env=build_environment())
