# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local NPU DSV4 Flash cases over the synchronous CAMP2P boundary.

DeepSeek V4 does not fit on a single Attention or FFN die, so each host runs
its own recorded launch profile: the A5 case runs Attention DP2/TP1 and FFN
DP2/TP1 with expert parallelism and graph capture on four dies, without the
native DBO its launch script records, while the A3 case shards by tensor
parallel on eight. The synchronous connector needs no CAM vendor package: the
plugin's own a2e/e2a operators carry the activations and, for the DeepSeek V4
Hash layers, the token ids that the FFN-side gate routes with.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import run_case_entry
from tests.e2e.environment import devices_from_env, model_case_command, required_env
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_SCENARIOS,
    DSV4_SYNC_HCCL_BUFFER_SIZE_MB,
    DSV4_SYNC_LOCAL_AFD_HOST,
    DSV4_SYNC_SHAPES,
    DSV4SyncShape,
)


def _afd_host(shape: DSV4SyncShape) -> str:
    """Return the AFD rendezvous host the selected profile expects.

    The A3 profile takes the caller's advertised address, which its recorded
    launch script passes explicitly. The A5 script starts both roles on the
    loopback address and needs no NIC variable.
    """
    if shape.nic_env_required:
        return required_env("HCCL_IF_IP")
    return os.environ.get("HCCL_IF_IP") or DSV4_SYNC_LOCAL_AFD_HOST


def build_runner_command(scenario: str, output_path: Path) -> list[str]:
    shape = DSV4_SYNC_SHAPES[scenario]
    if required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSV4 sync CAMP2P E2E requires AFD_E2E_BACKEND=npu")
    devices_from_env("AFD_E2E_DEVICES", shape.device_count)
    command = model_case_command(
        "tests.e2e.models.deepseek_v4_flash.cases", scenario, output_path, backend="npu"
    )
    command.extend(
        [
            "--afd-host",
            _afd_host(shape),
            "--api-port-base",
            os.environ.get("AFD_NPU_DSV4_SYNC_E2E_API_PORT", "19380"),
            "--afd-port",
            os.environ.get("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", "6456"),
            "--startup-timeout",
            os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        ]
    )
    return command


def build_environment(scenario: str) -> dict[str, str]:
    """Runtime environment for the operator transport on a multi-NIC host.

    Every entry follows the selected host profile's recorded launch script. The
    A3 profile requires the caller's NIC and drops any inherited HCCL_BUFFSIZE,
    because CAMP2P sizes its own AFD HCCL domains. The A5 profile keeps the
    script's global buffer size, its plain allocator setting, and the platform's
    default multiprocessing start method; its NIC variables stay optional, and
    the socket names are only forwarded when the caller supplies one. No CAM
    vendor package is involved on either host, so no CAM vendor variable is
    installed.
    """
    shape = DSV4_SYNC_SHAPES[scenario]
    env = os.environ.copy()
    if shape.nic_env_required:
        interface = required_env("HCCL_SOCKET_IFNAME")
        required_env("HCCL_IF_IP")
    else:
        interface = os.environ.get("HCCL_SOCKET_IFNAME", "")
    if shape.keep_hccl_buffsize:
        env.setdefault("HCCL_BUFFSIZE", str(DSV4_SYNC_HCCL_BUFFER_SIZE_MB))
    else:
        env.pop("HCCL_BUFFSIZE", None)
    if shape.force_spawn:
        env.update(
            {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            }
        )
    else:
        env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        env.pop("AFD_FORCE_SPAWN_MULTIPROCESSING", None)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", shape.npu_alloc_conf)
    env.update(
        {
            "HCCL_CONNECT_TIMEOUT": "1800",
            "HCCL_EXEC_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "3600000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
        }
    )
    if interface:
        env.update(
            {
                "GLOO_SOCKET_IFNAME": interface,
                "TP_SOCKET_IFNAME": interface,
            }
        )
    return env


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", DSV4_SYNC_CAMP2P_SCENARIOS)
def test_deepseek_v4_flash_sync_camp2p(scenario: str, tmp_path: Path) -> None:
    output_path = tmp_path / scenario
    run_case_entry(
        build_runner_command(scenario, output_path),
        output_path,
        env=build_environment(scenario),
    )
