# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU smoke test for DeepSeek-V2-Lite with async CAM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tests.conftest import run_runner
from tests.e2e.environment import (
    devices_from_env as _devices,
)
from tests.e2e.environment import (
    required_env as _required_env,
)
from tests.e2e.runner import (
    ASYNC_CAM_ATTENTION_RANKS,
    ASYNC_CAM_FFN_RANKS,
    ASYNC_CAM_SCENARIO,
    ASYNC_UBATCH_ATTENTION_RANKS,
    ASYNC_UBATCH_FFN_RANKS,
    ASYNC_UBATCH_SCENARIO,
)

CAM_HCCL_BUFFER_SIZE_MB = 4096
CAM_MAX_NUM_SEQUENCES = "8"
CAM_MAX_BATCHED_TOKENS = "8000"
CAM_MEMORY_UTILIZATION = "0.75"


def _async_cam_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HCCL_BUFFSIZE", None)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    env.setdefault("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "1")
    return env


def _connector_extra_config(model: str) -> str:
    dynamic_quant = int(
        os.environ.get(
            "AFD_NPU_ASYNC_CAM_E2E_DYNAMIC_QUANT",
            "1" if (Path(model) / "quant_model_description.json").is_file() else "0",
        ),
    )
    return json.dumps(
        {
            "dynamicQuant": dynamic_quant,
            "hccl_buffer_size": CAM_HCCL_BUFFER_SIZE_MB,
        },
        separators=(",", ":"),
    )


def build_runner_command() -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "npu":
        raise RuntimeError("async CAM E2E requires AFD_E2E_BACKEND=npu")

    device_count = ASYNC_CAM_ATTENTION_RANKS + ASYNC_CAM_FFN_RANKS
    devices = _devices("AFD_E2E_DEVICES", device_count)
    model = _required_env("AFD_NPU_E2E_MODEL")
    common_arguments = (
        "--trust-remote-code",
        "--max-num-seqs",
        CAM_MAX_NUM_SEQUENCES,
        "--max-num-batched-tokens",
        CAM_MAX_BATCHED_TOKENS,
        "--gpu-memory-utilization",
        CAM_MEMORY_UTILIZATION,
        "--no-enable-prefix-caching",
    )
    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--attention-devices",
        ",".join(devices[:ASYNC_CAM_ATTENTION_RANKS]),
        "--ffn-devices",
        ",".join(devices[ASYNC_CAM_ATTENTION_RANKS:]),
        "--scenario",
        ASYNC_CAM_SCENARIO,
        "--served-model-name-prefix",
        "cam-async",
        "--afd-connector-extra-config",
        _connector_extra_config(model),
        "--api-port-base",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_API_PORT", "19080"),
        "--afd-port",
        os.environ.get("AFD_NPU_ASYNC_CAM_E2E_AFD_PORT", "6453"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        *(f"--common-vllm-arg={argument}" for argument in common_arguments),
    ]
    max_model_len = os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_MODEL_LEN")
    if max_model_len:
        command.extend(
            [
                "--common-vllm-arg=--max-model-len",
                f"--common-vllm-arg={max_model_len}",
            ],
        )
    return command


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
def test_deepseek_v2_lite_async_cam() -> None:
    run_runner(build_runner_command(), env=_async_cam_env())


def build_ubatch_runner_command(gsm8k_output_path: Path) -> list[str]:
    """Build the DP1/TP2 Attention async MoE ubatching runner command.

    The fixed scenario owns the token-split and batch-size configuration. This
    entrypoint supplies the devices, ports, and GSM8K output path.
    """
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "npu":
        raise RuntimeError("async ubatch E2E requires AFD_E2E_BACKEND=npu")

    device_count = ASYNC_UBATCH_ATTENTION_RANKS + ASYNC_UBATCH_FFN_RANKS
    devices = _devices("AFD_E2E_DEVICES", device_count)
    model = _required_env("AFD_NPU_E2E_MODEL")
    return [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        model,
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--attention-devices",
        ",".join(devices[:ASYNC_UBATCH_ATTENTION_RANKS]),
        "--ffn-devices",
        ",".join(devices[ASYNC_UBATCH_ATTENTION_RANKS:]),
        "--scenario",
        ASYNC_UBATCH_SCENARIO,
        "--served-model-name-prefix",
        "cam-async-ubatch",
        "--afd-connector-extra-config",
        json.dumps(
            {"hccl_buffer_size": CAM_HCCL_BUFFER_SIZE_MB},
            separators=(",", ":"),
        ),
        "--api-port-base",
        os.environ.get("AFD_NPU_ASYNC_UBATCH_E2E_API_PORT", "19180"),
        "--afd-port",
        os.environ.get("AFD_NPU_ASYNC_UBATCH_E2E_AFD_PORT", "6454"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
        "--gsm8k-output-path",
        str(gsm8k_output_path),
    ]


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
def test_deepseek_v2_lite_async_ubatch(tmp_path: Path) -> None:
    command = build_ubatch_runner_command(tmp_path / ASYNC_UBATCH_SCENARIO)
    run_runner(command, env=_async_cam_env())
