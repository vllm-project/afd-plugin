# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the DSV4 async CAM and sync CAMP2P NPU E2E entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as async_entrypoint
from tests.e2e.models.deepseek_v4_flash import test_sync_camp2p_npu as sync_entrypoint
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_ASYNC_CAM_SCENARIO,
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
)

ASYNC = DSV4_ASYNC_CAM_SCENARIO
A5 = DSV4_SYNC_CAMP2P_A5_SCENARIO
A3 = DSV4_SYNC_CAMP2P_A3_SCENARIO
SCENARIOS = (ASYNC, A5, A3)
# Each host runs a fixed device count split evenly between Attention and FFN.
DEVICE_COUNTS = {ASYNC: 16, A5: 4, A3: 16}
AFD_HOST = "192.0.2.1"


def _build_command(scenario: str, output_path: Path) -> list[str]:
    if scenario == ASYNC:
        return async_entrypoint.build_runner_command(output_path)
    return sync_entrypoint.build_runner_command(scenario, output_path)


def _arguments(monkeypatch, tmp_path, scenario: str = ASYNC):
    """Parse the runner arguments the entrypoint builds for `scenario`."""
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    devices = ",".join(map(str, range(DEVICE_COUNTS[scenario])))
    monkeypatch.setenv("AFD_E2E_DEVICES", devices)
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/dsv4")
    monkeypatch.setenv("HCCL_IF_IP", AFD_HOST)
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_API_PORT", raising=False)
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", raising=False)
    command = _build_command(scenario, tmp_path / "responses.json")
    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    return runner.parse_args()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_entrypoint_rejects_wrong_devices(monkeypatch, tmp_path, scenario):
    count = DEVICE_COUNTS[scenario]
    for devices in (range(count - 1), range(count + 1), [0] * count):
        _arguments(monkeypatch, tmp_path, scenario)
        monkeypatch.setenv("AFD_E2E_DEVICES", ",".join(map(str, devices)))
        with pytest.raises(RuntimeError, match=rf"exactly {count} devices|unique"):
            _build_command(scenario, tmp_path / "responses.json")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_rejects_deployment_overrides(monkeypatch, tmp_path, scenario):
    for field, value, error in (
        ("common_vllm_arg", ["--max-num-batched-tokens=1"], "extra vLLM arguments"),
        ("attention_vllm_arg", ["--max-num-batched-tokens=1"], "extra vLLM arguments"),
        ("ffn_vllm_arg", ["--max-num-batched-tokens=1"], "extra vLLM arguments"),
        ("device_backend", "gpu", "DSV4 scenarios require NPU"),
    ):
        args = _arguments(monkeypatch, tmp_path, scenario)
        setattr(args, field, value)
        with pytest.raises(ValueError, match=error):
            runner.configure_scenario(args)
            runner.validate_topology(
                args,
                runner.parse_csv(args.attention_devices),
                runner.parse_csv(args.ffn_devices),
            )


# The A5 script announces the loopback address and needs no NIC variable; the
# other hosts require the caller's advertised rendezvous address.
@pytest.mark.parametrize(
    ("scenario", "afd_host"), [(ASYNC, None), (A5, "127.0.0.1"), (A3, None)]
)
def test_afd_host_without_caller_address(monkeypatch, tmp_path, scenario, afd_host):
    _arguments(monkeypatch, tmp_path, scenario)
    monkeypatch.delenv("HCCL_IF_IP")
    if afd_host is None:
        with pytest.raises(RuntimeError, match="HCCL_IF_IP"):
            _build_command(scenario, tmp_path / "responses.json")
    else:
        command = _build_command(scenario, tmp_path / "responses.json")
        assert command[command.index("--afd-host") + 1] == afd_host
