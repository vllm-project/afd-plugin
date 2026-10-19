# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the DSV4 async CAM and sync CAMP2P NPU E2E entrypoints."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import completions
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as async_entrypoint
from tests.e2e.models.deepseek_v4_flash import test_sync_camp2p_npu as sync_entrypoint
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_ASYNC_CAM_SCENARIO,
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    sync_shape,
)

ASYNC = DSV4_ASYNC_CAM_SCENARIO
A5 = DSV4_SYNC_CAMP2P_A5_SCENARIO
A3 = DSV4_SYNC_CAMP2P_A3_SCENARIO
SCENARIOS = (ASYNC, A5, A3)
# Each host runs a fixed device count split evenly between Attention and FFN.
DEVICE_COUNTS = {ASYNC: 16, A5: 4, A3: 16}
CONCURRENT_REQUESTS = 10
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


def _flag_value(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def _completion(content: str, finish_reason: str | None = "stop") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }


# scenario -> (connector, `--quantization` value, or None when omitted)
EXPECTED_DEPLOYMENT = {
    ASYNC: ("CAMAsyncAFDConnector", "ascend"),
    # The A5 launch script passes no `--quantization` at all.
    A5: ("CAMP2pAFDConnector", None),
    A3: ("CAMP2pAFDConnector", "ascend"),
}
ASYNC_ALL2ALL_BACKENDS = {
    "attention": "allgather_reducescatter",
    "ffn": "flashinfer_all2allv",
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixed_deployment(monkeypatch, tmp_path, scenario):
    args = _arguments(monkeypatch, tmp_path, scenario)
    is_async = scenario == ASYNC
    shape = sync_shape(scenario)
    use_graph = shape is not None and shape.use_graph
    verbatim = shape is not None and shape.verbatim_launch
    connector, quantization = EXPECTED_DEPLOYMENT[scenario]
    runner.configure_scenario(args)
    devices = {
        "attention": runner.parse_csv(args.attention_devices),
        "ffn": runner.parse_csv(args.ffn_devices),
    }
    runner.validate_topology(args, devices["attention"], devices["ffn"])
    ranks = {role: len(role_devices) for role, role_devices in devices.items()}
    assert ranks == dict.fromkeys(devices, DEVICE_COUNTS[scenario] // 2)
    # Only the async transport defers an FFN SIGKILL for a pending CAM receive.
    assert runner.uses_npu_async_process_cleanup(args) is is_async
    assert args.afd_async is args.compute_gate_on_attention is is_async
    assert args.gsm8k_output_path is None
    for role in devices:
        command = runner.build_vllm_command(args, role=role)
        dp = int(command[command.index("--data-parallel-size") + 1])
        tp = int(command[command.index("--tensor-parallel-size") + 1])
        assert dp * tp == ranks[role]
        assert _flag_value(command, "--quantization") == quantization
        assert "--enable-expert-parallel" in command
        assert "--enable-dbo" not in command
        assert "--kv-transfer-config" not in command
        assert ("--enforce-eager" in command) is not use_graph
        assert ("--seed" in command) is not verbatim
        assert _flag_value(command, "--all2all-backend") == (
            ASYNC_ALL2ALL_BACKENDS[role] if is_async else None
        )
        config = json.loads(command[command.index("--additional-config") + 1])
        assert config["enable_flashcomm1"] is (is_async and role == "attention")
        assert config["multistream_dsv4_dsa_overlap"] is False
        afd = config["afd"]
        assert (afd["role"], afd["connector"]) == (role, connector)
        assert afd["num_attention_ranks"] == ranks["attention"]
        assert afd["num_ffn_ranks"] == ranks["ffn"]
        # CAMP2P rejects both the asynchronous mode and gate-on-Attention.
        assert ("async" in afd) is ("compute_gate_on_attention" in afd) is is_async
        env = runner.build_env("0", args, role=role, e2e_run_id="test")
        assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        # Cleanup finds the service process trees through this marker.
        assert env[runner.E2E_RUN_ID_ENV] == "test"


def test_main_uses_concurrent_requests_and_longer_cleanup(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    cleanup_options = {}
    evaluations = []
    process = SimpleNamespace(pid=123, poll=lambda: None)
    fakes = {
        "parse_args": lambda: args,
        "start_process": lambda *_args: process,
        "stream_output": lambda *_args: SimpleNamespace(join=lambda **_kwargs: None),
        "wait_for_openai_api": lambda *_args: None,
        "run_concurrent_completion_evaluation": lambda _args: evaluations.append(1),
        "run_gsm8k_evaluation": lambda _args: pytest.fail("must not run GSM8K"),
        "terminate_processes": lambda _procs, **kwargs: cleanup_options.update(kwargs),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(runner, name, fake)
    assert runner.main() == 0
    assert evaluations == [1]
    assert cleanup_options["termination_timeout_s"] == 60
    kill_env = cleanup_options["force_kill_environment"]
    assert kill_env[runner.E2E_PROCESS_ROLE_ENV] == "ffn"


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
        assert _flag_value(command, "--afd-host") == afd_host


NIC = {"HCCL_IF_IP": AFD_HOST, "HCCL_SOCKET_IFNAME": "eth-test"}
A5_SCRIPT_ENV = {
    "HCCL_BUFFSIZE": "2048",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
}
FORWARDED_NIC = {"GLOO_SOCKET_IFNAME": "eth-test", "TP_SOCKET_IFNAME": "eth-test"}
SPAWN = ("VLLM_WORKER_MULTIPROC_METHOD", "AFD_FORCE_SPAWN_MULTIPROCESSING")


@pytest.mark.parametrize(
    ("scenario", "inherited", "expected"),
    [
        (ASYNC, NIC, {"HCCL_BUFFSIZE": "4096"} | FORWARDED_NIC),
        # A5 keeps its script's buffer size, allocator and start method, and the
        # NIC variables stay optional on a single host.
        (
            A5,
            {SPAWN[0]: "fork"},
            A5_SCRIPT_ENV | dict.fromkeys((*SPAWN, *FORWARDED_NIC)),
        ),
        # A caller who pins the buffer size or the NIC keeps them.
        (
            A5,
            {"HCCL_BUFFSIZE": "4096"} | NIC,
            {"HCCL_BUFFSIZE": "4096"} | FORWARDED_NIC,
        ),
    ],
)
def test_environment(monkeypatch, scenario, inherited, expected):
    """`expected` maps each checked variable to its value, None when unset."""
    for name in (*NIC, "HCCL_BUFFSIZE", "PYTORCH_NPU_ALLOC_CONF", SPAWN[1]):
        monkeypatch.delenv(name, raising=False)
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    env = (
        async_entrypoint.build_environment()
        if scenario == ASYNC
        else sync_entrypoint.build_environment(scenario)
    )
    assert {name: env.get(name) for name in expected} == expected
    assert env["AFD_FORCE_BALANCED_TOPK_IDS"] == "0"


@pytest.mark.parametrize(
    ("scenario", "check_answer"),
    # Both sync profiles are smoke cases; only the async case checks the answer.
    [(ASYNC, True), (A5, False), (A3, False)],
)
def test_oracle_uses_the_profile_answer_policy(
    monkeypatch, tmp_path, scenario, check_answer
):
    seen: dict = {}
    monkeypatch.setattr(runner, "evaluate_completions", seen.update)
    args = _arguments(monkeypatch, tmp_path, scenario)
    runner.configure_scenario(args)
    runner.run_concurrent_completion_evaluation(args)
    assert seen["check_answer"] is check_answer


@pytest.mark.parametrize(
    ("content", "finish_reason", "check_answer", "error"),
    [
        # A profile that skips the answer still requires a finished response.
        ("28", "stop", False, None),
        ("10 10:56:33 10:56:33", "length", False, None),
        ("I cannot compute that.", "stop", False, None),
        ("", "stop", False, "returned empty content"),
        ("28", None, False, "did not finish normally"),
        # A host that compares the answer keeps the exact terminal state.
        ("28", "length", True, "did not finish normally"),
    ],
)
def test_served_response_shape_checks(content, finish_reason, check_answer, error):
    response = _completion(content, finish_reason)
    if error is None:
        completions.validate_response(response, check_answer=check_answer)
    else:
        with pytest.raises(RuntimeError, match=error):
            completions.validate_response(response, check_answer=check_answer)


# failure -> `httpx.Response` arguments replacing the last request's reply.
FAILED_REPLIES: dict[str, dict[str, Any]] = {
    "empty": {"json": _completion("")},
    "wrong-answer": {"json": _completion("999")},
    "truncated": {"json": _completion("28", "length")},
    "choices": {"json": {"choices": []}},
    "http": {"status_code": 500, "text": "server error"},
    "json": {"text": "not-json"},
}


@pytest.mark.parametrize("failure", [None, *FAILED_REPLIES, "timeout"])
def test_ten_requests_overlap_and_validate_every_response(
    monkeypatch, tmp_path, failure
):
    args = _arguments(monkeypatch, tmp_path)
    runner.configure_scenario(args)
    # All ten must arrive before any response is released. Sequential clients
    # cannot pass even if they merely claim concurrency in metadata.
    arrived = asyncio.Event()
    request_count = 0

    async def respond(request):
        nonlocal request_count
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"thinking": False}
        operand = int(payload["messages"][0]["content"].split()[1])
        request_count += 1
        if request_count == CONCURRENT_REQUESTS:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), timeout=5)
        if operand != 21 or failure is None:
            return httpx.Response(200, json=_completion(str(operand + 7)))
        if failure == "timeout":
            raise httpx.ReadTimeout("server stalled", request=request)
        return httpx.Response(**{"status_code": 200, **FAILED_REPLIES[failure]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        completions.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    if failure:
        with pytest.raises(RuntimeError, match="Request 9"):
            runner.run_concurrent_completion_evaluation(args)
    else:
        runner.run_concurrent_completion_evaluation(args)
    results = json.loads((tmp_path / "responses.json").read_text())
    assert len(results) == CONCURRENT_REQUESTS
    successful = results[:-1] if failure else results
    assert all("error" not in row for row in successful)
    assert [
        row["response"]["choices"][0]["message"]["content"] for row in successful
    ] == [str(value) for value in range(19, 19 + len(successful))]
    if failure:
        assert results[-1]["error"]
    if failure in ("http", "json"):
        assert results[-1]["response_body"] == FAILED_REPLIES[failure]["text"]
