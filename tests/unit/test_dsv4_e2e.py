# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import completions
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as entrypoint


def _arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", ",".join(map(str, range(16))))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/dsv4")
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    command = entrypoint.build_runner_command(tmp_path / "responses.json")
    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    return runner.parse_args()


def test_dsv4_fixed_deployment_and_cleanup(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    runner.configure_scenario(args)
    runner.validate_topology(
        args,
        runner.parse_csv(args.attention_devices),
        runner.parse_csv(args.ffn_devices),
    )
    assert runner.uses_npu_async_process_cleanup(args)
    assert args.gsm8k_output_path is None
    for role, dp, tp in (("attention", "2", "4"), ("ffn", "8", "1")):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == dp
        assert command[command.index("--tensor-parallel-size") + 1] == tp
        assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
        assert command[command.index("--max-model-len") + 1] == "1048576"
        # The asynchronous 910C case keeps the Ascend quantization method; only
        # the synchronous case resolves it from the checkpoint.
        assert command[command.index("--quantization") + 1] == "ascend"
        assert "--enforce-eager" in command
        assert "--enable-expert-parallel" in command
        assert "--enable-dbo" not in command
        assert "--kv-transfer-config" not in command
        config = json.loads(command[command.index("--additional-config") + 1])
        assert config["enable_dsv4_shared_compressor_workspace"] is False
        assert config["enable_cpu_binding"] is True
        assert config["afd"] == {
            "role": role,
            "connector": "CAMAsyncAFDConnector",
            "host": "192.0.2.1",
            "port": 6455,
            "num_attention_ranks": 8,
            "num_ffn_ranks": 8,
            "async": True,
            "compute_gate_on_attention": True,
            "connector_extra_config": {
                "dynamicQuant": 1,
                "attn_ranks_per_dp": 4,
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": 2,
                "async_moe_split": "token",
            },
        }
        env = runner.build_env("0", args, role=role, e2e_run_id="test")
        assert env["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == (
            "1" if role == "attention" else "0"
        )
        assert env[runner.E2E_RUN_ID_ENV] == "test"


def test_dsv4_main_uses_concurrent_requests_and_longer_cleanup(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    cleanup_options = {}
    evaluations = []
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "start_process", lambda *_args: process)
    monkeypatch.setattr(
        runner,
        "stream_output",
        lambda *_args: SimpleNamespace(join=lambda **_kwargs: None),
    )
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "run_concurrent_completion_evaluation",
        lambda _args: evaluations.append("concurrent"),
    )
    monkeypatch.setattr(
        runner,
        "run_gsm8k_evaluation",
        lambda _args: pytest.fail("must not run GSM8K"),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **kwargs: cleanup_options.update(kwargs),
    )
    assert runner.main() == 0
    assert evaluations == ["concurrent"]
    assert cleanup_options["termination_timeout_s"] == 60
    assert (
        cleanup_options["force_kill_environment"][runner.E2E_PROCESS_ROLE_ENV] == "ffn"
    )


@pytest.mark.parametrize("devices", ["0,1,2,3", ",".join(["0"] * 16)])
def test_dsv4_entrypoint_rejects_wrong_devices(monkeypatch, tmp_path, devices):
    _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv("AFD_E2E_DEVICES", devices)
    with pytest.raises(RuntimeError, match="exactly 16 devices|devices must be unique"):
        entrypoint.build_runner_command(tmp_path / "responses.json")


@pytest.mark.parametrize(
    "field", ["common_vllm_arg", "attention_vllm_arg", "ffn_vllm_arg"]
)
def test_dsv4_rejects_deployment_overrides(monkeypatch, tmp_path, field):
    args = _arguments(monkeypatch, tmp_path)
    setattr(args, field, ["--max-num-batched-tokens=1"])
    with pytest.raises(ValueError, match="extra vLLM arguments"):
        runner.configure_scenario(args)


def test_dsv4_rejects_gpu(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path)
    args.device_backend = "gpu"
    runner.configure_scenario(args)
    with pytest.raises(ValueError, match="require NPU"):
        runner.validate_topology(
            args, list(map(str, range(8))), list(map(str, range(8, 16)))
        )


def test_dsv4_environment_uses_source_ops_and_preserves_network(monkeypatch):
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    env = entrypoint.build_environment()
    assert env["HCCL_BUFFSIZE"] == "4096"
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"
    assert env["AFD_FORCE_BALANCED_TOPK_IDS"] == "0"


@pytest.mark.parametrize(
    "failure",
    [None, "empty", "truncated", "choices", "http", "json", "timeout", "wrong-answer"],
)
def test_ten_requests_overlap_and_validate_every_response(
    monkeypatch,
    tmp_path,
    failure,
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
        if request_count == 10:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), timeout=5)
        result: dict = {
            "choices": [
                {
                    "message": {"content": str(operand + 7)},
                    "finish_reason": "stop",
                }
            ]
        }
        if operand == 21:
            if failure == "wrong-answer":
                result["choices"][0]["message"]["content"] = "999"
            elif failure == "empty":
                result["choices"][0]["message"]["content"] = ""
            elif failure == "truncated":
                result["choices"][0]["finish_reason"] = "length"
            elif failure == "choices":
                result["choices"] = []
            elif failure == "http":
                return httpx.Response(500, text="server error")
            elif failure == "json":
                return httpx.Response(200, text="not-json")
            elif failure == "timeout":
                raise httpx.ReadTimeout("server stalled", request=request)
        return httpx.Response(200, json=result)

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
    assert len(results) == 10
    successful = results[:-1] if failure else results
    assert all("error" not in row for row in successful)
    assert [
        row["response"]["choices"][0]["message"]["content"] for row in successful
    ] == [str(value) for value in range(19, 19 + len(successful))]
    assert all(row["finished_at"] >= row["started_at"] for row in results)
    if failure:
        assert results[-1]["error"]
    if failure in ("http", "json"):
        assert results[-1]["response_body"] == (
            "server error" if failure == "http" else "not-json"
        )


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cancel_blocked_requests_cleans_service_groups(monkeypatch, tmp_path, signum):
    _arguments(monkeypatch, tmp_path)
    arrived = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    request_count = 0

    class StalledServer(ThreadingHTTPServer):
        request_queue_size = 32

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                request_count += 1
                if request_count == 10:
                    arrived.set()
            release.wait(timeout=30)

    server = StalledServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    script = """
import json
import sys
from pathlib import Path
from tests.e2e import runner

pid_path = Path(sys.argv.pop(1))
pids = []
start = runner.start_process

def start_service(name, command, env):
    process = start(name, [sys.executable, '-c', 'import time; time.sleep(60)'], env)
    pids.append(process.pid)
    pid_path.write_text(json.dumps(pids))
    return process

runner.start_process = start_service
runner.wait_for_openai_api = lambda *args: None
sys.exit(runner.main())
"""
    pid_path = tmp_path / "services.json"
    command = entrypoint.build_runner_command(tmp_path / "responses.json")
    command.extend(["--api-port-base", str(server.server_port)])
    with (tmp_path / "cancel.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(pid_path), *command[3:]],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            assert arrived.wait(timeout=15), "ten requests never reached the server"
            process.send_signal(signum)
            # Much shorter than both the HTTP timeout (300s) and the outer
            # runner cleanup deadline (240s). Exercise real signal unwinding,
            # socket cancellation and separately launched service groups.
            process.wait(timeout=10)
            log.seek(0)
            assert process.returncode == 128 + signum, log.read()
            for pid in json.loads(pid_path.read_text()):
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            results = json.loads((tmp_path / "responses.json").read_text())
            assert len(results) == 10
            assert all("cancelled" in row["error"] for row in results)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if pid_path.exists():
                for pid in json.loads(pid_path.read_text()):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
            release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
