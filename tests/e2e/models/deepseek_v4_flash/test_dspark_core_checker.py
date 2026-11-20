# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host regressions for known DSpark acceptance failures."""

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.e2e.case import CaseEnvironment
from tests.e2e.models.deepseek_v4_flash import dspark_cases, dspark_eager


@pytest.mark.parametrize("mode,replays", [("eager", 0), ("full", 0)])
def test_graph_dispatch_requires_full_request_replay_on_every_ffn(mode, replays):
    before, after = {}, {}
    for rank in range(2, 6):
        name = f"rank-{rank}.json"
        state = {
            "rank": rank,
            "draft_wire_layer_ids": [43, 44, 45],
            "owns_routed_experts": True,
            "owns_attention": False,
            "owns_shared_experts": False,
            "ffn_execution": "layered",
            "executor_layer_ids": list(range(46)),
            "ffn_mode": mode if rank == 3 else "full",
            "graph_capture_count": 0 if mode == "eager" and rank == 3 else 1,
            "graph_replay_count": replays if rank == 3 else 1,
            "submitted_returns": {},
        }
        before[name] = {**state, "graph_replay_count": 0}
        after[name] = state
    with pytest.raises(AssertionError):
        dspark_eager.assert_dispatch(
            before, after, require_graph=True, require_layer_returns=False
        )
    after["rank-3.json"].update(
        ffn_mode="full", graph_capture_count=1, graph_replay_count=1
    )
    dspark_eager.assert_dispatch(
        before, after, require_graph=True, require_layer_returns=False
    )


@pytest.fixture
def service_observations(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    executable = runtime / "python"
    executable.write_text("interpreter a")
    modules = {}
    for name in (
        "afd_plugin",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "torch_npu",
        "transformers",
        "vllm",
        "vllm_ascend",
    ):
        package = runtime / name
        package.mkdir()
        origin = package / "__init__.py"
        origin.write_text(f"{name} content")
        modules[name] = str(origin)
    observation = {
        "executable": str(executable),
        "prefix": str(runtime),
        "base_prefix": str(runtime),
        "version": "test",
        "sys_path": [str(runtime)],
        "modules": modules,
    }

    def write_stage(stage, data):
        requests = stage / "requests"
        requests.mkdir(parents=True)
        (requests / "summary.json").write_text("[]")
        attention = [{"rank": rank, "runtime": data} for rank in range(2)]
        ffn = {
            f"rank-{rank}.json": {"rank": rank, "runtime": data} for rank in range(2, 6)
        }
        (requests / "attention.json").write_text(json.dumps({"results": attention}))
        (requests / "ffn-after.json").write_text(json.dumps(ffn))

    return observation, write_stage


def test_graph_reference_compares_worker_runtime(
    service_observations, monkeypatch, tmp_path
):
    observation, write_stage = service_observations
    reference_stage = tmp_path / "old-target"
    write_stage(reference_stage, observation)
    old_runtime = dspark_cases.service_runtime_identity(reference_stage)
    executable = tmp_path / "other-python"
    executable.write_text("interpreter b")
    new_observation = {**observation, "executable": str(executable)}
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "run.json").write_text(
        json.dumps(
            {
                "case_id": dspark_cases.EAGER,
                "acceptance_passed": True,
                "identity": {"same": True},
                "service_runtime": old_runtime,
                "stages": {"target": str(reference_stage)},
            }
        )
    )
    env = CaseEnvironment(
        model="/model",
        vllm_bin="same-wrapper",
        device_backend="npu",
        devices=tuple(map(str, range(6))),
        output_path=tmp_path / "graph",
    )
    monkeypatch.setattr(dspark_cases, "identity", lambda _env: {"same": True})

    def subrun(_case_id, env, name, **_kwargs):
        path = env.output_path / name
        write_stage(path, new_observation)
        return path

    monkeypatch.setattr(dspark_cases, "subrun", subrun)
    with pytest.raises(ValueError, match="service runtime"):
        dspark_cases.execute(dspark_cases.GRAPH, env, reference)


@pytest.fixture
def blocked_dspark_server():
    arrived = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    count = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal count
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result: dict[str, Any]
            if self.path == "/tokenize":
                result = {"tokens": list(range(600))}
            else:
                with lock:
                    count += 1
                    index = count
                    if index == 11:
                        arrived.set()
                if index > 7:
                    release.wait(timeout=30)
                prompt = payload["prompt"]
                batch = isinstance(prompt, list) and isinstance(prompt[0], list)
                result = {
                    "choices": [
                        {"token_ids": list(range(32))} for _ in range(2 if batch else 1)
                    ],
                    "usage": {},
                }
            body = json.dumps(result).encode()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", arrived, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("signum", [signal.SIGTERM])
def test_cancellation_reaches_cleanup_with_blocked_requests(
    blocked_dspark_server, tmp_path, signum
):  # long-test: real concurrent HTTP, child signals, and cleanup notification
    url, arrived, release = blocked_dspark_server
    script = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from tests.e2e.cancellation import cancellable_run
from tests.e2e.models.deepseek_v4_flash.dspark_eager import exercise
url, output, fd = sys.argv[1:]
args = SimpleNamespace(url=url, model='host', pd=False)
with cancellable_run(lambda: os.write(int(fd), b'cleanup')):
    exercise(args, url, Path(output), {}, False, '')
"""
    read_fd, write_fd = os.pipe()
    with (tmp_path / "child.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", script, url, str(tmp_path), str(write_fd)],
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
        os.close(write_fd)
        try:
            assert arrived.wait(timeout=15), "four concurrent requests did not arrive"
            process.send_signal(signum)
            ready, _, _ = select.select([read_fd], [], [], 5)
            assert ready and os.read(read_fd, 7) == b"cleanup", "cleanup did not start"
            process.wait(timeout=5)
            assert process.returncode == 128 + signum
        finally:
            release.set()
            if process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            os.close(read_fd)
