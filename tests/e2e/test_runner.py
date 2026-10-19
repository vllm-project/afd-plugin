# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Host-only harness integration: real sockets, processes and signals; no devices."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as entrypoint

CONCURRENT_REQUESTS = 10


CANCEL_SCRIPT = """
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


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cancel_blocked_requests_cleans_service_groups(
    monkeypatch, tmp_path, signum
):  # long-test: real HTTP cancellation requires owned subprocess setup and teardown
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", ",".join(map(str, range(16))))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/dsv4")
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
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
                if request_count == CONCURRENT_REQUESTS:
                    arrived.set()
            release.wait(timeout=30)

    server = StalledServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    pid_path = tmp_path / "services.json"
    command = entrypoint.build_runner_command(tmp_path / "responses.json")
    command.extend(["--api-port-base", str(server.server_port)])
    with (tmp_path / "cancel.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", CANCEL_SCRIPT, str(pid_path), *command[3:]],
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
            assert len(results) == CONCURRENT_REQUESTS
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
