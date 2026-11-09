# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Host-only harness integration: real sockets, processes and signals; no devices."""

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.e2e import runner
from tests.e2e.case import (
    CLEANUP_PROFILE_STRICT,
    CaseEnvironment,
    DboConfig,
    Mrv2Check,
    Phase,
)
from tests.e2e.models.deepseek_v2_lite import cases as v2_cases
from tests.e2e.models.deepseek_v4_flash import cases as dsv4_cases
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as entrypoint
from tests.e2e.models.deepseek_v4_flash.completions import CompletionTask

CONCURRENT_REQUESTS = 10


CANCEL_SCRIPT = """
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tests.e2e import runner
from tests.e2e.deployment import ServiceDeployment
from tests.e2e.models.deepseek_v4_flash.cases import build_case, NPU_ASYNC_IDS

pid_path = Path(sys.argv.pop(1))

def start_services(self):
    for role in self.resolved.launch_order:
        process = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            start_new_session=True,
        )
        self.processes.append(process)
        self.processes_by_role[role] = process
    pid_path.write_text(json.dumps([p.pid for p in self.processes]))

def host_case(case_id, environment):
    case = build_case(case_id, environment)
    return replace(case, id="host-completion-cancellation", deployment=replace(
        case.deployment, cleanup_profile="strict", termination_timeout_s=1,
    ))

# Isolate only device-service startup; execute the real task, signals and
# process cleanup. This host-only run does not qualify an NPU scenario.
ServiceDeployment.start = start_services
sys.exit(runner.case_main(host_case, NPU_ASYNC_IDS))
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
            results = json.loads(
                (
                    tmp_path / "responses.json" / "evaluation" / "completions.json"
                ).read_text()
            )
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


HOST_SERVICE = r"""
import json
import os
import re
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

arguments = sys.argv[1:]
model = arguments[arguments.index("--served-model-name") + 1]
with open(os.environ["AFD_HOST_PIDS"], "a") as log:
    log.write(str(os.getpid()) + "\n")
Path(os.environ["AFD_HOST_PIDS"]).with_name(f"{model}.json").write_text(json.dumps({
    "arguments": arguments,
    "logging_level": os.environ.get("VLLM_LOGGING_LEVEL"),
    "worker_shutdown_timeout": os.environ.get("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"),
}))
if (
    (os.environ.get("AFD_HOST_OUTCOME") == "startup-failure"
     and model.endswith("attention"))
    or (os.environ.get("AFD_HOST_OUTCOME") == "peer-startup-failure"
        and os.environ.get("AFD_E2E_POD_INDEX") == "1"
        and model.endswith("ffn"))
):
    raise SystemExit(3)
if os.environ.get("AFD_HOST_OUTCOME") == "cleanup-failure":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "--headless" in arguments or model.endswith("-ffn"):
    # Connector-driven FFN processes have no HTTP API.
    signal.pause()
    raise SystemExit(0)
port = int(arguments[arguments.index("--port") + 1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": []}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/collective_rpc":
            # Host services expose no device execution evidence.
            self.send_error(503, "host fixture provides no device evidence")
            return
        prompt = body["messages"][0]["content"]
        operands = re.search(r"Compute (\d+) \+ (\d+)", prompt)
        repeated = re.search(r"repeated (\d+) times", prompt)
        count = int(repeated[1]) if repeated else 1
        answer = str(int(operands[1]) + int(operands[2]))
        if os.environ.get("AFD_HOST_OUTCOME") == "wrong-answer":
            answer = "incorrect"
        choice = {"message": {"content": " ".join([answer] * count)},
                  "finish_reason": "stop"}
        response = {"choices": [choice]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""

HOST_EVALUATOR = r"""
import argparse
import json
import os
import signal
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--model_args")
parser.add_argument("--tasks")
parser.add_argument("--output_path")
parser.add_argument("--log_samples", action="store_true")
parser.add_argument("--num_fewshot", type=int)
parser.add_argument("--limit", type=int, default=1319)
args = parser.parse_args()
outcome = os.environ.get("AFD_HOST_OUTCOME")
if outcome == "cancel":
    print("AFD_HOST_EVALUATOR_WAITING", flush=True)
    signal.pause()
if outcome == "evaluator-error":
    raise SystemExit(7)
output = Path(args.output_path)
output.mkdir(parents=True, exist_ok=True)
result = {
    "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
    "n-samples": {"gsm8k": {"effective": args.limit - (outcome == "partial-result")}},
}
(output / "results.json").write_text(json.dumps(result))
(output / "invocation.json").write_text(json.dumps(vars(args)))
print(json.dumps(result), flush=True)
"""


@pytest.fixture
def host_case_runtime(monkeypatch, tmp_path):
    """Real child-process CLI boundaries; this fixture supplies no device evidence."""
    executable = tmp_path / "host-vllm"
    executable.write_text(f"#!{sys.executable}\n" + HOST_SERVICE)
    executable.chmod(0o700)
    package = tmp_path / "stubs" / "lm_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(HOST_EVALUATOR)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(package.parent), os.getcwd()))
    )
    monkeypatch.setenv("AFD_HOST_PIDS", str(tmp_path / "service-pids"))
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    monkeypatch.delenv("AFD_GSM8K_THRESHOLD", raising=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return CaseEnvironment(
        model="/models/host-test",
        vllm_bin=str(executable),
        device_backend="gpu",
        devices=("0", "1", "2", "3"),
        output_path=tmp_path / "result",
        api_port_base=port,
        startup_timeout=10,
    )


@pytest.mark.parametrize(
    "worker_override",
    [None, "75"],
    ids=["default-worker-timeout", "explicit-worker-timeout"],
)
def test_async_cam_shutdown_settings_reach_services_and_workers(
    monkeypatch, host_case_runtime, worker_override
):
    if worker_override is None:
        monkeypatch.delenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", worker_override)
    environment = replace(
        host_case_runtime,
        device_backend="npu",
        devices=tuple(str(index) for index in range(16)),
    )
    case = dsv4_cases.build_case(dsv4_cases.NPU_ASYNC_IDS[0], environment)
    # Host stubs never daemonize. Exercise owned-group cleanup here, not the
    # real CAM orphan/pidfd teardown, and keep this result out of device coverage.
    case = replace(
        case,
        id="host-async-shutdown",
        deployment=replace(case.deployment, cleanup_profile=CLEANUP_PROFILE_STRICT),
    )
    record = runner.run_case(case, environment)
    assert record["acceptance_passed"] is True
    assert record["cleanup"]["status"] == "passed"
    for role in ("attention", "ffn"):
        invocation = json.loads(
            (
                environment.output_path.parent
                / f"{case.served_model_name_prefix}-{role}.json"
            ).read_text()
        )
        arguments = invocation["arguments"]
        assert arguments[arguments.index("--shutdown-timeout") + 1] == "90"
        assert invocation["worker_shutdown_timeout"] == (worker_override or "60")
    assert record["canonical"]["deployment"]["worker_shutdown_timeout_s"] == 60
    assert record["effective"]["deployment"]["worker_shutdown_timeout_s"] == int(
        worker_override or "60"
    )
    assert record["effective"]["deployment"]["termination_timeout_s"] == 120
    if worker_override is not None:
        assert (
            record["overrides"]["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"]
            == worker_override
        )
    for service in record["processes"]:
        with pytest.raises(ProcessLookupError):
            os.kill(service["pid"], 0)


@pytest.mark.parametrize(
    "default_dbo,attention_dbo,check_roles,expected",
    [
        pytest.param(
            None,
            DboConfig(),
            ("attention",),
            {"attention": (None, None), "ffn": None},
            id="attention-only-native-defaults",
        ),
        pytest.param(
            DboConfig(2, 8),
            DboConfig(4, 16),
            ("attention", "ffn"),
            {"attention": (4, 16), "ffn": (2, 8)},
            id="role-override-and-inherited-default",
        ),
        pytest.param(None, DboConfig(), ("ffn",), None, id="reject-disabled-role"),
    ],
)
def test_role_dbo_reaches_service_cli_before_requiring_device_evidence(
    monkeypatch, host_case_runtime, default_dbo, attention_dbo, check_roles, expected
):  # long-test: exercise configuration through launched service CLIs and the probe RPC
    environment = host_case_runtime
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    source = v2_cases.build_case("afd-v2-eager-dp2", environment)
    case = replace(
        source,
        deployment=replace(
            source.deployment,
            dbo=default_dbo,
            attention=replace(source.deployment.attention, dbo=attention_dbo),
        ),
        phases=(
            Phase(
                "evaluation", source.phases[0].task, (Mrv2Check(check_roles, dbo=True),)
            ),
        ),
    )
    if expected is None:
        with pytest.raises(ValueError, match="DBO evidence requested"):
            runner.run_case(case, environment)
        assert not (environment.output_path.parent / "service-pids").exists()
        return

    # The real probe RPC must be reached; no host fixture can qualify DBO.
    with pytest.raises(urllib.error.HTTPError, match="503"):
        runner.run_case(case, environment)
    record = json.loads((environment.output_path / "run.json").read_text())
    assert record["acceptance_passed"] is False
    assert record["cleanup"]["status"] == "passed"
    for role, thresholds in expected.items():
        invocation = json.loads(
            (
                environment.output_path.parent
                / f"{case.served_model_name_prefix}-{role}.json"
            ).read_text()
        )
        arguments = invocation["arguments"]
        assert ("--enable-dbo" in arguments) == (thresholds is not None)
        if thresholds is not None:
            assert (
                tuple(
                    int(arguments[arguments.index(flag) + 1])
                    if flag in arguments
                    else None
                    for flag in (
                        "--dbo-decode-token-threshold",
                        "--dbo-prefill-token-threshold",
                    )
                )
                == thresholds
            )
        assert invocation["logging_level"] == (
            "DEBUG" if thresholds is not None else "INFO"
        )
    for service in record["processes"]:
        with pytest.raises(ProcessLookupError):
            os.kill(service["pid"], 0)


@pytest.mark.parametrize(
    "outcome,limit,threshold,error,eligible",
    [
        ("pass", None, None, None, True),
        ("debug-samples", "3", None, None, False),
        ("debug-threshold", None, "0.1", None, False),
        ("expanded", "28", None, None, True),
        ("partial-result", None, None, "expected 7", True),
        ("evaluator-error", None, None, "exited with code 7", True),
        ("cleanup-failure", None, None, "forced SIGKILL", True),
        ("startup-failure", None, None, "exited with 3", True),
    ],
)
def test_case_runs_public_clis_and_records_qualification(
    monkeypatch,
    host_case_runtime,
    outcome,
    limit,
    threshold,
    error,
    eligible,
):  # long-test: integrates service launch, evaluator CLI, failure and resource cleanup
    environment = host_case_runtime
    monkeypatch.setenv("AFD_HOST_OUTCOME", outcome)
    if limit is not None:
        monkeypatch.setenv("AFD_GSM8K_LIMIT", limit)
    if threshold is not None:
        monkeypatch.setenv("AFD_GSM8K_THRESHOLD", threshold)
    source_id = "afd-v2-eager-dp2" if outcome == "startup-failure" else "baseline-graph"
    source = v2_cases.build_case(source_id, environment)
    case = replace(
        source,
        id="host-harness",
        deployment=replace(source.deployment, termination_timeout_s=0.5),
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            runner.run_case(case, environment)
    else:
        runner.run_case(case, environment)

    record = json.loads((environment.output_path / "run.json").read_text())
    assert record["configuration"]["eligible"] is eligible
    assert record["execution"]["status"] == ("failed" if error else "passed")
    assert record["acceptance_passed"] == (eligible and error is None)
    assert record["cleanup"]["status"] == (
        "failed" if outcome == "cleanup-failure" else "passed"
    )
    for service in record["processes"]:
        with pytest.raises(ProcessLookupError):
            os.kill(service["pid"], 0)
    if error is None:
        invocation = json.loads(
            (environment.output_path / "evaluation" / "invocation.json").read_text()
        )
        settings = record["effective"]["phases"][0]["task"]["settings"]
        assert invocation["limit"] == (int(limit) if limit else 7)
        assert invocation["num_fewshot"] == 8
        assert "max_gen_toks=512" in invocation["model_args"]
        assert settings["threshold"] == (
            float(threshold) if threshold else (0.27 if limit == "28" else 0.25)
        )


def test_case_comparison_debug_preserves_floor_without_qualifying(
    monkeypatch, host_case_runtime
):
    """Exercise the published evaluator CLI with the existing debug policy."""
    monkeypatch.setenv("AFD_GSM8K_LIMIT", "7")
    environment = host_case_runtime
    case = replace(
        v2_cases.build_case("afd-v2-eager-dp2", environment),
        id="host-comparison-debug",
    )
    record = runner.run_case(case, environment)
    canonical = record["canonical"]["phases"][0]["task"]["settings"]
    effective = record["effective"]["phases"][0]["task"]["settings"]
    assert (canonical["sample_limit"], canonical["threshold"]) == (128, 0.27)
    assert (effective["sample_limit"], effective["threshold"]) == (24, 0.25)
    assert record["execution"]["status"] == "passed"
    assert record["configuration"]["eligible"] is False
    assert record["acceptance_passed"] is False
    assert record["cleanup"]["status"] == "passed"
    invocation = json.loads(
        (environment.output_path / "evaluation" / "invocation.json").read_text()
    )
    assert invocation["limit"] == 24
    assert "num_concurrent=12" in invocation["model_args"]
    for service in record["processes"]:
        with pytest.raises(ProcessLookupError):
            os.kill(service["pid"], 0)


CANCEL_CASE_SCRIPT = """
import os
from dataclasses import replace
from pathlib import Path
from tests.e2e.case import CaseEnvironment
from tests.e2e.models.deepseek_v2_lite.cases import build_case
from tests.e2e.runner import run_case

env = CaseEnvironment(
    model="/models/host-test", vllm_bin=os.environ["AFD_HOST_VLLM"],
    device_backend="gpu", devices=("0", "1", "2", "3"),
    output_path=Path(os.environ["AFD_HOST_OUTPUT"]),
    api_port_base=int(os.environ["AFD_HOST_PORT"]), startup_timeout=10,
)
case = replace(build_case("baseline-graph", env), id="host-harness")
run_case(case, env)
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_case_cancellation_reaps_evaluator_and_service(
    monkeypatch,
    host_case_runtime,
    signum,
):  # long-test: cancellation crosses evaluator, runner and owned service process groups
    environment = host_case_runtime
    monkeypatch.setenv("AFD_HOST_OUTCOME", "cancel")
    monkeypatch.setenv("AFD_HOST_VLLM", environment.vllm_bin)
    monkeypatch.setenv("AFD_HOST_OUTPUT", str(environment.output_path))
    monkeypatch.setenv("AFD_HOST_PORT", str(environment.api_port_base))
    waiting = threading.Event()
    process = subprocess.Popen(
        [sys.executable, "-c", CANCEL_CASE_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines = []

    def read_output():
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if "AFD_HOST_EVALUATOR_WAITING" in line:
                waiting.set()

    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()
    try:
        assert waiting.wait(timeout=15), "".join(lines)
        process.send_signal(signum)
        process.wait(timeout=15)
        thread.join(timeout=5)
        assert process.returncode == 128 + signum, "".join(lines)
        record = json.loads((environment.output_path / "run.json").read_text())
        assert record["execution"]["status"] == "cancelled"
        assert record["cleanup"]["status"] == "passed"
        assert record["acceptance_passed"] is False
        for service in record["processes"]:
            with pytest.raises(ProcessLookupError):
                os.kill(service["pid"], 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        pid_path = environment.output_path.parent / "service-pids"
        if pid_path.exists():
            for pid in pid_path.read_text().splitlines():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(int(pid), signal.SIGKILL)


@pytest.mark.parametrize("outcome", ["pass", "wrong-answer"])
def test_case_sequences_model_tasks_and_gsm8k(
    monkeypatch, host_case_runtime, outcome
):  # long-test: real concurrent HTTP task, evaluator subprocess and cleanup
    environment = host_case_runtime
    monkeypatch.setenv("AFD_HOST_OUTCOME", outcome)
    source = v2_cases.build_case("baseline-graph", environment)
    case = replace(
        source,
        id="host-model-tasks",
        phases=(
            Phase("first", CompletionTask(answer_repetitions=8)),
            Phase(
                "second", CompletionTask(first_request_index=10, answer_repetitions=8)
            ),
            replace(source.phases[0], name="gsm8k"),
        ),
    )
    if outcome == "wrong-answer":
        with pytest.raises(RuntimeError, match="wrong answer"):
            runner.run_case(case, environment)
    else:
        runner.run_case(case, environment)
    record = json.loads((environment.output_path / "run.json").read_text())
    assert record["cleanup"]["status"] == "passed"
    assert record["acceptance_passed"] is (outcome == "pass")
    if outcome == "pass":
        assert [phase["name"] for phase in record["phases"]] == [
            "first",
            "second",
            "gsm8k",
        ]
        for name, start in (("first", 0), ("second", 10)):
            results = json.loads(
                (environment.output_path / name / "completions.json").read_text()
            )
            assert [item["index"] for item in results] == list(range(start, start + 10))
            assert all("error" not in item for item in results)
        assert record["phases"][-1]["result"]["sample_count"] == 7
    else:
        assert [phase["name"] for phase in record["phases"]] == ["first"]
        assert not (environment.output_path / "second").exists()
        assert not (environment.output_path / "gsm8k").exists()


@pytest.fixture
def multi_pod_runtime(host_case_runtime, tmp_path):
    """Real TCPStore peers and owned service/evaluator CLIs; no device simulation."""

    def run(model, scenario, layout, outcome):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            store_port = listener.getsockname()[1]
        environment = host_case_runtime
        children = []
        threads = []
        lines: list[list[str]] = [[], []]
        waiting = threading.Event()
        other_run = None
        if model == "deepseek_v2_lite" and layout == "2A0F,0A2F" and outcome == "pass":
            # A different marker is not proof of a stale process or ownership
            # of this deployment's devices/ports on a shared host.
            other_run = subprocess.Popen(
                [sys.executable, "-c", "import signal; signal.pause()"],
                env={**os.environ, "AFD_E2E_RUN_ID": "host-independent-run"},
                start_new_session=True,
            )

        def read_output(index, child):
            for line in child.stdout:
                lines[index].append(line)
                if "AFD_HOST_EVALUATOR_WAITING" in line:
                    waiting.set()

        try:
            for index in range(2):
                child_env = os.environ.copy()
                child_env.update(
                    AFD_E2E_POD_INDEX=str(index),
                    AFD_HOST_OUTCOME=outcome,
                    AFD_HOST_PIDS=str(tmp_path / f"pod-{index}-pids"),
                )
                # Only the evaluator receives a weakened override. Its debug
                # verdict and effective workload must reach the other pod too.
                if outcome == "debug" and index == 0:
                    child_env["AFD_GSM8K_LIMIT"] = "3"
                command = [
                    sys.executable,
                    "-m",
                    f"tests.e2e.models.{model}.cases",
                    "multi-pod",
                    "--scenario",
                    scenario,
                    "--model",
                    environment.model,
                    "--vllm-bin",
                    environment.vllm_bin,
                    "--device-backend",
                    "gpu",
                    "--devices",
                    "0,1",
                    "--output-path",
                    str(tmp_path / f"pod-{index}"),
                    "--api-port-base",
                    str(environment.api_port_base),
                    "--pod-layout",
                    layout,
                    "--run-id",
                    tmp_path.name,
                    "--store-host",
                    "127.0.0.1",
                    "--store-port",
                    str(store_port),
                    "--pod-addresses",
                    "127.0.0.1,127.0.0.1",
                    "--launch-timeout",
                    "15",
                    "--serving-timeout",
                    "15",
                    "--verdict-timeout",
                    "20",
                ]
                child = subprocess.Popen(
                    command,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                children.append(child)
                thread = threading.Thread(
                    target=read_output, args=(index, child), daemon=True
                )
                thread.start()
                threads.append(thread)
            if outcome == "cancel":
                assert waiting.wait(timeout=25), "".join(sum(lines, []))
                children[0].send_signal(signal.SIGTERM)
            for child in children:
                child.wait(timeout=45)
            for thread in threads:
                thread.join(timeout=5)
            for index in range(2):
                assert (tmp_path / f"pod-{index}" / "run.json").exists(), "".join(
                    sum(lines, [])
                )
            records = [
                json.loads((tmp_path / f"pod-{index}" / "run.json").read_text())
                for index in range(2)
            ]
            # Assert before emergency fixture cleanup so it cannot hide a leak.
            for record in records:
                for service in record["processes"]:
                    with pytest.raises(ProcessLookupError):
                        os.kill(service["pid"], 0)
            if other_run is not None:
                assert other_run.poll() is None
            return children, records, lines
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            if other_run is not None:
                other_run.terminate()
                other_run.wait(timeout=5)
            for index in range(2):
                pids = tmp_path / f"pod-{index}-pids"
                if pids.exists():
                    for pid in pids.read_text().splitlines():
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(int(pid), signal.SIGKILL)

    return run


@pytest.mark.parametrize(
    "model,scenario,layout,outcome,samples",
    [
        ("deepseek_v2_lite", "afd-graph-2a2f", "2A0F,0A2F", "pass", 7),
        ("deepseek_v2_lite", "afd-graph-2a2f", "1A1F,1A1F", "pass", 7),
        ("qwen3_moe", "afd-deepep-m2n-graph-2a2f", "2A0F,0A2F", "pass", 100),
        ("deepseek_v2_lite", "afd-graph-2a2f", "2A0F,0A2F", "debug", 3),
        (
            "deepseek_v2_lite",
            "afd-graph-2a2f",
            "1A1F,1A1F",
            "peer-startup-failure",
            None,
        ),
        ("deepseek_v2_lite", "afd-graph-2a2f", "2A0F,0A2F", "evaluator-error", None),
        ("deepseek_v2_lite", "afd-graph-2a2f", "2A0F,0A2F", "cancel", None),
    ],
)
def test_multi_pod_case_coordinates_workload_and_owned_cleanup(
    multi_pod_runtime, model, scenario, layout, outcome, samples
):
    children, records, lines = multi_pod_runtime(model, scenario, layout, outcome)
    succeeded = samples is not None
    for child, record, log in zip(children, records, lines, strict=True):
        assert (child.returncode == 0) is succeeded, "".join(log)
        assert record["cleanup"]["status"] == "passed"
        assert record["acceptance_passed"] is (outcome == "pass")
        assert record["coverage"] == ["task_result", "local_liveness", "local_cleanup"]
        if succeeded:
            assert record["execution"]["status"] == "passed"
            assert record["configuration"]["eligible"] is (outcome == "pass")
            assert record["phases"][0]["result"]["sample_count"] == samples
            assert record["phases"][0]["checks"] == []
    if layout == "1A1F,1A1F":
        headless = records[1]["commands"]
        assert all("--headless" in command for command in headless.values())
        assert all("--port" not in command for command in headless.values())
    if outcome == "cancel":
        assert children[0].returncode == 128 + signal.SIGTERM
        assert records[0]["execution"]["status"] == "cancelled"
