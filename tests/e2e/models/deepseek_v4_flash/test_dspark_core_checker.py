# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host checks for DSpark acceptance oracles and deployment boundaries."""

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.case import CaseEnvironment
from tests.e2e.deployment import resolve
from tests.e2e.models.deepseek_v4_flash import dspark_cases, dspark_eager


def metrics(first: int, second: int, label: str = "") -> str:
    return "\n".join(
        f'counter{{engine="{rank}"{label}}} {value}'
        for rank, value in enumerate((first, second))
    )


def test_counter_checks_request_deltas_on_both_data_parallel_ranks():
    before = metrics(9, 10)
    assert dspark_eager.counter_delta(before, metrics(10, 12), "counter", 2) == {
        "0": 1,
        "1": 2,
    }
    with pytest.raises(AssertionError):
        dspark_eager.counter_delta(before, metrics(10, 10), "counter", 2)
    with pytest.raises(AssertionError):
        dspark_eager.counter_delta(
            before, metrics(10, 12).splitlines()[0], "counter", 2
        )


def test_external_kv_counter_ignores_other_sources():
    label = ',source="external_kv_transfer"'
    before = metrics(2, 3, label) + "\n" + metrics(100, 100, ',source="local_compute"')
    after = metrics(3, 5, label) + "\n" + metrics(1000, 1000, ',source="local_compute"')
    assert dspark_eager.counter_delta(before, after, "counter", 2, label) == {
        "0": 1,
        "1": 2,
    }
    with pytest.raises(AssertionError):
        dspark_eager.counter_delta(before, after, "counter", 2, 'source="missing"')


def test_rejects_native_libraries_loaded_from_another_checkout():
    source = Path(dspark_eager.__file__).resolve().parents[4] / "afd_plugin"
    loaded = [str(source / "_C_ascend.so"), str(source / "libcust_opapi.so")]
    dspark_eager.assert_native_source([{"native_maps": loaded}])
    loaded[0] = "/tmp/old-checkout/afd_plugin/_C_ascend.so"
    with pytest.raises(AssertionError):
        dspark_eager.assert_native_source([{"native_maps": loaded}])


def test_attention_ownership_requires_both_data_parallel_ranks():
    source = Path(dspark_eager.__file__).resolve().parents[4] / "afd_plugin"
    records = [
        {
            "rank": rank,
            "native_maps": [
                str(source / "_C_ascend.so"),
                str(source / "libcust_opapi.so"),
            ],
            "owns_routed_experts": False,
            "owns_attention": True,
            "owns_shared_experts": True,
        }
        for rank in (0, 1)
    ]
    dspark_eager.assert_attention_ownership(records, 2)
    with pytest.raises(AssertionError):
        dspark_eager.assert_attention_ownership(records[:1], 2)


def test_graph_replay_requires_nonempty_target_and_draft_on_every_attention_rank():
    event = {"tokens": 6, "real": [1]}
    records: list[dict[str, Any]] = [
        {
            "rank": rank,
            "events": {"target": [event], "draft": [event]},
        }
        for rank in range(2)
    ]
    dspark_eager.assert_attention_graph(records, 2)
    records[1]["events"]["draft"] = [{"tokens": 6, "real": [0]}]
    with pytest.raises(AssertionError):
        dspark_eager.assert_attention_graph(records, 2)


def test_serial_comparison_requires_same_tokens_and_prompts(tmp_path: Path):
    left, right = tmp_path / "left", tmp_path / "right"
    for path in (left, right):
        (path / "requests").mkdir(parents=True)
        (path / "requests/summary.json").write_text(
            json.dumps([{"name": "short", "token_ids": [[1, 2]]}])
        )
    dspark_cases.compare(left, right)
    (right / "requests/summary.json").write_text(
        json.dumps([{"name": "short", "token_ids": [[1, 3]]}])
    )
    with pytest.raises(AssertionError, match="token IDs differ"):
        dspark_cases.compare(left, right)


@pytest.mark.parametrize("case_id", dspark_cases.CASE_IDS)
def test_core_deployment_has_expected_roles_and_execution(case_id, tmp_path):
    env = CaseEnvironment(
        model="/model",
        vllm_bin="vllm",
        device_backend="npu",
        devices=tuple(map(str, range(10))),
        output_path=tmp_path,
    )
    case = dspark_cases.build_case(case_id, env)
    resolved = resolve(case, env)
    bindings = dict(resolved.role_devices)
    assert len(bindings["attention"]) == 2 and len(bindings["ffn"]) == 4
    assert len(bindings.get("prefill", ())) == (
        4 if case_id == dspark_cases.PD_GRAPH else 0
    )
    attention, ffn = resolved.command("attention"), resolved.command("ffn")
    assert "--speculative-config" in attention and "--speculative-config" in ffn
    for command in (attention, ffn):
        spec = json.loads(command[command.index("--speculative-config") + 1])
        assert spec["num_speculative_tokens"] == 5
    if case_id != dspark_cases.EAGER:
        a = json.loads(attention[attention.index("--compilation-config") + 1])
        f = json.loads(ffn[ffn.index("--compilation-config") + 1])
        assert (
            a["cudagraph_mode"] == "FULL_DECODE_ONLY" and f["cudagraph_mode"] == "FULL"
        )
    if case_id == dspark_cases.PD_GRAPH:
        prefill = resolved.command("prefill")
        assert "--kv-transfer-config" in prefill and "--kv-transfer-config" in attention
        assert "--worker-extension-cls" not in prefill
        assert "--prefiller-ports" in resolved.proxy_command()
    control = dspark_cases.build_case(case_id, env, speculative=False)
    assert "--speculative-config" not in resolve(control, env).command("attention")


def test_repeated_subruns_keep_evidence_isolated(monkeypatch, tmp_path):
    env = CaseEnvironment(
        model="/model",
        vllm_bin="vllm",
        device_backend="npu",
        devices=tuple(map(str, range(6))),
        output_path=tmp_path,
    )
    monkeypatch.setattr(
        dspark_cases, "run_case", lambda _case, _env: {"acceptance_passed": True}
    )
    first = dspark_cases.subrun(dspark_cases.GRAPH, env, "target")
    old_proof = first / "workers/attention-graph/rank-1.json"
    old_proof.parent.mkdir(parents=True)
    old_proof.write_text('{"rank": 1}')
    second = dspark_cases.subrun(dspark_cases.GRAPH, env, "target")
    assert old_proof.is_file()
    assert not (second / "workers/attention-graph/rank-1.json").exists()


@pytest.mark.parametrize("changed", ["dirty-runtime", "cann-kernel"])
def test_reference_identity_changes_with_runtime_contents(
    monkeypatch, tmp_path, changed
):
    root = tmp_path / "afd"
    kernel = root / "afd_plugin/_cann_ops_custom/vendors/afd-plugin/kernel/op.o"
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"first-kernel")
    (root / "tests/e2e").mkdir(parents=True)
    runtime = tmp_path / "runtime"
    package = runtime / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    runtime_file = package / "runner.py"
    runtime_file.write_text("value = 0\n")
    subprocess.run(["git", "init", "-q", str(runtime)], check=True)
    subprocess.run(["git", "-C", str(runtime), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(runtime),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    runtime_file.write_text("value = 1\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = CaseEnvironment(
        model=str(model),
        vllm_bin="vllm",
        device_backend="npu",
        devices=tuple(map(str, range(6))),
        output_path=tmp_path,
    )
    monkeypatch.setattr(dspark_cases, "ROOT", root)
    monkeypatch.setattr(
        dspark_cases.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    monkeypatch.setattr(
        dspark_cases.importlib.metadata, "version", lambda _name: "test"
    )
    before = dspark_cases.identity(env)
    if changed == "dirty-runtime":
        runtime_file.write_text("value = 2\n")
    else:
        kernel.write_bytes(b"second-kernel")
    assert dspark_cases.identity(env) != before


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


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
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
