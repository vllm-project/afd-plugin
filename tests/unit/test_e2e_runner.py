# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import argparse
import io
import signal
import subprocess
import sys
from collections.abc import Callable

import pytest

import tests.conftest as conftest
from tests.conftest import REPO_ROOT, RUNNER_CLEANUP_TIMEOUT_S, run_runner
from tests.e2e import runner
from tests.e2e.accuracy import gsm8k as helpers_gsm8k
from tests.e2e.models.qwen3_6 import test_qwen3_6 as qwen3_6_e2e


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="deepseek-ai/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=1239,
        served_model_name_prefix="deepseek-v2-lite-afd",
        num_attention_ranks=2,
        num_ffn_ranks=1,
        attention_devices="0,1",
        ffn_devices="2",
        device_backend="gpu",
        tp_size=1,
        attention_tp_size=None,
        ffn_tp_size=None,
        scenario="afd-eager-2a1f",
        baseline=False,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        use_decode_bench_connector=False,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        gsm8k_output_path="/tmp/gsm8k-results",
        completion_output_path="/tmp/dsv4-completions.json",
    )


# --- Model E2E entrypoints ---------------------------------------------------


def test_qwen3_6_entrypoint_rejects_non_gpu_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    with pytest.raises(RuntimeError, match="supports only the 'gpu' backend"):
        qwen3_6_e2e.build_runner_command("afd-eager-2a1f", tmp_path)


# --- run_runner (tests/conftest.py) ------------------------------------------


def test_run_runner_forwards_cancellation_and_reaps(
    monkeypatch,
):  # long-test: signal delivery and process reaping form one lifecycle contract
    command = [sys.executable, "-m", "tests.e2e.runner"]
    previous_handlers = {signal.SIGTERM: object(), signal.SIGINT: object()}
    installed_handlers: dict[int, Callable[[int, object], None]] = {}
    signal_calls = []
    kill_calls = []
    popen_calls = []

    class FakeProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
                pytest.fail("the first cancellation signal must unwind the call")
            if len(self.wait_calls) == 2:
                installed_handlers[signal.SIGINT](signal.SIGINT, None)
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fake_popen(actual_command, **kwargs):
        popen_calls.append((actual_command, kwargs))
        return process

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        installed_handlers[signum] = handler

    monkeypatch.setattr(conftest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        conftest.signal, "getsignal", lambda signum: previous_handlers[signum]
    )
    monkeypatch.setattr(conftest.signal, "signal", fake_signal)
    monkeypatch.setattr(
        conftest.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(conftest.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(SystemExit) as error:
        run_runner(command)

    assert error.value.code == 128 + signal.SIGTERM
    assert popen_calls == [
        (command, {"cwd": REPO_ROOT, "env": None, "start_new_session": True})
    ]
    assert kill_calls == [(321, signal.SIGTERM), (321, 9)]
    assert process.wait_calls == [None, RUNNER_CLEANUP_TIMEOUT_S, None]
    assert signal_calls[-2:] == list(previous_handlers.items())


def test_run_runner_forwards_signal_received_during_spawn(monkeypatch):
    handlers: dict[int, Callable[[int, object], None]] = {}
    forwarded = []

    class FakeProcess:
        pid = 321

        def poll(self):
            return 0

    def fake_popen(*_args, **_kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return FakeProcess()

    monkeypatch.setattr(conftest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(conftest.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(
        conftest.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        conftest.os,
        "killpg",
        lambda pid, sig: forwarded.append((pid, sig)),
        raising=False,
    )

    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        run_runner(["runner"])

    assert forwarded == [(321, signal.SIGTERM)]


def test_run_runner_fails_for_a_nonzero_exit(monkeypatch):
    class FailedProcess:
        pid = 321
        returncode = 17

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        conftest.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess()
    )
    monkeypatch.setattr(conftest.signal, "signal", lambda *_args: None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_runner([sys.executable, "-m", "tests.e2e.runner"])

    assert error.value.returncode == 17


# --- Runner scenario configuration and commands ------------------------------


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--num-attention-ranks", "4"],
        ["--tp-size", "2"],
        ["--enable-dbo"],
        ["--prompt", "Hello"],
        ["--expect-text", "answer"],
    ],
)
def test_parse_args_rejects_legacy_fixed_scenario_options(monkeypatch, legacy_args):
    argv = "runner --model model --scenario afd-eager-2a1f --gsm8k-output-path out"
    monkeypatch.setattr(sys, "argv", [*argv.split(), *legacy_args])

    with pytest.raises(SystemExit) as error:
        runner.parse_args()

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("scenario", "backend", "attention_devices", "ffn_devices", "error_message"),
    [
        ("baseline-graph", "gpu", ["0", "1", "2", "3"], [], None),
        ("afd-v2-eager-1a1f", "npu", ["0"], ["1"], "require GPU"),
        (runner.ASYNC_CAM_SCENARIO, "gpu", ["0", "1"], ["2", "3"], "require NPU"),
        (runner.ASYNC_UBATCH_SCENARIO, "gpu", ["0", "1"], ["2"], "require NPU"),
        ("afd-eager-2a1f", "gpu", ["0"], ["2"], "attention-devices .* exactly 2"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], [], "ffn-devices .* exactly 1"),
        ("afd-eager-2a1f", "gpu", ["0", "0"], ["2"], "Attention devices must be"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], ["2", "2"], "FFN devices must be"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], ["1"], "must not overlap"),
    ],
)
def test_validate_topology(
    scenario, backend, attention_devices, ffn_devices, error_message
):
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)

    if error_message is None:
        runner.validate_topology(args, attention_devices, ffn_devices)
    else:
        with pytest.raises(ValueError, match=error_message):
            runner.validate_topology(args, attention_devices, ffn_devices)


# --- lm-eval invocation and GSM8K gating -------------------------------------


def _inline_thread(on_join: Callable[[float | None], None], alive: bool = False):
    class InlineThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            on_join(timeout)

        def is_alive(self):
            return alive

    return InlineThread


def test_lm_eval_timeout_reaps_process_and_reader(  # long-test: reap and join
    monkeypatch, tmp_path
):
    popen_calls = []
    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []
    reader_join_calls: list[float | None] = []

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None
        direct_kill_calls = 0
        wait_calls = []

        def kill(self):
            self.direct_kill_calls += 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            group_alive = False

    monotonic_values = iter([100.0, 111.0, 8000.0, 8021.0, 9000.0, 9000.0])
    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        helpers_gsm8k.threading, "Thread", _inline_thread(reader_join_calls.append)
    )
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(helpers_gsm8k, "signal", signal, raising=False)
    monkeypatch.setattr(
        helpers_gsm8k.os,
        "kill",
        lambda pid, sig: leader_signal_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(TimeoutError, match="lm-eval exceeded 10s budget") as error:
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            timeout_s=10,
        )

    assert popen_calls[0][1]["start_new_session"] is True
    assert process.direct_kill_calls == 0
    assert leader_signal_calls == [(404, signal.SIGTERM)]
    assert group_signal_calls == [(404, 0), (404, 9), (404, 0)]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "forced SIGKILL" in str(error.value.__cause__)
    assert process.wait_calls == [helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S]
    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_rejects_a_reader_that_does_not_join(monkeypatch, tmp_path):
    reader_join_calls: list[float | None] = []

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO('{"results":{"gsm8k":{"exact_match":1.0}}}\n')
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        helpers_gsm8k.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess()
    )
    monkeypatch.setattr(
        helpers_gsm8k.threading,
        "Thread",
        _inline_thread(reader_join_calls.append, alive=True),
    )
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: 100.0)

    with pytest.raises(RuntimeError, match="reader thread"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000", "model", output_path=str(tmp_path / "results")
        )

    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_lm_eval_signal_waits_for_cleanup(  # long-test: signal and cleanup lifecycle
    monkeypatch, tmp_path
):
    events = []
    installed_handlers = {}
    group_alive = True

    def previous_term_handler(signum, _frame):
        events.append(("delegate", signum))
        raise SystemExit(128 + signum)

    previous_handlers = {
        signal.SIGTERM: previous_term_handler,
        signal.SIGINT: signal.SIG_IGN,
    }

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            raise RuntimeError("reap failed")

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler
        action = "restore" if handler is previous_handlers[signum] else "install"
        events.append((action, signum))

    def fake_kill(_pid, sig):
        nonlocal group_alive
        events.append(("cleanup-term", sig))
        group_alive = False
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)

    def fake_killpg(_pid, sig):
        events.append(("probe", sig))
        if not group_alive:
            raise ProcessLookupError

    process = FakeProcess()
    monotonic_values = iter([100.0, 111.0, 8000.0, 9000.0, 9000.0])
    monkeypatch.setattr(
        helpers_gsm8k.signal, "getsignal", lambda signum: previous_handlers[signum]
    )
    monkeypatch.setattr(helpers_gsm8k.signal, "signal", fake_signal)
    monkeypatch.setattr(
        helpers_gsm8k.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(
        helpers_gsm8k.threading,
        "Thread",
        _inline_thread(lambda timeout: events.append(("reader-join", timeout))),
    )
    monkeypatch.setattr(helpers_gsm8k.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(SystemExit) as error:
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            timeout_s=10,
        )

    assert error.value.code == 128 + signal.SIGTERM
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "reap failed" in str(error.value.__cause__)
    assert events == [
        ("install", signal.SIGTERM),
        ("install", signal.SIGINT),
        ("cleanup-term", signal.SIGTERM),
        ("probe", 0),
        ("wait", helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S),
        ("reader-join", helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S),
        ("restore", signal.SIGTERM),
        ("restore", signal.SIGINT),
        ("delegate", signal.SIGTERM),
    ]


def test_run_lm_eval_handles_signal_received_during_spawn(monkeypatch, tmp_path):
    handlers: dict[int, Callable[[int, object], None]] = {}
    cleaned = []

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")

    class UnstartedReader:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    def previous_handler(signum, _frame):
        raise SystemExit(128 + signum)

    def fake_popen(*_args, **_kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return FakeProcess()

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", UnstartedReader)
    monkeypatch.setattr(
        helpers_gsm8k.signal, "getsignal", lambda _signum: previous_handler
    )
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        helpers_gsm8k,
        "_finish_lm_eval_cleanup",
        lambda process, *_args, **_kwargs: cleaned.append(process.pid),
    )

    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000", "model", output_path=str(tmp_path), timeout_s=0
        )

    assert cleaned == [404]


# --- Runner process lifecycle -------------------------------------------------


def test_ensure_processes_alive_reports_exited_process_returncode():
    process = argparse.Namespace(poll=lambda: 17)

    with pytest.raises(RuntimeError, match="returncode=17"):
        runner.ensure_processes_alive([process])  # type: ignore[list-item]


def test_terminate_processes_rejects_cleanup_failure(monkeypatch):
    monkeypatch.setattr(
        runner, "terminate_process_groups", lambda *_args, **_kwargs: ["cleanup failed"]
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.terminate_processes([])


class _JoinRecorder:
    joined = False

    def join(self, timeout):
        self.joined = True


def _stub_main_pipeline(monkeypatch, process, log_thread: _JoinRecorder) -> None:
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", lambda *_args: None)


@pytest.mark.parametrize("native_handler", [False, True])
def test_main_checks_processes_and_restores_signal_handlers_when_cleanup_fails(
    monkeypatch, capsys, native_handler
):
    process = argparse.Namespace(poll=lambda: None)
    log_thread = _JoinRecorder()
    checked_processes = []
    installed_handlers = []
    previous_handlers = {
        signal.SIGTERM: None if native_handler else "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }
    _stub_main_pipeline(monkeypatch, process, log_thread)
    monkeypatch.setattr(
        runner,
        "ensure_processes_alive",
        lambda processes: checked_processes.append(list(processes)),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failed")
        ),
    )
    monkeypatch.setattr(
        runner.signal, "getsignal", lambda signum: previous_handlers[signum]
    )
    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda signum, handler: installed_handlers.append((signum, handler)),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.main()

    assert checked_processes == [[process, process]] * 2
    assert log_thread.joined is True
    assert installed_handlers[2:] == [
        (signum, signal.SIG_DFL if handler is None else handler)
        for signum, handler in previous_handlers.items()
    ]
    assert "E2E SCENARIO" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_error:
        installed_handlers[0][1](signal.SIGTERM, None)
    assert exit_error.value.code == 128 + signal.SIGTERM


def test_main_defers_a_first_signal_until_cleanup_failure_is_reported(monkeypatch):
    log_thread = _JoinRecorder()
    installed_handlers = {}
    cleanup_events = []
    previous_handlers = {
        signal.SIGTERM: "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler

    def cleanup_with_first_signal(_processes, **_kwargs):
        cleanup_events.append("started")
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        cleanup_events.append("completed")
        raise RuntimeError("cleanup failed")

    _stub_main_pipeline(monkeypatch, argparse.Namespace(poll=lambda: None), log_thread)
    monkeypatch.setattr(runner, "terminate_processes", cleanup_with_first_signal)
    monkeypatch.setattr(
        runner.signal, "getsignal", lambda signum: previous_handlers[signum]
    )
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 128 + signal.SIGTERM
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "cleanup failed"
    assert cleanup_events == ["started", "completed"]
    assert log_thread.joined is True
    assert installed_handlers == previous_handlers
