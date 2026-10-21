# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime activation of the existing public TorchNPU profiler."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.npu
PROBE_TIMEOUT_S = 120
KILL_GRACE_S = 5
MATRIX_SIZE = 256
ITERATIONS = 4
BUSY_MATRIX_SIZE = 4096
BUSY_ITERATIONS = 1024
CPU_MARKER = "CAM_RUNTIME_FULL_PRODUCER"


def _probe_cooperative(output: Path) -> None:
    import uuid
    from functools import partial

    import torch
    import torch_npu

    from afd_plugin.profiling.native import RuntimeProfiler
    from afd_plugin.profiling.npu import FFNDeviceWaiter
    from afd_plugin.profiling.session import WorkerSession

    os.environ["AFD_NPU_FFN_PROFILER_ENABLE"] = "0"
    os.environ["AFD_NPU_FFN_PROFILER_WITH_STACK"] = "0"
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch_npu.npu.set_device(device)

    def unexpected_tracer():
        raise AssertionError("FULL must not allocate LIGHTWEIGHT resources")

    session = WorkerSession(
        "ffn-test",
        "ffn",
        output,
        None,
        {},
        profiler_factory=partial(RuntimeProfiler, "ffn"),
        tracer_factory=unexpected_tracer,
    )
    waiter = FFNDeviceWaiter(device, session)
    session.device_waiter = waiter
    session.start(device)
    matrix = torch.ones(
        (BUSY_MATRIX_SIZE, BUSY_MATRIX_SIZE), dtype=torch.float16, device=device
    )
    target = torch.empty_like(matrix)
    control = torch.tensor([123, 5, 17], dtype=torch.int64, device=device)
    torch.mm(matrix, matrix, out=target)
    pending = torch_npu.npu.Event()
    pending.record()
    torch_npu.npu.synchronize(device)
    session_id = str(uuid.uuid4())
    session.prepare(
        {
            "session_id": session_id,
            "command_seq": 1,
            "mode": "FULL",
            "duration_s": 30,
        },
        backend_busy=False,
    )
    session.commit(
        {
            "session_id": session_id,
            "command_seq": 2,
            "start_at_ns": time.time_ns() + 10_000_000,
            "clock_by_worker": {"ffn-test": {"offset_ns": 0, "uncertainty_ns": 0}},
        }
    )
    runtime = session.capture.profiler
    observations: list[dict] = []
    wait_calls = 0
    diagnostics = {
        "start_ns": session.capture.start_ns,
        "deadline_ns": session.capture.deadline_ns,
    }
    advance_profiler = session.advance_profiler
    wait_for_stream = waiter.wait_for_stream

    def observe_copy() -> None:
        diagnostics["pending_after_copy"] = not pending.query()
        wait_for_stream()

    def observe_start(**kwargs) -> None:
        nonlocal wait_calls
        wait_calls += 1
        pending_before = not pending.query()
        if wait_calls == 1:
            diagnostics["pending_after_event_record"] = pending_before
            diagnostics["first_full_state"] = runtime.state
            diagnostics["first_session_state"] = session.state
        advance_profiler(**kwargs)
        if runtime.started_at_ns is not None and not observations:
            session.tick()
            observations.append(
                {
                    "pending_before_start": pending_before,
                    "pending_after_start": not pending.query(),
                    "producer_tid": threading.get_native_id(),
                    "session_state": session.state,
                }
            )

    session.advance_profiler = observe_start
    waiter.wait_for_stream = observe_copy
    waiter.begin_receive()
    queue_begin = time.monotonic()
    for _ in range(BUSY_ITERATIONS):
        torch.mm(matrix, matrix, out=target)
    pending.record()
    diagnostics["pending_before_copy"] = not pending.query()
    assert diagnostics["pending_before_copy"], "work completed before receive-wait"
    read_begin = time.monotonic()
    assert waiter.read_ffn_control(control) == [123, 5, 17]
    diagnostics.update(
        wait_calls=wait_calls,
        queue_seconds=read_begin - queue_begin,
        read_seconds=time.monotonic() - read_begin,
    )
    assert observations, {"diagnostics": diagnostics, "status": session.status()}
    assert observations[0]["pending_before_start"], observations
    assert observations[0]["pending_after_start"], observations
    assert observations[0]["session_state"] == "RUNNING", observations
    assert observations[0]["producer_tid"] == threading.get_native_id()
    assert pending.query(), "control data became readable before work completed"
    marker = f"{CPU_MARKER}_cooperative"
    with torch.profiler.record_function(marker):
        torch.mm(matrix, matrix, out=target)
    session.stop({"session_id": session_id, "command_seq": 3})
    torch.npu.current_stream().synchronize()
    runtime.step()
    assert runtime.state == "COMPLETED", runtime.snapshot()
    deadline = time.monotonic() + 5
    while session.state not in {"COMPLETED", "FAILED"}:
        session.tick()
        assert time.monotonic() < deadline, session.status()
        time.sleep(0.001)
    assert session.state == "COMPLETED", session.status()
    assert session.tracer is None
    records = [
        json.loads(line)
        for path in output.rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert records
    assert all(record["record_type"] in {"SESSION", "SUMMARY"} for record in records)
    traces = list(runtime.trace_dir.rglob("trace_view.json"))
    assert traces, runtime.snapshot()
    events = []
    for path in traces:
        trace = json.loads(path.read_text())
        events.extend(trace if isinstance(trace, list) else trace["traceEvents"])
    assert any(event.get("name") == marker for event in events), traces
    assert any(
        event.get("name") == "aten::mm"
        and event.get("tid") == threading.get_native_id()
        for event in events
    ), traces
    assert any(
        "Task Type" in event.get("args", {}) and "aclnnMm_" in event.get("name", "")
        for event in events
    ), traces
    (output / "probe-result.json").write_text(
        json.dumps(
            {
                "startup": observations[0],
                "diagnostics": diagnostics,
                "status": session.status(),
            },
            indent=2,
        )
    )


def _probe_cam_idle(output: Path) -> None:
    import torch
    import torch_npu  # noqa: F401

    from afd_plugin.profiling.model import transition
    from afd_plugin.profiling.native import RuntimeProfiler
    from afd_plugin.profiling.npu import FFNDeviceWaiter
    from afd_plugin.profiling.session import Capture
    from afd_plugin.profiling.tracing import DeviceTracer
    from tests.npu.async_cam_common import HarnessConfig, NativeHarness

    os.environ["AFD_NPU_FFN_PROFILER_WITH_STACK"] = "0"
    config = HarnessConfig(
        attn_ranks=1,
        moe_ranks=1,
        tp_size=1,
        lengths=(8,),
        capacity=16,
        max_seq_len=8,
        timeout_seconds=60,
    )
    results = []
    with NativeHarness(config) as harness:
        assert harness.comm_args.numel() == 1
        assert not harness.run_round(0)["failures"]
        original = harness.operations["dispatch_recv"]
        if harness.rank == 1:
            matrix = torch.ones(
                (MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float16, device=harness.device
            )
            target = torch.empty_like(matrix)
            waiter = FFNDeviceWaiter(
                harness.device, SimpleNamespace(advance_profiler=lambda **kwargs: None)
            )

        for mode, variant in (
            (mode, variant)
            for mode in ("FULL", "LIGHTWEIGHT")
            for variant in ("start_before_receive", "start_during_receive")
        ):

            def receive(*args, variant=variant, mode=mode):
                runtime = RuntimeProfiler("ffn", output / mode / variant, mode)
                if variant == "start_before_receive":
                    runtime.start()
                    runtime.step(startup_only=True)
                    with torch.profiler.record_function(CPU_MARKER):
                        for _ in range(ITERATIONS):
                            torch.mm(matrix, matrix, out=target)
                waiter.begin_receive()
                execution = None
                if mode == "LIGHTWEIGHT":
                    tracer = DeviceTracer(harness.device, "f0")
                    tracer.session = SimpleNamespace(
                        capture=Capture(
                            "idle-probe",
                            1,
                            SimpleNamespace(),
                            accepting=runtime.state == "RUNNING",
                            deadline_ns=time.monotonic_ns()
                            + int(config.timeout_seconds * 1e9),
                        )
                    )
                    execution = tracer.new_work()
                transition(execution, "WAIT")
                values = original(*args)
                completed = torch_npu.npu.Event()
                completed.record()
                assert not completed.query(), "sender arrived before the idle probe"
                if variant == "start_during_receive":
                    runtime.start()
                    runtime.step(startup_only=True)
                deadline = time.monotonic() + 10
                while not waiter.begin.query():
                    assert time.monotonic() < deadline
                    time.sleep(0)
                assert runtime.state == "RUNNING", runtime.snapshot()
                begin = time.monotonic()
                runtime.stop()
                runtime.step()
                seconds = time.monotonic() - begin
                assert runtime.state == "COMPLETED", runtime.snapshot()
                assert seconds < 10 and not completed.query(), (
                    seconds,
                    runtime.snapshot(),
                )
                events = []
                for path in runtime.trace_dir.rglob("trace_view.json"):
                    trace = json.loads(path.read_text())
                    events.extend(
                        trace if isinstance(trace, list) else trace["traceEvents"]
                    )
                if mode == "FULL" and variant == "start_before_receive":
                    kernels = sum(
                        "Task Type" in e.get("args", {})
                        and "aclnnMm_" in e.get("name", "")
                        for e in events
                    )
                    assert kernels == ITERATIONS, kernels
                    assert sum(e.get("name") == CPU_MARKER for e in events) == 1
                results.append(
                    {
                        "mode": mode,
                        "variant": variant,
                        "stop_seconds": seconds,
                        "state": runtime.snapshot(),
                    }
                )
                _, _, batch, counts = values
                control = torch.stack(
                    (batch[0], batch[2], counts.sum(dtype=torch.int64))
                )
                assert waiter.read_ffn_control(control) == [16, 1, 16]
                assert completed.query()
                return values

            harness.operations["dispatch_recv"] = receive

            def delay_sender(name):
                if harness.rank == 0 and name == "dispatch_send":
                    time.sleep(20)

            report = harness.run_round(1, before_op=delay_sender)
            assert not report["failures"], report
        if harness.rank == 1:
            (output / "probe-result.json").write_text(json.dumps(results, indent=2))


@pytest.mark.parametrize("probe", ["--cooperative-probe", "--cam-idle-probe"])
def test_runtime_full_preserves_public_profiler_outputs(tmp_path, probe):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    pytest.importorskip("torch_npu")
    command = [sys.executable]
    if probe == "--cam-idle-probe":
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if len([item for item in visible if item.strip()]) < 2:
            pytest.skip("blocked CAM receive probe requires two allocated devices")
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    command += [str(Path(__file__).resolve()), probe, str(tmp_path)]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                log, _ = process.communicate(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                log, _ = process.communicate(timeout=KILL_GRACE_S)
            (tmp_path / "probe.log").write_text(log)
            pytest.fail(f"Runtime FULL exceeded its deadline:\n{log}")
        (tmp_path / "probe.log").write_text(log)
        assert process.returncode == 0, log
    assert (tmp_path / "probe-result.json").exists()


if __name__ == "__main__":
    probes = {
        "--cooperative-probe": _probe_cooperative,
        "--cam-idle-probe": _probe_cam_idle,
    }
    if len(sys.argv) != 3 or sys.argv[1] not in probes:
        raise SystemExit("usage: test_cam_profile_runtime_full.py PROBE OUTPUT")
    probes[sys.argv[1]](Path(sys.argv[2]))
