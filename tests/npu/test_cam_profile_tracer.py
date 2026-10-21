# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native pipeline transitions, multiple streams and compressed capture reuse."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.npu
PROBE_TIMEOUT_S = 120
KILL_GRACE_S = 5


def _probe(output: Path) -> None:
    from functools import partial

    import torch
    from torch_npu.profiler.profiler import analyse

    from afd_plugin.profiling.analyze import export_chrome_trace
    from afd_plugin.profiling.model import transition
    from afd_plugin.profiling.native import RuntimeProfiler
    from afd_plugin.profiling.native_trace import native_directory, read_native_trace
    from afd_plugin.profiling.session import WorkerSession
    from afd_plugin.profiling.tracing import DeviceTracer

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    tracer = DeviceTracer(device, "a0")
    session = WorkerSession(
        "a0",
        "attention",
        output,
        tracer,
        {},
        profiler_factory=partial(RuntimeProfiler, "attention"),
    )
    session.start(device)
    x = torch.ones((256, 256), dtype=torch.float16, device=device)
    targets = [torch.empty_like(x), torch.empty_like(x)]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    torch.mm(x, x, out=targets[0])
    torch.npu.synchronize()
    results = []
    for mode in ("LIGHTWEIGHT", "FULL", "LIGHTWEIGHT"):
        session_id = str(uuid.uuid4())
        session.prepare(
            {
                "session_id": session_id,
                "command_seq": 1,
                "mode": mode,
                "duration_s": 30,
            },
            backend_busy=False,
        )
        session.commit(
            {
                "session_id": session_id,
                "command_seq": 2,
                "start_at_ns": time.time_ns() + 10_000_000,
                "clock_by_worker": {"a0": {"offset_ns": 0, "uncertainty_ns": 11}},
            }
        )
        deadline = time.monotonic() + 10
        while session.state != "RUNNING":
            assert time.monotonic() < deadline, session.status()
            session.advance_profiler()
            time.sleep(0.005)
        for index, stream in enumerate(streams):
            with torch.npu.stream(stream):
                if mode == "LIGHTWEIGHT":
                    execution = tracer.execution()
                    assert execution is not None and tracer.retain_execution(execution)
                    for phase in ("COMPUTE", "COMPUTE", "COMMUNICATION", "WAIT"):
                        transition(
                            execution,
                            phase,
                            f"{index}|7|3|9" if phase == "COMMUNICATION" else "",
                        )
                        torch.mm(x, x, out=targets[index])
                    tracer.release_execution(execution)
                else:
                    torch.mm(x, x, out=targets[index])
        # Bound this component test to completed work before the host END marker.
        torch.npu.synchronize()
        session.stop({"session_id": session_id, "command_seq": 3})
        session.advance_profiler()
        assert session.capture.profiler.state == "COMPLETED", session.status()
        deadline = time.monotonic() + 10
        while session.state not in {"COMPLETED", "FAILED"}:
            assert time.monotonic() < deadline, session.status()
            time.sleep(0.005)
        assert session.state == "COMPLETED", session.status()
        owner = session.capture.profiler
        if mode == "LIGHTWEIGHT":
            deadline = time.monotonic() + 10
            while owner.archive_state not in {"COMPLETED", "FAILED"}:
                assert time.monotonic() < deadline, owner.snapshot()
                time.sleep(0.005)
            assert owner.archive_state == "COMPLETED", owner.snapshot()
            assert not owner.trace_dir.exists()
            rows = [
                json.loads(line)
                for line in (output / session_id / "a0.jsonl").read_text().splitlines()
            ]
            assert [r["record_type"] for r in rows] == ["SESSION", "SUMMARY"]
            assert rows[1]["native_marks"] == 7
            with native_directory(owner.trace_dir) as root:
                for raw in root.glob("*_ascend_pt"):
                    analyse(str(raw), max_process_number=1)
                records = read_native_trace(root, rows[0])
                result = export_chrome_trace([*rows, *records])
                assert result["data_quality"] == "COMPLETE", result["errors"]
                phases = [r for r in records if r["record_type"] == "PIPELINE"]
                assert len({r["stream_id"] for r in phases}) == 2
                assert {r["kind"] for r in phases} == {
                    "COMPUTE",
                    "COMMUNICATION",
                    "WAIT",
                }
                assert all(r["end_ns"] > r["begin_ns"] for r in phases)
                annotations = [r["annotation"] for r in phases if "annotation" in r]
                assert sorted(annotations, key=lambda r: r["microbatch_id"]) == [
                    {
                        "microbatch_id": i,
                        "layer_idx": 7,
                        "num_requests": 3,
                        "num_tokens": 9,
                    }
                    for i in range(2)
                ]
                events = []
                for path in root.rglob("trace_view.json"):
                    raw = json.loads(path.read_text())
                    events.extend(raw if isinstance(raw, list) else raw["traceEvents"])
                assert not any(e.get("name", "").startswith("aten::") for e in events)
                assert not any("Task Type" in e.get("args", {}) for e in events)
        results.append({"mode": mode, "status": session.status()})
    (output / "probe-result.json").write_text(json.dumps(results, indent=2))


@pytest.mark.parametrize("probe", ["--probe", "--cam-pipeline"])
def test_native_lightweight_capture(tmp_path, probe):
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires an allocated NPU and runtime opt-in")
    pytest.importorskip("torch_npu")
    command = [sys.executable]
    if probe == "--cam-pipeline":
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
            pytest.fail(f"Native LIGHTWEIGHT exceeded its deadline:\n{log}")
        (tmp_path / "probe.log").write_text(log)
        assert process.returncode == 0, log
    assert (tmp_path / "probe-result.json").exists()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--probe":
        raise SystemExit("usage: test_cam_profile_tracer.py --probe OUTPUT")
    _probe(Path(sys.argv[2]))
