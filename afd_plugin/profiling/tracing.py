# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Record LIGHTWEIGHT stream phases through native MSTX transition marks.

Model/connector hooks reuse stream tracks and mark COMPUTE/COMMUNICATION/WAIT
only on transitions. WorkerSession controls admission/drain; this module bounds
pending work, emits END and reports losses for offline native_trace.py/pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch_npu

if TYPE_CHECKING:
    import torch

    from afd_plugin.profiling.session import Capture, WorkerSession

RANGE_PREFIX = "afd.cam"
PHASE_MESSAGES = {
    phase: f"afd.cam|{phase}" for phase in ("COMPUTE", "COMMUNICATION", "WAIT")
}
MAX_PIPELINE_STREAMS = 64
MAX_PENDING_EXECUTIONS = 1024


@dataclass(slots=True)
class Execution:
    """One reusable stream track per capture, without layer or request identity."""

    tracer: DeviceTracer
    capture: Capture
    stream: tuple[int, int, int]
    pending: int = 0
    phase: str = ""


class DeviceTracer:
    def __init__(self, device: torch.device, worker_id: str) -> None:
        self.device_index = (
            device.index if device.index is not None else torch_npu.npu.current_device()
        )
        self.worker_id = worker_id
        self.session: WorkerSession | None = None
        self._finalized_generation = 0
        self.begin_capture()

    def begin_capture(self) -> None:
        self._streams: dict[tuple[int, int, int], Execution] = {}
        self._recorded_marks = 0
        self._dropped_spans = 0
        self._unresolved = 0
        self._ended = False

    def execution(self) -> Execution | None:
        assert self.session is not None
        capture = self.session.capture
        if capture is None or not capture.admits():
            return None
        # shortcut: pinned TorchNPU bindings avoid Stream wrappers; revisit on upgrade.
        stream = torch_npu._C._npu_getCurrentStream(self.device_index)
        execution = self._streams.get(stream)
        if execution is None:
            if len(self._streams) >= MAX_PIPELINE_STREAMS:
                self._dropped_spans += 1
                return None
            execution = Execution(self, capture, stream)
            self._streams[stream] = execution
        return execution

    def new_work(self) -> Execution | None:
        return self.execution()

    def transition(
        self, execution: Execution, phase: str, annotation: str = ""
    ) -> None:
        if execution.phase == phase or self._ended or self.finalized(execution.capture):
            return
        try:
            torch_npu._C._mstx._mark(
                f"{PHASE_MESSAGES[phase]}|{annotation}"
                if annotation
                else PHASE_MESSAGES[phase],
                *execution.stream,
                RANGE_PREFIX,
            )
            execution.phase = phase
            self._recorded_marks += 1
        except Exception as error:
            execution.capture.error = f"native pipeline mark failed: {error}"
            self._dropped_spans += 1
            assert self.session is not None
            self.session._wake.set()

    def close_admission(self, capture: Capture) -> None:
        capture.accepting = False

    def retain_execution(self, execution: Execution) -> bool:
        if not execution.capture.admits() or self.finalized(execution.capture):
            return False
        if execution.pending >= MAX_PENDING_EXECUTIONS:
            self._dropped_spans += 1
            return False
        execution.pending += 1
        return True

    def release_execution(self, execution: Execution) -> None:
        if self.finalized(execution.capture):
            return
        execution.pending = max(0, execution.pending - 1)
        if execution.capture.stop_requested:
            assert self.session is not None
            self.session._wake.set()

    def idle(self, capture: Capture) -> None:
        if self.finalized(capture) or self._ended:
            return
        for execution in self._streams.values():
            if execution.capture is capture:
                self.transition(execution, "WAIT")

    def pending_count(self) -> int:
        return sum(execution.pending for execution in self._streams.values())

    def snapshot(self) -> dict:
        return {
            "pending_spans": self.pending_count(),
            "dropped_spans": self._dropped_spans,
            "dropped_metadata": 0,
            "unresolved_spans": self._unresolved,
            "native_marks": self._recorded_marks,
            "pipeline_streams": len(self._streams),
        }

    def finalized(self, capture: Capture) -> bool:
        return self._finalized_generation >= capture.generation

    def end_capture(self, capture: Capture) -> None:
        if self._ended or self.finalized(capture):
            return
        self._ended = True
        if self._recorded_marks:
            try:
                # Host END also closes streams blocked in CAM receive, without a sync.
                torch_npu._C._mstx._mark_on_host("afd.cam|END", RANGE_PREFIX)
                self._recorded_marks += 1
            except Exception as error:
                capture.error = f"native pipeline end failed: {error}"
                self._dropped_spans += 1

    def finish_capture(self, capture: Capture) -> None:
        self._finalized_generation = capture.generation
        self._unresolved = self.pending_count()
        if self._unresolved:
            capture.error = capture.error or "executions unresolved at drain deadline"
        capture.writer.submit(
            {
                "record_type": "SUMMARY",
                "worker_id": self.worker_id,
                **self.snapshot(),
                "dropped_records": capture.writer.snapshot()["dropped_records"],
                "error": capture.error,
            }
        )
        self._streams.clear()
