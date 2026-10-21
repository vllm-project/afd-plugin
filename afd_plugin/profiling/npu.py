# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Provide FFN model-thread safe points around asynchronous CAM receive.

FFNDeviceWaiter queries two untimed fences: before receive and after control
copy/submitted work. It advances session.py's native lifecycle only at safe
points and reads pinned control values after their nonblocking copy completes.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from afd_plugin.profiling.session import WorkerSession


FFN_CONTROL_FIELDS = 3


class FFNDeviceWaiter:
    def __init__(self, device: torch.device, session: WorkerSession) -> None:
        import torch
        import torch_npu

        self.device = device
        self.session = session
        self._control = torch.empty(
            (FFN_CONTROL_FIELDS,), dtype=torch.int64, pin_memory=True
        )
        self._ready = torch_npu.npu.Event()
        # Warm the wait event before the FFN receive loop can block the device.
        self._ready.record()
        self._ready.synchronize()
        self.begin = torch_npu.npu.Event()
        self.begin.record()
        self.begin.synchronize()

    def begin_receive(self) -> None:
        import torch_npu

        self.stream = torch_npu.npu.current_stream(self.device)
        # The receive's begin event also fences all preceding FFN computation.
        self.begin.record(self.stream)

    def wait_for_stream(self) -> None:
        import torch_npu

        self._ready.record(torch_npu.npu.current_stream(self.device))
        while not self._ready.query():
            self.session.advance_profiler(stop_ready=self.begin.query)
            # Release the GIL without adding a timer delay to every receive.
            os.sched_yield()

    def read_ffn_control(self, control: torch.Tensor) -> list[int]:
        target = self._control[: control.numel()]
        target.copy_(control, non_blocking=True)
        self.wait_for_stream()
        self.session.advance_profiler()
        return target.tolist()
