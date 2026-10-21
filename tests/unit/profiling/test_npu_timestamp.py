# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Only completion fences remain; no custom timing backend is required."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from afd_plugin.profiling import npu as npu_module
from afd_plugin.profiling.npu import FFNDeviceWaiter


def test_receive_wait_yields_and_stops_only_after_pre_receive_fence(monkeypatch):
    device = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda _: "stream"))
    monkeypatch.setitem(sys.modules, "torch_npu", device)
    yields, progressed = [], []
    monkeypatch.setattr(npu_module.os, "sched_yield", lambda: yields.append(True))
    waiter = FFNDeviceWaiter.__new__(FFNDeviceWaiter)
    waiter.device = "npu:0"
    ready, fences = iter((False, False, True)), iter((False, True))
    waiter._ready = SimpleNamespace(record=lambda _: None, query=lambda: next(ready))
    waiter.begin = SimpleNamespace(query=lambda: next(fences))
    waiter.session = SimpleNamespace(
        advance_profiler=lambda **kw: progressed.append(kw["stop_ready"]())
    )
    waiter.wait_for_stream()
    assert yields == [True, True]
    assert progressed == [False, True]


def test_control_copy_supports_basic_and_profile_metadata(monkeypatch):
    import torch

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    waiter = FFNDeviceWaiter.__new__(FFNDeviceWaiter)
    waiter._control = torch.empty(3, dtype=torch.int64)
    waiter.wait_for_stream = lambda: None
    waiter.session = SimpleNamespace(advance_profiler=lambda: None)
    for values in ([123, 4, 17],):
        assert waiter.read_ffn_control(torch.tensor(values)) == values
