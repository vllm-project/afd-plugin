# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The driver ABI is an external protocol, independently testable on CPU."""

import ctypes

import pytest

from afd_plugin.connectors.gpu import cuda_rt


class DriverWait:
    def __init__(self, status=0):
        self.status = status
        self.messages = []

    def __call__(self, stream, pointer, value, flags):
        self.messages.append((stream.value, pointer.value, value.value, flags.value))
        return self.status


def test_driver_wait_uses_v2_exact_marker_abi(monkeypatch):
    wait = DriverWait()

    class DriverLibrary:
        cuStreamWaitValue32_v2 = wait  # noqa: N815 -- exact CUDA ABI

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: DriverLibrary())
    driver = cuda_rt.CudaStreamMemoryOps()
    driver.stream_wait_value32(0x1234, 0xABCDEF1234, 1)
    assert wait.messages == [(0x1234, 0xABCDEF1234, 1, 1)]


def test_driver_status_failure_remains_visible(monkeypatch):
    class DriverLibrary:
        cuStreamWaitValue32_v2 = DriverWait(status=801)  # noqa: N815 -- CUDA ABI

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: DriverLibrary())
    driver = cuda_rt.CudaStreamMemoryOps()
    with pytest.raises(RuntimeError, match="cuStreamWaitValue32_v2.*801"):
        driver.stream_wait_value32(1, 2, 1)
