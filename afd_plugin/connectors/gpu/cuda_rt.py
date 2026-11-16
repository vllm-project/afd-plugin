# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Instance-owned CUDA driver stream waits for peer-written window flags."""

from __future__ import annotations

import ctypes

CU_STREAM_WAIT_VALUE_EQ = 0x1


class CudaStreamMemoryOps:
    """Bind CUDA's V2 memory-wait ABI once, before graph capture.

    CUDA 13 supports the V2 entry point directly. The deprecated V1 device
    capability attribute says nothing about this API and must not gate it.
    """

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcuda.so.1")
        self.wait_value32 = self.library.cuStreamWaitValue32_v2
        self.wait_value32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self.wait_value32.restype = ctypes.c_int

    def stream_wait_value32(self, stream: int, device_ptr: int, value: int) -> None:
        """Queue an exact ready-marker wait; this call never polls on the host."""
        status = self.wait_value32(
            ctypes.c_void_p(stream),
            ctypes.c_ulonglong(device_ptr),
            ctypes.c_uint32(value & 0xFFFFFFFF),
            ctypes.c_uint32(CU_STREAM_WAIT_VALUE_EQ),
        )
        if status:
            raise RuntimeError(f"cuStreamWaitValue32_v2 failed with status {status}")
