# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Stream memory operations, for waiting on a peer's flag without the host.

A receive is a wait on one 32-bit word that a peer writes. Doing that wait on
the host -- copy the flag down, look at it, try again -- costs a synchronize per
attempt and, far worse, stops the host from queueing anything behind the wait.
A profile of the async connector showed 86% of kernel launches starting within
5us of being issued: the GPU was idle waiting to be fed one kernel at a time,
because every layer blocked the host twice.

``cuStreamWaitValue32`` moves the wait onto the stream. The host enqueues "wait
until this word reaches N" and keeps going, so the work behind the wait is
already queued when the flag arrives. It is a driver API with no runtime API
equivalent and no PyTorch binding, hence ctypes -- the same approach
``nvshmem_rt`` already takes for the symmetric allocator.
"""

from __future__ import annotations

import ctypes
from typing import Final

# CUstreamWaitValue_flags. GEQ is a cyclic comparison, so a monotonically
# increasing sequence number keeps working across 32-bit wraparound.
CU_STREAM_WAIT_VALUE_GEQ: Final[int] = 0x0
CU_STREAM_WAIT_VALUE_EQ: Final[int] = 0x1
# CUdevice_attribute: stream memory ops must be supported by the device.
_CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS: Final[int] = 74

_lib: ctypes.CDLL | None = None
_wait_value32 = None
_checked_devices: set[int] = set()


def _load() -> ctypes.CDLL:
    global _lib, _wait_value32
    if _lib is not None:
        return _lib

    lib = ctypes.CDLL("libcuda.so.1")
    # CUDA 11.7 renamed the entry point; the unsuffixed symbol still exists on
    # some builds, so take whichever this driver exports.
    for symbol in ("cuStreamWaitValue32_v2", "cuStreamWaitValue32"):
        fn = getattr(lib, symbol, None)
        if fn is not None:
            fn.argtypes = [
                ctypes.c_void_p,  # CUstream
                ctypes.c_ulonglong,  # CUdeviceptr
                ctypes.c_uint32,  # value
                ctypes.c_uint32,  # flags
            ]
            fn.restype = ctypes.c_int
            _wait_value32 = fn
            break
    else:
        raise RuntimeError(
            "libcuda.so.1 exports no cuStreamWaitValue32; this driver cannot "
            "wait on a flag from a stream",
        )
    lib.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cuDeviceGetAttribute.restype = ctypes.c_int
    _lib = lib
    return lib


def require_stream_mem_ops(device_index: int) -> None:
    """Fail loudly at setup if the device cannot wait on memory from a stream."""
    if device_index in _checked_devices:
        return
    lib = _load()
    supported = ctypes.c_int(0)
    status = lib.cuDeviceGetAttribute(
        ctypes.byref(supported),
        _CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
        device_index,
    )
    if status != 0:
        raise RuntimeError(
            f"cuDeviceGetAttribute failed with status {status} while checking "
            "stream memory op support",
        )
    if not supported.value:
        raise RuntimeError(
            f"CUDA device {device_index} does not support stream memory "
            "operations, which the async GPU connector needs to wait on a "
            "peer's flag without blocking the host",
        )
    _checked_devices.add(device_index)


def stream_wait_value32(
    stream: int,
    device_ptr: int,
    value: int,
    *,
    flags: int = CU_STREAM_WAIT_VALUE_GEQ,
) -> None:
    """Enqueue "block this stream until ``*device_ptr`` reaches ``value``"."""
    if _wait_value32 is None:
        _load()
    assert _wait_value32 is not None
    status = _wait_value32(
        ctypes.c_void_p(stream),
        ctypes.c_ulonglong(device_ptr),
        ctypes.c_uint32(value & 0xFFFFFFFF),
        ctypes.c_uint32(flags),
    )
    if status != 0:
        raise RuntimeError(
            f"cuStreamWaitValue32 failed with status {status} "
            f"(ptr={device_ptr:#x}, value={value})",
        )


__all__ = [
    "CU_STREAM_WAIT_VALUE_EQ",
    "CU_STREAM_WAIT_VALUE_GEQ",
    "require_stream_mem_ops",
    "stream_wait_value32",
]
