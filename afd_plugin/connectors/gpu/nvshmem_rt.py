# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Minimal NVSHMEM host-library binding for the async GPU connector.

``torch.distributed._symmetric_memory`` cannot serve AFD: its NVSHMEM backend
bootstraps on the *default* process group and carves teams out of
``NVSHMEM_TEAM_WORLD`` with ``nvshmem_team_split_strided``. Each AFD role runs
as its own ``vllm serve`` whose default group covers only that role's ranks, so
the cross-role AFD group is never a strided subset of it and team creation
fails.

Bootstrapping NVSHMEM ourselves from a unique id exchanged over the AFD group's
store makes the AFD world *be* ``NVSHMEM_TEAM_WORLD``, so no team split is
needed. Logical AFD ranks are translated to node-contiguous PEs for NVSHMEM's
built-in local-rank teams. The host library's ABI is pinned by its headers
(``uniqueid`` 128 B, ``init_attr`` 144 B, version = ``(1 << 16) + sizeof``),
which permits ctypes for the mapped-peer path. The optional CUDA extension
initializes NVSHMEM device state as well, enabling remote put-with-signal.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import site
import socket
import sysconfig
from types import ModuleType
from typing import TYPE_CHECKING, Final

import torch

from afd_plugin.connectors.gpu.async_gpu_protocol import nvshmem_pe_mapping

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup, Store

NVSHMEM_ABI_VERSION: Final[int] = 1 << 16
NVSHMEM_UNIQUE_ID_BYTES: Final[int] = 128
NVSHMEM_INIT_ARGS_BYTES: Final[int] = 128
NVSHMEM_INIT_ATTR_BYTES: Final[int] = 144
UNIQUEID_PADDING: Final[int] = 124
# 128 (init_args) - 4 (version) - 24 (uid_args) - 4 (trailing alignment)
INIT_ARGS_PADDING: Final[int] = 96
NVSHMEMX_INIT_WITH_UNIQUEID: Final[int] = 1 << 3
_UID_STORE_KEY: Final[str] = "afd_nvshmem_uid"
_LIB_RELATIVE: Final[str] = "nvidia/nvshmem/lib/libnvshmem_host.so.3"


class _UniqueId(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("internal", ctypes.c_char * UNIQUEID_PADDING),
    )


class _UniqueIdArgs(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("id", ctypes.POINTER(_UniqueId)),
        ("myrank", ctypes.c_int),
        ("nranks", ctypes.c_int),
    )


class _InitArgs(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("uid_args", _UniqueIdArgs),
        ("content", ctypes.c_char * INIT_ARGS_PADDING),
    )


class _InitAttr(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("mpi_comm", ctypes.c_void_p),
        ("args", _InitArgs),
    )


def _find_host_library() -> str:
    """Locate ``libnvshmem_host.so.3`` next to the installed nvshmem wheel."""
    roots = [sysconfig.get_paths()["purelib"], *site.getsitepackages()]
    for root in roots:
        candidate = os.path.join(root, _LIB_RELATIVE)
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "AFD async GPU connector requires the NVSHMEM host library; "
        f"{_LIB_RELATIVE} was not found under {roots}. Install "
        "nvidia-nvshmem-cu13 matching the installed torch build.",
    )


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_host_library(), mode=ctypes.RTLD_GLOBAL)
    lib.nvshmemx_get_uniqueid.argtypes = [ctypes.POINTER(_UniqueId)]
    lib.nvshmemx_get_uniqueid.restype = ctypes.c_int
    lib.nvshmemx_set_attr_uniqueid_args.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_UniqueId),
        ctypes.POINTER(_InitAttr),
    ]
    lib.nvshmemx_set_attr_uniqueid_args.restype = ctypes.c_int
    lib.nvshmemx_hostlib_init_attr.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(_InitAttr),
    ]
    lib.nvshmemx_hostlib_init_attr.restype = ctypes.c_int
    lib.nvshmem_malloc.argtypes = [ctypes.c_size_t]
    lib.nvshmem_malloc.restype = ctypes.c_void_p
    lib.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.nvshmem_ptr.restype = ctypes.c_void_p
    lib.nvshmem_free.argtypes = [ctypes.c_void_p]
    lib.nvshmem_free.restype = None
    lib.nvshmemx_hostlib_finalize.argtypes = []
    lib.nvshmemx_hostlib_finalize.restype = None
    lib.nvshmem_my_pe.restype = ctypes.c_int
    lib.nvshmem_n_pes.restype = ctypes.c_int
    return lib


class NvshmemRuntime:
    """Own one AFD NVSHMEM world and release it after its receiver drains.

    A worker owns one connector and one runtime. There is no module-level
    world cache: independent vLLM workers live in independent processes.
    """

    def __init__(self, pg: ProcessGroup, rank: int, world_size: int) -> None:
        # Lazy upstream import: only worker initialization needs distributed state.
        from torch.distributed.distributed_c10d import _get_process_group_store

        if (
            ctypes.sizeof(_UniqueId),
            ctypes.sizeof(_InitArgs),
            ctypes.sizeof(_InitAttr),
        ) != (
            NVSHMEM_UNIQUE_ID_BYTES,
            NVSHMEM_INIT_ARGS_BYTES,
            NVSHMEM_INIT_ATTR_BYTES,
        ):
            raise RuntimeError("NVSHMEM unique-id ABI requires a 64-bit host")
        self.library = _load_library()
        self.bridge: ModuleType | None = None
        if importlib.util.find_spec("afd_plugin._C_gpu_async") is not None:
            self.bridge = importlib.import_module("afd_plugin._C_gpu_async")
        self.closed = False
        self.store: Store = _get_process_group_store(pg)
        self.store.set(f"afd_nvshmem_host/{rank}", socket.gethostname())
        hosts = [
            self.store.get(f"afd_nvshmem_host/{peer}").decode()
            for peer in range(world_size)
        ]
        self.pe_by_rank = nvshmem_pe_mapping(hosts)
        pe_rank = self.pe_by_rank[rank]
        unique_id = _UniqueId()
        unique_id.version = NVSHMEM_ABI_VERSION + ctypes.sizeof(_UniqueId)
        if rank == 0:
            status = self.library.nvshmemx_get_uniqueid(ctypes.byref(unique_id))
            if status:
                raise RuntimeError(f"nvshmemx_get_uniqueid failed: {status}")
            self.store.set(_UID_STORE_KEY, bytes(memoryview(unique_id).cast("B")))
        else:
            raw = self.store.get(_UID_STORE_KEY)
            if len(raw) != ctypes.sizeof(_UniqueId):
                raise RuntimeError("NVSHMEM unique id has an invalid byte length")
            ctypes.memmove(ctypes.byref(unique_id), raw, len(raw))
        attr = _InitAttr()
        attr.version = NVSHMEM_ABI_VERSION + ctypes.sizeof(_InitAttr)
        attr.args.version = NVSHMEM_ABI_VERSION + ctypes.sizeof(_InitArgs)
        attr.args.uid_args.version = NVSHMEM_ABI_VERSION + ctypes.sizeof(_UniqueIdArgs)
        status = self.library.nvshmemx_set_attr_uniqueid_args(
            pe_rank,
            world_size,
            ctypes.byref(unique_id),
            ctypes.byref(attr),
        )
        if status:
            raise RuntimeError(f"nvshmemx_set_attr_uniqueid_args failed: {status}")
        if self.bridge is None:
            status = self.library.nvshmemx_hostlib_init_attr(
                NVSHMEMX_INIT_WITH_UNIQUEID, ctypes.byref(attr)
            )
        else:
            status = self.bridge.initialize(ctypes.addressof(attr))
        if status:
            raise RuntimeError(f"nvshmemx_hostlib_init_attr failed: {status}")
        actual = (self.library.nvshmem_my_pe(), self.library.nvshmem_n_pes())
        if actual != (pe_rank, world_size):
            raise RuntimeError(
                f"NVSHMEM PE numbering {actual} differs from {(pe_rank, world_size)}",
            )

    def malloc(self, nbytes: int) -> int:
        pointer = self.library.nvshmem_malloc(nbytes)
        if not pointer:
            raise RuntimeError(
                f"nvshmem_malloc({nbytes}) failed; raise NVSHMEM_SYMMETRIC_SIZE",
            )
        return int(pointer)

    def peer_ptr(self, local_ptr: int, pe: int) -> int | None:
        pointer = self.library.nvshmem_ptr(
            ctypes.c_void_p(local_ptr), self.pe_by_rank[pe]
        )
        return int(pointer) if pointer else None

    def close(self, pointer: int) -> None:
        """Collective: callers must finish the producer/receiver handshake first."""
        if self.closed:
            return
        self.library.nvshmem_free(ctypes.c_void_p(pointer))
        if self.bridge is None:
            self.library.nvshmemx_hostlib_finalize()
        else:
            self.bridge.finalize()
        self.closed = True


class _DeviceBuffer:
    """Hand a raw device pointer to torch via ``__cuda_array_interface__``."""

    def __init__(self, pointer: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "data": (pointer, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "version": 3,
            "strides": None,
        }


def tensor_from_ptr(
    base_ptr: int,
    *,
    byte_offset: int,
    sizes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """View symmetric memory as a tensor without copying.

    The buffer is exposed as bytes and then reinterpreted, because
    ``__cuda_array_interface__`` has no type string for dtypes like bfloat16.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    numel = 1
    for size in sizes:
        numel *= size
    if numel == 0:
        # A zero-length __cuda_array_interface__ buffer is rejected by the CUDA
        # runtime (cudaErrorInvalidValue); an empty slot is legitimate whenever
        # routing sends a peer nothing, so hand back a plain empty tensor.
        return torch.empty(sizes, dtype=dtype, device=device)
    nbytes = numel * itemsize
    if byte_offset % itemsize:
        raise ValueError(
            f"byte offset {byte_offset} is not aligned to {itemsize}-byte {dtype}",
        )
    raw = torch.as_tensor(
        _DeviceBuffer(base_ptr + byte_offset, nbytes),
        device=device,
    )
    return raw.view(dtype).reshape(sizes)


__all__ = ["NvshmemRuntime", "tensor_from_ptr"]
