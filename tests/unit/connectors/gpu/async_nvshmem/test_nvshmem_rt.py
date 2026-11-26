# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NVSHMEM allocation errors and pointer ABI, without a CUDA device."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch.distributed import TCPStore  # noqa: E402
from torch.distributed import distributed_c10d as c10d  # noqa: E402

from afd_plugin.connectors.gpu.async_nvshmem import nvshmem_rt  # noqa: E402
from afd_plugin.connectors.gpu.async_nvshmem.nvshmem_rt import (  # noqa: E402
    NvshmemRuntime,
    tensor_from_ptr,
)


@pytest.fixture
def bootstrap_stores(monkeypatch):
    timeout = timedelta(seconds=5)
    master = TCPStore("127.0.0.1", 0, is_master=True, timeout=timeout)
    peer = TCPStore("127.0.0.1", master.port, is_master=False, timeout=timeout)
    monkeypatch.setattr(c10d, "_get_process_group_store", lambda pg: pg)
    return (master, peer)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("host", "rank 1: OSError: libnvshmem_host dependency missing"),
        ("bridge", "rank 1: ImportError: CUDA bridge dependency missing"),
        ("absent", "CUDA/NVSHMEM extension is missing on ranks [1]"),
    ],
)
def test_native_load_failure_reaches_every_rank(
    monkeypatch, bootstrap_stores, failure, expected
):
    rank: ContextVar[int] = ContextVar("rank")

    def load_library():
        if rank.get() == 1 and failure == "host":
            raise OSError("libnvshmem_host dependency missing")
        return SimpleNamespace()

    def load_bridge(name):
        if rank.get() == 1 and failure == "bridge":
            raise ImportError("CUDA bridge dependency missing")
        return SimpleNamespace()

    monkeypatch.setattr(nvshmem_rt, "_load_library", load_library)
    monkeypatch.setattr(
        nvshmem_rt,
        "importlib",
        SimpleNamespace(
            util=SimpleNamespace(
                find_spec=lambda name: (
                    None if rank.get() == 1 and failure == "absent" else True
                )
            ),
            import_module=load_bridge,
        ),
    )

    def bootstrap(local_rank):
        rank.set(local_rank)
        NvshmemRuntime(bootstrap_stores[local_rank], local_rank, 2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bootstrap, local_rank) for local_rank in range(2)]
        messages = [str(future.exception()) for future in futures]
    assert messages[0] == messages[1]
    assert expected in messages[0]


def test_native_allocation_failure_is_explicit():
    runtime = NvshmemRuntime.__new__(NvshmemRuntime)
    runtime.library = SimpleNamespace(
        nvshmem_malloc=lambda _size: 0,
        nvshmem_ptr=lambda _pointer, _peer: 0,
    )
    with pytest.raises(RuntimeError, match="nvshmem_malloc.*NVSHMEM_SYMMETRIC_SIZE"):
        runtime.malloc(4096)


def test_unmapped_peer_selects_remote_transport():
    runtime = NvshmemRuntime.__new__(NvshmemRuntime)
    runtime.library = SimpleNamespace(nvshmem_ptr=lambda _pointer, _peer: 0)
    runtime.pe_by_rank = (0, 1)
    assert runtime.peer_ptr(0x12340000, 1) is None


def test_collective_release_is_idempotent():
    messages: list[tuple] = []
    runtime = NvshmemRuntime.__new__(NvshmemRuntime)
    runtime.closed = False
    runtime.bridge = None
    runtime.library = SimpleNamespace(
        nvshmem_free=lambda pointer: messages.append(("free", pointer.value)),
        nvshmemx_hostlib_finalize=lambda: messages.append(("finalize",)),
    )
    runtime.close(0x12340000)
    runtime.close(0x12340000)
    assert messages == [("free", 0x12340000), ("finalize",)]


def test_collective_release_frees_non_zero_pointers_once():
    messages: list[tuple] = []
    runtime = NvshmemRuntime.__new__(NvshmemRuntime)
    runtime.closed = False
    runtime.bridge = None
    runtime.library = SimpleNamespace(
        nvshmem_free=lambda pointer: messages.append(("free", pointer.value)),
        nvshmemx_hostlib_finalize=lambda: messages.append(("finalize",)),
    )
    runtime.close(0x56780000, 0, 0x12340000)
    runtime.close(0x56780000, 0, 0x12340000)
    assert messages == [("free", 0x56780000), ("free", 0x12340000), ("finalize",)]


def test_misaligned_raw_pointer_view_is_rejected_before_cuda_access():
    with pytest.raises(ValueError, match="not aligned"):
        tensor_from_ptr(
            0x12340000,
            byte_offset=1,
            sizes=(4,),
            dtype=torch.int32,
            device=torch.device("cuda", 0),
        )
