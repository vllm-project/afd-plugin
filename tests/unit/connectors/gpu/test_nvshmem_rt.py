# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NVSHMEM allocation errors and pointer ABI, without a CUDA device."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from afd_plugin.connectors.gpu.nvshmem_rt import (  # noqa: E402
    NvshmemRuntime,
    tensor_from_ptr,
)


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


def test_misaligned_raw_pointer_view_is_rejected_before_cuda_access():
    with pytest.raises(ValueError, match="not aligned"):
        tensor_from_ptr(
            0x12340000,
            byte_offset=1,
            sizes=(4,),
            dtype=torch.int32,
            device=torch.device("cuda", 0),
        )
