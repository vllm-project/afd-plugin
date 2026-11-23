# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contract tests for ``SymmWindow`` sizing, outbox packing, and cleanup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from afd_plugin.connectors.gpu.async_nvshmem import symm_window  # noqa: E402
from afd_plugin.connectors.gpu.async_nvshmem.protocol import (  # noqa: E402
    SIGNAL_WORD_BYTES,
    SlotLayout,
    align_bytes,
    decode_header_words,
)


def _layout() -> SlotLayout:
    return SlotLayout.build(
        expert_per_rank=4,
        partial_cap=16,
        token_cap=8,
        shared_cap=2,
        hidden_size=4,
        payload_itemsize=2,
    )


class _FakeCudaStream:
    def __init__(self, device=None) -> None:
        self.device = device
        self.cuda_stream = 101

    def wait_stream(self, _other) -> None:
        return None


class _FakeRuntime:
    def __init__(
        self,
        *,
        rank: int,
        mapped_peers: set[int],
        bridge: object | None,
        buffers: dict[int, Tensor],
        allocations: list[int],
        ptrs: list[int],
        freed: list[int],
        fail_on_alloc: int | None = None,
    ) -> None:
        self.rank = rank
        self.mapped_peers = mapped_peers
        self.bridge = bridge
        self.buffers = buffers
        self.allocations = allocations
        self.ptrs = ptrs
        self.freed = freed
        self.fail_on_alloc = fail_on_alloc
        self.closed = False
        self.pe_by_rank = tuple(range(16))

    def malloc(self, nbytes: int) -> int:
        if (
            self.fail_on_alloc is not None
            and len(self.allocations) + 1 == self.fail_on_alloc
        ):
            raise RuntimeError(f"nvshmem_malloc({nbytes}) failed")
        buf = torch.full((nbytes,), 0x5A, dtype=torch.uint8)
        ptr = buf.data_ptr()
        self.buffers[ptr] = buf
        self.allocations.append(nbytes)
        self.ptrs.append(ptr)
        return ptr

    def peer_ptr(self, local_ptr: int, pe: int) -> int | None:
        return local_ptr if pe in self.mapped_peers else None

    def close(self, *pointers: int) -> None:
        if self.closed:
            return
        for ptr in pointers:
            if ptr:
                self.freed.append(ptr)
        self.closed = True


@pytest.fixture
def window_harness(monkeypatch):
    def build(
        *,
        world_size: int = 8,
        mapped_peers: set[int] | None = None,
        max_remote_peers: int | None = None,
        has_bridge: bool = True,
        fail_on_alloc: int | None = None,
    ):
        buffers: dict[int, Tensor] = {}
        allocations: list[int] = []
        ptrs: list[int] = []
        freed: list[int] = []
        put_calls: list[tuple] = []
        mapped = set(range(world_size)) if mapped_peers is None else set(mapped_peers)

        def put_signal(dest, src, fields, signal, seq_ptr, value, peer, stream):
            put_calls.append((dest, src, fields, signal, seq_ptr, value, peer, stream))

        bridge = SimpleNamespace(put_signal=put_signal) if has_bridge else None

        def make_runtime(_group, r, _world):
            return _FakeRuntime(
                rank=r,
                mapped_peers=mapped,
                bridge=bridge,
                buffers=buffers,
                allocations=allocations,
                ptrs=ptrs,
                freed=freed,
                fail_on_alloc=fail_on_alloc,
            )

        def tensor_from_ptr(base_ptr, *, byte_offset, sizes, dtype, device):
            itemsize = torch.empty(0, dtype=dtype).element_size()
            numel = 1
            for size in sizes:
                numel *= size
            if numel == 0:
                return torch.empty(sizes, dtype=dtype, device=device)
            raw = buffers[base_ptr][byte_offset : byte_offset + numel * itemsize]
            return raw.view(dtype).reshape(sizes)

        def all_gather_into_tensor(gathered, local, group=None):
            gathered.copy_(local.unsqueeze(0).expand(gathered.shape[0], -1))

        def all_reduce(tensor, op=None, group=None):
            if max_remote_peers is not None:
                tensor.fill_(max_remote_peers)

        fake_cuda = SimpleNamespace(
            synchronize=lambda *_args: None,
            Stream=_FakeCudaStream,
            current_stream=lambda _device=None: _FakeCudaStream(),
            stream=lambda _stream: SimpleNamespace(
                __enter__=lambda self: None, __exit__=lambda self, *exc: False
            ),
        )
        fake_dist = SimpleNamespace(
            ReduceOp=SimpleNamespace(MAX="MAX"),
            all_gather_into_tensor=all_gather_into_tensor,
            all_reduce=all_reduce,
            barrier=lambda group=None: None,
        )
        monkeypatch.setattr(
            symm_window,
            "nvshmem_rt",
            SimpleNamespace(
                NvshmemRuntime=make_runtime,
                tensor_from_ptr=tensor_from_ptr,
            ),
        )
        monkeypatch.setattr(symm_window, "dist", fake_dist)
        monkeypatch.setattr(symm_window.torch, "cuda", fake_cuda)
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
        return SimpleNamespace(
            buffers=buffers,
            allocations=allocations,
            ptrs=ptrs,
            freed=freed,
            put_calls=put_calls,
        )

    return build


def _expected_receive_bytes(
    num_regions: int, ring_depth: int, layout: SlotLayout
) -> int:
    num_flags = num_regions * ring_depth
    return align_bytes(num_flags * SIGNAL_WORD_BYTES) + num_flags * layout.slot_bytes


def test_single_node_with_bridge_skips_outbox_allocation(window_harness):
    harness = window_harness(
        world_size=4,
        mapped_peers={0, 1, 2, 3},
        max_remote_peers=0,
        has_bridge=True,
    )
    layout = _layout()
    window = symm_window.SymmWindow(
        num_regions=2,
        ring_depth=2,
        layout=layout,
        payload_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        group=SimpleNamespace(),
        rank=0,
        world_size=4,
        wait_timeout_s=5.0,
        send_peers=(2, 3),
    )

    expected_receive = _expected_receive_bytes(2, 2, layout)
    assert window.remote_peers == frozenset()
    assert window.outbox_bytes == 0
    assert window.total_bytes == expected_receive
    assert window.local_bytes_view().numel() == expected_receive
    assert harness.allocations == [expected_receive]

    window.close()
    assert harness.freed == [harness.ptrs[0]]


def test_multinode_4a4f_sizes_outboxes_per_remote_peer_and_ring(window_harness):
    harness = window_harness(
        world_size=8,
        mapped_peers={0, 1, 2, 3},
        max_remote_peers=4,
        has_bridge=True,
    )
    layout = _layout()
    window = symm_window.SymmWindow(
        num_regions=4,
        ring_depth=2,
        layout=layout,
        payload_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        group=SimpleNamespace(),
        rank=1,
        world_size=8,
        wait_timeout_s=5.0,
        send_peers=range(4, 8),
    )

    expected_receive = _expected_receive_bytes(4, 2, layout)
    expected_outbox = 4 * 2 * layout.slot_bytes
    assert window.remote_peers == frozenset({4, 5, 6, 7})
    assert window.outbox_bytes == expected_outbox
    assert window.total_bytes == expected_receive + expected_outbox
    assert harness.allocations == [expected_receive, expected_outbox]

    window.close()
    assert harness.freed == [harness.ptrs[1], harness.ptrs[0]]


def test_remote_write_slot_packs_compact_outbox_and_targets_region_ring(
    window_harness,
):
    harness = window_harness(
        world_size=8,
        mapped_peers={0, 1, 2, 3},
        max_remote_peers=4,
        has_bridge=True,
    )
    layout = _layout()
    window = symm_window.SymmWindow(
        num_regions=4,
        ring_depth=2,
        layout=layout,
        payload_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        group=SimpleNamespace(),
        rank=2,
        world_size=8,
        wait_timeout_s=5.0,
        send_peers=range(4, 8),
    )
    header = symm_window.encode_header(
        layout,
        layer_idx=3,
        num_tokens=2,
        routed_tokens=3,
        flags=0,
        expert_counts=[1, 0, 2, 0],
        segment_start=1,
    )
    routed_x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    window.write_slot(
        peer=5,
        region=2,
        ring=1,
        header=header,
        expand_idx=torch.tensor([0, 1, 0], dtype=torch.int32),
        weights=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
        routed_x=routed_x,
        shared_x=torch.full((1, 4), 7.0, dtype=torch.bfloat16),
        flag_value=99,
    )

    assert len(harness.put_calls) == 1
    dest, src, fields, signal, _seq_ptr, value, pe, _stream = harness.put_calls[0]
    receive_ptr, outbox_ptr = harness.ptrs
    flag_bytes = align_bytes(4 * 2 * SIGNAL_WORD_BYTES)
    assert dest == receive_ptr + flag_bytes + (2 * 2 + 1) * layout.slot_bytes
    expected_src_off = (1 * 2 + 1) * layout.slot_bytes
    assert src == outbox_ptr + expected_src_off
    assert signal == receive_ptr + (2 * 2 + 1) * SIGNAL_WORD_BYTES
    assert (value, pe) == (99, 5)
    assert fields[0] == (layout.header_off, layout.header_words * 4)

    outbox_buf = harness.buffers[outbox_ptr]
    decoded = decode_header_words(
        outbox_buf[expected_src_off : expected_src_off + layout.header_words * 4]
        .view(torch.int32)
        .tolist(),
        layout,
    )
    assert (decoded.layer_idx, decoded.num_tokens, decoded.routed_tokens) == (3, 2, 3)
    assert decoded.expert_counts == (1, 0, 2, 0)
    packed_routed = (
        outbox_buf[
            expected_src_off + layout.routed_x_off : expected_src_off
            + layout.routed_x_off
            + 2 * 4 * 2
        ]
        .view(torch.bfloat16)
        .reshape(2, 4)
    )
    assert torch.equal(packed_routed, routed_x)


def test_outbox_malloc_failure_frees_receive_window(window_harness):
    harness = window_harness(
        world_size=4,
        mapped_peers={0, 1},
        max_remote_peers=2,
        has_bridge=True,
        fail_on_alloc=2,
    )
    with pytest.raises(RuntimeError, match="nvshmem_malloc"):
        symm_window.SymmWindow(
            num_regions=2,
            ring_depth=1,
            layout=_layout(),
            payload_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            group=SimpleNamespace(),
            rank=0,
            world_size=4,
            wait_timeout_s=5.0,
            send_peers=(2, 3),
        )
    assert harness.freed == [harness.ptrs[0]]


def test_unconfigured_send_peers_are_rejected(window_harness):
    window_harness(
        world_size=4,
        mapped_peers={0, 1, 2, 3},
        max_remote_peers=0,
        has_bridge=True,
    )
    layout = _layout()
    with pytest.raises(ValueError, match="send peer 4 is outside"):
        symm_window.SymmWindow(
            num_regions=2,
            ring_depth=1,
            layout=layout,
            payload_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            group=SimpleNamespace(),
            rank=0,
            world_size=4,
            wait_timeout_s=5.0,
            send_peers=(4,),
        )

    window = symm_window.SymmWindow(
        num_regions=2,
        ring_depth=1,
        layout=layout,
        payload_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        group=SimpleNamespace(),
        rank=0,
        world_size=4,
        wait_timeout_s=5.0,
        send_peers=(2, 3),
    )
    with pytest.raises(ValueError, match="not a configured send peer"):
        window.write_slot(
            peer=1,
            region=0,
            ring=0,
            header=None,
            expand_idx=None,
            weights=None,
            routed_x=None,
            shared_x=None,
            flag_value=1,
        )
    window.close()
