# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Symmetric-memory window used by the async GPU AFD connector.

The window mirrors the CAM shared-window substrate: every rank allocates an
identically sized symmetric buffer, senders write one-sided into the receiver's
buffer, and arrival is announced by a magic-stamped flag word that the receiver
polls. Only the allocation is collective; once it is done a sender never needs
the receiver to participate.

Layout of one window::

    [flag[num_regions * ring_depth] | slot(0, 0) | slot(0, 1) | ...]

``slot(region, ring)`` holds a fixed header followed by the routed/shared
payloads. Dispatch (A -> F) and combine (F -> A) use the same slot layout, so a
single spec sizes both directions. Shared-expert tokens are a contiguous range
of the batch, so the slot carries only their count -- the range is implied by
which FFN rank the slot belongs to.

Every slot is written at capacity: the payload carries the whole batch
(``num_tokens`` rows) and the ``expand_idx``/``weights`` arrays carry every
partial, with ``segment_start``/``routed_tokens`` in the header telling a
destination which run of partials is its own. Sizing the writes by the routing
instead would save a fraction of the bytes and cost a device-to-host readback
per layer to learn the sizes, which measured far more than the bytes are worth:
with ``topk`` slots spread over ``ffn_size`` destinations a token misses a given
destination only ``(1 - 1/ffn_size) ** topk`` of the time, so at 2A2F and
``topk=6`` the capacity payload is about 1.6% larger than the exact one.

``expand_idx`` and ``weights`` are 4 bytes a row against the payload's
``hidden_size`` elements, so shipping them whole to every peer costs well under
a percent of the slot.

``shared_x`` is the exception, and is sized ``shared_cap`` rather than
``token_cap``: shared-expert tokens are split across the FFN ranks, so a slot
never holds more than ``ceil(token_cap / ffn_size)`` of them in either
direction, and a model with no shared experts does not need the field at all.
Sizing it like ``routed_x`` reserved a second whole-batch payload per slot that
nothing could ever fill.

Flag words are written *after* the payload on the same stream. Same-stream
device-to-device copies complete in issue order, so a visible flag implies a
complete payload. That holds for NVLink-mapped peer memory; a cross-node
transport would need an explicit fence here.

The receive side reads flags and headers on its own stream. What it reads was
produced by a peer, never by anything this rank queued, so the read needs no
ordering against local compute -- and staying off the compute stream is what
lets an arrival be noticed while the previous kernel is still running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from afd_plugin.connectors.gpu import cuda_rt, nvshmem_rt

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

# Header words shared by dispatch and combine, followed by expert_counts.
#
# Only what a receiver cannot work out for itself is on the wire. A slot's
# region already names the sender, its ring already names the stage, and the
# shared-token split is the same function of ``num_tokens`` on both sides, so
# carrying any of them would be shipping a value the reader could compute --
# and, since a CUDA graph records this prefix once per capture, shipping it on
# every replay forever. What is left is the layer to run, the batch size, and
# the shutdown bit, none of which is derivable.
HEADER_MAGIC = 0x41464447  # "AFDG"
# Version 4 dropped seq, src_role_rank, stage_idx, shared_tokens, topk and
# echo_seq. Both roles must be on the same version; the magic/version check in
# decode_header is what catches a mismatch.
HEADER_VERSION = 4
_H_MAGIC = 0
_H_VERSION = 1
_H_LAYER_IDX = 2
_H_NUM_TOKENS = 3
_H_FLAGS = 4
# Everything from here on is routing, which only the device knows. Keeping those
# words in one contiguous tail is what lets a sender fill the host-known prefix
# with a single copy and the rest straight from the plan, so no dispatch ever
# reads the routing back to the host.
HEADER_HOST_WORDS = 5
H_ROUTED_TOKENS = 5  # partials: one per (token, topk slot) landing here
H_SEGMENT_START = 6  # where those partials begin in the shipped index arrays
HEADER_FIXED_WORDS = 7

FLAG_EMPTY = 0
FLAG_SHUTDOWN_BIT = 1 << 1

# Every field starts on this boundary so a byte offset stays divisible by the
# element size of whichever dtype views it.
_FIELD_ALIGN = 256


def _align(offset: int) -> int:
    return (offset + _FIELD_ALIGN - 1) // _FIELD_ALIGN * _FIELD_ALIGN


@dataclass(frozen=True, slots=True)
class SlotLayout:
    """Byte offsets and element counts for the fields inside one slot."""

    header_words: int
    partial_cap: int
    token_cap: int
    shared_cap: int
    hidden_size: int
    payload_itemsize: int

    header_off: int
    expand_idx_off: int
    weights_off: int
    routed_x_off: int
    shared_x_off: int
    slot_bytes: int

    @classmethod
    def build(
        cls,
        *,
        expert_per_rank: int,
        partial_cap: int,
        token_cap: int,
        shared_cap: int,
        hidden_size: int,
        payload_itemsize: int,
    ) -> SlotLayout:
        header_words = HEADER_FIXED_WORDS + expert_per_rank
        header_off = 0
        expand_idx_off = _align(header_off + header_words * 4)
        weights_off = _align(expand_idx_off + partial_cap * 4)
        routed_x_off = _align(weights_off + partial_cap * 4)
        shared_x_off = _align(routed_x_off + token_cap * hidden_size * payload_itemsize)
        slot_bytes = _align(shared_x_off + shared_cap * hidden_size * payload_itemsize)
        return cls(
            header_words=header_words,
            partial_cap=partial_cap,
            token_cap=token_cap,
            shared_cap=shared_cap,
            hidden_size=hidden_size,
            payload_itemsize=payload_itemsize,
            header_off=header_off,
            expand_idx_off=expand_idx_off,
            weights_off=weights_off,
            routed_x_off=routed_x_off,
            shared_x_off=shared_x_off,
            slot_bytes=slot_bytes,
        )


def encode_header_host_words(
    *,
    layer_idx: int,
    num_tokens: int,
    flags: int,
) -> np.ndarray:
    """Build the host-known prefix of one slot header as int32.

    The routing tail -- ``routed_tokens``, ``segment_start`` and the per-expert
    counts -- is not here: a dispatch fills it straight from the plan on the
    device, which is what keeps the send path free of a readback.

    Every word here is fixed by the dispatch shape, so a sender can build this
    once per ``(layer, stage, token count)`` and keep it rather than rebuild it
    per transfer.
    """
    words = np.empty(HEADER_HOST_WORDS, dtype=np.int32)
    words[:] = (HEADER_MAGIC, HEADER_VERSION, layer_idx, num_tokens, flags)
    return words


def fill_header_prefix(
    headers: torch.Tensor,
    *,
    layer_idx: int,
    num_tokens: int,
    flags: int,
) -> None:
    """Write the host-known prefix into device-resident headers, in place.

    Five scalar fills rather than a staged copy, because every one of them is a
    constant and none of them needs the host. The copy this replaces was
    pageable, which makes it synchronous: it blocked the calling thread until
    its stream drained, and a blocking CUDA call inside a ubatch thread
    deadlocks against the peer thread it has not yielded to yet.

    Runs once per dispatch shape, so five tiny kernels cost nothing.
    """
    headers[:, _H_MAGIC] = HEADER_MAGIC
    headers[:, _H_VERSION] = HEADER_VERSION
    headers[:, _H_LAYER_IDX] = layer_idx
    headers[:, _H_NUM_TOKENS] = num_tokens
    headers[:, _H_FLAGS] = flags


def encode_header(
    layout: SlotLayout,
    *,
    layer_idx: int,
    num_tokens: int,
    routed_tokens: int,
    flags: int,
    expert_counts: list[int],
    segment_start: int = 0,
) -> torch.Tensor:
    """Build a whole slot header as a CPU int32 tensor.

    Used where the routing is already on the host: the FFN reply, which decoded
    it from the dispatch it answers, and the shutdown announcement, which has no
    routing at all.
    """
    expert_per_rank = layout.header_words - HEADER_FIXED_WORDS
    if len(expert_counts) != expert_per_rank:
        raise ValueError(
            f"expert_counts must have {expert_per_rank} entries, "
            f"got {len(expert_counts)}",
        )
    header = np.empty(layout.header_words, dtype=np.int32)
    header[:HEADER_HOST_WORDS] = encode_header_host_words(
        layer_idx=layer_idx,
        num_tokens=num_tokens,
        flags=flags,
    )
    header[H_ROUTED_TOKENS] = routed_tokens
    header[H_SEGMENT_START] = segment_start
    if expert_per_rank:
        header[HEADER_FIXED_WORDS:] = expert_counts
    # torch.from_numpy shares the buffer, so the wrap is free.
    return torch.from_numpy(header)


@dataclass(frozen=True, slots=True)
class SlotHeader:
    """Decoded slot header.

    Carries only the wire fields. A receiver that needs the sender's rank, the
    stage, or its shared-token count derives them from the slot it arrived in;
    see ``GpuAsyncAFDConnector.recv_attn_output``.
    """

    layer_idx: int
    num_tokens: int
    routed_tokens: int
    flags: int
    segment_start: int
    expert_counts: list[int]

    @property
    def is_shutdown(self) -> bool:
        return bool(self.flags & FLAG_SHUTDOWN_BIT)


def decode_header(header: torch.Tensor) -> SlotHeader:
    """Decode a CPU int32 header tensor, validating magic and version."""
    values = header.tolist()
    if values[_H_MAGIC] != HEADER_MAGIC:
        raise RuntimeError(
            f"AFD async GPU header magic mismatch: got {values[_H_MAGIC]:#x}, "
            f"expected {HEADER_MAGIC:#x}",
        )
    if values[_H_VERSION] != HEADER_VERSION:
        raise RuntimeError(
            f"AFD async GPU header version {values[_H_VERSION]} is not supported "
            f"(expected {HEADER_VERSION})",
        )
    return SlotHeader(
        layer_idx=values[_H_LAYER_IDX],
        num_tokens=values[_H_NUM_TOKENS],
        routed_tokens=values[H_ROUTED_TOKENS],
        flags=values[_H_FLAGS],
        segment_start=values[H_SEGMENT_START],
        expert_counts=values[HEADER_FIXED_WORDS:],
    )


@dataclass(slots=True)
class ArrivedSlot:
    """One arrival found by ``SymmWindow.poll``."""

    region: int
    ring: int
    header: SlotHeader


def _row_count(src: torch.Tensor | None, rows: torch.Tensor | None) -> int:
    """Rows a payload field will occupy, gathered or copied whole."""
    if src is None:
        return 0
    return int(src.shape[0]) if rows is None else int(rows.numel())


class SymmWindow:
    """Symmetric receive window plus the one-sided writes that fill peers'."""

    def __init__(
        self,
        *,
        num_regions: int,
        ring_depth: int,
        layout: SlotLayout,
        payload_dtype: torch.dtype,
        device: torch.device,
        group: ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        if num_regions <= 0 or ring_depth <= 0:
            raise ValueError("num_regions and ring_depth must be positive")
        self.num_regions = num_regions
        self.ring_depth = ring_depth
        self.layout = layout
        self.payload_dtype = payload_dtype
        self.device = device
        self.rank = rank

        self.num_flags = num_regions * ring_depth
        self._flag_bytes = _align(self.num_flags * 4)
        self.total_bytes = self._flag_bytes + self.num_flags * layout.slot_bytes

        nvshmem_rt.init(group, rank, world_size)
        self._base = nvshmem_rt.malloc(self.total_bytes)
        # Peer mappings are stable for the life of the allocation, so resolve
        # them once instead of per transfer.
        self._peer_base = {
            pe: (self._base if pe == rank else nvshmem_rt.peer_ptr(self._base, pe))
            for pe in range(world_size)
        }
        self.local_bytes_view().zero_()
        torch.cuda.synchronize()

        # The layout is static, so every window view is built once and then
        # sliced. Rebuilding them per transfer meant a __cuda_array_interface__
        # import on every field of every message, which dominated the data path.
        self._view_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._flag_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._flags_local = nvshmem_rt.tensor_from_ptr(
            self._base,
            byte_offset=0,
            sizes=(self.num_flags,),
            dtype=torch.int32,
            device=device,
        )

        # Pinned staging keeps header transfers asynchronous; a pageable source
        # forces a blocking copy, and there is one header per peer per layer.
        # One row per (peer, ring): a single shared row would be overwritten on
        # the host by the next peer in the send loop while its own asynchronous
        # copy was still in flight, delivering another peer's token counts.
        self._header_send = torch.zeros(
            (world_size, ring_depth, layout.header_words),
            dtype=torch.int32,
        ).pin_memory()
        self._header_recv = torch.zeros(
            layout.header_words,
            dtype=torch.int32,
        ).pin_memory()
        # Host mirror of the local flag array; one D2H per poll refreshes it.
        self._flag_host = torch.zeros(self.num_flags, dtype=torch.int32).pin_memory()
        self._seen = [FLAG_EMPTY] * self.num_flags
        # Polls read flags and headers that a *peer* wrote into our window, so
        # they depend on nothing this rank has queued. On the compute stream the
        # poll would still queue behind our own previous kernel, which serializes
        # every receive against the compute it is supposed to overlap with.
        self._poll_stream = torch.cuda.Stream(device=device)
        self._warm_views()

    def _warm_views(self) -> None:
        """Import every window pointer now, so the data path never does.

        Building a view goes through ``__cuda_array_interface__``, which is a
        CUDA runtime call on the host. Left lazy, the first use of each
        ``(peer, region, ring, field)`` pays it on the layer path -- and under
        vLLM's ubatching that is fatal rather than slow: a host-blocking CUDA
        call in a ubatch thread never returns if the peer thread it has not
        yielded to yet is the one that would let its stream drain, and the two
        deadlock.

        The set is small and fixed -- world size times regions times rings,
        times the five slot fields plus one flag each -- so importing it up
        front costs a moment at startup and removes the class of hang entirely.
        """
        layout = self.layout
        field_offs = (
            layout.header_off,
            layout.expand_idx_off,
            layout.weights_off,
            layout.routed_x_off,
            layout.shared_x_off,
        )
        for peer in range(len(self._peer_base)):
            for flag_idx in range(self.num_flags):
                self._flag_view(peer, flag_idx)
            for region in range(self.num_regions):
                for ring in range(self.ring_depth):
                    for field_off in field_offs:
                        self._capacity_view(peer, region, ring, field_off)

    def local_bytes_view(self) -> torch.Tensor:
        return nvshmem_rt.tensor_from_ptr(
            self._base,
            byte_offset=0,
            sizes=(self.total_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )

    def _slot_byte_off(self, region: int, ring: int) -> int:
        return (
            self._flag_bytes
            + (region * self.ring_depth + ring) * self.layout.slot_bytes
        )

    def _capacity_view(
        self,
        peer: int,
        region: int,
        ring: int,
        field_off: int,
    ) -> torch.Tensor:
        """Return the cached full-capacity view of one slot field."""
        key = (peer, region, ring, field_off)
        view = self._view_cache.get(key)
        if view is not None:
            return view

        layout = self.layout
        hidden = layout.hidden_size
        sizes: tuple[int, ...]
        if field_off == layout.header_off:
            sizes, dtype = (layout.header_words,), torch.int32
        elif field_off == layout.expand_idx_off:
            sizes, dtype = (layout.partial_cap,), torch.int32
        elif field_off == layout.weights_off:
            sizes, dtype = (layout.partial_cap,), torch.float32
        elif field_off == layout.routed_x_off:
            sizes, dtype = (layout.token_cap, hidden), self.payload_dtype
        elif field_off == layout.shared_x_off:
            sizes, dtype = (layout.shared_cap, hidden), self.payload_dtype
        else:
            raise ValueError(f"unknown slot field offset {field_off}")

        view = nvshmem_rt.tensor_from_ptr(
            self._peer_base[peer],
            byte_offset=self._slot_byte_off(region, ring) + field_off,
            sizes=sizes,
            dtype=dtype,
            device=self.device,
        )
        self._view_cache[key] = view
        return view

    def _view(
        self,
        peer: int,
        region: int,
        ring: int,
        field_off: int,
        sizes: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Slicing a cached capacity view costs no CUDA calls, unlike importing
        # a fresh pointer for every field of every message.
        return self._capacity_view(peer, region, ring, field_off)[: sizes[0]]

    def _flag_view(self, peer: int, flag_idx: int) -> torch.Tensor:
        key = (peer, flag_idx)
        view = self._flag_cache.get(key)
        if view is None:
            view = nvshmem_rt.tensor_from_ptr(
                self._peer_base[peer],
                byte_offset=flag_idx * 4,
                sizes=(1,),
                dtype=torch.int32,
                device=self.device,
            )
            self._flag_cache[key] = view
        return view

    # ------------------------------------------------------------------
    # Send side: every write targets ``peer``'s window, one-sided.
    # ------------------------------------------------------------------

    def write_slot(
        self,
        *,
        peer: int,
        region: int,
        ring: int,
        header: torch.Tensor | None,
        expand_idx: torch.Tensor | None,
        weights: torch.Tensor | None,
        routed_x: torch.Tensor | None,
        shared_x: torch.Tensor | None,
        flag_value: int | torch.Tensor,
        routed_rows: torch.Tensor | None = None,
        shared_rows: torch.Tensor | None = None,
    ) -> None:
        """Write one slot into ``peer``'s window, then stamp its flag.

        The flag copy is issued last on the same stream, so a peer that observes
        the flag also observes the payload.

        ``routed_rows``/``shared_rows`` are row indices into ``routed_x`` /
        ``shared_x``: the gather then lands directly in the peer's window.
        Materializing the gathered rows locally first would write and re-read
        every payload byte for nothing.

        ``flag_value`` is what the flag is stamped with. A dispatch stamps a
        rising sequence number, which is how the receiver's poll notices it; a
        reply stamps a constant ready marker, so the rank waiting for it knows
        the value to wait for before it arrives -- that is what lets the wait
        happen on a stream rather than on the host. It is explicit because the
        header no longer carries a sequence number to fall back on, and reading
        one off a device-resident header would synchronize.

        A device tensor is accepted there as well, and is what a CUDA graph
        needs: a Python int becomes an immediate baked into the captured kernel,
        so every replay would stamp the same number, while a one-element int32
        tensor the graph itself advances stamps a fresh one each time.

        ``header`` may live on the device, which is how a dispatch ships routing
        it never read back. It may also be ``None``, which is how a reply ships
        nothing but rows: the rank waiting for one already knows the shape it
        is going to read, so a header there would be written and never read.
        """
        layout = self.layout
        routed_count = _row_count(routed_x, routed_rows)
        shared_count = _row_count(shared_x, shared_rows)
        partial_count = _row_count(expand_idx, None)
        if routed_count > layout.token_cap:
            raise RuntimeError(
                f"payload rows {routed_count} exceed token_cap {layout.token_cap}",
            )
        if partial_count > layout.partial_cap:
            raise RuntimeError(
                f"partials {partial_count} exceed partial_cap {layout.partial_cap}",
            )
        if shared_count > layout.shared_cap:
            raise RuntimeError(
                f"shared tokens {shared_count} exceed shared_cap {layout.shared_cap}",
            )

        if header is not None:
            header_view = self._capacity_view(peer, region, ring, layout.header_off)
            if header.is_cuda:
                header_view.copy_(header, non_blocking=True)
            else:
                # Stage through pinned memory so the copy is asynchronous: a
                # pageable source would force a blocking transfer, and there is
                # one header per peer per layer.
                staging = self._header_send[peer][ring]
                staging.copy_(header)
                header_view.copy_(staging, non_blocking=True)

        def write_field(
            field_off: int,
            trailing_sizes: tuple[int, ...],
            dtype: torch.dtype,
            src: torch.Tensor | None,
            count: int,
            rows: torch.Tensor | None = None,
        ) -> None:
            if src is None or not count:
                return
            view = self._view(
                peer,
                region,
                ring,
                field_off,
                (count, *trailing_sizes),
                dtype,
            )
            if rows is None:
                view.copy_(src, non_blocking=True)
            else:
                torch.index_select(src, 0, rows, out=view)

        write_field(
            layout.expand_idx_off,
            (),
            torch.int32,
            expand_idx,
            partial_count,
        )
        write_field(
            layout.weights_off,
            (),
            torch.float32,
            weights,
            _row_count(weights, None),
        )
        write_field(
            layout.routed_x_off,
            (layout.hidden_size,),
            self.payload_dtype,
            routed_x,
            routed_count,
            routed_rows,
        )
        write_field(
            layout.shared_x_off,
            (layout.hidden_size,),
            self.payload_dtype,
            shared_x,
            shared_count,
            shared_rows,
        )

        flag_idx = region * self.ring_depth + ring
        flag_view = self._flag_view(peer, flag_idx)
        if isinstance(flag_value, torch.Tensor):
            flag_view.copy_(flag_value)
        else:
            flag_view.fill_(flag_value)

    # ------------------------------------------------------------------
    # Receive side.
    # ------------------------------------------------------------------

    def wait(self, *, timeout_s: float | None = None) -> ArrivedSlot | None:
        """Spin until a slot arrives, or ``timeout_s`` passes.

        The spin is hot on purpose. A poll is a D2H plus a stream synchronize,
        about 40us, and a profile shows this loop issuing ~25k of them a second
        per rank -- more copy-engine time than the expert GEMM itself. Backing
        off looks like the obvious fix and is not: every layer is a serialized
        A->F->A round trip, so detection latency lands directly on the critical
        path twice per layer, and on the FFN side it also delays picking up the
        next rank's dispatch. Measured on 2A2F, sleeping between attempts (4 hot
        tries, then 50us doubling to 1ms) made mean TTFT worse at every rate:
        346 -> 431 ms at 32 rps, 1869 -> 3906 ms at 64 rps.

        ponytail: hot spin, and it costs real GPU copy-engine time. The fix is
        not to poll less often but to stop polling from the host: a device-side
        ``wait_any`` kernel spinning on the flag array would notice an arrival
        in ~1us and let the host block on one synchronize. The larger win is to
        take the round trip off the critical path entirely (ubatching), after
        which detection latency stops being paid per layer.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            arrived = self.poll()
            if arrived is not None:
                return arrived
            if deadline is not None and time.monotonic() >= deadline:
                return None

    def stream_wait(self, region: int, ring: int, value: int) -> None:
        """Block the current stream until this slot's flag reaches ``value``.

        The host returns immediately, so whatever it queues next -- the combine
        that consumes this slot, the next layer -- is already on the stream when
        the peer's write lands. Nothing here tells the host that the data
        arrived, so the caller must already know the shapes it is going to read.
        """
        cuda_rt.require_stream_mem_ops(self.device.index)
        flag_idx = region * self.ring_depth + ring
        cuda_rt.stream_wait_value32(
            torch.cuda.current_stream(self.device).cuda_stream,
            self._base + flag_idx * 4,
            value,
        )

    def clear_flag(self, region: int, ring: int) -> None:
        """Put one of this rank's own flags back to ``FLAG_EMPTY``.

        Queued on the compute stream, so it lands after whatever was enqueued to
        read the slot and before whatever is enqueued to reuse it. That ordering
        is the whole point: a stream wait compares against a value baked in when
        the graph was captured, so a peer cannot signal with a fresh number
        every time. It signals with a constant, and this puts the flag back
        below it so the next wait blocks again.

        Only a rank that waits on a flag may clear it. ``poll`` recognizes an
        arrival by the flag differing from what it last saw, which a reset
        would defeat, so a role either polls its flags or stream-waits and
        clears them -- never both.
        """
        self._flags_local[region * self.ring_depth + ring].zero_()

    def poll(self) -> ArrivedSlot | None:
        """Return the first slot whose flag advanced past what we consumed.

        One D2H per call, on the poll stream. Callers should use ``wait``
        instead of spinning on this.
        """
        with torch.cuda.stream(self._poll_stream):
            self._flag_host.copy_(self._flags_local, non_blocking=False)
        host = self._flag_host.tolist()
        for idx in range(self.num_flags):
            if host[idx] != self._seen[idx]:
                self._seen[idx] = host[idx]
                region, ring = divmod(idx, self.ring_depth)
                return ArrivedSlot(
                    region=region,
                    ring=ring,
                    header=self.read_header(region, ring),
                )
        return None

    def read_header(self, region: int, ring: int) -> SlotHeader:
        # Reuse the pinned mirror instead of allocating a fresh host tensor on
        # every arrival, and read it on the poll stream for the same reason the
        # flag is read there.
        with torch.cuda.stream(self._poll_stream):
            self._header_recv.copy_(
                self._capacity_view(self.rank, region, ring, self.layout.header_off),
            )
        return decode_header(self._header_recv)

    def local_expert_counts(self, region: int, ring: int) -> torch.Tensor:
        """Device view of the arrived header's per-expert counts.

        The counts already sit in device memory as the header's trailing words,
        so the grouped GEMM can read them straight from the slot. Rebuilding the
        tensor from the decoded host list would cost one blocking H2D per work
        item.
        """
        header = self._capacity_view(
            self.rank,
            region,
            ring,
            self.layout.header_off,
        )
        return header[HEADER_FIXED_WORDS:]

    def local_expand_idx(
        self,
        region: int,
        ring: int,
        count: int,
        start: int = 0,
    ) -> torch.Tensor:
        """Device view of this destination's run of partial indices.

        Senders ship the whole array, so a destination's own partials start at
        the ``segment_start`` its header carries.
        """
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.expand_idx_off,
            (start + count,),
            torch.int32,
        )[start:]

    def local_weights(
        self,
        region: int,
        ring: int,
        count: int,
        start: int = 0,
    ) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.weights_off,
            (start + count,),
            torch.float32,
        )[start:]

    def local_routed(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.routed_x_off,
            (count, self.layout.hidden_size),
            self.payload_dtype,
        )

    def local_shared(self, region: int, ring: int, count: int) -> torch.Tensor:
        return self._view(
            self.rank,
            region,
            ring,
            self.layout.shared_x_off,
            (count, self.layout.hidden_size),
            self.payload_dtype,
        )

    def close(self) -> None:
        # ponytail: the symmetric allocation is left to process teardown.
        # nvshmem_free is collective, so freeing here would need both roles to
        # shut down in lockstep; add it if windows are ever recreated in-process.
        self._peer_base = {}
