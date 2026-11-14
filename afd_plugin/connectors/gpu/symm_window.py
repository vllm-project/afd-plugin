# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Symmetric-memory window used by the async GPU AFD connector.

The window mirrors the CAM shared-window substrate: every rank allocates an
identically sized symmetric buffer, senders write one-sided into the receiver's
buffer, and arrival is announced by a magic-stamped flag word that the receiver
polls. Only the allocation is collective; once it is done a sender never needs
the receiver to participate.

Layout of one window::

    [uint64 signals | receive slots | symmetric per-peer/ring outboxes]

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

Mapped peers receive direct writes followed by a 64-bit signal. Remote peers
receive one contiguous slot via NVSHMEM put-with-signal from a symmetric
outbox, after local dtype conversion. The CUDA kernel reads the dispatch
counter on every graph replay and completes RMA before the outbox is reused.
NVSHMEM wait/test supplies the corresponding receive visibility guarantee.

The receive side reads flags and headers on its own stream. What it reads was
produced by a peer, never by anything this rank queued, so the read needs no
ordering against local compute -- and staying off the compute stream is what
lets an arrival be noticed while the previous kernel is still running.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from afd_plugin.connectors.gpu import cuda_rt, nvshmem_rt
from afd_plugin.connectors.gpu.async_gpu_protocol import (
    FLAG_EMPTY,
    HEADER_FIXED_WORDS,
    HEADER_MAGIC,
    HEADER_VERSION,
    SIGNAL_WORD_BYTES,
    SlotHeader,
    SlotLayout,
    align_bytes,
    decode_header_words,
    encode_header_words,
)

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


def fill_header_prefix(
    headers: torch.Tensor, *, layer_idx: int, num_tokens: int, flags: int
) -> None:
    headers[:, 0] = HEADER_MAGIC
    headers[:, 1] = HEADER_VERSION
    headers[:, 2] = layer_idx
    headers[:, 3] = num_tokens
    headers[:, 4] = flags


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
    header = SlotHeader(
        layer_idx, num_tokens, routed_tokens, flags, segment_start, tuple(expert_counts)
    )
    return torch.tensor(encode_header_words(header, layout), dtype=torch.int32)


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
        num_stages: int = 1,
    ) -> None:
        if num_regions <= 0 or ring_depth <= 0:
            raise ValueError("num_regions and ring_depth must be positive")
        self.num_regions = num_regions
        self.ring_depth = ring_depth
        self.layout = layout
        self.payload_dtype = payload_dtype
        self.device = device
        self.rank = rank
        self.num_stages = num_stages

        self.num_flags = num_regions * ring_depth
        self._flag_bytes = align_bytes(self.num_flags * SIGNAL_WORD_BYTES)
        self._receive_bytes = self._flag_bytes + self.num_flags * layout.slot_bytes
        self._has_bridge = (
            importlib.util.find_spec("afd_plugin._C_gpu_async") is not None
        )
        # One outbox per (peer, region, ring). The owning stream cannot repack
        # it until its blocking put/quiet kernel has completed. All PEs reserve
        # the same geometry, even when only some peers require RMA.
        self.total_bytes = self._receive_bytes
        if self._has_bridge:
            self.total_bytes += world_size * self.num_flags * layout.slot_bytes

        # A sender writes into a peer's window using its own idea of the slot
        # geometry, and the receiver only ever checks magic and version -- so a
        # configuration difference between the two role deployments is invisible
        # at the wire. Unequal heaps fault inside the transport; equal heaps with
        # different field offsets read a header out of the wrong words, and a
        # smaller capacity silently clamps a partial read. Compare the geometry
        # once, here, where the parameter that differs can still be named.
        self._validate_uniform_geometry(group, world_size)

        self.runtime = nvshmem_rt.NvshmemRuntime(group, rank, world_size)
        self.driver = cuda_rt.CudaStreamMemoryOps()
        self._base = self.runtime.malloc(self.total_bytes)
        # Peer mappings are stable for the life of the allocation, so resolve
        # them once instead of per transfer.
        mappings = {
            pe: (self._base if pe == rank else self.runtime.peer_ptr(self._base, pe))
            for pe in range(world_size)
        }
        self.remote_peers = frozenset(
            pe for pe, pointer in mappings.items() if pointer is None
        )
        if self.remote_peers and not self._has_bridge:
            raise RuntimeError(
                "Remote GPU peers require the CUDA/NVSHMEM extension. Reinstall "
                "with AFD_BUILD_GPU_ASYNC_OPS=1 (see csrc/gpu/README.md)."
            )
        # Remote views refer to our symmetric outboxes; only the final put uses
        # the target PE. Never pass a mapped peer VA as an NVSHMEM RMA target.
        self._peer_base: dict[int, int] = {
            pe: pointer
            if pointer is not None
            else (
                self._base
                + self._receive_bytes
                + pe * self.num_flags * layout.slot_bytes
                - self._flag_bytes
            )
            for pe, pointer in mappings.items()
        }
        self.local_bytes_view().zero_()
        torch.cuda.synchronize()
        # The collective malloc orders the allocation, not this reset. A peer
        # that returned from its own malloc first can already be writing its
        # first dispatch into this window while the zero above is still queued,
        # and the zero would erase the arriving flag -- leaving that peer
        # waiting on a reply for a slot this rank never saw. Nobody may write
        # until every rank has finished resetting, so hold the group here.
        dist.barrier(group=group)
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
            dtype=torch.int64,
            device=device,
        )
        self._flag_observations = torch.zeros(
            self.num_flags, dtype=torch.int64, device=device
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
        self._flag_host = torch.zeros(self.num_flags, dtype=torch.int64).pin_memory()
        self._seen = [FLAG_EMPTY] * self.num_flags
        self._poll_cursor = 0
        self._closed = False
        # Polls read flags and headers that a *peer* wrote into our window, so
        # they depend on nothing this rank has queued. On the compute stream the
        # poll would still queue behind our own previous kernel, which serializes
        # every receive against the compute it is supposed to overlap with.
        self._poll_stream = torch.cuda.Stream(device=device)
        self._poll_stream.wait_stream(torch.cuda.current_stream(device))
        self._warm_views()

    def _validate_uniform_geometry(
        self,
        group: ProcessGroup,
        world_size: int,
    ) -> None:
        """Raise unless every rank in the group agrees on the slot geometry.

        Named in field order so the error can point at the one parameter that
        differs rather than at a byte count nobody configured directly.
        """
        layout = self.layout
        mine = {
            "header_version": HEADER_VERSION,
            "signal_bytes": SIGNAL_WORD_BYTES,
            "gpu_extension": int(self._has_bridge),
            "num_regions": self.num_regions,
            "num_stages": self.num_stages,
            "ring_depth": self.ring_depth,
            "header_words": layout.header_words,
            "partial_cap": layout.partial_cap,
            "token_cap": layout.token_cap,
            "shared_cap": layout.shared_cap,
            "hidden_size": layout.hidden_size,
            "payload_itemsize": layout.payload_itemsize,
            "payload_dtype": (torch.float16, torch.bfloat16, torch.float32).index(
                self.payload_dtype
            ),
            "slot_bytes": layout.slot_bytes,
            "total_bytes": self.total_bytes,
        }
        names = tuple(mine)
        local = torch.tensor(
            [mine[name] for name in names],
            dtype=torch.int64,
            device=self.device,
        )
        gathered = torch.empty(
            (world_size, local.numel()),
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_gather_into_tensor(gathered, local, group=group)
        rows = gathered.cpu()
        for column, name in enumerate(names):
            values = rows[:, column].tolist()
            if len(set(values)) == 1:
                continue
            raise ValueError(
                f"AFD symmetric window {name} differs across the group: "
                f"rank {self.rank} has {mine[name]}, group has {values}. "
                "Both roles must be launched with the same window geometry.",
            )

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
            if peer not in self.remote_peers:
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
                byte_offset=flag_idx * SIGNAL_WORD_BYTES,
                sizes=(1,),
                dtype=torch.int64,
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
        """Pack one slot and publish it to ``peer`` with an ordered signal.

        The flag copy is issued last on the same stream, so a peer that observes
        the flag also observes the payload.

        ``routed_rows``/``shared_rows`` are row indices into ``routed_x`` /
        ``shared_x``: the gather then lands directly in the peer's window.
        Mapped peers receive the gather directly; remote peers receive a
        contiguous copy of the local symmetric outbox after all fields finish.

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
        if (
            partial_count > layout.partial_cap
            or _row_count(weights, None) > layout.partial_cap
        ):
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
        if peer in self.remote_peers:
            bridge = self.runtime.bridge
            assert bridge is not None
            byte_off = self._slot_byte_off(region, ring)
            bridge.put_signal(
                self._base + byte_off,
                self._peer_base[peer] + byte_off,
                layout.slot_bytes,
                self._base + flag_idx * SIGNAL_WORD_BYTES,
                flag_value.data_ptr() if isinstance(flag_value, torch.Tensor) else 0,
                0 if isinstance(flag_value, torch.Tensor) else flag_value,
                self.runtime.pe_by_rank[peer],
                torch.cuda.current_stream(self.device).cuda_stream,
            )
            return
        flag_view = self._flag_view(peer, flag_idx)
        if isinstance(flag_value, torch.Tensor):
            flag_view.copy_(flag_value)
        else:
            flag_view.fill_(flag_value)

    # ------------------------------------------------------------------
    # Receive side.
    # ------------------------------------------------------------------

    def wait(self, *, timeout_s: float | None = None) -> ArrivedSlot | None:
        """Poll on the dedicated receive stream until arrival or timeout.

        shortcut: host polling uses copy-engine bandwidth. Replace it with a
        device arrival queue if transport polling dominates measured runtime.
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
        flag_idx = region * self.ring_depth + ring
        if self.runtime.bridge is not None:
            self.runtime.bridge.wait_signal(
                self._base + flag_idx * SIGNAL_WORD_BYTES,
                value,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
            return
        self.driver.stream_wait_value32(
            torch.cuda.current_stream(self.device).cuda_stream,
            self._base + flag_idx * SIGNAL_WORD_BYTES,
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
            if self.runtime.bridge is None:
                flags = self._flags_local
            else:
                self.runtime.bridge.fetch_signals(
                    self._base,
                    self._flag_observations.data_ptr(),
                    self.num_flags,
                    self._poll_stream.cuda_stream,
                )
                flags = self._flag_observations
            self._flag_host.copy_(flags, non_blocking=False)
        host = self._flag_host.tolist()
        for offset in range(self.num_flags):
            idx = (self._poll_cursor + offset) % self.num_flags
            if host[idx] != self._seen[idx]:
                self._seen[idx] = host[idx]
                self._poll_cursor = (idx + 1) % self.num_flags
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
        return decode_header_words(self._header_recv.tolist(), self.layout)

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
        """Collective release, after the connector's producer/receiver handshake."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self._view_cache.clear()
        self._flag_cache.clear()
        self._peer_base.clear()
        self.runtime.close(self._base)
        self._closed = True
