# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-slot Async CAM receive/compute and FIFO-send execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from afd_plugin.connectors.npu.async_cam import AFDAsyncTransferState

if TYPE_CHECKING:
    from afd_plugin.connectors.metadata import AFDA2FTransferPayload
    from afd_plugin.connectors.npu.async_cam import (
        AFDAsyncExtraInfo,
        CAMAsyncAFDConnector,
    )
    from afd_plugin.model_executor.npu.async_cam_layered import AsyncCAMLayeredExecutor


@dataclass
class _Slot:
    ready: torch.npu.Event
    done: torch.npu.Event
    submitted: bool = False
    payload: AFDA2FTransferPayload | None = None
    output: torch.Tensor | None = None
    tensors: list[torch.Tensor] = field(default_factory=list)
    compute_graph: torch.npu.NPUGraph | None = None
    send_graph: torch.npu.NPUGraph | None = None
    pools: tuple | None = None

    def release_transaction(self) -> None:
        self.payload = self.output = None
        self.tensors.clear()


class AsyncFFNTwoStream:
    def __init__(
        self,
        device: torch.device,
        config: AFDAsyncExtraInfo,
        connector: CAMAsyncAFDConnector,
        executor: AsyncCAMLayeredExecutor,
    ) -> None:
        self.connector, self.executor = connector, executor
        self.sequence = self.capture_count = self.replay_count = 0
        self.pending = self.closed = False
        self.failure: Exception | None = None
        self.compute = torch.npu.Stream(device=device)
        self.send = torch.npu.Stream(device=device)
        limits = torch.npu.get_stream_limit(self.compute)
        if (
            config.ffn_compute_aic_num > limits["cube_core_num"]
            or 2 * config.ffn_compute_aic_num > config.ffn_compute_aiv_num
            or config.ffn_compute_aiv_num + config.ffn_combine_aiv_num
            > limits["vector_core_num"]
        ):
            raise ValueError(
                "FFN two-stream core limits exceed the available 1:2 budget"
            )
        torch.npu.set_stream_limit(
            self.compute,
            cube_num=config.ffn_compute_aic_num,
            vector_num=config.ffn_compute_aiv_num,
        )
        torch.npu.set_stream_limit(self.send, vector_num=config.ffn_combine_aiv_num)
        self.compute.wait_stream(torch.npu.current_stream(device))
        self.slots = [_Slot(torch.npu.Event(), torch.npu.Event()) for _ in range(2)]

    def _check_live(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "Async FFN two-stream failed; retain resources"
            ) from self.failure
        if self.closed:
            raise RuntimeError("Async FFN two-stream is closed")

    def _compute(self, slot: _Slot) -> None:
        slot.payload = self.connector.recv_attn_output()
        state = cast(AFDAsyncTransferState, slot.payload.context.states)
        slot.output = self.executor(
            slot.payload.hidden_states,
            state.dynamic_scales,
            state.group_list,
            state.token_nums_rankid_layeridx,
            retained_tensors=slot.tensors,
        )

    def _send(self, slot: _Slot) -> None:
        assert slot.payload is not None and slot.output is not None
        self.connector.send_ffn_output(slot.output, slot.payload.context)

    def capture(self) -> None:
        self._check_live()
        if self.pending or self.capture_count:
            raise RuntimeError("Async FFN two-stream is not ready for capture")
        try:
            for slot in self.slots:
                slot.compute_graph, slot.send_graph = (
                    torch.npu.NPUGraph(),
                    torch.npu.NPUGraph(),
                )
                slot.pools = (
                    torch.npu.graph_pool_handle(),
                    torch.npu.graph_pool_handle(),
                )
                with torch.npu.graph(
                    slot.compute_graph, pool=slot.pools[0], stream=self.compute
                ):
                    self._compute(slot)
                with torch.npu.graph(
                    slot.send_graph, pool=slot.pools[1], stream=self.send
                ):
                    self._send(slot)
                self.capture_count += 2
        except Exception as exc:
            self.failure = exc
            raise
        self.sequence = self.replay_count = 0

    def submit(self, count: int) -> None:
        self._check_live()
        if self.pending or count < 1:
            raise RuntimeError("Async FFN two-stream is not ready for submission")
        self.pending = True  # Even a failed call can leave queued native work.
        try:
            for _ in range(count):
                slot = self.slots[self.sequence % 2]
                captured = slot.compute_graph is not None
                if slot.submitted and not captured:
                    # An idle DR has no startup deadline; retain until CS completes.
                    slot.done.synchronize()
                    slot.release_transaction()
                with torch.npu.stream(self.compute):
                    if slot.submitted:
                        self.compute.wait_event(slot.done)
                    if captured:
                        assert slot.compute_graph is not None
                        slot.compute_graph.replay()
                        self.replay_count += 1
                    else:
                        self._compute(slot)
                    slot.ready.record(self.compute)
                with torch.npu.stream(self.send):
                    self.send.wait_event(slot.ready)
                    if captured:
                        assert slot.send_graph is not None
                        slot.send_graph.replay()
                        self.replay_count += 1
                    else:
                        self._send(slot)
                    slot.done.record(self.send)
                slot.submitted = True
                self.sequence += 1
        except Exception as exc:
            self.failure = exc
            raise

    def complete(
        self, *, wait: Callable[[torch.npu.Event], None] | None = None
    ) -> bool:
        self._check_live()
        if not self.pending:
            raise RuntimeError("Async FFN two-stream has no pending submission")
        try:
            if wait is None:
                # Match the existing bounded-batch completion policy.
                torch.npu.synchronize()
            else:
                # FIFO send completion also covers all preceding compute work.
                wait(self.slots[(self.sequence - 1) % len(self.slots)].done)
        except Exception as exc:
            self.failure = exc
            raise
        self.pending = False
        if not self.capture_count:
            for slot in self.slots:
                slot.release_transaction()
        return True

    def close(self, *, receiver_exited: bool) -> None:
        if self.closed:
            return
        self._check_live()
        if self.pending or not receiver_exited:
            raise RuntimeError(
                "Two-stream resources require receiver exit and device completion"
            )
        for slot in self.slots:
            if slot.compute_graph is not None:
                slot.compute_graph.reset()
            if slot.send_graph is not None:
                slot.send_graph.reset()
            slot.release_transaction()
            slot.compute_graph = slot.send_graph = None
            slot.pools = None
        self.slots.clear()
        del self.compute, self.send, self.connector, self.executor
        self.closed = True
