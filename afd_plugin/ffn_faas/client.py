# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention transport client; model routing/shared experts stay with Attention."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from afd_plugin.ffn_faas.npu import (
    STORE_POLL_SECONDS,
    NativeCompletion,
    NativeDeviceError,
    NPUExecutor,
)
from afd_plugin.ffn_faas.protocol import MAX_GENERATION, Mode, Result, SourceChunkPlan


@dataclass(frozen=True)
class PendingWork:
    sequence: int
    layer: int
    chunks: int
    x: torch.Tensor
    ids: torch.Tensor
    probabilities: torch.Tensor
    exchange: NativeCompletion


class AttentionSession:
    """One ordered A rank, one pending model call, finite transport chunks."""

    def __init__(self, executor: NPUExecutor, *, tp_rows_aligned: bool = False) -> None:
        if executor.is_ffn:
            raise ValueError("AttentionSession requires an Attention executor")
        self.executor = executor
        self.sequence = 0
        self.pending: PendingWork | None = None
        self._closed = False
        self._calls = 0
        self.tp_rows_aligned = tp_rows_aligned
        self._in_model_scope = False
        self._scope_failed = False

    @contextmanager
    def model_execution_scope(self) -> Iterator[None]:
        """Validate all eager model work before its caller can return a result.

        Failure invalidates this session, including speculative Attention/KV
        state. The connector must terminate its worker after bounded cleanup.
        """
        if (
            self._closed
            or self._scope_failed
            or self.pending is not None
            or self._in_model_scope
        ):
            raise RuntimeError("model scope requires an idle, open Attention session")
        self._in_model_scope = True
        try:
            yield
            if self.pending is not None:
                raise NativeDeviceError("model scope ended with an unconsumed FFN call")
            self.executor.flush_scoped()
        except BaseException as error:
            self._scope_failed = True
            self.executor.publish_failure(error)
            raise
        finally:
            self._in_model_scope = False

    def _agree_chunks(self, tokens: int, layer: int) -> int:
        executor, spec = self.executor, self.executor.spec
        local_chunks = max(1, (tokens + spec.capacity - 1) // spec.capacity)
        if self.tp_rows_aligned:
            # Production uses prepare_cam_dispatch_payload with SP disabled:
            # every TP member has equal padded rows and the same layer/call
            # order. Native selection still validates all TP identities. This
            # path needs no per-layer host Store traffic or polling.
            if self.sequence + local_chunks > MAX_GENERATION:
                raise ValueError("chunked work exhausts the session generation range")
            return local_chunks
        control = executor.control
        assert control is not None
        self._calls += 1
        plan = SourceChunkPlan(
            self._calls,
            self.sequence + 1,
            layer,
            local_chunks,
        )
        first_rank = (executor.rank // spec.attention_tp) * spec.attention_tp
        keys = [
            f"chunk-shape/{peer}"
            for peer in range(first_rank, first_rank + spec.attention_tp)
        ]
        # One overwritable cell per A rank: no per-call Store growth. A fast TP
        # cannot finish a call's first chunk before every TP has read this call's
        # plan and published, so it cannot overwrite a plan another TP still needs.
        control.write(f"chunk-shape/{executor.rank}", plan.to_json())
        deadline = time.monotonic() + executor.timeout
        while True:
            executor.check_peers()
            if control.ready(keys):
                peers = [
                    SourceChunkPlan.from_json(control.read(key).decode())
                    for key in keys
                ]
                if any(peer.call > plan.call for peer in peers):
                    raise ValueError("Attention TP advanced beyond an unagreed call")
                if all(peer.call == plan.call for peer in peers):
                    return plan.agree(peers)
            if time.monotonic() >= deadline:
                raise TimeoutError("Attention TP chunk-shape agreement timed out")
            time.sleep(STORE_POLL_SECONDS)

    def _combine(
        self,
        sequence: int,
        layer: int,
        x: torch.Tensor,
        ids: torch.Tensor,
        probabilities: torch.Tensor,
        exchange: NativeCompletion,
    ) -> torch.Tensor:
        deadline = time.monotonic() + self.executor.timeout
        completion = exchange
        while True:
            self.executor.check_peers()
            code = self.executor.retire(completion)
            if code == Result.OK:
                if completion.output.shape == x.shape:
                    return completion.output
                return completion.output[: x.shape[0]]
            if code not in (Result.BACKPRESSURE, Result.TIMED_OUT) or (
                code == Result.BACKPRESSURE and completion.mode != Mode.EXCHANGE
            ):
                raise RuntimeError(f"unexpected combine result: {code.name}")
            if time.monotonic() > deadline:
                raise TimeoutError("FFN expert result did not arrive")
            # EXCHANGE backpressure means publication did not occur. Timeout
            # means publication succeeded, so only retry the combine phase.
            completion = self.executor.submit(
                Mode.EXCHANGE if code == Result.BACKPRESSURE else Mode.COMBINE,
                sequence=sequence,
                layer=layer,
                x=x,
                expert_ids=ids,
                probabilities=probabilities,
            )

    def warmup(self) -> None:
        executor, spec = self.executor, self.executor.spec
        control = executor.control
        assert control is not None
        executor.wait_keys([f"prepared/{rank}" for rank in range(spec.ffn_ranks)])
        # Pair at least two physical layers where available. Zeros give an
        # exact zero-input oracle for bias-free routed experts, including combine.
        x = torch.zeros(
            (1, spec.hidden_size), dtype=torch.bfloat16, device=executor.device
        )
        ids = torch.tensor(
            [
                [
                    (route % spec.ffn_ranks) * spec.local_experts
                    for route in range(spec.top_k)
                ]
            ],
            dtype=torch.int32,
            device=executor.device,
        )
        probabilities = torch.full(
            (1, spec.top_k),
            1.0 / spec.top_k,
            dtype=torch.float32,
            device=executor.device,
        )
        for index, layer in enumerate(spec.layer_indices[:2]):
            # First work repeats one expert: other F ranks must complete zero
            # local work. The second work exercises every F on another layer.
            routes = torch.zeros_like(ids) if index == 0 else ids
            self.send(x, routes, probabilities, layer=layer)
            output = self.receive()
            if torch.count_nonzero(output).item() != 0:
                raise RuntimeError("paired FFN warmup violated the zero-input oracle")
        control.write(f"warmup/{executor.rank}", "1")
        executor.wait_keys([f"ready/{rank}" for rank in range(spec.ffn_ranks)])

    def send(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        layer: int,
    ) -> None:
        if self._closed or self._scope_failed:
            raise RuntimeError("Attention session is closed")
        if self.pending is not None:
            raise RuntimeError("consume the previous model call before another send")
        spec = self.executor.spec
        if x.dtype != torch.bfloat16 or tuple(x.shape[1:]) != (spec.hidden_size,):
            raise ValueError("FFN input must be a BF16 [tokens, hidden_size] tensor")
        if (
            tuple(ids.shape) != (x.shape[0], spec.top_k)
            or probabilities.shape != ids.shape
        ):
            raise ValueError("FFN routing shape must be [tokens, top_k]")
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        if probabilities.dtype not in (torch.float32, torch.bfloat16):
            probabilities = probabilities.float()
        if not ids.is_contiguous():
            ids = ids.contiguous()
        if not probabilities.is_contiguous():
            probabilities = probabilities.contiguous()
        if not x.is_contiguous():
            x = x.contiguous()
        spec.validate_work(self.sequence + 1, layer, min(x.shape[0], spec.capacity))
        chunks = self._agree_chunks(x.shape[0], layer)
        self.sequence += 1
        if chunks == 1:
            first_x, first_ids, first_probabilities = x, ids, probabilities
        else:
            first = slice(0, spec.capacity)
            first_x, first_ids, first_probabilities = (
                x[first],
                ids[first],
                probabilities[first],
            )
        self.executor.check_peers()
        # Unaligned TP uses one overwritable Store plan cell. It must finish
        # consuming the current call before a fast member can replace that cell.
        scoped = self._in_model_scope and self.tp_rows_aligned and chunks == 1
        if not scoped:
            self.executor.flush_scoped()
        exchange = self.executor.submit(
            Mode.SCOPED_EXCHANGE if scoped else Mode.EXCHANGE,
            sequence=self.sequence,
            layer=layer,
            x=first_x,
            expert_ids=first_ids,
            probabilities=first_probabilities,
        )
        self.pending = PendingWork(
            self.sequence, layer, chunks, x, ids, probabilities, exchange
        )

    def receive(self) -> torch.Tensor:
        if self._closed or self._scope_failed:
            raise RuntimeError("Attention session is closed")
        pending = self.pending
        if pending is None:
            raise RuntimeError("no FFN model call is pending")
        if pending.exchange.mode == Mode.SCOPED_EXCHANGE:
            output = self.executor.consume_scoped(pending.exchange)
            self.pending = None
            return output[: pending.x.shape[0]] if not pending.x.shape[0] else output
        if pending.chunks == 1:
            output = self._combine(
                pending.sequence,
                pending.layer,
                pending.x,
                pending.ids,
                pending.probabilities,
                pending.exchange,
            )
            self.pending = None
            return output
        spec = self.executor.spec
        chunks = []
        exchange = pending.exchange
        for chunk in range(pending.chunks):
            offset = chunk * spec.capacity
            part = slice(offset, offset + spec.capacity)
            x, ids, probabilities = (
                pending.x[part],
                pending.ids[part],
                pending.probabilities[part],
            )
            if offset:
                self.sequence += 1
                exchange = self.executor.submit(
                    Mode.EXCHANGE,
                    sequence=self.sequence,
                    layer=pending.layer,
                    x=x,
                    expert_ids=ids,
                    probabilities=probabilities,
                )
            chunks.append(
                self._combine(
                    self.sequence, pending.layer, x, ids, probabilities, exchange
                )
            )
        self.pending = None
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        executor = self.executor
        abort = abort or self._scope_failed or self.pending is not None
        if not abort:
            try:
                executor.check_peers()
                code = executor.retire(
                    executor.submit(Mode.END_SOURCE, sequence=self.sequence)
                )
                if code != Result.OK:
                    raise RuntimeError(f"FFN source end failed: {code.name}")
            except BaseException as error:
                executor.publish_failure(error)
                abort = True
        executor.close(abort=abort)
        self._closed = True
