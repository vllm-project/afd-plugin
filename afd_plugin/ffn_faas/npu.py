# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend native executor. This module has no vLLM EngineCore dependency."""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _new_process_group_helper, _world

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded
from afd_plugin.ffn_faas.control import SessionControl, SessionPeerFailureError
from afd_plugin.ffn_faas.protocol import (
    DEFAULT_COMBINE_POLLS,
    DEFAULT_IDLE_POLLS,
    DEFAULT_SCOPED_COMBINE_POLLS,
    DEFAULT_SERVE_POLLS,
    MAX_SCOPED_EXCHANGES,
    MAX_WORK_PER_SERVE,
    Mode,
    Result,
    SessionSpec,
)

HEARTBEAT_INTERVAL_SECONDS = 0.5
MILLISECONDS_PER_SECOND = 1000
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WINDOW_MB = 64
MB_BYTES = 1024 * 1024
STORE_POLL_SECONDS = 0.05
EXPERT_TILE_ROWS = 16
CACHE_LINE_BYTES = 64
SCRATCH_HEADER_WORDS = 16
INT32_BYTES = 4
BF16_BYTES = 2


class UnsafeSessionCleanupError(RuntimeError):
    """The owning process must exit; remote access cannot be disproved."""


class NativeDeviceError(RuntimeError):
    """A native submission/wait failed; completion cannot be assumed."""


@dataclass(frozen=True)
class NativeWeightView:
    """Borrowed tensors; their FFNModel owner outlives every submission."""

    w13: torch.Tensor
    w2: torch.Tensor
    layer_map: torch.Tensor


@dataclass
class NativeCompletion:
    mode: Mode
    output: torch.Tensor
    status: torch.Tensor
    request: torch.Tensor
    retained: tuple[torch.Tensor, ...]
    stream: torch.npu.Stream
    completed_work: int | None = None
    native: _ScopedNativeWork | None = None

    def wait(self, timeout: float) -> Result:
        try:
            with torch.npu.stream(self.stream):
                milliseconds = math.ceil(timeout * MILLISECONDS_PER_SECOND)
                if self.mode == Mode.SERVE:
                    code, completed = torch.ops.afd_ascend.ffn_faas_wait_served(
                        self.status, milliseconds
                    )
                else:
                    code = torch.ops.afd_ascend.ffn_faas_wait(self.status, milliseconds)
            result = Result(code)
            if self.mode == Mode.SERVE:
                if (
                    result
                    not in (
                        Result.OK,
                        Result.IDLE,
                        Result.DRAINED,
                        Result.INVALID,
                        Result.PEER_FAILED,
                    )
                    or not 0 <= completed <= MAX_WORK_PER_SERVE
                    or (result == Result.OK and completed == 0)
                    or (result == Result.IDLE and completed != 0)
                ):
                    raise NativeDeviceError(
                        f"invalid fused receipt: {result.name}, completed={completed}"
                    )
                self.completed_work = completed
            return result
        except (RuntimeError, ValueError) as error:
            raise NativeDeviceError("native FFN FaaS completion failed") from error


class _ScopedNativeWork(Protocol):
    def consume(self) -> torch.Tensor: ...


class _ScopedNativeSubmitter(Protocol):
    def submit(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        probabilities: torch.Tensor,
        request_host: torch.Tensor,
        request: torch.Tensor,
        sequence: int,
        layer: int,
        tokens: int,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor, _ScopedNativeWork]: ...

    def consume(self, work: _ScopedNativeWork) -> torch.Tensor: ...


class NPUExecutor:
    """Own communication windows, stream, state and bounded native submissions."""

    WEIGHT_QUANTIZATION = "none"

    def __init__(
        self,
        spec: SessionSpec,
        *,
        rank: int,
        device: int,
        host: str,
        port: int,
        weights: NativeWeightView | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        window_mb: int = DEFAULT_WINDOW_MB,
        startup_timeout: float | None = None,
        auto_initialize: bool = True,
    ) -> None:
        spec.native_config(rank, DEFAULT_IDLE_POLLS)
        if spec.required_window_bytes > window_mb * MB_BYTES:
            raise ValueError("HCCL window is smaller than the negotiated slot layout")
        if window_mb <= 0 or timeout <= 0:
            raise ValueError("window size and timeout must be positive")
        self.spec = spec
        self.rank = rank
        self.timeout = timeout
        self.startup_timeout = timeout if startup_timeout is None else startup_timeout
        if self.startup_timeout <= 0:
            raise ValueError("startup timeout must be positive")
        self._host, self._port, self._window_mb = host, port, window_mb
        self.is_ffn = rank >= spec.attention_ranks
        if self.is_ffn and spec.weight_quantization != self.WEIGHT_QUANTIZATION:
            raise ValueError(
                "FFN weight quantization does not match this native executor"
            )
        self._active: NativeCompletion | None = None
        self._scoped: list[NativeCompletion] = []
        self._untracked_retained: tuple[torch.Tensor, ...] = ()
        self._untracked_submission = False
        self._untracked_native: _ScopedNativeWork | None = None
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_error: BaseException | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._peer_values: dict[int, tuple[bytes, float]] = {}
        self._last_peer_check = 0.0
        self._pg: dist.ProcessGroup | None = None
        self._group_creation_attempted = False
        self._initialized = False
        self._failed = False
        self._failure_published = False
        self._drained = False
        self.store: dist.Store | None = None
        self.control: SessionControl | None = None
        torch.npu.set_device(device)
        os.environ["HCCL_BUFFSIZE"] = str(window_mb)
        ensure_afd_ascend_ops_loaded()
        self.device = torch.device(f"npu:{device}")
        self.stream = torch.npu.Stream(device=self.device)
        self._anchor = torch.zeros(
            (1, spec.hidden_size), dtype=torch.bfloat16, device=self.device
        )
        self._ids = torch.zeros((1, spec.top_k), dtype=torch.int32, device=self.device)
        self._probabilities = torch.zeros(
            (1, spec.top_k), dtype=torch.float32, device=self.device
        )
        self.state = torch.zeros(
            (1 + spec.source_count,), dtype=torch.int64, device=self.device
        )
        self._empty_request = torch.zeros((3,), dtype=torch.int64, device=self.device)
        self._request_host = torch.empty(
            (3,), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._request_device = torch.empty((3,), dtype=torch.int64, device=self.device)
        self._scoped_requests = [
            (
                torch.empty((3,), dtype=torch.int64, device="cpu", pin_memory=True),
                torch.empty((3,), dtype=torch.int64, device=self.device),
            )
            for _ in range(0 if self.is_ffn else MAX_SCOPED_EXCHANGES)
        ]
        self._scoped_submitter: _ScopedNativeSubmitter | None = None
        padded_rows = (
            spec.attention_tp * spec.capacity * spec.top_k
            + (EXPERT_TILE_ROWS - 1) * spec.local_experts
        )
        metadata_bytes = (
            SCRATCH_HEADER_WORDS
            + 3 * spec.local_experts
            + 1
            + spec.attention_tp
            + padded_rows
        ) * INT32_BYTES
        metadata_bytes = (
            (metadata_bytes + CACHE_LINE_BYTES - 1)
            // CACHE_LINE_BYTES
            * CACHE_LINE_BYTES
        )
        scratch_bytes = (
            metadata_bytes
            + padded_rows
            * (2 * spec.hidden_size + 3 * spec.intermediate_size)
            * BF16_BYTES
        )
        self.scratch = torch.empty(
            (scratch_bytes if self.is_ffn else 1,),
            dtype=torch.uint8,
            device=self.device,
        )
        if weights is None:
            if self.is_ffn:
                raise ValueError("FFN executor requires borrowed expert weights")
            layer_map = torch.full((spec.model_layers,), -1, dtype=torch.int32)
            for compact, layer in enumerate(spec.layer_indices):
                layer_map[layer] = compact
            weights = NativeWeightView(
                torch.zeros((1,), dtype=torch.bfloat16, device=self.device),
                torch.zeros((1,), dtype=torch.bfloat16, device=self.device),
                layer_map.to(self.device),
            )
        self.weights: NativeWeightView | None = weights
        self.stream.wait_stream(torch.npu.current_stream(self.device))
        if auto_initialize:
            self.initialize()

    def initialize(self, *, stop: threading.Event | None = None) -> None:
        if self.store is not None or self._closed:
            raise RuntimeError("executor initialization may only be attempted once")
        spec, rank = self.spec, self.rank
        duration = timedelta(seconds=self.startup_timeout)
        # TCPStore is explicitly constructed: torchrun's agent-store environment
        # does not take ownership of the A/F session rendezvous.
        self.store = dist.TCPStore(
            self._host,
            self._port,
            spec.world_size,
            rank == 0,
            duration,
            wait_for_workers=False,
        )
        self.control = SessionControl(self.store)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat, name="ffn-faas-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()
        try:
            self.store.set(f"hello/{rank}", spec.to_json())
            self.wait_keys([f"hello/{r}" for r in range(spec.world_size)], stop=stop)
            for peer in range(spec.world_size):
                hello = self.store.get(f"hello/{peer}").decode()
                if hello != spec.to_json():
                    raise ValueError(
                        f"FFN FaaS session mismatch at rank {peer}: {hello}"
                    )
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = {"hccl_buffer_size": self._window_mb}
            self._group_creation_attempted = True
            self._pg, _ = _new_process_group_helper(
                spec.world_size,
                rank,
                [],
                dist.Backend("hccl"),
                dist.PrefixStore("ffn-faas-hccl", self.store),
                group_name="afd_ffn_faas",
                backend_options=options,
                timeout=duration,
            )
            _world.pg_group_ranks[self._pg] = {r: r for r in range(spec.world_size)}
            backend = self._pg._get_backend(self.device)
            self.group_name = str(backend.get_hccl_comm_name(rank))
            if not self.is_ffn:
                # The communicator and borrowed weights/state must exist before
                # caching this immutable submission context. Construction only
                # retains CPU handles; it does not enqueue device work.
                from afd_plugin import _C_ascend

                assert self.weights is not None
                self._scoped_submitter = _C_ascend.FFNFaasScopedSubmitter(
                    self.weights.w13,
                    self.weights.w2,
                    self.weights.layer_map,
                    self.state,
                    self.scratch,
                    spec.native_config(rank, DEFAULT_SCOPED_COMBINE_POLLS),
                    self.group_name,
                    self.stream.stream_id,
                    self.stream.device_index,
                    self.stream.device_type,
                )
            initialized = self.submit(Mode.INITIALIZE)
            if self.retire(initialized) != Result.OK:
                raise RuntimeError("native FFN FaaS window initialization failed")
            self.store.set(f"initialized/{rank}", "1")
            self.wait_keys(
                [f"initialized/{r}" for r in range(spec.world_size)], stop=stop
            )
            self._initialized = True
        except BaseException as error:
            self.publish_failure(error)
            raise

    def bind_host_threads(self) -> None:
        """Apply the backend CPU plan to this F process, without NUMA/IRQ edits."""
        from vllm_ascend.cpu_binding import GLOBAL_SLICE_MODE, CpuAlloc, is_arm_cpu

        if not self.is_ffn:
            raise RuntimeError("standalone host binding belongs to FFN ranks")
        if not is_arm_cpu():
            return
        # CpuAlloc uses npu-smi chip_logic_id keys, filtered by this same
        # runtime visibility list. Do not substitute a host physical chip ID.
        device_id = self.device.index
        assert device_id is not None
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if visible:
            device_id = tuple(map(int, visible.split(",")))[device_id]
        binder = CpuAlloc(self.rank - self.spec.attention_ranks, npu_id=device_id)
        binder.build_cpu_node_map()
        # build_cpu_pools/run_all may bind shared UVB threads or mutate IRQ and
        # memory placement. Call only the plan builders, then this PID's threads.
        if binder._uses_cluster_cpu_topology():
            if not binder.build_ascend_950_cpu_pools():
                return
        elif binder._binding_mode() == GLOBAL_SLICE_MODE:
            binder.build_global_slice_cpu_pool()
        else:
            binder._build_topo_affinity_cpu_pool()
        binder.allocate()
        binder.bind_threads()

    def _heartbeat(self) -> None:
        assert self.store is not None
        try:
            count = 0
            while not self._heartbeat_stop.is_set():
                self.store.set(f"heartbeat/{self.rank}", str(count))
                count += 1
                self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS)
        except BaseException as error:
            self._heartbeat_error = error

    def check_peers(self) -> None:
        control = self.control
        assert control is not None
        now = time.monotonic()
        if now - self._last_peer_check < HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_peer_check = now
        if self._heartbeat_error is not None:
            raise SessionPeerFailureError(
                "FFN FaaS heartbeat writer lost the session Store"
            ) from self._heartbeat_error
        for peer in range(self.spec.world_size):
            if control.ready([f"failure/{peer}"]):
                raise SessionPeerFailureError(control.read(f"failure/{peer}").decode())
            if not control.ready([f"heartbeat/{peer}"]):
                # A process may still be joining during startup. Handshake waits
                # have a separate, bounded startup deadline.
                if not self._initialized:
                    continue
                raise SessionPeerFailureError(f"rank {peer} has no session heartbeat")
            value = control.read(f"heartbeat/{peer}")
            previous, changed = self._peer_values.get(peer, (value, now))
            if previous != value:
                changed = now
            elif now - changed > self.timeout:
                raise SessionPeerFailureError(f"rank {peer} session heartbeat expired")
            self._peer_values[peer] = (value, changed)

    def publish_failure(self, error: BaseException) -> None:
        self._failed = True
        if self.store is not None and not self._failure_published:
            # Keep the original failure if the store itself has disconnected.
            with suppress(RuntimeError):
                self.store.set(
                    f"failure/{self.rank}",
                    f"rank {self.rank}: {type(error).__name__}: {error}",
                )
                self._failure_published = True

    def wait_keys(
        self,
        keys: list[str],
        *,
        timeout: float | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        control = self.control
        assert control is not None
        deadline = time.monotonic() + (
            self.startup_timeout if timeout is None else timeout
        )
        while not control.ready(keys):
            if stop is not None and stop.is_set():
                raise InterruptedError("FFN FaaS was stopped during startup")
            self.check_peers()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FFN FaaS handshake timed out: {keys}")
            if stop is None:
                time.sleep(STORE_POLL_SECONDS)
            else:
                stop.wait(STORE_POLL_SECONDS)

    def submit(
        self,
        mode: Mode,
        *,
        sequence: int = 0,
        layer: int = 0,
        x: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        tokens: int | None = None,
    ) -> NativeCompletion:
        scoped = mode == Mode.SCOPED_EXCHANGE
        if (
            self._closed
            or self._untracked_submission
            or (self._failed and mode != Mode.ABORT_SESSION)
            or self._active is not None
        ):
            raise RuntimeError("executor is closed or already has an active submission")
        if self._scoped and not scoped:
            raise RuntimeError("validate the model scope before another native mode")
        if scoped:
            if self.is_ffn:
                raise ValueError("scoped exchanges belong to Attention")
            if len(self._scoped) == MAX_SCOPED_EXCHANGES:
                self.flush_scoped()
        if self._drained and mode != Mode.ABORT_SESSION:
            raise RuntimeError("FFN FaaS source/session has already drained")
        logical_tokens = 0 if x is None else x.shape[0]
        if tokens is not None:
            logical_tokens = tokens
        identity = (sequence, layer, logical_tokens)
        if mode in (
            Mode.PUBLISH,
            Mode.COMBINE,
            Mode.EXCHANGE,
            Mode.SCOPED_EXCHANGE,
            Mode.PEEK_RESULTS,
        ):
            self.spec.validate_work(sequence, layer, logical_tokens)
        if x is None or logical_tokens == 0:
            x, expert_ids, probabilities = self._anchor, self._ids, self._probabilities
        assert expert_ids is not None and probabilities is not None
        budget = DEFAULT_IDLE_POLLS
        if mode in (Mode.COMBINE, Mode.EXCHANGE, Mode.PEEK_RESULTS):
            budget = DEFAULT_COMBINE_POLLS
        elif mode == Mode.SERVE:
            budget = DEFAULT_SERVE_POLLS
        elif scoped:
            budget = DEFAULT_SCOPED_COMBINE_POLLS
        assert self.weights is not None
        request_host, request_device = (
            self._scoped_requests[len(self._scoped)]
            if scoped
            else (self._request_host, self._request_device)
        )
        self._untracked_submission = True
        self._untracked_retained = (
            x,
            expert_ids,
            probabilities,
            request_host,
            request_device,
        )
        try:
            native = None
            if scoped:
                assert self._scoped_submitter is not None
                # The Python-owned submitter retains pending work before any
                # enqueue, including failure before Python receives this tuple.
                request = request_device
                y, status, native = self._scoped_submitter.submit(
                    x,
                    expert_ids,
                    probabilities,
                    request_host,
                    request,
                    sequence,
                    layer,
                    logical_tokens,
                    len(self._scoped),
                )
                self._untracked_native = native
            else:
                config = self.spec.native_config(self.rank, budget)
                producer_stream = torch.npu.current_stream(self.device)
                with torch.npu.stream(self.stream):
                    if mode in (
                        Mode.PUBLISH,
                        Mode.COMBINE,
                        Mode.EXCHANGE,
                        Mode.PEEK_RESULTS,
                        Mode.END_SOURCE,
                    ):
                        request_host.copy_(
                            torch.tensor(identity, dtype=torch.int64, device="cpu")
                        )
                        request_device.copy_(request_host, non_blocking=True)
                        request = request_device
                    else:
                        request = self._empty_request
                    if not self.is_ffn or mode == Mode.INITIALIZE:
                        self.stream.wait_stream(producer_stream)
                    y, status = self._submit_native(
                        x,
                        expert_ids,
                        probabilities,
                        request,
                        mode,
                        config,
                    )
            self._untracked_retained += (y, status)
            completion = NativeCompletion(
                mode,
                y,
                status,
                request,
                self._untracked_retained,
                self.stream,
                native=native,
            )
            if scoped:
                self._scoped.append(completion)
            else:
                self._active = completion
            self._untracked_submission = False
            self._untracked_retained = ()
            self._untracked_native = None
            return completion
        except RuntimeError as error:
            raise NativeDeviceError("native FFN FaaS submission failed") from error

    def _submit_native(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        probabilities: torch.Tensor,
        request: torch.Tensor,
        mode: Mode,
        config: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.weights is not None
        return torch.ops.afd_ascend.ffn_faas(
            x,
            expert_ids,
            probabilities,
            self.weights.w13,
            self.weights.w2,
            self.weights.layer_map,
            request,
            self.state,
            self.scratch,
            int(mode),
            config,
            self.group_name,
        )

    def consume_scoped(self, completion: NativeCompletion) -> torch.Tensor:
        if completion.mode != Mode.SCOPED_EXCHANGE or not any(
            work is completion for work in self._scoped
        ):
            raise RuntimeError("completion does not belong to the model scope")
        assert completion.native is not None
        assert self._scoped_submitter is not None
        return self._scoped_submitter.consume(completion.native)

    def flush_scoped(self) -> None:
        if not self._scoped:
            return
        try:
            with torch.npu.stream(self.stream):
                codes = torch.ops.afd_ascend.ffn_faas_wait_many(
                    [work.status for work in self._scoped],
                    math.ceil(self.timeout * MILLISECONDS_PER_SECOND),
                )
            if len(codes) != len(self._scoped):
                raise NativeDeviceError("invalid model scope completion count")
            results = [Result(code) for code in codes]
        except (RuntimeError, ValueError) as error:
            # Unknown completion keeps every tensor/request alive through exit.
            raise NativeDeviceError("native model scope completion failed") from error
        self._scoped.clear()
        for result in results:
            if result != Result.OK:
                failure = NativeDeviceError(
                    f"model scope exchange failed: {result.name}"
                )
                self.publish_failure(failure)
                raise failure

    def retire(self, completion: NativeCompletion) -> Result:
        if completion is not self._active:
            raise RuntimeError("completion does not belong to the active submission")
        code = completion.wait(self.timeout)
        self._active = None
        if (completion.mode == Mode.END_SOURCE and code == Result.OK) or (
            completion.mode in (Mode.EXECUTE, Mode.SERVE) and code == Result.DRAINED
        ):
            self._drained = True
        if code == Result.PEER_FAILED:
            raise SessionPeerFailureError("native FFN FaaS reported a peer abort")
        if code == Result.INVALID:
            raise RuntimeError(f"native FFN FaaS failed: {code.name}")
        return code

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        try:
            if self._untracked_submission:
                raise NativeDeviceError(
                    "native submission failed before creating a completion"
                )
            if self._group_creation_attempted and self._pg is None:
                raise RuntimeError(
                    "HCCL group creation failed without returning ownership"
                )
            if self._active is not None:
                self.retire(self._active)
            if self._scoped:
                self.flush_scoped()
            if self._initialized and (abort or self._failed or not self._drained):
                self.publish_failure(
                    InterruptedError("session aborted before normal drain")
                )
                self.retire(self.submit(Mode.ABORT_SESSION))
            if self.store is not None and self._pg is not None:
                # Acknowledge only after our last native submission completes.
                # Failure notifications do not imply peer quiescence.
                self.store.set(f"quiescent/{self.rank}", "1")
                self.store.wait(
                    [f"quiescent/{r}" for r in range(self.spec.world_size)],
                    timedelta(seconds=self.timeout),
                )
            if self._pg is not None:
                dist.destroy_process_group(self._pg)
                self._pg = None
            self._heartbeat_stop.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=self.timeout)
                if self._heartbeat_thread.is_alive():
                    raise TimeoutError("session heartbeat did not stop")
            if self.store is not None and self._initialized:
                # Rank zero owns TCPStore's server. Keep it alive until every
                # client has completed group teardown and its final Store use.
                self.store.set(f"closed/{self.rank}", "1")
                if self.rank == 0:
                    self.store.wait(
                        [f"closed/{rank}" for rank in range(self.spec.world_size)],
                        timedelta(seconds=self.timeout),
                    )
            # Drop borrowed model references only after all peer accesses end.
            self.weights = None
            self._closed = True
        except BaseException as error:
            self.publish_failure(error)
            raise UnsafeSessionCleanupError(
                "session could not prove peer quiescence; exit the owning process"
            ) from error
        finally:
            self._heartbeat_stop.set()
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=self.timeout)
