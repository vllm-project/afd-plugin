# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cooperative execution of two complete CAMAsync MoE call stacks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, Thread, get_ident
from time import monotonic
from typing import TYPE_CHECKING, Generic, TypeVar, cast

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext

CAM_ASYNC_EXECUTION_KEY = "afd_cam_async_execution"
CAM_ASYNC_SCHEDULER_KEY = "afd_cam_async_scheduler"
STAGE_COUNT = 2
SHUTDOWN_TIMEOUT_S = 5.0

_Result = TypeVar("_Result")


class CAMAsyncPhase(Enum):
    ROUTED = "routed"
    DISPATCHED = "dispatched"
    LAYER_DONE = "layer_done"


class CAMAsyncTaskState(Enum):
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CAMAsyncEvent:
    run_id: int
    stage_idx: int
    layer_id: int | None
    phase: CAMAsyncPhase | CAMAsyncTaskState


@dataclass(frozen=True)
class CAMAsyncExecutionContext:
    run_id: int
    stage_idx: int
    num_stages: int
    use_sequence_parallel: bool
    scheduler: CAMAsyncUbatchScheduler | None = None
    cancel_check: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def checkpoint(self, layer_id: int, phase: CAMAsyncPhase) -> None:
        if self.scheduler is not None:
            self.scheduler.checkpoint(self, layer_id, phase)
        elif self.cancel_check is not None:
            self.cancel_check()

    def layer_done(self, layer_id: int) -> None:
        self.checkpoint(layer_id, CAMAsyncPhase.LAYER_DONE)


def require_cam_async_execution_context() -> CAMAsyncExecutionContext:
    # Lazy imports keep the control scheduler usable without vLLM or an NPU.
    from vllm.forward_context import get_forward_context

    execution = get_forward_context().additional_kwargs.get(CAM_ASYNC_EXECUTION_KEY)
    if execution is None:
        raise RuntimeError("CAMAsync requires an explicit execution context")
    return execution


class _StageCancelledError(Exception):
    pass


class CAMAsyncUbatchScheduler(Generic[_Result]):
    """Reuse two threads; only the stage holding permission may run model code."""

    def __init__(self, *, wait_timeout: float | None = None) -> None:
        self._condition = Condition()
        self._threads: list[Thread] = []
        self._run_id = 0
        self._active = False
        self._abandoned = False
        self._closed = False
        # Failures are terminal: partially completed CAM transfers cannot be retried.
        self._failed = False
        self._failure_reason: str | None = None
        self._cancelled = False
        self._remaining = 0
        self._permission: int | None = None
        self._events: list[CAMAsyncEvent | None] = [None, None]
        self._results: list[_Result | None] = [None, None]
        self._error: BaseException | None = None
        self._task: Callable[[CAMAsyncExecutionContext], _Result] | None = None
        self._activate: Callable[[CAMAsyncExecutionContext], None] | None = None
        self._thread_context: Callable[[], AbstractContextManager] = nullcontext
        self._executions: list[CAMAsyncExecutionContext] = []
        self._wait_timeout = wait_timeout

    @property
    def quiescent(self) -> bool:
        with self._condition:
            return self._remaining == 0 and not self._active

    def _check_available(self) -> None:
        if self._failed:
            raise RuntimeError(
                "CAMAsync scheduler is FAILED and cannot be reused: "
                f"{self._failure_reason}"
            )
        if self._active or self._closed:
            raise RuntimeError("CAMAsync scheduler is active or closed")

    def _check_cancelled(self) -> None:
        if self._cancelled or self._closed:
            raise _StageCancelledError()

    def _wait_permission(self, stage_idx: int) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: (
                    self._permission == stage_idx or self._cancelled or self._closed
                ),
                timeout=self._wait_timeout,
            ):
                raise TimeoutError("CAMAsync stage timed out waiting for permission")
            self._check_cancelled()
            execution = self._executions[stage_idx]
            activate = self._activate
        # Activation runs on the permitted worker, outside the control lock.
        if activate is not None:
            activate(execution)

    def checkpoint(
        self, execution: CAMAsyncExecutionContext, layer_id: int, phase: CAMAsyncPhase
    ) -> None:
        with self._condition:
            self._check_cancelled()
            stage_idx = self._permission
            if (
                stage_idx is None
                or self._threads[stage_idx].ident != get_ident()
                or execution != self._executions[stage_idx]
            ):
                raise RuntimeError("CAMAsync checkpoint has the wrong run or stage")
            if self._events[stage_idx] is not None:
                raise RuntimeError("CAMAsync stage published a duplicate event")
            self._events[stage_idx] = CAMAsyncEvent(
                execution.run_id, execution.stage_idx, layer_id, phase
            )
            self._permission = None
            self._condition.notify_all()
        self._wait_permission(stage_idx)

    def _fail(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error
            # Retain only text after _clear_run releases the request traceback.
            self._failure_reason = f"{type(error).__name__}: {error}"
        self._failed = True
        self._cancelled = True
        self._permission = None
        self._condition.notify_all()

    def _worker(self, stage_idx: int) -> None:
        last_run = 0

        def has_pending_run() -> bool:
            return self._task is not None and self._run_id != last_run

        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or has_pending_run())
                if not has_pending_run():
                    return
                last_run = self._run_id
                task = self._task
                execution = self._executions[stage_idx]
                thread_context = self._thread_context
            state = CAMAsyncTaskState.DONE
            result = None
            try:
                self._wait_permission(stage_idx)
                with thread_context():
                    assert task is not None
                    result = task(execution)
            except _StageCancelledError:
                state = CAMAsyncTaskState.CANCELLED
            except BaseException as error:
                state = CAMAsyncTaskState.FAILED
                with self._condition:
                    self._fail(error)
            finally:
                # Do not retain request closures or tensors in idle worker frames.
                task = None
                del execution
                thread_context = nullcontext
                with self._condition:
                    self._results[stage_idx] = result
                    result = None
                    self._events[stage_idx] = CAMAsyncEvent(
                        last_run, stage_idx, None, state
                    )
                    self._permission = None
                    self._remaining -= 1
                    if self._remaining == 0 and self._abandoned:
                        self._clear_run()
                    self._condition.notify_all()

    def _advance(
        self,
        stage_idx: int,
        layer_id: int | None,
        expected_phase: CAMAsyncPhase | CAMAsyncTaskState,
    ) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._cancelled or self._closed:
                raise RuntimeError("CAMAsync execution was cancelled")
            if self._events[stage_idx] is not None:
                raise RuntimeError("CAMAsync stage has an unconsumed event")
            self._permission = stage_idx
            self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: (
                    self._events[stage_idx] is not None
                    or self._error is not None
                    or self._cancelled
                    or self._closed
                ),
                timeout=self._wait_timeout,
            ):
                raise TimeoutError("CAMAsync stage did not reach its checkpoint")
            if self._error is not None:
                raise self._error
            if self._cancelled or self._closed:
                raise RuntimeError("CAMAsync execution was cancelled")
            event = self._events[stage_idx]
            expected = CAMAsyncEvent(self._run_id, stage_idx, layer_id, expected_phase)
            if event != expected:
                raise RuntimeError(f"CAMAsync expected {expected}, received {event}")
            self._events[stage_idx] = None

    def run(
        self,
        task: Callable[[CAMAsyncExecutionContext], _Result],
        layer_ids: Sequence[int],
        *,
        use_sequence_parallel: bool = False,
        activate: Callable[[CAMAsyncExecutionContext], None] | None = None,
        thread_context: Callable[[], AbstractContextManager] = nullcontext,
    ) -> tuple[_Result, _Result]:
        if not layer_ids:
            raise ValueError("CAMAsync scheduling requires at least one MoE layer")
        startup_error: BaseException | None = None
        with self._condition:
            self._check_available()
            try:
                if not self._threads:
                    for idx in range(STAGE_COUNT):
                        thread = Thread(
                            target=self._worker,
                            args=(idx,),
                            name=f"afd-cam-stage-{idx}",
                            daemon=True,
                        )
                        thread.start()
                        self._threads.append(thread)
            except BaseException as error:
                startup_error = error
                self._failure_reason = f"{type(error).__name__}: {error}"
                self._failed = True
                self._closed = True
                self._cancelled = True
                self._condition.notify_all()
            else:
                # Publish only when both workers exist; idle workers hold no task.
                self._run_id += 1
                self._task = task
                self._activate = activate
                self._thread_context = thread_context
                self._executions = [
                    CAMAsyncExecutionContext(
                        self._run_id, idx, STAGE_COUNT, use_sequence_parallel, self
                    )
                    for idx in range(STAGE_COUNT)
                ]
                self._active = True
                self._remaining = STAGE_COUNT
                self._condition.notify_all()
        if startup_error is not None:
            # A started worker needs the condition lock to observe closure and exit.
            try:
                self.shutdown()
            except RuntimeError as cleanup_error:
                raise cleanup_error from startup_error
            raise startup_error
        try:
            first = layer_ids[0]
            self._advance(0, first, CAMAsyncPhase.ROUTED)
            self._advance(0, first, CAMAsyncPhase.DISPATCHED)
            for index, layer_id in enumerate(layer_ids):
                self._advance(1, layer_id, CAMAsyncPhase.ROUTED)
                self._advance(0, layer_id, CAMAsyncPhase.LAYER_DONE)
                self._advance(1, layer_id, CAMAsyncPhase.DISPATCHED)
                has_next = index + 1 < len(layer_ids)
                if has_next:
                    self._advance(0, layer_ids[index + 1], CAMAsyncPhase.ROUTED)
                self._advance(1, layer_id, CAMAsyncPhase.LAYER_DONE)
                if has_next:
                    self._advance(0, layer_ids[index + 1], CAMAsyncPhase.DISPATCHED)
            self._advance(0, None, CAMAsyncTaskState.DONE)
            self._advance(1, None, CAMAsyncTaskState.DONE)
            return cast(_Result, self._results[0]), cast(_Result, self._results[1])
        except BaseException as error:
            with self._condition:
                self._fail(error)
            raise
        finally:
            with self._condition:
                stopped = self._condition.wait_for(
                    lambda: self._remaining == 0,
                    timeout=(
                        self._wait_timeout
                        if self._wait_timeout is not None
                        else SHUTDOWN_TIMEOUT_S
                    ),
                )
                if stopped:
                    self._clear_run()
                    self._condition.notify_all()
                else:
                    # The controller must propagate an unrecoverable error now.
                    # The last late worker releases references when it finally exits.
                    self._abandoned = True
                    raise RuntimeError(
                        "CAMAsync worker is still executing; process cleanup required"
                    ) from self._error

    def _clear_run(self) -> None:
        # Called under the condition lock only after both task frames are released.
        self._task = None
        self._activate = None
        self._thread_context = nullcontext
        self._executions = []
        self._events = [None, None]
        self._results = [None, None]
        self._error = None
        self._permission = None
        self._active = False
        self._abandoned = False

    @contextmanager
    def single_stage(
        self, *, use_sequence_parallel: bool
    ) -> Iterator[CAMAsyncExecutionContext]:
        with self._condition:
            self._check_available()
            self._active = True
            self._run_id += 1
            execution = CAMAsyncExecutionContext(
                self._run_id,
                0,
                1,
                use_sequence_parallel,
                cancel_check=self._check_single_stage_cancelled,
            )
        try:
            yield execution
        except BaseException as error:
            with self._condition:
                self._failure_reason = f"{type(error).__name__}: {error}"
                self._failed = True
            raise
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def _check_single_stage_cancelled(self) -> None:
        with self._condition:
            if self._closed or self._cancelled:
                raise RuntimeError("CAMAsync execution was cancelled")

    def cancel(self) -> None:
        with self._condition:
            self._closed = True
            self._cancelled = True
            self._condition.notify_all()

    def shutdown(self, *, timeout: float = SHUTDOWN_TIMEOUT_S) -> None:
        self.cancel()
        deadline = monotonic() + timeout
        with self._condition:
            if not self._condition.wait_for(lambda: not self._active, timeout=timeout):
                raise RuntimeError(
                    "CAMAsync task did not stop; process cleanup required"
                )
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("CAMAsync worker did not stop; process cleanup required")


class CAMAsyncRuntimeContext:
    """Capture parent modes and reactivate the shared stream on every resume."""

    def __init__(
        self, contexts: Sequence[ForwardContext], device: torch.device
    ) -> None:
        import torch

        self.contexts = contexts
        self.device = device
        self.inference_enabled = torch.is_inference_mode_enabled()
        self.grad_enabled = torch.is_grad_enabled()
        self.autocast_enabled = torch.is_autocast_enabled(device.type)
        self.autocast_dtype = torch.get_autocast_dtype(device.type)
        self.autocast_cache = torch.is_autocast_cache_enabled()
        self.stream = None
        if device.type == "npu":
            import torch_npu  # noqa: F401

            self.stream = torch.npu.current_stream(device)

    def activate(self, execution: CAMAsyncExecutionContext) -> None:
        import vllm.forward_context as forward_context_module

        if self.stream is not None:
            import torch

            # A fresh worker's default device may be 0; bind before NPU queries.
            torch.npu.set_device(self.device)
            torch.npu.set_stream(self.stream)
        context = self.contexts[execution.stage_idx]
        context.additional_kwargs[CAM_ASYNC_EXECUTION_KEY] = execution
        # Only the permitted stage writes this process-global pinned vLLM field.
        forward_context_module._forward_context = context

    @contextmanager
    def thread_context(self) -> Iterator[None]:
        import torch

        with (
            torch.inference_mode(self.inference_enabled),
            torch.set_grad_enabled(self.grad_enabled),
            torch.autocast(
                self.device.type,
                enabled=self.autocast_enabled,
                dtype=self.autocast_dtype,
                cache_enabled=self.autocast_cache,
            ),
        ):
            yield
