# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only order and lifecycle contracts for complete CAMAsync call stacks."""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from afd_plugin.model_executor.npu.async_cam_execution import (
    CAMAsyncExecutionContext,
    CAMAsyncPhase,
    CAMAsyncRuntimeContext,
    CAMAsyncUbatchScheduler,
    require_cam_async_execution_context,
)

TEST_WAIT_SECONDS = 2.0
FIRST_LAYER_ID = 7


@pytest.fixture
def scheduler():
    value = CAMAsyncUbatchScheduler(wait_timeout=TEST_WAIT_SECONDS)
    yield value
    value.shutdown(timeout=TEST_WAIT_SECONDS)


def _baseline_trace(layer_ids):
    # Frozen from 88c50cb7's model-owned warmup, steady state and final drain.
    trace = [("route", layer_ids[0], 0), ("dispatch", layer_ids[0], 0)]
    for current, following in zip(layer_ids[:-1], layer_ids[1:], strict=True):
        trace.extend(
            [
                ("route", current, 1),
                ("combine", current, 0),
                ("dispatch", current, 1),
                ("route", following, 0),
                ("combine", current, 1),
                ("dispatch", following, 0),
            ]
        )
    trace.extend(
        [
            ("route", layer_ids[-1], 1),
            ("combine", layer_ids[-1], 0),
            ("dispatch", layer_ids[-1], 1),
            ("combine", layer_ids[-1], 1),
        ]
    )
    return trace


@pytest.mark.parametrize("layer_count", [1, 2, 3, 8, 61])
@pytest.mark.parametrize("use_sequence_parallel", [False, True])
def test_complete_calls_preserve_baseline_order_and_reuse_threads(
    scheduler, layer_count, use_sequence_parallel
):
    layer_ids = tuple(range(FIRST_LAYER_ID, FIRST_LAYER_ID + layer_count))
    threads_by_stage: dict[int, int] = {}
    run_ids: list[int] = []
    assert scheduler._threads == []

    class SharedRunner:
        def forward(self, execution, layer_id, hidden):
            routed = hidden * 3 + execution.stage_idx
            trace.append(("route", layer_id, execution.stage_idx))
            execution.checkpoint(layer_id, CAMAsyncPhase.ROUTED)
            dispatched = routed + 11
            shared = hidden / 2
            trace.append(("dispatch", layer_id, execution.stage_idx))
            execution.checkpoint(layer_id, CAMAsyncPhase.DISPATCHED)
            trace.append(("combine", layer_id, execution.stage_idx))
            return dispatched + shared

    runner = SharedRunner()
    for offset in (2, 5, 9):
        trace: list[tuple[str, int, int]] = []
        active: list[CAMAsyncExecutionContext] = []
        entered: list[int] = []
        exited: list[int] = []

        @contextmanager
        def thread_context(entered=entered, exited=exited):
            thread_id = threading.get_ident()
            entered.append(thread_id)
            try:
                yield
            finally:
                exited.append(thread_id)

        def activate(execution, active=active):
            active[:] = [execution]

        def task(execution, offset=offset, active=active):
            assert execution.num_stages == 2
            assert execution.use_sequence_parallel is use_sequence_parallel
            thread_id = threading.get_ident()
            assert (
                threads_by_stage.setdefault(execution.stage_idx, thread_id) == thread_id
            )
            run_ids.append(execution.run_id)
            hidden = offset + execution.stage_idx
            for layer_id in layer_ids:
                assert active == [execution]
                hidden = runner.forward(execution, layer_id, hidden)
                assert active == [execution]
                execution.layer_done(layer_id)
            return hidden

        outputs = scheduler.run(
            task,
            layer_ids,
            use_sequence_parallel=use_sequence_parallel,
            activate=activate,
            thread_context=thread_context,
        )
        assert trace == _baseline_trace(layer_ids)
        for stage_idx, actual in enumerate(outputs):
            expected = float(offset + stage_idx)
            for _ in layer_ids:
                expected = expected * 3 + stage_idx + 11 + expected / 2
            assert actual == expected
        assert sorted(entered) == sorted(exited) == sorted(threads_by_stage.values())
        assert scheduler.quiescent
        assert scheduler._task is None
        assert scheduler._executions == []
        assert scheduler._events == [None, None]
        assert scheduler._results == [None, None]
    assert run_ids == [1, 1, 2, 2, 3, 3]
    assert len(set(threads_by_stage.values())) == 2


@pytest.mark.parametrize(
    "invalid_event",
    ["phase", "layer", "stage", "run", "duplicate", "early_done"],
)
def test_invalid_events_fail_closed(scheduler, invalid_event):
    def task(execution):
        if invalid_event == "phase":
            execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.DISPATCHED)
        elif invalid_event == "layer":
            execution.checkpoint(FIRST_LAYER_ID + 1, CAMAsyncPhase.ROUTED)
        elif invalid_event == "stage":
            replace(execution, stage_idx=1 - execution.stage_idx).checkpoint(
                FIRST_LAYER_ID, CAMAsyncPhase.ROUTED
            )
        elif invalid_event == "run":
            replace(execution, run_id=execution.run_id + 1).checkpoint(
                FIRST_LAYER_ID, CAMAsyncPhase.ROUTED
            )
        elif invalid_event == "duplicate":
            execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.ROUTED)
            execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.ROUTED)
        return "unexpected completion"

    with pytest.raises(RuntimeError, match="CAMAsync"):
        scheduler.run(task, [FIRST_LAYER_ID])
    assert scheduler.quiescent
    with pytest.raises(RuntimeError, match="FAILED"):
        scheduler.run(task, [FIRST_LAYER_ID])


@pytest.mark.parametrize("stage_idx", [0, 1])
@pytest.mark.parametrize("phase", list(CAMAsyncPhase))
@pytest.mark.parametrize("when", ["before", "after"])
def test_exception_at_every_checkpoint_cancels_peer(scheduler, stage_idx, phase, when):
    error = ValueError(f"stage {stage_idx}: {when} {phase.value}")
    events: list[str | tuple[int, CAMAsyncPhase]] = []

    def task(execution):
        for checkpoint in CAMAsyncPhase:
            if (
                execution.stage_idx == stage_idx
                and checkpoint is phase
                and when == "before"
            ):
                events.append("failed")
                raise error
            events.append((execution.stage_idx, checkpoint))
            execution.checkpoint(FIRST_LAYER_ID, checkpoint)
            if (
                execution.stage_idx == stage_idx
                and checkpoint is phase
                and when == "after"
            ):
                events.append("failed")
                raise error
        return execution.stage_idx

    with pytest.raises(ValueError) as caught:
        scheduler.run(task, [FIRST_LAYER_ID])
    assert caught.value is error
    assert events[-1] == "failed"
    assert scheduler.quiescent
    with pytest.raises(RuntimeError, match="FAILED"):
        scheduler.run(task, [FIRST_LAYER_ID])


@pytest.mark.parametrize("during", ["activate", "context_enter", "context_exit"])
def test_context_failure_cancels_waiters(scheduler, during):
    error = RuntimeError(during)

    def activate(_execution):
        if during == "activate":
            raise error

    @contextmanager
    def thread_context():
        if during == "context_enter":
            raise error
        yield
        if during == "context_exit":
            raise error

    def task(execution):
        for phase in CAMAsyncPhase:
            execution.checkpoint(FIRST_LAYER_ID, phase)
        return execution.stage_idx

    with pytest.raises(RuntimeError) as caught:
        scheduler.run(
            task, [FIRST_LAYER_ID], activate=activate, thread_context=thread_context
        )
    assert caught.value is error
    assert scheduler.quiescent


@pytest.mark.parametrize("stop_method", ["cancel", "shutdown"])
def test_cancellation_wakes_waiters_without_granting_permission(scheduler, stop_method):
    stage_one_entered = threading.Event()
    release_stage_one = threading.Event()
    errors = []
    completed = []

    def task(execution):
        if execution.stage_idx == 1:
            stage_one_entered.set()
            assert release_stage_one.wait(TEST_WAIT_SECONDS)
        execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.ROUTED)
        execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.DISPATCHED)
        completed.append(execution.stage_idx)
        execution.layer_done(FIRST_LAYER_ID)
        return execution.stage_idx

    def run():
        try:
            scheduler.run(task, [FIRST_LAYER_ID])
        except BaseException as error:
            errors.append(error)

    controller = threading.Thread(target=run, daemon=True)
    controller.start()
    try:
        assert stage_one_entered.wait(TEST_WAIT_SECONDS)
        if stop_method == "cancel":
            scheduler.cancel()
        else:
            with pytest.raises(RuntimeError, match="process cleanup required"):
                scheduler.shutdown(timeout=0.01)
        release_stage_one.set()
        controller.join(TEST_WAIT_SECONDS)
        assert not controller.is_alive()
        assert len(errors) == 1
        assert "cancelled" in str(errors[0])
        assert completed == []
        assert scheduler.quiescent
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
        assert all(not thread.is_alive() for thread in scheduler._threads)
        with pytest.raises(RuntimeError, match="FAILED.*cancelled"):
            scheduler.run(task, [FIRST_LAYER_ID])
    finally:
        release_stage_one.set()
        controller.join(TEST_WAIT_SECONDS)


def test_success_releases_task_context_and_results(scheduler):
    class Payload:
        pass

    class Task:
        def __init__(self):
            self.payload = Payload()

        def __call__(self, execution):
            for phase in CAMAsyncPhase:
                execution.checkpoint(FIRST_LAYER_ID, phase)
            return self.payload

    task = Task()
    task_ref = weakref.ref(task)
    payload_ref = weakref.ref(task.payload)
    results = scheduler.run(task, [FIRST_LAYER_ID])
    assert results[0] is results[1] is task.payload
    del task, results
    gc.collect()
    assert task_ref() is None
    assert payload_ref() is None
    assert scheduler._activate is None
    assert scheduler._error is None


def test_scheduler_rejects_empty_layer_sequence_without_starting_threads(scheduler):
    with pytest.raises(ValueError, match="at least one MoE layer"):
        scheduler.run(lambda execution: execution.stage_idx, [])
    assert scheduler._threads == []
    assert scheduler.quiescent


def test_single_stage_context_checkpoints_are_local():
    execution = CAMAsyncExecutionContext(0, 0, 1, False)
    for phase in CAMAsyncPhase:
        execution.checkpoint(FIRST_LAYER_ID, phase)
    execution.layer_done(FIRST_LAYER_ID)


def test_scheduler_import_needs_no_torch_or_vllm():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import afd_plugin.model_executor.npu.async_cam_execution; "
            "assert 'torch' not in sys.modules; "
            "assert 'torch_npu' not in sys.modules; "
            "assert 'vllm' not in sys.modules; "
            "assert 'vllm_ascend' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=TEST_WAIT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _install_forward_context_module(monkeypatch, parent):
    module = SimpleNamespace(_forward_context=parent)
    module.get_forward_context = lambda: module._forward_context
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(forward_context=module))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)
    return module


def test_runtime_context_preserves_cpu_modes_and_stage_identity(scheduler, monkeypatch):
    torch = pytest.importorskip("torch")
    parent = SimpleNamespace(additional_kwargs={"request": "parent"})
    context_module = _install_forward_context_module(monkeypatch, parent)
    original_modes = (
        torch.is_inference_mode_enabled(),
        torch.is_grad_enabled(),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_cache_enabled(),
    )
    modes = [
        (False, True, False, torch.bfloat16, True),
        (True, False, True, torch.bfloat16, False),
        (False, False, True, torch.float16, True),
        (False, True, False, torch.bfloat16, True),
    ]
    worker_ids: set[int] = set()
    restored_worker_ids: list[int] = []

    def mode_snapshot():
        return (
            torch.is_inference_mode_enabled(),
            torch.is_grad_enabled(),
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        )

    for request_id, expected in enumerate(modes):
        inference, grad, autocast, dtype, cache = expected
        contexts = [
            SimpleNamespace(additional_kwargs={"request": request_id, "stage": stage})
            for stage in range(2)
        ]
        with (
            torch.inference_mode(inference),
            torch.set_grad_enabled(grad),
            torch.autocast("cpu", enabled=autocast, dtype=dtype, cache_enabled=cache),
        ):
            runtime = CAMAsyncRuntimeContext(contexts, torch.device("cpu"))

            @contextmanager
            def checked_thread_context(runtime=runtime):
                before = mode_snapshot()
                with runtime.thread_context():
                    yield
                assert mode_snapshot() == before
                restored_worker_ids.append(threading.get_ident())

            def check_context(execution, expected=expected, contexts=contexts):
                worker_ids.add(threading.get_ident())
                assert (
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                    torch.is_autocast_enabled("cpu"),
                    torch.get_autocast_dtype("cpu"),
                    torch.is_autocast_cache_enabled(),
                ) == expected
                assert context_module._forward_context is contexts[execution.stage_idx]
                assert require_cam_async_execution_context() is execution
                value = torch.ones((2, 2))
                product = value @ value
                assert product.dtype == (expected[3] if expected[2] else torch.float32)

            def task(execution, check_context=check_context):
                for phase in CAMAsyncPhase:
                    check_context(execution)
                    execution.checkpoint(FIRST_LAYER_ID, phase)
                    check_context(execution)
                return execution.stage_idx

            try:
                assert scheduler.run(
                    task,
                    [FIRST_LAYER_ID],
                    activate=runtime.activate,
                    thread_context=checked_thread_context,
                ) == (0, 1)
            finally:
                # Parent model code restores this global only after workers quiesce.
                assert scheduler.quiescent
                context_module._forward_context = parent
            assert torch.is_inference_mode_enabled() is inference
            assert torch.is_grad_enabled() is grad
            assert torch.is_autocast_enabled("cpu") is autocast
            assert torch.get_autocast_dtype("cpu") is dtype
            assert torch.is_autocast_cache_enabled() is cache
        assert context_module._forward_context is parent
        assert (
            torch.is_inference_mode_enabled(),
            torch.is_grad_enabled(),
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
            torch.is_autocast_cache_enabled(),
        ) == original_modes
    assert len(worker_ids) == 2
    assert len(restored_worker_ids) == 2 * len(modes)
    assert set(restored_worker_ids) == worker_ids


def test_runtime_context_binds_npu_before_reusing_parent_stream(scheduler, monkeypatch):
    torch = pytest.importorskip("torch")
    parent_thread = threading.get_ident()
    parent = SimpleNamespace(additional_kwargs={"request": "parent"})
    context_module = _install_forward_context_module(monkeypatch, parent)
    parent_stream = object()
    device = SimpleNamespace(type="npu", index=3)
    events = []

    def current_stream(requested_device):
        assert requested_device is device
        assert threading.get_ident() == parent_thread
        events.append(("capture", parent_thread, parent_stream))
        return parent_stream

    def set_device(requested_device):
        assert requested_device is device
        assert threading.get_ident() != parent_thread
        events.append(("device", threading.get_ident(), requested_device))

    def set_stream(stream):
        assert stream is parent_stream
        assert events[-1] == ("device", threading.get_ident(), device)
        events.append(("stream", threading.get_ident(), stream))

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=current_stream, set_device=set_device, set_stream=set_stream
        ),
        raising=False,
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _device: False)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda _device: torch.float16)
    contexts = [
        SimpleNamespace(additional_kwargs={"stage": stage}) for stage in range(2)
    ]
    runtime = CAMAsyncRuntimeContext(contexts, device)
    activation_counts = []

    def task(execution):
        for phase in CAMAsyncPhase:
            assert context_module._forward_context is contexts[execution.stage_idx]
            assert require_cam_async_execution_context() is execution
            execution.checkpoint(FIRST_LAYER_ID, phase)
            assert context_module._forward_context is contexts[execution.stage_idx]
            assert require_cam_async_execution_context() is execution
        return execution.stage_idx

    for _ in range(2):
        try:
            assert scheduler.run(task, [FIRST_LAYER_ID], activate=runtime.activate) == (
                0,
                1,
            )
        finally:
            assert scheduler.quiescent
            context_module._forward_context = parent
        activation_counts.append(sum(event[0] == "stream" for event in events))
    assert events[0] == ("capture", parent_thread, parent_stream)
    assert [event[0] for event in events[1:]] == ["device", "stream"] * 16
    assert activation_counts == [8, 16]
    assert context_module._forward_context is parent


def test_single_stage_guard_is_lazy_and_does_not_wake_reused_workers(
    scheduler, monkeypatch
):
    errors: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    single_stage_live = threading.Event()
    worker_rechecked = [threading.Event(), threading.Event()]
    original_wait_for = scheduler._condition.wait_for

    def observe_wait_for(predicate, timeout=None):
        def observed_predicate():
            if single_stage_live.is_set():
                for stage_idx, worker in enumerate(scheduler._threads):
                    if worker is threading.current_thread():
                        worker_rechecked[stage_idx].set()
            return predicate()

        return original_wait_for(observed_predicate, timeout)

    monkeypatch.setattr(scheduler._condition, "wait_for", observe_wait_for)
    with scheduler.single_stage(use_sequence_parallel=True) as execution:
        assert (execution.run_id, execution.stage_idx, execution.num_stages) == (
            1,
            0,
            1,
        )
        assert execution.use_sequence_parallel
        assert scheduler._threads == []
        for phase in CAMAsyncPhase:
            execution.checkpoint(FIRST_LAYER_ID, phase)
        with (
            pytest.raises(RuntimeError, match="active"),
            scheduler.single_stage(use_sequence_parallel=False),
        ):
            pytest.fail("nested execution was admitted")

    def task(execution):
        for phase in CAMAsyncPhase:
            execution.checkpoint(FIRST_LAYER_ID, phase)
        return execution.run_id, execution.stage_idx

    assert scheduler.run(task, [FIRST_LAYER_ID]) == ((2, 0), (2, 1))
    workers = tuple(scheduler._threads)
    with scheduler.single_stage(use_sequence_parallel=False) as execution:
        assert execution.run_id == 3
        assert not execution.use_sequence_parallel
        single_stage_live.set()
        # Observe both idle workers reject this single-stage task after a wakeup.
        with scheduler._condition:
            scheduler._condition.notify_all()
        for rechecked in worker_rechecked:
            assert rechecked.wait(TEST_WAIT_SECONDS)
        for phase in CAMAsyncPhase:
            execution.checkpoint(FIRST_LAYER_ID, phase)
        single_stage_live.clear()
    assert scheduler.quiescent
    assert tuple(scheduler._threads) == workers
    assert scheduler.run(task, [FIRST_LAYER_ID]) == ((4, 0), (4, 1))
    assert errors == []
    assert all(worker.is_alive() for worker in workers)


def test_single_stage_failure_poisoning_is_shared_with_two_stage(scheduler):
    error = ValueError("single stage failed after dispatch")
    with (
        pytest.raises(ValueError) as caught,
        scheduler.single_stage(use_sequence_parallel=False) as execution,
    ):
        execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.ROUTED)
        execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.DISPATCHED)
        raise error
    assert caught.value is error
    assert scheduler.quiescent
    assert scheduler._threads == []
    with (
        pytest.raises(RuntimeError, match="FAILED"),
        scheduler.single_stage(use_sequence_parallel=False),
    ):
        pytest.fail("failed scheduler accepted single stage")
    with pytest.raises(RuntimeError, match="FAILED"):
        scheduler.run(lambda execution: execution.stage_idx, [FIRST_LAYER_ID])
    scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
    scheduler.shutdown(timeout=TEST_WAIT_SECONDS)


@pytest.mark.parametrize("stop_method", ["cancel", "shutdown"])
def test_single_stage_cancellation_prevents_dispatch_and_waits_for_exit(
    scheduler, stop_method
):
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    dispatched = []

    def run():
        try:
            with scheduler.single_stage(use_sequence_parallel=False) as execution:
                assert execution.scheduler is None
                entered.set()
                assert release.wait(TEST_WAIT_SECONDS)
                execution.checkpoint(FIRST_LAYER_ID, CAMAsyncPhase.ROUTED)
                dispatched.append(True)
        except BaseException as error:
            errors.append(error)

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    try:
        assert entered.wait(TEST_WAIT_SECONDS)
        assert not scheduler.quiescent
        if stop_method == "cancel":
            scheduler.cancel()
        else:
            with pytest.raises(RuntimeError, match="process cleanup required"):
                scheduler.shutdown(timeout=0.01)
        assert not scheduler.quiescent
        assert caller.is_alive()
        release.set()
        caller.join(TEST_WAIT_SECONDS)
        assert not caller.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "cancelled" in str(errors[0])
        assert dispatched == []
        assert scheduler.quiescent
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
        with (
            pytest.raises(RuntimeError, match="FAILED.*cancelled"),
            scheduler.single_stage(use_sequence_parallel=False),
        ):
            pytest.fail("closed scheduler accepted single stage")
    finally:
        release.set()
        caller.join(TEST_WAIT_SECONDS)


def test_late_worker_exit_releases_abandoned_run_before_shutdown():
    scheduler = CAMAsyncUbatchScheduler(wait_timeout=0.05)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def task(execution):
        if execution.stage_idx == 1:
            entered.set()
            assert release.wait(TEST_WAIT_SECONDS)
        for phase in CAMAsyncPhase:
            execution.checkpoint(FIRST_LAYER_ID, phase)
        return execution.stage_idx

    def run():
        try:
            scheduler.run(task, [FIRST_LAYER_ID])
        except BaseException as error:
            errors.append(error)

    controller = threading.Thread(target=run, daemon=True)
    controller.start()
    try:
        assert entered.wait(TEST_WAIT_SECONDS)
        controller.join(TEST_WAIT_SECONDS)
        assert not controller.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "process cleanup required" in str(errors[0])
        assert isinstance(errors[0].__cause__, TimeoutError)
        assert errors[0].__cause__ is scheduler._error
        assert not scheduler.quiescent
        with pytest.raises(RuntimeError, match="active|FAILED"):
            scheduler.run(task, [FIRST_LAYER_ID])

        # The controller has returned; only the last worker can release the run.
        release.set()
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
        assert scheduler.quiescent
        assert all(not worker.is_alive() for worker in scheduler._threads)
        assert scheduler._task is None
        assert scheduler._activate is None
        assert scheduler._executions == []
        assert scheduler._events == [None, None]
        assert scheduler._results == [None, None]
        assert scheduler._error is None
        with pytest.raises(RuntimeError, match="closed|FAILED"):
            scheduler.run(task, [FIRST_LAYER_ID])
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
    finally:
        release.set()
        controller.join(TEST_WAIT_SECONDS)
        scheduler.shutdown(timeout=TEST_WAIT_SECONDS)


@pytest.mark.parametrize("failed_start", [1, 2])
def test_thread_start_failure_is_terminal_without_publishing_task(
    scheduler, monkeypatch, failed_start
):
    error = RuntimeError("can't start new thread")
    original_start = threading.Thread.start
    attempted: list[threading.Thread] = []
    task_calls = []

    def fail_start(thread):
        attempted.append(thread)
        assert scheduler._task is None
        assert scheduler._executions == []
        assert scheduler._remaining == 0
        if len(attempted) == failed_start:
            raise error
        original_start(thread)

    def task(execution):
        task_calls.append(execution.stage_idx)
        return execution.stage_idx

    @contextmanager
    def thread_context():
        pytest.fail("failed startup entered the task context")
        yield

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError) as caught:
        scheduler.run(
            task,
            [FIRST_LAYER_ID],
            activate=lambda execution: None,
            thread_context=thread_context,
        )
    assert caught.value is error
    assert task_calls == []
    assert len(attempted) == failed_start
    assert scheduler._run_id == 0
    assert scheduler._thread_context is nullcontext
    assert len(scheduler._threads) == failed_start - 1
    assert all(not thread.is_alive() for thread in attempted)
    assert scheduler.quiescent
    assert scheduler._failed
    assert scheduler._closed
    assert scheduler._remaining == 0
    assert scheduler._task is None
    assert scheduler._activate is None
    assert scheduler._executions == []
    assert scheduler._events == [None, None]
    assert scheduler._results == [None, None]
    assert scheduler._error is None
    scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
    scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
    with pytest.raises(RuntimeError, match="closed|FAILED"):
        scheduler.run(task, [FIRST_LAYER_ID])
    with (
        pytest.raises(RuntimeError, match="closed|FAILED"),
        scheduler.single_stage(use_sequence_parallel=False),
    ):
        pytest.fail("failed startup admitted a single-stage forward")


@pytest.mark.parametrize("failed_start", [1, 2])
def test_startup_cleanup_failure_preserves_startup_cause(
    scheduler, monkeypatch, failed_start
):
    startup_error = RuntimeError("can't start new thread")
    cleanup_error = RuntimeError("CAMAsync worker did not stop")
    original_start = threading.Thread.start
    starts = 0

    def fail_start(thread):
        nonlocal starts
        starts += 1
        if starts == failed_start:
            raise startup_error
        original_start(thread)

    def fail_shutdown():
        raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_start)
        patch.setattr(scheduler, "shutdown", fail_shutdown)
        with pytest.raises(RuntimeError) as caught:
            scheduler.run(lambda execution: execution.stage_idx, [FIRST_LAYER_ID])
        assert caught.value is cleanup_error
        assert caught.value.__cause__ is startup_error
    scheduler.shutdown(timeout=TEST_WAIT_SECONDS)
    assert scheduler.quiescent
    assert all(not worker.is_alive() for worker in scheduler._threads)


@pytest.mark.parametrize("single_stage", [False, True])
def test_failed_scheduler_keeps_reason_without_retaining_request(
    scheduler, single_stage
):
    class Payload:
        pass

    payload_refs = []

    def fail(execution):
        payload = Payload()
        payload_refs.append(weakref.ref(payload))
        raise ValueError("request failed after dispatch")

    with pytest.raises(ValueError, match="request failed after dispatch"):
        if single_stage:
            with scheduler.single_stage(use_sequence_parallel=False) as execution:
                fail(execution)
        else:
            scheduler.run(fail, [FIRST_LAYER_ID])
    assert scheduler.quiescent
    assert scheduler._error is None
    gc.collect()
    assert payload_refs
    assert all(reference() is None for reference in payload_refs)

    reason = "FAILED.*ValueError: request failed after dispatch"
    with pytest.raises(RuntimeError, match=reason):
        scheduler.run(fail, [FIRST_LAYER_ID])
    with (
        pytest.raises(RuntimeError, match=reason),
        scheduler.single_stage(use_sequence_parallel=False),
    ):
        pytest.fail("failed scheduler accepted single stage")
