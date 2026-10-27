# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import afd_plugin.v1.worker.ffn_model_runner as ffn_model_runner_module  # noqa: E402
from afd_plugin.connectors import AFDDPMetadata  # noqa: E402
from afd_plugin.connectors.gpu.fastafd import M2NShape  # noqa: E402
from afd_plugin.v1.worker import fastafd_ffn_model_runner as runner_module  # noqa: E402
from afd_plugin.v1.worker.fastafd_ffn_model_runner import (  # noqa: E402
    FastAFDExpertExecutor,
    FastAFDFFNModelRunner,
)

_SHAPE = M2NShape(
    num_attention_ranks=2,
    num_ffn_ranks=2,
    num_layers=40,
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=512,
    max_tokens_per_rank=1024,
)


def _lane(name):
    return SimpleNamespace(
        buffer=f"{name}.buffer",
        peer_ptrs=[f"{name}.peer"],
        max_tokens_per_rank=1152,
    )


class _Event:
    """CUDA event double; ``done`` is what ``query()`` reports."""

    done = True

    def record(self):
        pass

    def query(self):
        return self.done


class _Connector(SimpleNamespace):
    def close(self):
        self.closed = True
        self.lanes = []


def _runner(monkeypatch, *, num_lanes=2):
    launches = []
    ops = SimpleNamespace(mega_moe_m2n_eg=lambda *args: launches.append(args))
    monkeypatch.setattr(runner_module, "fastafd_ops", lambda: ops)
    monkeypatch.setattr(runner_module.torch.cuda, "Event", _Event)
    monkeypatch.setattr(
        runner_module,
        "FastAFDExpertWeights",
        lambda *_args: SimpleNamespace(
            descriptors="descriptors",
            l1_pointers="l1_pointers",
            num_layers=40,
            l1_nbytes=4096,
        ),
    )
    connector = _Connector(
        shape=_SHAPE,
        role_rank=1,
        union_rank=3,
        lanes=[_lane(f"lane{i}") for i in range(num_lanes)],
        closed=False,
    )
    runner = object.__new__(FastAFDFFNModelRunner)
    runner.prof = None
    runner.connector = connector
    runner.executor = FastAFDExpertExecutor(SimpleNamespace(), connector)
    return runner, launches


def _plan(*stage_tokens):
    return {
        stage: AFDDPMetadata(num_tokens_across_dp_cpu=tokens)
        for stage, tokens in enumerate(stage_tokens)
    }


def test_one_launch_serves_every_layer_and_lane(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.executor.submit(_plan([3, 9], [5, 0]))

    assert len(launches) == 1
    args = launches[0]
    assert args[:7] == ("descriptors", "l1_pointers", 40, 128, 2048, 512, 4096)
    assert args[9:13] == ("lane0.buffer", "lane1.buffer", None, None)
    assert args[13:17] == (
        ["lane0.peer"],
        ["lane1.peer"],
        ["lane0.peer"],
        ["lane0.peer"],
    )
    # union rank, M/N ranks, aligned capacity, experts, top-k, expected tokens.
    assert args[17:24] == (3, 2, 2, 1152, 256, 8, 9)


def test_single_stage_step_uses_only_the_first_lane(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.executor.submit(_plan([0, 0]))

    args = launches[0]
    assert args[9:13] == ("lane0.buffer", None, None, None)
    assert args[23] == 1


def test_graph_capture_steps_do_not_launch_the_expert_kernel(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.execute_model(dp_metadata_list=_plan([4, 4]), is_graph_capturing=True)
    assert launches == []

    runner.execute_model(dp_metadata_list=_plan([4, 4]))
    assert len(launches) == 1


def test_capture_model_serves_attention_warmup_only(monkeypatch):
    runner, launches = _runner(monkeypatch)

    assert runner.capture_model(_plan([4, 4]), is_warmup=False) == 0
    assert launches == []
    assert runner.capture_model(_plan([4, 4]), is_warmup=True) == 0
    assert len(launches) == 1


def test_steps_require_a_plan(monkeypatch):
    runner, _launches = _runner(monkeypatch)

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_dummy_weights_are_rejected():
    runner = object.__new__(FastAFDFFNModelRunner)

    with pytest.raises(ValueError, match="needs checkpoint expert weights"):
        runner.load_model(load_dummy_weights=True)


def test_step_plans_with_more_stages_than_lanes_are_rejected(monkeypatch):
    runner, launches = _runner(monkeypatch, num_lanes=1)

    with pytest.raises(RuntimeError, match="2 stages, the connector 1 lanes"):
        runner.executor.submit(_plan([1, 1], [1, 1]))
    assert launches == []


def test_steps_before_load_model_fail_clearly(monkeypatch):
    runner, _launches = _runner(monkeypatch)
    runner.executor = None

    with pytest.raises(RuntimeError, match="before load_model"):
        runner.execute_model(dp_metadata_list=_plan([1, 1]))


def _shutdown_ready(monkeypatch, runner):
    """State the base runner shutdown releases, with no graphs or model."""
    runner._cuda_graphs = {}
    runner._graph_memory_pool = None
    runner.model = None
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(ffn_model_runner_module, "_ROPE_DICT", {})
    monkeypatch.setattr(
        ffn_model_runner_module, "reset_workspace_manager", lambda: None
    )


def test_shutdown_after_a_finished_step_closes_connector_and_drops_experts(
    monkeypatch,
):
    runner, _launches = _runner(monkeypatch)
    _shutdown_ready(monkeypatch, runner)
    runner.execute_model(dp_metadata_list=_plan([1, 1]))

    runner.shutdown()

    assert runner.connector.closed
    assert runner.executor is None


def test_shutdown_keeps_connector_and_experts_while_a_step_is_running(monkeypatch):
    runner, _launches = _runner(monkeypatch)
    _shutdown_ready(monkeypatch, runner)
    executor = runner.executor
    runner.execute_model(dp_metadata_list=_plan([1, 1]))
    monkeypatch.setattr(_Event, "done", False)
    monkeypatch.setattr(runner_module, "_SHUTDOWN_DRAIN_S", 0.0)

    runner.shutdown()

    assert not runner.connector.closed
    assert runner.executor is executor


def test_drain_without_any_step_reports_finished(monkeypatch):
    runner, _launches = _runner(monkeypatch)

    assert runner.executor.drain(0.0)


def test_steps_after_drain_are_refused_before_any_launch(monkeypatch):
    runner, launches = _runner(monkeypatch)
    runner.executor.drain(0.0)

    with pytest.raises(RuntimeError, match="shutting down"):
        runner.executor.submit(_plan([1, 1]))
    assert launches == []


class _ObservedLock:
    """Lock that reports, by thread name, each thread that starts waiting."""

    def __init__(self):
        self._lock = threading.Lock()
        self.waiters: queue.Queue[str] = queue.Queue()

    def __enter__(self):
        self.waiters.put(threading.current_thread().name)
        self._lock.acquire()
        return self

    def __exit__(self, *exc_info):
        self._lock.release()


# Bounds the waits below so a regression fails instead of hanging the suite;
# the ordering itself is proven by the events and the observed lock.
_WATCHDOG_S = 10.0


def test_drain_started_during_a_launch_reports_that_launch(monkeypatch):
    """A drain on the shutdown thread must not miss a launch already admitted."""
    runner, _launches = _runner(monkeypatch)
    executor = runner.executor
    lock = _ObservedLock()
    executor._lock = lock  # observe who waits; the executor builds a plain Lock
    entered, release = threading.Event(), threading.Event()

    def blocking_launch(*args):
        entered.set()
        release.wait(_WATCHDOG_S)

    ops = SimpleNamespace(mega_moe_m2n_eg=blocking_launch)
    monkeypatch.setattr(runner_module, "fastafd_ops", lambda: ops)
    # The launch that is in flight when the drain starts never finishes.
    monkeypatch.setattr(_Event, "done", False)
    drained: list[bool] = []
    submitter = threading.Thread(
        target=executor.submit, args=(_plan([1, 1]),), name="submitter"
    )
    drainer = threading.Thread(
        target=lambda: drained.append(executor.drain(0.0)), name="drainer"
    )
    submitter.start()
    try:
        assert entered.wait(_WATCHDOG_S), "the launch never started"
        assert lock.waiters.get(timeout=_WATCHDOG_S) == "submitter"
        drainer.start()
        try:
            waiter = lock.waiters.get(timeout=_WATCHDOG_S)
        except queue.Empty:
            pytest.fail("drain did not wait for the launch in progress")
        assert waiter == "drainer"
    finally:
        release.set()
        submitter.join(_WATCHDOG_S)
        if drainer.ident is not None:
            drainer.join(_WATCHDOG_S)

    assert drained == [False]
