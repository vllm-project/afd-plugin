# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.base import FFNExecutionSession
from tests.unit.fakes.dp_metadata import (  # noqa: E402
    DPMetadataValues as _FakeDPMetadata,
)
from tests.unit.fakes.dp_metadata import dp_tokens as _tokens
from tests.unit.fakes.graph import Graph as _FakeGraph  # noqa: E402
from tests.unit.fakes.vllm_config import (  # noqa: E402
    npu_vllm_config as _vllm_config,
)


def _ffn_payload(hidden_states, metadata, states=None):
    return AFDA2FTransferPayload(
        hidden_states=hidden_states,
        context=AFDTransferContext(
            metadata=metadata,
            states=states if states is not None else AFDTransferState(),
        ),
    )


@contextmanager
def _fake_ffn_ascend_forward_context(**_kwargs):
    """Stand in for the vLLM-Ascend forward context in FFN runner unit tests.

    The real ``ascend_forward_context`` delegates to vLLM-Ascend internals that
    read many ``vllm_config`` fields; these unit tests only exercise the FFN
    runner's recv/compute/send orchestration, so a minimal fake context is used.
    """
    yield SimpleNamespace(additional_kwargs={}, dp_metadata=None, all_moe_layers={})


def _patch_ffn_forward_context(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )


class _FakeFFNConnector:
    def __init__(self, *, attn_size=1, ffn_size=1, role_rank=0, world_rank=0):
        self.dp_metadata_list = {}
        self.attn_outputs: deque[
            AFDA2FTransferPayload | tuple[object, AFDTransferMetadata]
        ] = deque()
        self.ffn_outputs = []
        self.updates = []
        self.attn_size = attn_size
        self.ffn_size = ffn_size
        self.world_rank = world_rank
        self.topology = SimpleNamespace(role_rank=role_rank)
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_list = dict(payload.dp_metadata_list)
        self.updates.append(
            (
                dict(payload.dp_metadata_list),
                {
                    "is_graph_capturing": payload.is_graph_capturing,
                    "is_warmup": payload.is_warmup,
                },
            ),
        )

    def recv_attn_output(self, ubatch_idx=None, **kwargs):
        for item in tuple(self.attn_outputs):
            payload = (
                item
                if isinstance(item, AFDA2FTransferPayload)
                else _ffn_payload(item[0], item[1])
            )
            if payload.context.metadata.stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return payload
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, context, **kwargs):
        self.ffn_outputs.append((ffn_output, context.metadata, kwargs))

    def close(self):
        return None


class _FakeModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeStructuredFFNModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **_kwargs):
        return AFDF2ATransferPayload(
            routed_output=f"routed({hidden_states}, layer={layer_idx})",
            shared_output=f"shared({hidden_states}, layer={layer_idx})",
        )


def _require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")


def _new_ffn_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

    # object.__new__ bypasses __init__, which is where the runner would set up
    # the profiler and device; provide inert defaults the runtime paths expect.
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.prof = None
    runner.device = SimpleNamespace(type="npu")
    runner._is_shutdown = False
    runner.effective_mode = "eager"
    # _ffn_layer_indices reads these; production runs with the gate on the
    # Attention side, so every layer index is a MoE FFN layer here.
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=True)
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            n_routed_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
        ),
    )
    return runner


def _new_ffn_worker():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

    worker = object.__new__(AFDNPUFFNWorker)
    worker._ffn_loop_error = None
    # Most tests exercise the daemon loop rather than CPU placement. Tests for
    # the startup binding path explicitly reset this guard.
    worker._cpu_binding_attempted = True
    return worker


def test_npu_ffn_runner_executes_eager_ffn_step(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert len(runner.connector.updates) == 1
    update_metadata, update_flags = runner.connector.updates[0]
    assert update_metadata == {0: runner.connector.dp_metadata_list[0]}
    assert update_flags == {"is_graph_capturing": False, "is_warmup": False}
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


@pytest.mark.parametrize(
    ("attention_counts", "ffn_counts"),
    [
        ([[6], [7]], [[6], [7]]),
        ([[2, 3, 5, 7], [3, 8, 1, 7]], [[7, 10], [4, 15]]),
    ],
)
def test_npu_ffn_runner_builds_forward_context_for_each_dbo_stage(
    monkeypatch, attention_counts, ffn_counts
):  # long-test: native Ascend context consumes distinct stage token counts
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    context_calls = []

    @contextmanager
    def fake_ascend_forward_context(**kwargs):
        context_calls.append(kwargs)
        native_dp_metadata = object()
        context = SimpleNamespace(
            additional_kwargs={},
            dp_metadata=native_dp_metadata,
            all_moe_layers={},
        )
        yield context
        assert context.dp_metadata is native_dp_metadata

    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        fake_ascend_forward_context,
    )
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.vllm_config.parallel_config.data_parallel_size = len(ffn_counts[0])
    runner.connector = _FakeFFNConnector(
        attn_size=max(2, len(attention_counts[0])), ffn_size=2
    )
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 16
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    for stage_idx, counts in enumerate(ffn_counts):
        metadata = AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=stage_idx,
            seq_len=counts[0],
        )
        runner.connector.attn_outputs.append((f"hidden-{stage_idx}", metadata))

    runner.execute_model(
        dp_metadata_list={
            stage: _FakeDPMetadata(counts)
            for stage, counts in enumerate(attention_counts)
        },
        is_profile=True,
    )

    assert [call["num_tokens"] for call in context_calls] == [
        counts[0] for counts in ffn_counts
    ]
    assert [call["in_profile_run"] for call in context_calls] == [True, True]
    assert [
        call["num_tokens_across_dp"].tolist() for call in context_calls
    ] == ffn_counts


def test_npu_ffn_runner_sends_structured_shared_output(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeStructuredFFNModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        (
            "routed(hidden, layer=0)",
            metadata,
            {
                "ubatch_idx": 0,
                "expand_x_shared": "shared(hidden, layer=0)",
            },
        ),
    ]


def test_npu_ffn_runner_replays_acl_graph_when_key_exists():
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([1])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}

    runner.execute_model(
        dp_metadata_list=dp_metadata,
        is_graph_replaying=True,
    )

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_npu_ffn_runner_skips_replay_when_attention_is_eager(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    dp_metadata = {0: _FakeDPMetadata([1])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 0
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_falls_back_to_eager_on_acl_graph_miss(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_warmup_uses_eager_forward_without_graph(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    capture_flags: list[bool] = []

    def fail_capture_stream(*args, **kwargs):
        raise AssertionError("warmup must not create a capture stream")

    monkeypatch.setattr(ffn_model_runner.torch.npu, "Stream", fail_capture_stream)
    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "set_cudagraph_capturing_enabled",
        capture_flags.append,
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "mem_get_info", lambda: (0, 0))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_warmup=True,
    )

    assert capture_flags == [True, False]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_capture_stores_acl_graph_and_skips_duplicate_state_update(
    monkeypatch,
):  # long-test: native capture setup observes one peer control publication
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _FakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = True
    runner._acl_graphs = {}
    runner.graph_pool = None
    monkeypatch.setattr(
        ffn_model_runner,
        "ascend_forward_context",
        _fake_ffn_ascend_forward_context,
    )
    stream = SimpleNamespace(wait_stream=lambda current: None)
    monkeypatch.setattr(ffn_model_runner.torch.npu, "Stream", lambda **kw: stream)
    monkeypatch.setattr(
        ffn_model_runner.torch.npu, "current_stream", lambda device: None
    )
    monkeypatch.setattr(
        ffn_model_runner.torch.npu, "stream", lambda stream: nullcontext()
    )
    monkeypatch.setattr(
        ffn_model_runner.torch.npu,
        "graph",
        lambda graph, pool: nullcontext(),
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "NPUGraph", _FakeGraph)
    monkeypatch.setattr(
        ffn_model_runner,
        "set_cudagraph_capturing_enabled",
        lambda enabled: None,
    )
    monkeypatch.setattr(ffn_model_runner.torch.npu, "mem_get_info", lambda: (0, 0))
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner.connector.attn_outputs.append(("hidden", metadata))

    runner.execute_ffn_step(
        dp_metadata_list=dp_metadata,
        is_graph_capturing=True,
    )

    assert len(runner.connector.updates) == 1
    update_metadata, update_flags = runner.connector.updates[0]
    assert sorted(update_metadata) == [0]
    assert _tokens(update_metadata[0]) == [1]
    assert update_flags == {"is_graph_capturing": True, "is_warmup": False}


def test_npu_ffn_capture_cache_hit_still_updates_state_and_executes(monkeypatch):
    _require_npu_runtime()
    from vllm.config import CUDAGraphMode

    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    runner.connector = _FakeFFNConnector()
    runner.max_num_tokens = 8
    dp_metadata = {0: _FakeDPMetadata([4]), 1: _FakeDPMetadata([4])}
    graph = _FakeGraph()
    runner._acl_graphs = {runner._make_graph_key(dp_metadata): {"graph": graph}}

    def reject_new_capture():
        raise AssertionError("an identical FFN shape must reuse its graph")

    monkeypatch.setattr(ffn_model_runner.torch.npu, "NPUGraph", reject_new_capture)
    # Different Attention descriptors can share this FFN key. Each payload
    # nevertheless requires FFN communication, including repeated capture.
    for is_capturing in (True, False, True):
        runner._capture_graphs(
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            dp_metadata_list=dp_metadata,
            is_attn_graph_capturing=is_capturing,
        )

    assert graph.replay_count == 3
    assert [flags["is_graph_capturing"] for _, flags in runner.connector.updates] == [
        True,
        False,
        True,
    ]


def test_npu_ffn_runner_shutdown_is_idempotent(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    runner = _new_ffn_runner()
    close_calls = []
    profiler_calls = []
    free_calls: list[SimpleNamespace] = []
    reset_calls = []
    runner.vllm_config = _vllm_config(role="ffn")
    runner._acl_graphs = {
        ((0, (1,)),): {"graph": SimpleNamespace(reset=lambda: reset_calls.append(True))}
    }
    runner.graph_pool = "pool"
    runner.model = _FakeModel()
    runner.connector = SimpleNamespace(
        is_initialized=True,
        close=lambda: close_calls.append(True),
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "stop_afd_npu_profiler",
        lambda profiler: profiler_calls.append(profiler),
    )
    from vllm.v1.worker.gpu import shutdown

    monkeypatch.setattr(shutdown, "free_before_shutdown", free_calls.append)

    runner.shutdown()
    runner.shutdown()

    assert close_calls == [True]
    assert profiler_calls == [None]
    assert free_calls == [runner.vllm_config]
    assert reset_calls == [True]
    assert runner.graph_pool is None
    assert runner.model is None


def test_npu_ffn_worker_scheduler_execute_model_fails_fast():
    worker = _new_ffn_worker()

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_npu_ffn_worker_reports_zero_compilation_times():
    worker = _new_ffn_worker()

    compilation_times = worker.compile_or_warm_up_model()

    assert compilation_times.language_model == 0.0
    assert compilation_times.encoder == 0.0


@pytest.mark.parametrize("enable_cpu_binding", [False, True])
def test_npu_ffn_worker_start_binds_physical_npu_once_before_daemon(
    monkeypatch,
    enable_cpu_binding,
):  # long-test: native physical-device binding precedes daemon creation
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    worker._cpu_binding_attempted = False
    worker._ffn_thread = None
    worker.local_rank = 3
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True, control_plane=object()),
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        ffn_worker,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_cpu_binding=enable_cpu_binding),
    )

    def map_physical_npu(rank):
        events.append(("map", rank))
        return 11

    monkeypatch.setattr(
        ffn_worker,
        "current_platform",
        SimpleNamespace(
            device_id_to_physical_device_id=map_physical_npu,
        ),
    )

    def record_binding(rank, *, npu_id):
        events.append(("bind", (rank, npu_id)))

    class _NonRunningThread:
        def __init__(self, *, target, name, daemon):
            events.append(("thread", (target, name, daemon)))

        def start(self):
            events.append(("start", None))

        def is_alive(self):
            return False

    monkeypatch.setattr(ffn_worker, "bind_cpus", record_binding)
    monkeypatch.setattr(ffn_worker.threading, "Thread", _NonRunningThread)

    worker.start_ffn_server_loop()
    worker.start_ffn_server_loop()

    binding_events = [event for event in events if event[0] in ("map", "bind")]
    expected_binding_events = (
        [("map", 3), ("bind", (3, 11))] if enable_cpu_binding else []
    )
    assert binding_events == expected_binding_events
    first_thread_index = next(
        i for i, event in enumerate(events) if event[0] == "thread"
    )
    if enable_cpu_binding:
        assert events.index(("bind", (3, 11))) < first_thread_index
    assert sum(event[0] == "start" for event in events) == 2


def test_npu_ffn_worker_cpu_binding_failure_does_not_abort_daemon_start(
    monkeypatch,
    caplog,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    worker._cpu_binding_attempted = False
    worker._ffn_thread = None
    worker.local_rank = 5
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True, control_plane=object()),
    )
    thread_starts: list[bool] = []
    monkeypatch.setattr(
        ffn_worker,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_cpu_binding=True),
    )
    monkeypatch.setattr(
        ffn_worker,
        "current_platform",
        SimpleNamespace(device_id_to_physical_device_id=lambda _rank: 13),
    )

    def fail_binding(_local_rank, *, npu_id):
        raise RuntimeError("binding unavailable")

    class _NonRunningThread:
        def __init__(self, *, target, name, daemon):
            pass

        def start(self):
            thread_starts.append(True)

        def is_alive(self):
            return False

    monkeypatch.setattr(ffn_worker, "bind_cpus", fail_binding)
    monkeypatch.setattr(ffn_worker.threading, "Thread", _NonRunningThread)

    with caplog.at_level(logging.WARNING, logger=ffn_worker.__name__):
        worker.start_ffn_server_loop()
        worker.start_ffn_server_loop()

    assert thread_starts == [True, True]
    assert "Bind cpus failed in rank5: binding unavailable" in caplog.text
    assert caplog.text.count("Bind cpus failed") == 1


def test_npu_ffn_worker_loop_error_is_propagated(caplog):
    worker = _new_ffn_worker()
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True, control_plane=object()),
    )

    expected_error = RuntimeError("boom")

    def fail_loop():
        raise expected_error

    worker._run_ffn_server_loop = fail_loop

    with caplog.at_level(
        logging.ERROR,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker.start_ffn_server_loop()
        cast(threading.Thread, worker._ffn_thread).join(timeout=5)

    with pytest.raises(RuntimeError, match="AFD NPU FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()

    assert exc.value.__cause__ is expected_error
    assert "AFD NPU FFN worker loop failed" in caplog.text


def test_npu_ffn_worker_ignores_receive_error_during_shutdown(caplog):
    worker = _new_ffn_worker()
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True, control_plane=object()),
    )

    def stop_while_receiving():
        worker._ffn_shutdown_event.set()
        raise RuntimeError("CAM recv interrupted by connector close")

    worker._run_ffn_server_loop = stop_while_receiving

    with caplog.at_level(
        logging.ERROR,
        logger="afd_plugin.v1.worker.npu.ffn_worker",
    ):
        worker.start_ffn_server_loop()
        cast(threading.Thread, worker._ffn_thread).join(timeout=5)

    worker.raise_ffn_loop_error_if_any()
    assert "AFD NPU FFN worker loop failed" not in caplog.text


def test_npu_ffn_worker_stops_loop_before_parent_shutdown(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    worker._ffn_shutdown_event = threading.Event()
    calls = []

    class _StoppingThread:
        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return False

    connector = SimpleNamespace(
        control_plane=object(), close=lambda: calls.append(("close", None))
    )
    worker._ffn_thread = _StoppingThread()
    worker.model_runner = SimpleNamespace(connector=connector)
    monkeypatch.setattr(
        ffn_worker.NPUWorker,
        "shutdown",
        lambda self: calls.append(("parent", None)),
    )

    worker.shutdown()

    assert calls == [
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("parent", None),
    ]


def test_npu_ffn_worker_preserves_live_thread_after_shutdown_timeout(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_worker

    worker = _new_ffn_worker()
    shutdown_event = threading.Event()
    worker._ffn_shutdown_event = shutdown_event
    calls = []

    class _StoppingThread:
        alive = True

        def join(self, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return self.alive

    thread = _StoppingThread()
    worker._ffn_thread = thread
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(
            control_plane=object(), close=lambda: calls.append(("close", None))
        ),
    )
    monkeypatch.setattr(
        ffn_worker.NPUWorker,
        "shutdown",
        lambda self: calls.append(("parent", None)),
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        worker.shutdown()

    assert shutdown_event.is_set()
    assert ("parent", None) not in calls

    thread.alive = False
    worker.shutdown()

    assert calls == [
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("close", None),
        ("join", ffn_worker.FFN_SHUTDOWN_TIMEOUT_SECONDS),
        ("parent", None),
    ]


def test_npu_ffn_worker_uses_connector_driven_loop_for_async_connector(monkeypatch):
    worker = _new_ffn_worker()
    from afd_plugin.v1.worker.npu import ffn_worker

    event = threading.Event()
    calls: list[str | tuple[str, SimpleNamespace]] = []

    def execute_connector_driven_step(shutdown_requested):
        if shutdown_requested:
            return False
        calls.append("step")
        event.set()
        return True

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = object.__new__(ffn_worker.AsyncNPUFFNModelRunner)
    worker.model_runner.connector = SimpleNamespace(control_plane=None)
    worker.model_runner.session = FFNExecutionSession()
    worker.model_runner.effective_mode = "eager"
    worker.model_runner.execute_connector_driven_step = execute_connector_driven_step

    monkeypatch.setattr(
        ffn_worker.torch.npu,
        "set_device",
        lambda device: calls.append(("bind", device)),
    )
    monkeypatch.setattr(
        ffn_worker.torch.npu, "synchronize", lambda: calls.append("sync")
    )
    worker._run_ffn_server_loop()

    assert calls == [("bind", worker.device), "step", "sync"]


class _LifecycleConnector:
    """Connector fake recording lifecycle events for load-order tests."""

    def __init__(self, events, control_plane=None):
        self.events = events
        self.control_plane = control_plane
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    def init_afd_connector(self):
        self.events.append("connector_init")
        self._initialized = True


@pytest.mark.parametrize("use_ubatching", [False, True])
def test_npu_attention_runner_load_model_initializes_connector_after_weights(
    monkeypatch,
    use_ubatching,
):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import (
        attention_model_runner_v2 as attention_model_runner,
    )

    events: list[str] = []
    connector = _LifecycleConnector(events)
    runner = object.__new__(attention_model_runner.AFDNPUAttentionModelRunnerV2)
    runner.connector = connector
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=use_ubatching),
    )
    monkeypatch.setattr(
        attention_model_runner.NPUModelRunnerV2,
        "load_model",
        lambda self, load_dummy_weights=False: events.append("model_load"),
    )

    runner.load_model()

    assert events == ["model_load", "connector_init"]
    assert connector.is_initialized is True
