from __future__ import annotations

import logging
import threading
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import afd_plugin.v1.worker.ffn_model_runner as ffn_model_runner_module  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key  # noqa: E402
from afd_plugin.v1.worker.ffn_model_runner import (  # noqa: E402
    GPUFFNModelRunner,
    _set_moe_layer_index,
)
from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker  # noqa: E402


class _FakeConnector:
    def __init__(self):
        self.attn_outputs = deque()
        self.ffn_outputs = []
        self.expert_routing_specs = []
        self.dp_metadata_updates = []
        self.closed = False
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_updates.append(
            (
                dict(payload.dp_metadata_list),
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )

    def recv_attn_output(self, ubatch_idx=None, routing_spec=None):
        if routing_spec is not None:
            self.expert_routing_specs.append(routing_spec)
        if ubatch_idx is None:
            return self.attn_outputs.popleft()
        for item in tuple(self.attn_outputs):
            if item.context.metadata.stage_idx == ubatch_idx:
                self.attn_outputs.remove(item)
                return item
        raise IndexError(ubatch_idx)

    def send_ffn_output(self, ffn_output, context):
        self.ffn_outputs.append((ffn_output, context.metadata))

    def close(self):
        self.closed = True


class _ConnectorDrivenFakeConnector(_FakeConnector):
    def __init__(self):
        super().__init__()
        self.control_plane = None


class _FakeModel:
    def get_experts_layer_indices(self):
        return ()

    def compute_ffn_output(self, hidden_states, layer_idx):
        return f"ffn({hidden_states}, layer={layer_idx})"


class _StepProfiler:
    def __init__(self):
        self.steps = 0
        self.stopped = False

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


def _metadata():
    return AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )


def _metadata_for_stage(stage_idx):
    return AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=stage_idx,
        seq_len=1,
    )


def _payload(hidden_states, metadata):
    return AFDA2FTransferPayload(
        hidden_states=hidden_states,
        context=AFDTransferContext(metadata=metadata),
    )


def _runner_with_connector_and_model(model, *, num_layers=1):
    runner = object.__new__(GPUFFNModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            is_moe_model=True,
            use_sequence_parallel_moe=False,
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )
    runner.connector = _FakeConnector()
    runner.model = model
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=False)
    runner.afd_cudagraph_policy = SimpleNamespace(enabled=False)
    runner.num_layers = num_layers
    runner.use_cuda_graph = False
    runner._cuda_graphs = {}
    runner.prof = None
    return runner


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0
        self.reset_count = 0

    def replay(self):
        self.replay_count += 1

    def reset(self):
        self.reset_count += 1


def test_ffn_runner_executes_model_compute_ffn_output():
    runner = _runner_with_connector_and_model(_FakeModel())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert len(runner.connector.dp_metadata_updates) == 1
    dp_metadata_update, is_graph_capturing, is_warmup = (
        runner.connector.dp_metadata_updates[0]
    )
    assert sorted(dp_metadata_update) == [0]
    assert _tokens(dp_metadata_update[0]) == [1]
    assert is_graph_capturing is False
    assert is_warmup is False
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]
    assert metadata.layer_idx == 0


def test_ffn_runner_exposes_missing_model_contract():
    runner = _runner_with_connector_and_model(SimpleNamespace())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    with pytest.raises(AttributeError, match="get_experts_layer_indices"):
        runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})


def test_ffn_runner_processes_each_ubatch_for_each_layer():
    runner = _runner_with_connector_and_model(_FakeModel(), num_layers=2)
    metadata_0_layer_0 = _metadata_for_stage(0)
    metadata_1_layer_0 = _metadata_for_stage(1)
    metadata_0_layer_1 = _metadata_for_stage(0)
    metadata_1_layer_1 = _metadata_for_stage(1)
    runner.connector.attn_outputs.extend(
        [
            _payload("hidden-1-l0", metadata_1_layer_0),
            _payload("hidden-0-l0", metadata_0_layer_0),
            _payload("hidden-1-l1", metadata_1_layer_1),
            _payload("hidden-0-l1", metadata_0_layer_1),
        ],
    )

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([1]),
            1: _FakeDPMetadata([1]),
        },
    )

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden-0-l0, layer=0)", metadata_0_layer_0),
        ("ffn(hidden-1-l0, layer=0)", metadata_1_layer_0),
        ("ffn(hidden-0-l1, layer=1)", metadata_0_layer_1),
        ("ffn(hidden-1-l1, layer=1)", metadata_1_layer_1),
    ]


def test_ffn_side_gate_mixes_dense_and_experts_protocols():
    class _MixedModel(_FakeModel):
        def __init__(self):
            self.calls = []

        def get_experts_layer_indices(self):
            return (1,)

        def compute_ffn_output(self, hidden_states, layer_idx):
            self.calls.append((hidden_states, layer_idx))
            return f"ffn({hidden_states}, layer={layer_idx})"

    model = _MixedModel()
    runner = _runner_with_connector_and_model(model, num_layers=2)
    dense_metadata = _metadata()
    expert_context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=1,
            stage_idx=0,
            seq_len=1,
        ),
    )
    runner.connector.attn_outputs.append(_payload("dense-hidden", dense_metadata))
    runner.connector.attn_outputs.append(
        AFDA2FTransferPayload(
            hidden_states="moe-hidden",
            context=expert_context,
        ),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert model.calls == [("dense-hidden", 0), ("moe-hidden", 1)]
    assert runner.connector.ffn_outputs == [
        ("ffn(dense-hidden, layer=0)", dense_metadata),
        ("ffn(moe-hidden, layer=1)", expert_context.metadata),
    ]


def test_attention_side_gate_processes_only_experts_layers():
    router_logits = object()

    class _AttentionGateModel(_FakeModel):
        def __init__(self):
            self.calls = []

        def get_experts_layer_indices(self):
            return (1,)

        def get_experts_routing_spec(self, layer_idx):
            return AFDExpertRoutingSpec(
                router_logits_width=4,
                router_logits_dtype=torch.float32,
            )

        def compute_experts_output(
            self,
            hidden_states,
            layer_idx,
            received_router_logits,
        ):
            self.calls.append(
                (hidden_states, layer_idx, received_router_logits),
            )
            return "expert-output"

    model = _AttentionGateModel()
    runner = _runner_with_connector_and_model(model, num_layers=2)
    runner.afd_config.compute_gate_on_attention = True
    expert_context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=1,
            stage_idx=0,
            seq_len=1,
        ),
    )
    runner.connector.attn_outputs.append(
        AFDA2FTransferPayload(
            hidden_states="moe-hidden",
            context=expert_context,
            router_logits=router_logits,
        ),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert model.calls == [("moe-hidden", 1, router_logits)]
    assert runner.connector.ffn_outputs == [
        ("expert-output", expert_context.metadata),
    ]


def test_experts_graph_capture_passes_model_routing_spec():
    routing_spec = AFDExpertRoutingSpec(
        router_logits_width=4,
        router_logits_dtype=torch.float32,
    )
    router_logits = object()

    class _GraphModel(_FakeModel):
        def get_experts_layer_indices(self):
            return (1,)

        def get_experts_routing_spec(self, layer_idx):
            assert layer_idx == 1
            return routing_spec

        def compute_experts_output(
            self,
            hidden_states,
            layer_idx,
            received_router_logits,
        ):
            assert (hidden_states, layer_idx, received_router_logits) == (
                "moe-hidden",
                1,
                router_logits,
            )
            return "expert-output"

    runner = _runner_with_connector_and_model(_GraphModel(), num_layers=2)
    runner.afd_config.compute_gate_on_attention = True
    runner.afd_cudagraph_policy.enabled = True
    expert_context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=1,
            stage_idx=0,
            seq_len=1,
        ),
    )
    runner.connector.attn_outputs.append(
        AFDA2FTransferPayload(
            hidden_states="moe-hidden",
            context=expert_context,
            router_logits=router_logits,
        ),
    )

    runner._ffn_forward(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_graph_capturing=True,
    )

    assert runner.connector.expert_routing_specs == [routing_spec]
    assert runner.connector.ffn_outputs == [
        ("expert-output", expert_context.metadata),
    ]


@pytest.mark.parametrize(
    "is_warmup",
    [False, True],
    ids=["compiled-policy", "warmup"],
)
def test_experts_pass_static_routing_spec_for_each_stage(is_warmup):
    routing_spec = AFDExpertRoutingSpec(
        router_logits_width=4,
        router_logits_dtype=torch.float32,
    )

    class _ExpertsModel(_FakeModel):
        def get_experts_layer_indices(self):
            return (1,)

        def get_experts_routing_spec(self, layer_idx):
            assert layer_idx == 1
            return routing_spec

        def compute_experts_output(
            self,
            hidden_states,
            layer_idx,
            received_router_logits,
        ):
            return f"experts({hidden_states}, {layer_idx}, {received_router_logits})"

    runner = _runner_with_connector_and_model(_ExpertsModel(), num_layers=2)
    runner.afd_config.compute_gate_on_attention = True
    runner.afd_cudagraph_policy.enabled = True
    contexts = []
    for stage_idx in (1, 0):
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=1,
                stage_idx=stage_idx,
                seq_len=1,
            ),
        )
        contexts.append(context)
        runner.connector.attn_outputs.append(
            AFDA2FTransferPayload(
                hidden_states=f"hidden-{stage_idx}",
                context=context,
                router_logits=f"router-{stage_idx}",
            ),
        )

    runner.execute_model(
        dp_metadata_list={
            0: _FakeDPMetadata([1]),
            1: _FakeDPMetadata([1]),
        },
        is_warmup=is_warmup,
    )

    assert len(runner.connector.dp_metadata_updates) == 1
    dp_metadata_update, is_graph_capturing, reported_is_warmup = (
        runner.connector.dp_metadata_updates[0]
    )
    assert sorted(dp_metadata_update) == [0, 1]
    assert is_graph_capturing is False
    assert reported_is_warmup is is_warmup
    assert runner.connector.expert_routing_specs == [routing_spec, routing_spec]
    assert runner.connector.ffn_outputs == [
        ("experts(hidden-0, 1, router-0)", contexts[1].metadata),
        ("experts(hidden-1, 1, router-1)", contexts[0].metadata),
    ]


def test_ffn_runner_requires_dp_metadata_list():
    runner = object.__new__(GPUFFNModelRunner)
    runner.prof = None

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_ffn_runner_makes_original_style_graph_key():
    key = make_ffn_graph_key(
        {
            1: _FakeDPMetadata([5, 7]),
            0: _FakeDPMetadata([2, 3]),
        },
    )

    assert key == ((0, (2, 3)), (1, (5, 7)))


def test_ffn_runner_replays_cuda_graph_when_key_exists():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    graph = _FakeGraph()
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner._cuda_graphs = {
        make_ffn_graph_key(dp_metadata): {"graph": graph},
    }

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_ffn_runner_cuda_graph_miss_falls_back_to_eager():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_ffn_runner_steps_gpu_profiler():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.prof = _StepProfiler()
    runner.connector.attn_outputs.append(_payload("hidden", _metadata()))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.prof.steps == 1


def test_ffn_runner_releases_owned_runtime_state_on_shutdown(monkeypatch):
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.prof = _StepProfiler()
    graph = _FakeGraph()
    runner._cuda_graphs = {("graph",): {"graph": graph}}
    runner._graph_memory_pool = object()
    runner.vllm_config.compilation_config.static_forward_context["layer"] = object()
    rope_cache = {"rope": object()}
    workspace_resets = []
    monkeypatch.setattr(ffn_model_runner_module, "_ROPE_DICT", rope_cache)
    monkeypatch.setattr(
        ffn_model_runner_module,
        "reset_workspace_manager",
        lambda: workspace_resets.append(True),
    )

    runner.shutdown()

    assert graph.reset_count == 1
    assert runner._cuda_graphs == {}
    assert runner._graph_memory_pool is None
    assert runner.vllm_config.compilation_config.static_forward_context == {}
    assert runner.model is None
    assert rope_cache == {}
    assert workspace_resets == [True]
    assert runner.prof.stopped is True
    assert runner.connector.closed is True


def test_ffn_forward_can_skip_connector_state_update_for_capture():
    runner = _runner_with_connector_and_model(_FakeModel())
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner._ffn_forward(
        dp_metadata_list={0: _FakeDPMetadata([1])},
        is_graph_capturing=True,
        update_connector_state=False,
    )

    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_set_moe_layer_index_resets_for_current_layer():
    forward_context = SimpleNamespace(
        all_moe_layers=[
            "model.layers.1.mlp.experts",
            "model.layers.2.mlp.experts",
            "model.layers.3.mlp.experts",
        ],
        moe_layer_index=99,
    )

    _set_moe_layer_index(forward_context, 2)

    assert forward_context.moe_layer_index == 1


def test_ffn_worker_scheduler_execute_model_fails_fast():
    worker = object.__new__(AFDFFNWorker)

    with pytest.raises(RuntimeError, match="connector-driven"):
        worker.execute_model(scheduler_output=object())


def test_ffn_worker_reports_zero_compilation_times():
    worker = object.__new__(AFDFFNWorker)

    compilation_times = worker.compile_or_warm_up_model()

    assert compilation_times.language_model == 0.0
    assert compilation_times.encoder == 0.0


def test_ffn_worker_loop_rejects_connector_without_control_plane():
    worker = object.__new__(AFDFFNWorker)
    event = threading.Event()

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = SimpleNamespace(
        connector=_ConnectorDrivenFakeConnector(),
    )

    with pytest.raises(NotImplementedError, match="control-plane-driven"):
        worker._run_ffn_server_loop()


def test_ffn_worker_loop_logs_unexpected_thread_errors(caplog):
    worker = object.__new__(AFDFFNWorker)
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True),
    )

    expected_error = RuntimeError("boom")

    def fail_loop():
        raise expected_error

    worker._run_ffn_server_loop = fail_loop

    with caplog.at_level(logging.ERROR, logger="afd_plugin.v1.worker.ffn_worker"):
        worker.start_ffn_server_loop()
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

    assert worker._ffn_loop_error is expected_error
    assert "AFD FFN worker loop failed" in caplog.text
    with pytest.raises(RuntimeError, match="AFD FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()
    assert exc.value.__cause__ is expected_error
