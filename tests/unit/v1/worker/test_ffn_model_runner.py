# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.forward_context import get_forward_context  # noqa: E402
from vllm.v1.worker import gpu_worker as gpu_worker_module  # noqa: E402

import afd_plugin.v1.worker.ffn_model_runner as ffn_model_runner_module  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.base import AFDControlPlaneClosedError  # noqa: E402
from afd_plugin.eplb.lifecycle import FFNEplbLifecycle  # noqa: E402
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
)
from afd_plugin.v1.worker.cuda_graph import make_ffn_graph_key  # noqa: E402
from afd_plugin.v1.worker.ffn_model_runner import (  # noqa: E402
    GPUFFNModelRunner,
    _set_moe_layer_index,
)
from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker  # noqa: E402
from tests.unit.fakes.dp_metadata import (  # noqa: E402
    DPMetadataValues as _FakeDPMetadata,
)
from tests.unit.fakes.dp_metadata import dp_tokens as _tokens  # noqa: E402
from tests.unit.fakes.graph import Graph as _FakeGraph  # noqa: E402
from tests.unit.fakes.profiler import StepProfiler as _StepProfiler  # noqa: E402


@pytest.fixture
def loop_worker():
    def make(control_plane):
        worker = object.__new__(AFDFFNWorker)
        worker._ffn_thread = None
        worker._ffn_shutdown_event = threading.Event()
        worker._ffn_loop_error = None
        worker.device = SimpleNamespace(type="cpu")
        worker.model_runner = SimpleNamespace(
            connector=SimpleNamespace(control_plane=control_plane, is_initialized=True),
            use_cuda_graph=False,
            model="owned weights",
        )
        return worker

    return make


def test_ffn_worker_stops_receiver_before_native_resource_release(
    monkeypatch, loop_worker
):
    receiving = threading.Event()
    release = threading.Event()

    def receive():
        receiving.set()
        if not release.wait(timeout=5):
            raise TimeoutError("receiver was not cancelled")
        raise AFDControlPlaneClosedError("local close")

    plane = SimpleNamespace(recv_dp_metadata_list=receive, close=release.set)
    worker = loop_worker(plane)
    worker.start_ffn_server_loop()
    thread = worker._ffn_thread

    def native_shutdown(self):
        if thread.is_alive():
            raise RuntimeError("receiver still uses runner resources")
        self.model_runner.model = None

    monkeypatch.setattr(gpu_worker_module.Worker, "shutdown", native_shutdown)
    try:
        assert receiving.wait(timeout=5)
        worker.shutdown()
        assert worker.model_runner.model is None
    finally:
        release.set()
        thread.join(timeout=5)


def test_ffn_worker_timeout_retains_resources_and_remains_a_failure(
    monkeypatch, loop_worker
):
    worker = loop_worker(SimpleNamespace(close=lambda: None))
    worker._ffn_thread = SimpleNamespace(
        join=lambda timeout: None, is_alive=lambda: True
    )

    def native_shutdown(self):
        self.model_runner.model = None

    monkeypatch.setattr(gpu_worker_module.Worker, "shutdown", native_shutdown)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="did not stop"):
            worker.shutdown()
        assert worker.model_runner.model == "owned weights"


@pytest.mark.parametrize(
    "error, request_stop, cancelled",
    [
        (AFDControlPlaneClosedError("closed"), True, True),
        (AFDControlPlaneClosedError("closed"), False, False),
        (RuntimeError("NCCL failure"), True, False),
        (ValueError("invalid metadata"), True, False),
    ],
)
def test_ffn_worker_only_consumes_explicit_receive_cancellation(
    loop_worker, error, request_stop, cancelled
):
    plane = SimpleNamespace()
    worker = loop_worker(plane)

    def receive():
        if request_stop:
            worker._ffn_shutdown_event.set()
        raise error

    plane.recv_dp_metadata_list = receive

    if cancelled:
        worker._run_ffn_server_loop()
    else:
        with pytest.raises(type(error)) as caught:
            worker._run_ffn_server_loop()
        assert caught.value is error


def test_ffn_worker_releases_native_resources_while_reporting_loop_failure(
    monkeypatch, loop_worker
):
    worker = loop_worker(SimpleNamespace(close=lambda: None))
    original = RuntimeError("recorded FFN failure")
    worker._ffn_loop_error = original

    def native_shutdown(self):
        self.model_runner.model = None

    monkeypatch.setattr(gpu_worker_module.Worker, "shutdown", native_shutdown)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="FFN worker loop failed") as caught:
            worker.shutdown()
        assert caught.value.__cause__ is original
        assert worker.model_runner.model is None


def test_native_weight_update_finishes_without_ffn_lora_state(monkeypatch):
    events = []
    worker = object.__new__(AFDFFNWorker)
    worker.model_runner = object.__new__(GPUFFNModelRunner)
    worker.vllm_config = SimpleNamespace()
    worker._weight_update_active = True
    worker._weight_update_is_draft = False
    worker.weight_transfer_engine = SimpleNamespace(
        finish_weight_update=lambda: events.append("finish"),
        reset_weight_update_target=lambda: events.append("reset_target"),
    )
    monkeypatch.setattr(worker, "_check_weight_transfer_engine", lambda: None)
    monkeypatch.setattr(
        gpu_worker_module, "set_current_vllm_config", lambda config: nullcontext()
    )

    worker.finish_weight_update()

    assert events == ["finish", "reset_target"]
    assert worker.model_runner.lora_config is None


class _FakeConnector:
    def __init__(self):
        self.attn_outputs: deque[AFDA2FTransferPayload] = deque()
        self.ffn_outputs = []
        self.expert_routing_specs = []
        self.recv_input_ids = []
        self.dp_metadata_updates = []
        self.closed = False
        self.attn_size = 1
        self.ffn_size = 1
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane: _FakeConnector | None = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_updates.append(
            (
                dict(payload.dp_metadata_list),
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )

    def recv_attn_output(
        self,
        ubatch_idx=None,
        routing_spec=None,
        recv_input_ids=False,
    ):
        if routing_spec is not None:
            self.expert_routing_specs.append(routing_spec)
        self.recv_input_ids.append(recv_input_ids)
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


class _FakeEplbBackend:
    """Records the lifecycle calls the FFN runner makes per connector step."""

    def __init__(self, model_config=None):
        self.state = None
        self.model_config = model_config
        self.prepare_loads = 0
        self.registrations = []
        self.async_loop_starts = []
        self.steps = []
        self.prepared_forwards = []

    def initialize_model(self, model, *, load_dummy_weights=False):
        self.prepare_loads += 1
        self.registrations.append((model, self.model_config))
        self.async_loop_starts.append((True, load_dummy_weights))

    def step(self, *, is_dummy=False):
        self.steps.append((is_dummy, False))
        return False

    def prepare_forward(self, num_unpadded_tokens, *, is_dummy=False):
        self.prepared_forwards.append(num_unpadded_tokens)

    def get_eplb_state(self):
        return self.state


class _FakeModel:
    def get_experts_layer_indices(self):
        return ()

    def compute_ffn_output(self, hidden_states, layer_idx):
        return f"ffn({hidden_states}, layer={layer_idx})"


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
        # FFN keeps connector-driven execution within a GPU MRV2 deployment.
        use_v2_model_runner=True,
        # A real VllmConfig always carries a model config. The forward context asks
        # it for the MX scale algorithm, which reads exactly two fields: the
        # architecture list and the HF text config's model type. Supplying the two
        # keeps this fixture honest about the contract instead of omitting the whole
        # object; both default to "not MiniMax M3", which is what these tests are.
        model_config=SimpleNamespace(architectures=None, hf_text_config=None),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            enable_eplb=False,
            is_moe_model=True,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )
    runner.connector = _FakeConnector()
    runner.model = model
    runner.model_config = SimpleNamespace()
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=False)
    runner.afd_cudagraph_policy = SimpleNamespace(enabled=False)
    runner.num_layers = num_layers
    runner.use_cuda_graph = False
    runner._cuda_graphs = {}
    runner._graph_memory_pool = None
    runner.load_config = SimpleNamespace()
    runner.device = torch.device("cpu")
    runner.prof = None
    runner.eplb_lifecycle = FFNEplbLifecycle(
        runner.vllm_config,
        runner.afd_config,
        _FakeEplbBackend(model_config=runner.model_config),
    )
    return runner


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
    assert runner.connector.recv_input_ids == [False]
    assert metadata.layer_idx == 0


def test_v2_ffn_runner_keeps_hidden_state_only_connector_contract():
    class _V2Backbone:
        def __init__(self):
            self.calls = []

        def get_experts_layer_indices(self):
            return ()

        def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
            self.calls.append((hidden_states, layer_idx, kwargs))
            return hidden_states

    backbone = _V2Backbone()
    model = object.__new__(AFDDeepseekV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = backbone
    runner = _runner_with_connector_and_model(model)
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("v2-hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert not getattr(model, "afd_requires_input_ids", False)
    assert runner.connector.recv_input_ids == [False]
    assert runner.connector.ffn_outputs == [("v2-hidden", metadata)]


def test_ffn_runner_forwards_payload_input_ids_to_model():
    class _InputIdsModel:
        get_experts_layer_indices = _FakeModel.get_experts_layer_indices
        afd_requires_input_ids = True

        def __init__(self):
            self.calls = []

        def compute_ffn_output(self, hidden_states, layer_idx, *, input_ids):
            self.calls.append((hidden_states, layer_idx, input_ids))
            return input_ids

    model = _InputIdsModel()
    runner = _runner_with_connector_and_model(model)
    metadata = _metadata()
    input_ids = torch.tensor([11], dtype=torch.int32)
    runner.connector.attn_outputs.append(
        AFDA2FTransferPayload(
            hidden_states="hidden",
            context=AFDTransferContext(metadata=metadata),
            input_ids=input_ids,
        ),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [(input_ids, metadata)]
    assert runner.connector.recv_input_ids == [True]


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


def test_ffn_runner_aggregates_each_ubatch_metadata_for_ffn_ranks():
    class _MetadataModel(_FakeModel):
        def __init__(self):
            self.dp_counts = []

        def compute_ffn_output(self, hidden_states, layer_idx):
            metadata = get_forward_context().dp_metadata
            self.dp_counts.append(metadata.num_tokens_across_dp_cpu.tolist())
            return hidden_states

    model = _MetadataModel()
    runner = _runner_with_connector_and_model(model)
    runner.connector.attn_size = 4
    runner.connector.ffn_size = 2
    runner.vllm_config.parallel_config.data_parallel_size = 2
    runner.connector.attn_outputs.extend(
        [
            _payload("hidden-0", _metadata_for_stage(0)),
            _payload("hidden-1", _metadata_for_stage(1)),
        ],
    )
    dp_metadata_list = {
        0: _FakeDPMetadata([10, 11, 12, 13]),
        1: _FakeDPMetadata([1, 2, 3, 4]),
    }

    runner.execute_model(dp_metadata_list=dp_metadata_list)

    assert model.dp_counts == [[21, 25], [3, 7]]
    control_metadata = runner.connector.dp_metadata_updates[0][0]
    assert _tokens(control_metadata[0]) == [10, 11, 12, 13]
    assert _tokens(control_metadata[1]) == [1, 2, 3, 4]


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

    assert runner.connector.ffn_outputs == [
        ("ffn(dense-hidden, layer=0)", dense_metadata),
        ("ffn(moe-hidden, layer=1)", expert_context.metadata),
    ]


def test_experts_graph_capture_passes_model_routing_spec():  # long-test: routing ABI
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
def test_experts_pass_static_routing_spec_for_each_stage(
    is_warmup,
):  # long-test: two stage payloads must retain their own routing data
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


def test_ffn_runner_profiles_zero_cudagraph_memory():
    """Native Worker.determine_available_memory calls this on CUDA when
    CUDA graphs are enabled; FFN graphs are captured lazily from the connector
    loop into their own pool, so the engine estimate must stay zero."""

    runner = object.__new__(GPUFFNModelRunner)
    assert GPUFFNModelRunner.profile_cudagraph_memory(runner) == 0


def test_ffn_runner_replays_cuda_graph_when_key_exists():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    graph = _FakeGraph()
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner._cuda_graphs = {
        make_ffn_graph_key(dp_metadata): {"graph": graph},
    }

    runner.execute_model(
        dp_metadata_list=dp_metadata,
        is_graph_replaying=True,
    )

    assert graph.replay_count == 1
    assert runner.connector.ffn_outputs == []


def test_ffn_runner_skips_replay_when_attention_is_eager():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    graph = _FakeGraph()
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner._cuda_graphs = {
        make_ffn_graph_key(dp_metadata): {"graph": graph},
    }
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list=dp_metadata)

    assert graph.replay_count == 0
    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_ffn_runner_cuda_graph_miss_falls_back_to_eager():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    metadata = _metadata()
    runner.connector.attn_outputs.append(_payload("hidden", metadata))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.connector.ffn_outputs == [
        ("ffn(hidden, layer=0)", metadata),
    ]


def test_ffn_runner_loads_model_and_registers_eplb(monkeypatch):
    runner = _runner_with_connector_and_model(_FakeModel())
    model = runner.model
    runner.model = None
    loader = SimpleNamespace(
        load_model=lambda **_kwargs: model,
        load_weights=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ffn_model_runner_module, "get_model_loader", lambda _config: loader
    )

    runner.load_model()

    assert runner.model is model
    assert runner.eplb_lifecycle.backend.prepare_loads == 1
    assert runner.eplb_lifecycle.backend.registrations == [(model, runner.model_config)]
    assert runner.eplb_lifecycle.backend.async_loop_starts == [(True, False)]


def test_lifecycle_noops_without_a_backend():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.eplb_lifecycle = FFNEplbLifecycle(
        runner.vllm_config,
        runner.afd_config,
        None,
    )

    assert runner.eplb_lifecycle.is_enabled is False
    assert runner.eplb_lifecycle.step(is_dummy=False) is False
    assert runner.eplb_lifecycle.get_eplb_state() is None


def test_ffn_runner_steps_eplb_once_per_real_step():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.connector.attn_outputs.append(_payload("hidden", _metadata()))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.eplb_lifecycle.backend.steps == [(False, False)]


def test_ffn_runner_marks_warmup_and_capturing_steps_dummy():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.connector.attn_outputs.extend(
        _payload("hidden", _metadata()) for _ in range(2)
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])}, is_warmup=True)
    runner.execute_model(
        dp_metadata_list={0: _FakeDPMetadata([1])}, is_graph_capturing=True
    )

    assert runner.eplb_lifecycle.backend.steps == [(True, False), (True, False)]


def test_ffn_runner_replay_steps_eplb_as_real_step():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    graph = _FakeGraph()
    dp_metadata = {0: _FakeDPMetadata([1])}
    runner._cuda_graphs = {
        make_ffn_graph_key(dp_metadata): {"graph": graph},
    }

    runner.execute_model(dp_metadata_list=dp_metadata, is_graph_replaying=True)

    assert graph.replay_count == 1
    assert runner.eplb_lifecycle.backend.steps == [(False, False)]


def test_ffn_runner_prepares_eplb_unpadded_tokens_from_dp_metadata():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.vllm_config.parallel_config.enable_eplb = True
    runner.connector.attn_size = 4
    runner.connector.attn_outputs.append(_payload("hidden", _metadata()))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([4, 6])})

    # Two DP counts of 4 and 6 tokens, each replicated across two Attention
    # TP peers, give 20 real tokens for the router's record kernel.
    assert runner.eplb_lifecycle.backend.prepared_forwards == [20]


def test_ffn_runner_skips_eplb_prepare_when_eplb_is_disabled():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.connector.attn_outputs.append(_payload("hidden", _metadata()))

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([4, 6])})

    assert runner.eplb_lifecycle.backend.prepared_forwards == []


def test_ffn_runner_rejects_multistage_eplb_payload():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.vllm_config.parallel_config.enable_eplb = True
    two_stage = {0: _FakeDPMetadata([1]), 1: _FakeDPMetadata([1])}

    with pytest.raises(RuntimeError, match="exactly one stage"):
        runner.execute_model(dp_metadata_list=two_stage)


def test_ffn_runner_capture_prepared_and_steps_eplb_as_dummy(monkeypatch):
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.use_cuda_graph = True
    runner.vllm_config.parallel_config.enable_eplb = True
    monkeypatch.setattr(runner, "_dummy_run", lambda **_kwargs: None)
    free_gpu_memory = [100, 90]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_gpu_memory.pop(0), 0))
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: object())
    monkeypatch.setattr(
        ffn_model_runner_module, "graph_capture", lambda **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        ffn_model_runner_module, "set_cudagraph_capturing_enabled", lambda _v: None
    )

    cuda_graph_size = runner.capture_model(dp_metadata_list={0: _FakeDPMetadata([2])})

    assert cuda_graph_size == 10
    assert runner.eplb_lifecycle.backend.steps == [(True, False)]
    assert runner.eplb_lifecycle.backend.prepared_forwards == [2]


class _FakeEplbModelState:
    """The EplbModelState fields the placement log reads."""

    def __init__(self, placement: list[list[int]]):
        self.physical_to_logical_map = torch.tensor(placement)
        self.expert_load_window = torch.zeros(
            (2, len(placement), len(placement[0])), dtype=torch.int32
        )
        self.model_name = "test-model"
        self.rebalanced = False


class _FakeEplbState:
    """Steps the counter and commits a new placement at the boundary."""

    def __init__(self, placements: list[list[list[int]]]):
        self.model_states = {"model-hash": _FakeEplbModelState(placements[0])}
        self.expert_rearrangement_step_interval = 2
        self.expert_rearrangement_step = 0
        self._placements = placements
        self._rearrangements = 0

    def step(self, is_dummy=False, is_profile=False):
        self.expert_rearrangement_step += 1
        if self.expert_rearrangement_step < self.expert_rearrangement_step_interval:
            return
        self.expert_rearrangement_step = 0
        self._rearrangements += 1
        self.model_states["model-hash"].physical_to_logical_map = torch.tensor(
            self._placements[min(self._rearrangements, len(self._placements) - 1)]
        )


def _runner_for_placement_logging(state, *, log_placement: bool):
    runner = _runner_with_connector_and_model(_FakeModel())
    # The runner's EPLB seam needs only the state and its step; two FFN EP
    # ranks split the four-slot placement.
    backend = _FakeEplbBackend()
    backend.state = state
    backend.step = state.step
    runner.eplb_lifecycle = FFNEplbLifecycle(
        runner.vllm_config,
        runner.afd_config,
        backend,
    )
    runner.vllm_config.parallel_config.data_parallel_size = 2
    runner.vllm_config.parallel_config.eplb_config = SimpleNamespace(
        log_balancedness=log_placement,
    )
    return runner


def test_ffn_runner_logs_placement_when_a_rearrangement_commits(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """With log_balancedness the FFN logs the committed expert placement
    when EplbState.step rearranges, and only then.

    Why: vLLM's own log lists load stats, not which expert went where.
    """
    monkeypatch.setattr(
        ffn_model_runner_module,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    state = _FakeEplbState(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 2, 1, 3], [0, 2, 1, 3]],
        ]
    )
    runner = _runner_for_placement_logging(state, log_placement=True)

    with caplog.at_level(logging.INFO, logger=ffn_model_runner_module.logger.name):
        runner.eplb_step()
        # Not due yet: this step only advanced the counter.
        assert caplog.messages == []
        runner.eplb_step()

    message = caplog.messages[0]
    assert "AFD FFN EPLB chose experts for model test-model" in message
    assert "rank 0=[0, 2] | rank 1=[1, 3]" in message
    # A step without a further rearrangement stays silent.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=ffn_model_runner_module.logger.name):
        runner.eplb_step()
    assert caplog.messages == []


def test_ffn_runner_placement_logging_needs_balancedness_flag(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Without log_balancedness no placement is logged, even at a
    rearrangement boundary."""
    monkeypatch.setattr(
        ffn_model_runner_module,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    state = _FakeEplbState([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    runner = _runner_for_placement_logging(state, log_placement=False)

    with caplog.at_level(logging.INFO, logger=ffn_model_runner_module.logger.name):
        runner.eplb_step()
        runner.eplb_step()

    assert caplog.messages == []


def test_ffn_runner_placement_logging_only_on_ep_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Every FFN EP rank holds the same map; only EP rank 0 logs it."""
    monkeypatch.setattr(
        ffn_model_runner_module,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    state = _FakeEplbState([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    runner = _runner_for_placement_logging(state, log_placement=True)

    with caplog.at_level(logging.INFO, logger=ffn_model_runner_module.logger.name):
        runner.eplb_step()
        runner.eplb_step()

    assert caplog.messages == []


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

    assert "AFD FFN worker loop failed" in caplog.text
    with pytest.raises(RuntimeError, match="AFD FFN worker loop failed") as exc:
        worker.raise_ffn_loop_error_if_any()
    assert exc.value.__cause__ is expected_error
