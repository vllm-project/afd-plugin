# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import logging
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.forward_context import get_forward_context  # noqa: E402

import afd_plugin.v1.worker.ffn_model_runner as ffn_model_runner_module  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDExpertRoutingSpec,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.gpu.async_gpu import ConnectorShutdown  # noqa: E402
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDDeepseekV2ForCausalLM,
)
from afd_plugin.v1.worker.cuda_graph import (  # noqa: E402
    make_ffn_graph_key,
    padded_ffn_graph_buckets,
)
from afd_plugin.v1.worker.ffn_model_runner import (  # noqa: E402
    GPUFFNModelRunner,
    _set_moe_layer_index,
)
from afd_plugin.v1.worker.ffn_worker import AFDFFNWorker  # noqa: E402


class _FakeConnector:
    def __init__(self):
        self.attn_outputs: deque = deque()
        self.ffn_outputs = []
        self.expert_routing_specs = []
        self.recv_input_ids = []
        self.dp_metadata_updates = []
        self.closed = False
        self.attn_size = 1
        self.ffn_size = 1
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
        self.control_plane: Any = None


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
        # vLLM 0.26 VllmConfig field read by the Ascend platform's
        # set_additional_forward_context hook on NPU test environments.
        use_v2_model_runner=False,
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
    assert backbone.calls == [("v2-hidden", 0, {})]
    assert runner.connector.ffn_outputs == [("v2-hidden", metadata)]


def test_ffn_runner_forwards_payload_input_ids_to_model():
    class _InputIdsModel(_FakeModel):
        afd_requires_input_ids = True

        def __init__(self):
            self.calls = []

        def compute_ffn_output(  # type: ignore[override]
            self, hidden_states, layer_idx, *, input_ids
        ):
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

    assert model.calls == [("hidden", 0, input_ids)]
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


@pytest.mark.parametrize(
    ("attention_counts", "expected_ffn_counts"),
    [
        ([0, 4, 5, 6], [5, 11]),
        ([], [2, 2]),
        ([3, 4, 5], [7, 6]),
    ],
)
def test_ffn_runner_matches_p2p_token_count_aggregation(
    attention_counts,
    expected_ffn_counts,
):
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.connector.attn_size = 4
    runner.connector.ffn_size = 2
    runner.vllm_config.parallel_config.data_parallel_size = 2

    metadata = runner._make_ffn_dp_metadata(_FakeDPMetadata(attention_counts))

    assert _tokens(metadata) == expected_ffn_counts


def test_ffn_runner_projects_tensor_parallel_counts_to_dp_metadata():
    runner = _runner_with_connector_and_model(_FakeModel())
    runner.connector.attn_size = 2
    runner.connector.ffn_size = 2

    metadata = runner._make_ffn_dp_metadata(_FakeDPMetadata([8]))

    assert _tokens(metadata) == [8]


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


def test_ffn_runner_graph_key_preserves_attention_peer_shapes():
    first_key = make_ffn_graph_key({0: _FakeDPMetadata([1, 3, 5, 7])})
    second_key = make_ffn_graph_key({0: _FakeDPMetadata([2, 2, 6, 6])})

    assert first_key == ((0, (1, 3, 5, 7)),)
    assert second_key == ((0, (2, 2, 6, 6)),)
    assert first_key != second_key


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


def test_ffn_worker_loop_drives_connector_without_control_plane():
    worker = object.__new__(AFDFFNWorker)
    event = threading.Event()
    steps = []

    def execute_connector_driven_step():
        steps.append(1)
        # The connector-driven step returns on an idle poll; the loop must come
        # back to the shutdown event rather than block forever.
        if len(steps) == 3:
            event.set()

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = SimpleNamespace(
        connector=_ConnectorDrivenFakeConnector(),
        execute_connector_driven_step=execute_connector_driven_step,
    )

    worker._run_ffn_server_loop()

    assert len(steps) == 3


def test_ffn_worker_loop_exits_cleanly_when_peer_announces_shutdown():
    worker = object.__new__(AFDFFNWorker)

    def execute_connector_driven_step():
        raise ConnectorShutdown("peer left")

    worker._ffn_shutdown_event = threading.Event()
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = SimpleNamespace(
        connector=_ConnectorDrivenFakeConnector(),
        execute_connector_driven_step=execute_connector_driven_step,
    )

    # A peer shutdown is an ordinary exit, not a loop failure.
    worker._run_ffn_server_loop()


def test_ffn_worker_loop_logs_unexpected_thread_errors(caplog):
    worker = object.__new__(AFDFFNWorker)
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=True),
        capture_padded_ffn_graphs=lambda: 0,
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


# ----------------------------------------------------------------------
# Connector-driven padded graphs
#
# The FFN side never learns the next work item's shape ahead of time, which is
# why it ran eagerly. Padding removes the need to: the grouping is device data,
# so one captured row count serves every smaller item.
# ----------------------------------------------------------------------


class _RecordingPaddedGraph:
    def __init__(self, routed_out, shared_out=None):
        self.routed_out = routed_out
        self.shared_out = shared_out
        self.replays = 0

    @property
    def graph(self):
        return self

    def replay(self):
        self.replays += 1


class _RecordingFFNModel:
    """Stands in for the model; records eager compute calls."""

    def __init__(self):
        self.eager_calls = []

    def compute_ffn_output(self, *, hidden_states, layer_idx, group_list, **kwargs):
        self.eager_calls.append((layer_idx, int(hidden_states.shape[0])))
        return torch.zeros_like(hidden_states)


def _padded_runner(*, max_routed=8, max_shared=2, expert_per_rank=2, buckets=None):
    runner = object.__new__(GPUFFNModelRunner)
    runner.model = _RecordingFFNModel()
    runner._padded_max_routed = max_routed
    runner._padded_max_shared = max_shared
    runner._padded_hidden = torch.zeros(max_routed, 4)
    runner._padded_counts = torch.zeros(expert_per_rank, dtype=torch.int32)
    runner._padded_shared = torch.zeros(max_shared, 4) if max_shared else None
    runner._padded_graphs = {}
    runner._padded_buckets = (
        buckets if buckets is not None else padded_ffn_graph_buckets(max_routed)
    )
    runner._padded_rows_real = 0
    runner._padded_rows_charged = 0
    runner._padded_replays = 0
    runner._padded_eager_items = 0
    return runner


def _work_item(layer_idx, routed, shared, expert_per_rank=2, staged=False):
    counts = torch.zeros(expert_per_rank, dtype=torch.int32)
    counts[0] = routed
    states = SimpleNamespace(
        routed_tokens=routed,
        shared_tokens=shared,
        group_list=counts,
        staged_routed=staged,
        expand_x_shared=torch.ones(shared, 4) if shared else None,
    )
    item = SimpleNamespace(layer_idx=layer_idx, hidden_states=torch.ones(routed, 4))
    return item, states


def test_padded_graph_replays_and_slices_off_the_padding():
    # 1000 rows into a 1024 bucket: 2.4% padding, inside the ratio that makes a
    # replay worth more than the launches eager would pay.
    runner = _padded_runner(max_routed=1024, max_shared=2, buckets=(1024,))
    graph = _RecordingPaddedGraph(
        routed_out=torch.arange(1024 * 4, dtype=torch.float32).reshape(1024, 4),
        shared_out=torch.zeros(2, 4),
    )
    runner._padded_graphs[(3, runner._padded_buckets[-1])] = graph
    item, states = _work_item(3, routed=1000, shared=1)

    payload = GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert graph.replays == 1
    assert runner.model.eager_calls == []
    # The reply only ever sees the real rows.
    assert payload.routed_output.shape[0] == 1000
    assert payload.shared_output.shape[0] == 1
    # Counts must sum to the captured row count, with the pad on the last
    # expert -- otherwise the grouping and the row count disagree.
    assert runner._padded_counts.tolist() == [1000, 24]


def test_padded_graph_stages_the_real_rows_at_the_front():
    runner = _padded_runner(max_routed=1024, max_shared=0, buckets=(1024,))
    runner._padded_graphs[(0, 1024)] = _RecordingPaddedGraph(
        routed_out=torch.zeros(1024, 4)
    )
    item, states = _work_item(0, routed=1000, shared=0)
    item.hidden_states = torch.full((1000, 4), 7.0)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert torch.equal(runner._padded_hidden[:1000], torch.full((1000, 4), 7.0))


def test_rows_already_gathered_into_the_buffer_are_not_copied_again():
    # The arrival's gather can write straight into the graph's input buffer,
    # and it had to write somewhere regardless. Copying afterwards would be a
    # second pass over the whole payload, once per layer -- which measured as
    # the reason capturing was slower than running eagerly.
    runner = _padded_runner(max_routed=1024, max_shared=0, buckets=(1024,))
    runner._padded_graphs[(0, 1024)] = _RecordingPaddedGraph(
        routed_out=torch.zeros(1024, 4)
    )
    runner._padded_hidden[:1000] = 7.0
    item, states = _work_item(0, routed=1000, shared=0, staged=True)
    # What a stale copy would put there instead.
    item.hidden_states = torch.full((1000, 4), -1.0)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert torch.equal(runner._padded_hidden[:1000], torch.full((1000, 4), 7.0))


@pytest.mark.parametrize(
    ("routed", "shared", "why"),
    [
        (9, 0, "more routed rows than the capture"),
        (4, 5, "more shared rows than the capture"),
        (0, 0, "nothing routed here at all"),
    ],
)
def test_work_that_does_not_fit_the_capture_runs_eagerly(routed, shared, why):
    runner = _padded_runner(max_routed=8, max_shared=2)
    graph = _RecordingPaddedGraph(routed_out=torch.zeros(8, 4))
    runner._padded_graphs[(1, runner._padded_buckets[-1])] = graph
    item, states = _work_item(1, routed=routed, shared=shared)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert graph.replays == 0, why
    assert runner.model.eager_calls == [(1, routed)], why


def test_a_layer_without_a_captured_graph_runs_eagerly():
    runner = _padded_runner()
    item, states = _work_item(7, routed=4, shared=1)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert runner.model.eager_calls == [(7, 4)]


@pytest.mark.parametrize(
    ("use_cuda_graph", "connector_driven"),
    [(False, True), (True, False), (False, False)],
)
def test_capture_is_skipped_unless_graphs_and_connector_driven(
    use_cuda_graph,
    connector_driven,
):
    # The padded path is for the connector-driven connector only. With a
    # control plane the runner already has a shape-keyed graph cache, and with
    # graphs off nothing should allocate.
    runner = object.__new__(GPUFFNModelRunner)
    runner.use_cuda_graph = use_cuda_graph
    runner.is_connector_driven = connector_driven
    runner._padded_graphs = {}
    runner.model = None  # would raise if capture got past the guard

    assert GPUFFNModelRunner.capture_padded_ffn_graphs(runner) == 0
    assert runner._padded_graphs == {}


def test_capture_does_not_run_twice():
    # start_ffn_server_loop is callable more than once; a second capture would
    # leak the first set of graphs and their pool.
    runner = object.__new__(GPUFFNModelRunner)
    runner.use_cuda_graph = True
    runner.is_connector_driven = True
    runner._padded_graphs = {0: object()}
    runner.model = None  # would raise if capture got past the guard

    assert GPUFFNModelRunner.capture_padded_ffn_graphs(runner) == 0
    assert list(runner._padded_graphs) == [0]


def _ordering_worker(order, *, initialized):
    worker = object.__new__(AFDFFNWorker)
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker.model_runner = SimpleNamespace(
        connector=SimpleNamespace(is_initialized=initialized),
        capture_padded_ffn_graphs=lambda: order.append("capture") or 0,
        initialize_afd_connector=lambda: order.append("rendezvous"),
    )
    return worker


def test_graphs_are_captured_before_the_connector_joins():
    # The connector's process group is the only barrier between the roles: the
    # Attention rank profiles, and so dispatches, the moment it clears. Joining
    # first and capturing after leaves those dispatches landing in a slot
    # nobody is polling, which wedges the Attention rank mid-write with a flag
    # unstamped -- fatal under ubatching, where the blocked thread never
    # reaches its next yield.
    order: list[str] = []
    worker = _ordering_worker(order, initialized=False)
    worker._run_ffn_server_loop = lambda: None

    worker.start_ffn_server_loop()
    worker._ffn_thread.join(timeout=5)

    assert order == ["capture", "rendezvous"]


def test_graphs_are_captured_before_the_serving_thread_starts():
    # Capture has to happen on an idle stream, and an AFD FFN EngineCore is a
    # daemon that reaches start_ffn_server_loop by collective_rpc and never
    # runs initialize_from_config -- so this is the only point that sees both
    # entry paths.
    order: list[str] = []
    worker = _ordering_worker(order, initialized=True)
    started = threading.Event()

    def serve_loop():
        order.append("serve")
        started.set()

    worker._run_ffn_server_loop = serve_loop

    worker.start_ffn_server_loop()
    assert started.wait(timeout=5)
    worker._ffn_thread.join(timeout=5)

    assert order == ["capture", "serve"]


def test_item_padded_far_past_its_bucket_runs_eagerly():
    # A replay costs its whole bucket, so an item far below one is cheaper run
    # eagerly. Measured on DeepSeek-V4: eager beat a 1.18x-padded replay by 6%.
    runner = _padded_runner(max_routed=1024, max_shared=0, buckets=(1024,))
    runner._padded_graphs[(0, 1024)] = _RecordingPaddedGraph(
        routed_out=torch.zeros(1024, 4)
    )
    item, states = _work_item(0, routed=600, shared=0)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert runner.model.eager_calls == [(0, 600)]


def test_small_item_replays_the_small_bucket_not_the_largest():
    # The regression this guards: one graph at the upper bound charged every
    # item the full-batch row count, which is what made a DBO ubatch -- a
    # quarter of the rows -- cost the same as a whole batch.
    runner = _padded_runner(max_routed=4096, max_shared=0, buckets=(1024, 2048, 4096))
    graphs = {
        bucket: _RecordingPaddedGraph(routed_out=torch.zeros(bucket, 4))
        for bucket in (1024, 2048, 4096)
    }
    for bucket, graph in graphs.items():
        runner._padded_graphs[(0, bucket)] = graph
    item, states = _work_item(0, routed=2020, shared=0)

    payload = GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert graphs[2048].replays == 1, "should take the smallest bucket that fits"
    assert graphs[1024].replays == 0
    assert graphs[4096].replays == 0
    assert runner.model.eager_calls == []
    assert payload.routed_output.shape[0] == 2020
    # Counts sum to the chosen bucket, not to the largest one.
    assert runner._padded_counts.tolist() == [2020, 28]


def test_item_larger_than_every_bucket_still_runs_eagerly():
    runner = _padded_runner(max_routed=8, max_shared=0, buckets=(2, 4, 8))
    runner._padded_graphs[(0, 8)] = _RecordingPaddedGraph(routed_out=torch.zeros(8, 4))
    item, states = _work_item(0, routed=9, shared=0)

    GPUFFNModelRunner._compute_work_item(runner, item, states)

    assert runner.model.eager_calls == [(0, 9)]
