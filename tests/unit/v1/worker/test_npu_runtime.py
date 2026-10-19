# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import logging
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from afd_plugin.compat.npu import (
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDF2ATransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
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


class _RecordingConnector:
    world_rank = 0

    def __init__(self):
        self.dp_metadata_updates = []
        self.sent_dp_metadata_lists = []
        self.dp_metadata_payloads = []
        self.sent_dp_metadata_payloads = []
        # The runners reach the control plane through connector.control_plane;
        # the fake serves as both.
        self.control_plane: _RecordingConnector | None = self

    def update_state_from_dp_metadata(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.dp_metadata_payloads.append(payload)
        self.dp_metadata_updates.append(
            (
                payload.dp_metadata_list,
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )

    def send_dp_metadata_list(self, payload):
        assert isinstance(payload, AFDControlPayload)
        self.sent_dp_metadata_payloads.append(payload)
        self.sent_dp_metadata_lists.append(
            (
                payload.dp_metadata_list,
                payload.is_graph_capturing,
                payload.is_warmup,
            ),
        )


class _AsyncRecordingConnector(_RecordingConnector):
    def __init__(self):
        super().__init__()
        self.control_plane = None


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


class _RecordingFakeModel:
    def __init__(self):
        self.calls = []

    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        self.calls.append((hidden_states, layer_idx, kwargs))
        return f"npu-ffn({hidden_states}, layer={layer_idx})"


class _FakeStructuredFFNModel:
    def compute_ffn_output(self, hidden_states, layer_idx, **_kwargs):
        return AFDF2ATransferPayload(
            routed_output=f"routed({hidden_states}, layer={layer_idx})",
            shared_output=f"shared({hidden_states}, layer={layer_idx})",
        )


class _FakeDPMetadata:
    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def _parallel_config(**overrides):
    values = {
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
        "enable_dbo": False,
        "use_ubatching": False,
        "num_ubatches": 1,
        "ubatch_size": 0,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "use_sequence_parallel_moe": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "dbo_decode_token_threshold": 1,
        "dbo_prefill_token_threshold": 1,
        "worker_cls": "unused",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(
    *,
    role="attention",
    connector="CAMP2pAFDConnector",
    extra_config=None,
    use_mla=False,
    use_v2_model_runner=True,
    cudagraph_mode="FULL",
    speculative_config=None,
    **parallel_overrides,
):
    async_dp = bool(parallel_overrides.pop("async_dp", False))
    compute_gate_on_attention = bool(
        parallel_overrides.pop("compute_gate_on_attention", False),
    )
    num_attention_ranks = int(parallel_overrides.pop("num_attention_ranks", 1))
    num_ffn_ranks = int(parallel_overrides.pop("num_ffn_ranks", 1))
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        additional_config={
            "afd": {
                "role": role,
                "connector": connector,
                "async": async_dp,
                "compute_gate_on_attention": compute_gate_on_attention,
                "num_attention_ranks": num_attention_ranks,
                "num_ffn_ranks": num_ffn_ranks,
                "connector_extra_config": extra_config or {},
            },
        },
        parallel_config=_parallel_config(**parallel_overrides),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_text_config=SimpleNamespace(model_type="deepseek_v2"),
            use_mla=use_mla,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(
                name=cudagraph_mode,
                has_full_cudagraphs=lambda: (
                    cudagraph_mode in {"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}
                ),
            ),
            fast_moe_cold_start=False,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )


def _async_moe_config(
    *,
    role="attention",
    compute_gate_on_attention=True,
    num_attention_ranks=1,
    num_ffn_ranks=1,
    tensor_parallel_size=1,
    prefill_context_parallel_size=1,
    decode_context_parallel_size=1,
    **extra_config,
):
    return _vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        use_mla=True,
        async_dp=True,
        compute_gate_on_attention=compute_gate_on_attention,
        num_attention_ranks=num_attention_ranks,
        num_ffn_ranks=num_ffn_ranks,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
        extra_config={
            "async_moe_ubatching": True,
            "attn_ranks_per_dp": tensor_parallel_size,
            **extra_config,
        },
    )


def _require_npu_runtime():
    pytest.importorskip("vllm", reason="NPU runtime tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="NPU runtime tests require vLLM-Ascend")
    pytest.importorskip("torch_npu", reason="NPU runtime tests require torch-npu")


def _new_attention_runner():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )

    runner = object.__new__(AFDNPUAttentionModelRunnerV2)
    runner.vllm_config = _vllm_config()
    return runner


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


def test_npu_attention_runner_builds_and_sets_metadata():
    torch = pytest.importorskip("torch")
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    runner.ubatch_slices = None
    runner._afd_suppress_metadata_send = False
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([5])),
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=5),
        cudagraph_runtime_mode=None,
    )

    runner.install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [5]
    assert len(runner.connector.dp_metadata_updates) == 1
    assert len(runner.connector.sent_dp_metadata_lists) == 1


def test_npu_attention_async_connector_skips_dp_metadata_control_plane():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        data_parallel_size=2,
    )
    runner.connector = _AsyncRecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False
    runner._afd_pending_metadata = None
    runner._afd_transaction_counter = 0
    forward_context = SimpleNamespace(
        additional_kwargs={},
        dp_metadata=None,
        ubatch_slices=None,
        batch_descriptor=SimpleNamespace(num_tokens=3),
    )

    runner.install_afd_metadata_on_forward_context(forward_context)

    metadata = forward_context.additional_kwargs["afd_metadata"]
    assert metadata.tokens_lens == [3]
    assert runner.connector.dp_metadata_updates == []
    assert runner.connector.sent_dp_metadata_lists == []


def test_npu_attention_runner_builds_dp_fallback():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = object()
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 7)

    dp_metadata = runner._ensure_dp_metadata(None)

    tokens = dp_metadata.num_tokens_across_dp_cpu
    if not isinstance(tokens, list):
        tokens = tokens.tolist()
    assert tokens == [7]


def test_npu_attention_runner_sends_graph_flags():
    torch = pytest.importorskip("torch")
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(role="attention")
    runner.connector = _RecordingConnector()
    runner._is_warmup = True
    runner._afd_is_graph_capturing = True
    runner._afd_transaction_counter = 0
    runner._afd_pending_metadata = runner.build_afd_metadata(None, 3)

    runner.send_dp_metadata(
        SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3])),
        None,
    )

    assert runner.connector.dp_metadata_updates[0][1:] == (True, True)
    assert runner.connector.sent_dp_metadata_lists[0][1:] == (True, True)


def test_npu_attention_runner_sends_per_ubatch_dp_metadata():
    runner = _new_attention_runner()
    runner.vllm_config = _vllm_config(
        role="attention",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    runner.connector = _RecordingConnector()
    runner._is_warmup = False
    runner._afd_is_graph_capturing = False

    ubatch_slices = [
        SimpleNamespace(
            request_slice=slice(0, 2),
            token_slice=slice(0, 4),
            num_tokens=4,
        ),
        SimpleNamespace(
            request_slice=slice(2, 3),
            token_slice=slice(4, 7),
            num_tokens=3,
        ),
    ]

    runner.send_dp_metadata(
        None,
        ubatch_slices,
    )

    dp_metadata_list = runner.connector.dp_metadata_updates[0][0]
    assert sorted(dp_metadata_list) == [0, 1]
    assert _tokens(dp_metadata_list[0]) == [4]
    assert _tokens(dp_metadata_list[1]) == [3]
    sent_dp_metadata_list = runner.connector.sent_dp_metadata_lists[0][0]
    assert sorted(sent_dp_metadata_list) == [0, 1]
    assert _tokens(sent_dp_metadata_list[0]) == [4]
    assert _tokens(sent_dp_metadata_list[1]) == [3]


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
):
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


def test_npu_ffn_runner_dp_path_invokes_model_with_hidden_states_and_layer(monkeypatch):
    from afd_plugin.connectors.npu.async_cam import AFDAsyncTransferState

    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = _RecordingFakeModel()
    runner.num_layers = 1
    runner.max_num_tokens = 1
    runner.use_aclgraph = False
    runner._acl_graphs = {}
    metadata = AFDTransferMetadata.create_attention_metadata(
        layer_idx=0,
        stage_idx=0,
        seq_len=1,
    )
    # The DP-metadata FFN path forwards only hidden states and the layer index
    # to the model; backend transfer state stays on the context and is consumed
    # by the connector, not spread into compute_ffn_output kwargs.
    runner.connector.attn_outputs.append(
        _ffn_payload(
            "hidden",
            metadata,
            states=AFDAsyncTransferState(
                group_list="groups",
                dynamic_scales="scales",
            ),
        ),
    )

    runner.execute_model(dp_metadata_list={0: _FakeDPMetadata([1])})

    assert runner.model.calls == [("hidden", 0, {})]


@pytest.mark.parametrize("layer_sequence", [(7,), (7, 7, 7, 8)])
def test_npu_ffn_worker_consumes_chunks_across_runner_steps(
    monkeypatch, layer_sequence
):
    _require_npu_runtime()
    from afd_plugin.connectors.npu.async_cam import (
        AFDAsyncFFNWorkItem,
        AFDAsyncTransferState,
    )
    from afd_plugin.v1.worker.npu import async_ffn_model_runner, ffn_worker

    context_calls = []
    sent_outputs = []

    @contextmanager
    def fake_ascend_forward_context(**kwargs):
        context_calls.append(kwargs)
        yield SimpleNamespace(
            additional_kwargs={},
            dp_metadata="dp",
            all_moe_layers={},
        )

    monkeypatch.setattr(
        async_ffn_model_runner,
        "ascend_forward_context",
        fake_ascend_forward_context,
    )
    runner = object.__new__(async_ffn_model_runner.AsyncNPUFFNModelRunner)
    runner._layered_executor = None
    runner._graph = None
    runner.effective_mode = "eager"
    runner.layer_ids = tuple(sorted(set(layer_sequence)))
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=True)
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = SimpleNamespace(control_plane=None, expert_per_rank=2)
    runner.model = _RecordingFakeModel()
    runner.num_layers = len(set(layer_sequence))
    runner.prof = None
    runner.max_num_tokens = 16
    metadata = AFDTransferMetadata.create_ffn_metadata(
        layer_idx=7,
        stage_idx=0,
        seq_lens=[5],
    )
    states = AFDAsyncTransferState(
        batch_size=5,
        hidden_size=16,
        topk=2,
        layer_idx=7,
        group_list="groups",
        dynamic_scales="scales[:5]",
    )
    context = AFDTransferContext(metadata=metadata, states=states)
    recv_output = AFDA2FTransferPayload(
        hidden_states="recv-hidden",
        context=context,
    )
    work_item = AFDAsyncFFNWorkItem(
        hidden_states="hidden[:5]",
        context=context,
        recv_output=recv_output,
        layer_idx=7,
        stage_idx=0,
        num_tokens=5,
        total_num_tokens=7,
    )

    # Three chunks for layer 7 cross the two-item runner-call boundary.
    # The real daemon must continue receiving rather than treat that boundary
    # as completion of the layer or of the request.
    work_items = [
        replace(
            work_item,
            layer_idx=layer_idx,
            hidden_states=f"chunk-{index}",
            context=AFDTransferContext(
                metadata=replace(metadata, layer_idx=layer_idx),
                states=replace(states, layer_idx=layer_idx),
            ),
        )
        for index, layer_idx in enumerate(layer_sequence)
    ]
    pending = iter(work_items)
    event = threading.Event()
    step_boundaries = []

    def recv_ffn_work_item(*, stage_idx, max_num_tokens):
        assert stage_idx == 0
        assert max_num_tokens == 16
        return next(pending)

    def send_ffn_work_item_output(sent_work_item, ffn_output):
        sent_outputs.append((sent_work_item, ffn_output))
        if len(sent_outputs) == len(work_items):
            event.set()
        return ffn_output

    runner.connector.recv_ffn_work_item = recv_ffn_work_item
    runner.connector.send_ffn_work_item_output = send_ffn_work_item_output

    worker = _new_ffn_worker()
    worker.model_runner = runner
    worker.device = SimpleNamespace(type="cpu")
    worker._ffn_shutdown_event = event
    bound_devices: list[SimpleNamespace] = []
    monkeypatch.setattr(ffn_worker.torch.npu, "set_device", bound_devices.append)
    monkeypatch.setattr(
        ffn_worker.torch.npu,
        "synchronize",
        lambda: step_boundaries.append(len(sent_outputs)),
    )
    worker._run_ffn_server_loop()

    assert runner.model.calls == [
        (
            item.hidden_states,
            item.layer_idx,
            {"group_list": "groups", "dynamic_scales": "scales[:5]"},
        )
        for item in work_items
    ]
    assert sent_outputs == [
        (item, f"npu-ffn({item.hidden_states}, layer={item.layer_idx})")
        for item in work_items
    ]
    assert bound_devices == [worker.device]
    assert step_boundaries == ([1] if len(layer_sequence) == 1 else [2, 4])
    assert context_calls[0]["num_tokens"] == 5
    assert context_calls[0]["skip_mc2_mask"] is True
    assert context_calls[0]["afd_metadata"].tokens_lens == [5]


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


def test_npu_ffn_runner_filters_dense_layers_when_gate_runs_on_attention():
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu.ffn_model_runner import _ffn_layer_indices

    runner = _new_ffn_runner()
    runner.num_layers = 5
    runner.afd_config = SimpleNamespace(compute_gate_on_attention=True)
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            n_routed_experts=8,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        ),
    )

    assert _ffn_layer_indices(runner) == [2, 4]


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


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


@pytest.mark.parametrize(
    ("attention_counts", "ffn_counts"),
    [
        ([12] * 8, (24, 24, 24, 24)),
        ([2, 3, 5, 7], (7, 10)),
        ([4, 1, 4, 8], (8, 9)),
    ],
)
def test_npu_ffn_runner_graph_key_uses_ffn_aggregated_token_counts(
    attention_counts, ffn_counts
):
    runner = _new_ffn_runner()
    runner.connector = _FakeFFNConnector(
        attn_size=len(attention_counts), ffn_size=len(ffn_counts)
    )
    runner.max_num_tokens = 24

    assert runner._make_graph_key({0: _FakeDPMetadata(attention_counts)}) == (
        (0, ffn_counts),
    )


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

    assert runner._acl_graphs == {}
    assert capture_flags == [True, False]
    assert runner.connector.ffn_outputs == [
        ("npu-ffn(hidden, layer=0)", metadata, {"ubatch_idx": 0}),
    ]


def test_npu_ffn_runner_capture_stores_acl_graph_and_skips_duplicate_state_update(
    monkeypatch,
):
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

    assert runner._make_graph_key(dp_metadata) in runner._acl_graphs
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


def test_npu_ffn_runner_requires_compute_hook(monkeypatch):
    _patch_ffn_forward_context(monkeypatch)
    runner = _new_ffn_runner()
    runner.vllm_config = _vllm_config(role="ffn")
    runner.connector = _FakeFFNConnector()
    runner.model = SimpleNamespace()
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

    with pytest.raises(AttributeError, match="compute_ffn_output"):
        runner.execute_ffn_step(dp_metadata_list={0: _FakeDPMetadata([1])})


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
    assert runner._acl_graphs == {}
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
):
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
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

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
        assert worker._ffn_thread is not None
        worker._ffn_thread.join(timeout=5)

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

    assert worker._ffn_thread is thread
    assert worker._ffn_shutdown_event is shutdown_event
    assert shutdown_event.is_set()
    assert ("parent", None) not in calls

    thread.alive = False
    worker.shutdown()

    assert worker._ffn_thread is None
    assert worker._ffn_shutdown_event is None
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

    def execute_connector_driven_step():
        calls.append("step")
        event.set()
        return True

    worker._ffn_shutdown_event = event
    worker.device = SimpleNamespace(type="cpu")
    worker.model_runner = object.__new__(ffn_worker.AsyncNPUFFNModelRunner)
    worker.model_runner.connector = _AsyncRecordingConnector()
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


def test_npu_feature_validation_rejects_unsupported_switches():
    for extra_config, message in [
        ({"compute_gate_on_attention": True}, "compute_gate_on_attention"),
        ({"quant_mode": 1}, "quant_mode=0"),
    ]:
        with pytest.raises((RuntimeError, ValueError), match=message):
            fail_if_unsupported_npu_afd_features(
                _vllm_config(extra_config=extra_config),
            )


def test_npu_feature_validation_uses_selected_connector_extra_info_parser():
    with pytest.raises(ValueError, match="does not support connector_extra_config"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="P2pNcclAFDConnector",
                extra_config={"core_num": 8},
            ),
        )


def test_npu_feature_validation_allows_two_ubatches_only():
    config = _vllm_config(
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    fail_if_unsupported_npu_afd_features(config)
    assert npu_afd_num_ubatches(config) == 2

    with pytest.raises(RuntimeError, match="exactly two ubatches"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                enable_dbo=True,
                use_ubatching=True,
                num_ubatches=4,
                ubatch_size=4,
            ),
        )

    config = _vllm_config()
    config.model_config.enforce_eager = False
    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("cudagraph_mode", ["FULL", "FULL_AND_PIECEWISE"])
def test_npu_feature_validation_requires_decode_only_full_graph_for_mla_dbo(
    cudagraph_mode,
):
    config = _vllm_config(
        use_mla=True,
        cudagraph_mode=cudagraph_mode,
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )

    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        fail_if_unsupported_npu_afd_features(config)

    fail_if_unsupported_npu_afd_features(
        _vllm_config(
            use_mla=True,
            cudagraph_mode="FULL_DECODE_ONLY",
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
            ubatch_size=4,
        ),
    )

    sparse_config = _vllm_config(
        use_mla=True,
        cudagraph_mode="FULL",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    sparse_config.model_config.hf_text_config = SimpleNamespace(index_topk=8)
    fail_if_unsupported_npu_afd_features(sparse_config)

    # A compressor model such as DeepSeek V4 selects the DSA backend, whose
    # graph-params update is a no-op, so the MLA DBO full graph rules do not
    # apply to it either.
    compressor_config = _vllm_config(
        use_mla=True,
        cudagraph_mode="FULL",
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )
    compressor_config.model_config.hf_text_config = SimpleNamespace(
        index_topk=512,
        compress_ratios=[0, 4, 128],
    )
    fail_if_unsupported_npu_afd_features(compressor_config)


def test_npu_feature_validation_rejects_speculative_mla_dbo_full_graph():
    config = _vllm_config(
        use_mla=True,
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_config=object(),
        enable_dbo=True,
        use_ubatching=True,
        num_ubatches=2,
        ubatch_size=4,
    )

    with pytest.raises(RuntimeError, match="does not support speculative decoding"):
        fail_if_unsupported_npu_afd_features(config)


def test_npu_async_feature_validation_requires_async_config_and_eager():
    with pytest.raises(RuntimeError, match="async=true"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(connector="CAMAsyncAFDConnector", async_dp=False),
        )

    config = _vllm_config(connector="CAMAsyncAFDConnector", async_dp=True)
    config.model_config.enforce_eager = False
    with pytest.raises(RuntimeError, match="only eager"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "parallel_override",
    [
        {"use_ubatching": True},
        {"enable_dbo": True},
    ],
)
def test_npu_async_feature_validation_requires_two_native_ubatches(parallel_override):
    with pytest.raises(RuntimeError, match="DBO requires exactly two ubatches"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                **parallel_override,
            ),
        )


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize(
    ("method", "rejected"),
    [
        ("mtp", True),
        ("eagle3", True),
        ("draft_model", True),
        ("ngram", False),
        ("ngram_gpu", False),
        ("suffix", False),
    ],
)
def test_npu_async_feature_validation_rejects_native_draft_models(
    use_v2_model_runner,
    method,
    rejected,
):
    _require_npu_runtime()
    from vllm.config.speculative import SpeculativeConfig

    # Use the pinned classifiers without loading a draft model's HF config.
    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.method = method
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        use_mla=True,
        use_v2_model_runner=use_v2_model_runner,
        speculative_config=speculative_config,
    )
    if rejected:
        with pytest.raises(RuntimeError, match="native draft models or MTP"):
            fail_if_unsupported_npu_afd_features(config)
    else:
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    ("factor", "max_num_batched_tokens", "rejected"),
    [
        ("0.015625", 8192, True),
        ("0.015625", 4096, False),
        ("0.5", 8192, False),
        (None, 262144, False),
        ("0", 16, True),
        ("nan", 16, True),
        ("invalid", 16, True),
        ("0.01_5625", 4096, True),
    ],
)
def test_npu_async_feature_validation_checks_receive_capacity(
    monkeypatch,
    use_v2_model_runner,
    role,
    factor,
    max_num_batched_tokens,
    rejected,
):
    if factor is None:
        monkeypatch.delenv("BATCH_SIZE_FACTOR", raising=False)
    else:
        monkeypatch.setenv("BATCH_SIZE_FACTOR", factor)
    config = _vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        use_mla=True,
        use_v2_model_runner=use_v2_model_runner,
        num_attention_ranks=2,
        tensor_parallel_size=2 if role == "attention" else 1,
        extra_config={"attn_ranks_per_dp": 2},
    )
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    if rejected:
        with pytest.raises(RuntimeError, match="BATCH_SIZE_FACTOR"):
            fail_if_unsupported_npu_afd_features(config)
    else:
        fail_if_unsupported_npu_afd_features(config)


def test_npu_async_feature_validation_allows_dynamic_quant_zero_or_one():
    for dynamic_quant in (0, "1"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                use_mla=True,
                extra_config={"dynamicQuant": dynamic_quant},
            ),
        )

    with pytest.raises(RuntimeError, match="dynamicQuant"):
        fail_if_unsupported_npu_afd_features(
            _vllm_config(
                connector="CAMAsyncAFDConnector",
                async_dp=True,
                use_mla=True,
                extra_config={"dynamicQuant": 2},
            ),
        )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_async_moe_config(), id="request-tp1"),
        pytest.param(
            _async_moe_config(
                tensor_parallel_size=2,
                num_attention_ranks=2,
                async_moe_split="token",
            ),
            id="token-attention-tp2",
        ),
        pytest.param(
            _async_moe_config(role="ffn", async_moe_split="token"),
            id="token-ffn-tp1",
        ),
    ],
)
def test_npu_async_moe_ubatching_validation_accepts_supported_shape(config):
    fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            _async_moe_config(compute_gate_on_attention=False),
            "compute_gate_on_attention",
            id="missing-attention-gate",
        ),
        pytest.param(
            _async_moe_config(async_moe_num_ubatches=3),
            "exactly two",
            id="three-stages",
        ),
        pytest.param(
            _async_moe_config(decode_context_parallel_size=2),
            "context parallel",
            id="decode-context-parallel",
        ),
        pytest.param(
            _async_moe_config(prefill_context_parallel_size=2),
            "context parallel",
            id="prefill-context-parallel",
        ),
        pytest.param(
            _async_moe_config(
                tensor_parallel_size=2,
                attn_ranks_per_dp=1,
            ),
            "attn_ranks_per_dp",
            id="attention-cam-tp-mismatch",
        ),
    ],
)
def test_npu_async_moe_ubatching_validation_rejects_unsupported_shape(
    config,
    error,
):
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


def _tokens(dp_metadata):
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


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


def _fake_npu_connector_factory(monkeypatch, connector):
    from afd_plugin.v1.worker.npu import (
        attention_model_runner_v2 as attention_model_runner,
    )

    monkeypatch.setattr(
        attention_model_runner,
        "_resolve_world_ranks",
        lambda: (3, 0),
    )
    calls = []

    def create_connector(rank, local_rank, vllm_config, afd_config):
        calls.append((rank, local_rank))
        return connector

    monkeypatch.setattr(
        attention_model_runner.AFDConnectorFactory,
        "create_connector",
        create_connector,
    )
    return calls


def test_npu_attention_runner_constructor_does_not_initialize_connector(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import (
        attention_model_runner_v2 as attention_model_runner,
    )

    events: list[str] = []
    connector = _LifecycleConnector(events, control_plane=object())

    def fake_native_init(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device

    monkeypatch.setattr(
        attention_model_runner.NPUModelRunnerV2,
        "__init__",
        fake_native_init,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "fail_if_unsupported_npu_afd_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "validate_npu_model_runner_v2_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        attention_model_runner,
        "create_afd_npu_profiler",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        attention_model_runner.AFDNPUAttentionModelRunnerV2,
        "parse_config",
        staticmethod(
            lambda _vllm_config: SimpleNamespace(
                connector="CAMP2pAFDConnector",
                async_dp=False,
            )
        ),
    )
    connector_calls = _fake_npu_connector_factory(monkeypatch, connector)

    runner = attention_model_runner.AFDNPUAttentionModelRunnerV2(
        _vllm_config(),
        SimpleNamespace(index=2, type="npu"),
    )

    assert runner.connector is connector
    assert connector_calls == [(3, 2)]
    # Connector construction is device-light; the collective rendezvous is
    # deferred to load_model() so Attention and FFN weight loading overlap.
    assert events == []
    assert connector.is_initialized is False


def test_npu_ffn_connector_uses_model_device_index_for_dp_worker(monkeypatch):
    _require_npu_runtime()
    from afd_plugin.v1.worker.npu import ffn_model_runner

    connector = _LifecycleConnector([], control_plane=object())
    calls = []

    def create_connector(rank, local_rank, vllm_config, afd_config):
        calls.append((rank, local_rank))
        return connector

    monkeypatch.setattr(ffn_model_runner, "_resolve_world_ranks", lambda: (7, 0))
    monkeypatch.setattr(
        ffn_model_runner.AFDConnectorFactory, "create_connector", create_connector
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "validate_npu_model_runner_v2_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ffn_model_runner,
        "fail_if_unsupported_npu_afd_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ffn_model_runner, "create_afd_npu_profiler", lambda _role: None)
    monkeypatch.setattr(ffn_model_runner, "_use_npu_aclgraph", lambda _config: False)
    config = _vllm_config(role="ffn")
    config.model_config.hf_config = SimpleNamespace(num_hidden_layers=43)

    runner = ffn_model_runner.AFDNPUFFNModelRunner(
        config, SimpleNamespace(index=3, type="npu")
    )

    assert runner.connector is connector
    assert calls == [(7, 3)]


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


@pytest.mark.parametrize(
    "extra_config,additional_config,model_type,use_mla,error",
    [
        (
            {},
            {"enable_shared_expert_dp": True},
            "deepseek_v2",
            True,
            "shared-expert DP",
        ),
        *[
            (
                {},
                {"finegrained_tp_config": {key: 2}},
                "deepseek_v2",
                True,
                "finegrained TP",
            )
            for key in (
                "oproj_tensor_parallel_size",
                "mlp_tensor_parallel_size",
                "embedding_tensor_parallel_size",
                "lmhead_tensor_parallel_size",
            )
        ],
        ({}, {}, "qwen3_moe", True, "DeepSeek-V2/V3 MLA"),
        ({}, {}, "deepseek_v2", False, "DeepSeek-V2/V3 MLA"),
    ],
)
def test_npu_async_mrv2_feature_boundaries(
    extra_config, additional_config, model_type, use_mla, error
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        extra_config=extra_config,
    )
    config.additional_config.update(additional_config)
    config.model_config.hf_text_config.model_type = model_type
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "model_type,use_mla",
    [("deepseek_v2", True), ("deepseek_v3", True), ("deepseek_v4", False)],
)
@pytest.mark.parametrize("tp_size,use_sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("local_ubatching", [False, True])
def test_npu_async_mrv2_features_allow_model_layouts(
    model_type, use_mla, tp_size, use_sp, local_ubatching
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        tensor_parallel_size=tp_size,
        use_sequence_parallel_moe=use_sp,
        num_attention_ranks=2 * tp_size,
        data_parallel_size=2,
        extra_config={
            "attn_ranks_per_dp": tp_size,
            "dynamicQuant": 1,
            "async_moe_ubatching": local_ubatching,
            "async_moe_split": "token" if tp_size > 1 else "request",
        },
    )
    config.model_config.hf_text_config.model_type = model_type
    config.additional_config["enable_flashcomm1"] = use_sp
    fail_if_unsupported_npu_afd_features(config)


def test_npu_async_mrv2_rejects_fractional_tp_capacity():
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=True,
        tensor_parallel_size=2,
        num_attention_ranks=2,
        extra_config={"attn_ranks_per_dp": 2},
    )
    config.scheduler_config.max_num_batched_tokens = 5
    with pytest.raises(RuntimeError, match="divisible by attn_ranks_per_dp"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "native_flag,local_flag,dp_size,expected",
    [
        (False, True, 2, True),
        (True, False, 2, True),
        (False, False, 2, False),
        (False, True, 1, True),
    ],
)
def test_async_cam_dbo_configuration_preserves_thresholds(
    native_flag, local_flag, dp_size, expected
):
    from afd_plugin.compat.npu.runtime_config import configure_async_cam_dbo

    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        data_parallel_size=dp_size,
        use_ubatching=native_flag,
        num_ubatches=2,
        extra_config={"async_moe_ubatching": local_flag},
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=8,
    )
    configure_async_cam_dbo(config)
    assert config.parallel_config.enable_dbo is expected
    config.parallel_config.use_ubatching = expected
    assert config.parallel_config.dbo_decode_token_threshold == 2
    assert config.parallel_config.dbo_prefill_token_threshold == 8


@pytest.mark.parametrize("dp_size", [1, 2])
def test_async_moe_flag_initializes_two_workspace_lanes_before_runner(
    monkeypatch, dp_size
):
    _require_npu_runtime()
    from vllm.config import ParallelConfig

    from afd_plugin.v1.worker.npu import attention_worker

    parallel = object.__new__(ParallelConfig)
    parallel.enable_dbo = False
    parallel.ubatch_size = 1
    parallel.data_parallel_size = dp_size
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        extra_config={"async_moe_ubatching": True},
    )
    config.parallel_config = parallel
    worker = SimpleNamespace(vllm_config=config, _init_device=lambda: "device")
    events = []
    monkeypatch.setattr(
        attention_worker, "assert_compatible_afd_stack", lambda *a, **k: None
    )
    monkeypatch.setattr(
        attention_worker,
        "fail_if_unsupported_npu_afd_features",
        lambda cfg: events.append(("validate", cfg.parallel_config.enable_dbo)),
    )
    monkeypatch.setattr(
        attention_worker, "fix_all2all_backend_for_afd", lambda *a: None
    )
    monkeypatch.setattr(
        attention_worker, "validate_npu_model_runner_v2_config", lambda *a, **k: None
    )
    monkeypatch.setattr(attention_worker, "get_afd_model_config", lambda cfg, **kw: cfg)
    monkeypatch.setattr(
        attention_worker,
        "init_workspace_manager",
        lambda device, lanes: events.append(("workspace", lanes)),
    )

    def runner(cfg, device):
        assert cfg.parallel_config.enable_dbo
        events.append(("runner", cfg.parallel_config.num_ubatches))
        return object()

    monkeypatch.setattr(attention_worker, "AFDNPUAttentionModelRunnerV2", runner)
    attention_worker.AFDNPUAttentionWorker.init_device(worker)
    assert events == [("validate", True), ("workspace", 2), ("runner", 2)]
