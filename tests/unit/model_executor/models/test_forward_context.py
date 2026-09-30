# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
)
from vllm.forward_context import (  # noqa: E402
    get_forward_context as get_current_forward_context,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE  # noqa: E402

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor.models import (  # noqa: E402
    get_afd_metadata_from_forward_context,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
    get_async_moe_ubatch_metadata_from_forward_context,
)
from afd_plugin.model_executor.npu.async_cam_execution import (  # noqa: E402
    CAM_ASYNC_EXECUTION_KEY,
    CAM_ASYNC_SCHEDULER_KEY,
    CAMAsyncExecutionContext,
    CAMAsyncPhase,
    CAMAsyncUbatchScheduler,
    require_cam_async_execution_context,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
    plan_async_moe_stages,
)


@pytest.fixture
def cam_scheduler():
    scheduler = CAMAsyncUbatchScheduler(wait_timeout=2)
    yield scheduler
    scheduler.shutdown()


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None


def test_get_async_moe_ubatch_metadata_from_additional_kwargs():
    sidecar = {"ubatch_slices": ["stage0", "stage1"]}
    forward_context = SimpleNamespace(
        additional_kwargs={ASYNC_MOE_UBATCH_METADATA_KEY: sidecar},
    )

    assert (
        get_async_moe_ubatch_metadata_from_forward_context(forward_context) is sidecar
    )


@pytest.mark.parametrize(
    ("is_first_rank", "is_last_rank"),
    [(True, False), (False, True)],
)
def test_async_model_forward_preserves_pp_boundaries(
    monkeypatch,
    is_first_rank,
    is_last_rank,
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    class FakeIntermediateTensors(dict):
        pass

    monkeypatch.setattr(async_forward, "IntermediateTensors", FakeIntermediateTensors)
    monkeypatch.setattr(
        async_forward,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=is_last_rank,
        ),
    )
    forward_context = SimpleNamespace()
    afd_metadata = SimpleNamespace()
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        async_forward,
        "get_afd_metadata_from_forward_context",
        lambda context: afd_metadata if context is forward_context else None,
    )
    monkeypatch.setattr(
        async_forward,
        "get_async_moe_ubatch_metadata_from_forward_context",
        lambda context: None,
    )

    schedule_calls: list[tuple[Any, ...]] = []

    def run_schedule(
        model,
        hidden_states,
        residual,
        positions,
        received_metadata,
        llama_4_scaling,
    ):
        schedule_calls.append(
            (
                model,
                hidden_states,
                residual,
                positions,
                received_metadata,
                llama_4_scaling,
            )
        )
        next_residual = (
            torch.zeros_like(hidden_states) if residual is None else residual + 2
        )
        return hidden_states + 1, next_residual

    monkeypatch.setattr(
        async_forward,
        "run_attention_gate_afd_forward",
        run_schedule,
    )
    norm_calls = []

    def run_norm(hidden_states, residual):
        norm_calls.append((hidden_states, residual))
        return hidden_states + residual, None

    model = SimpleNamespace(
        aux_hidden_state_layers=(),
        embed_input_ids=lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1),
        _get_llama_4_scaling=lambda positions: None,
        norm=run_norm,
    )
    positions = torch.arange(2)
    if is_first_rank:
        input_ids = torch.tensor([3, 4])
        intermediate_tensors = None
        expected_hidden_states = model.embed_input_ids(input_ids)
        expected_residual = None
    else:
        input_ids = None
        expected_hidden_states = torch.full((2, 1), 5.0)
        expected_residual = torch.full((2, 1), 7.0)
        intermediate_tensors = FakeIntermediateTensors(
            {
                "hidden_states": expected_hidden_states,
                "residual": expected_residual,
            }
        )

    output = async_forward.run_model_forward(
        model,
        input_ids,
        positions,
        intermediate_tensors,
    )

    assert len(schedule_calls) == 1
    assert torch.equal(schedule_calls[0][1], expected_hidden_states)
    assert schedule_calls[0][2] is expected_residual
    assert schedule_calls[0][4] is afd_metadata
    scheduled_hidden_states = expected_hidden_states + 1
    scheduled_residual = (
        torch.zeros_like(expected_hidden_states)
        if expected_residual is None
        else expected_residual + 2
    )
    if is_last_rank:
        assert len(norm_calls) == 1
        assert torch.equal(output, scheduled_hidden_states + scheduled_residual)
    else:
        assert isinstance(output, FakeIntermediateTensors)
        assert torch.equal(output["hidden_states"], scheduled_hidden_states)
        assert torch.equal(output["residual"], scheduled_residual)


@pytest.mark.parametrize("in_profile_run", [False, True], ids=["regular", "profile"])
@pytest.mark.parametrize(
    "dense_prefix", [False, True], ids=["moe-only", "dense-prefix"]
)
def test_async_cam_profile_forward_runs_matched_connector_io(
    monkeypatch,
    in_profile_run,
    dense_prefix,
    cam_scheduler,
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    execution = CAMAsyncExecutionContext(0, 0, 1, True)
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        additional_kwargs={
            CAM_ASYNC_EXECUTION_KEY: execution,
            CAM_ASYNC_SCHEDULER_KEY: cam_scheduler,
        },
    )
    forward_context.in_profile_run = in_profile_run
    forward_context.ubatch_idx = 0
    forward_context.num_ubatches = 1
    forward_context.flash_comm_v1_enabled = True
    native_calls = []
    runner_calls = []

    events: list[tuple[Any, ...]] = []
    dispatch_layouts: list[object] = []
    restored_layouts: list[object] = []
    pending = []
    completed_layer_idx = None

    def prepare_dispatch_payload(
        hidden_states,
        topk_weights,
        topk_ids,
        router_logits,
        *,
        use_sequence_parallel,
    ):
        assert use_sequence_parallel is True
        layout = object()
        dispatch_layouts.append(layout)
        return SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            layout=layout,
        )

    def restore_dispatch_output(local_output, layout):
        events.append(("restore", completed_layer_idx, 0))
        restored_layouts.append(layout)
        return local_output

    monkeypatch.setattr(
        npu_remote_moe,
        "prepare_cam_dispatch_payload",
        prepare_dispatch_payload,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "restore_cam_dispatch_output",
        restore_dispatch_output,
    )

    def send_attn_output(hidden_states, context, **kwargs):
        metadata = context.metadata
        assert metadata.seq_lens == [hidden_states.shape[0]]
        assert kwargs["topk_weights"].shape == (hidden_states.shape[0], 1)
        assert kwargs["topk_ids"].dtype == torch.int32
        events.append(("send", metadata.layer_idx, metadata.stage_idx))
        pending.append((metadata.layer_idx, hidden_states))

    def recv_ffn_output(ref_tensor, ubatch_idx):
        nonlocal completed_layer_idx
        completed_layer_idx, dispatched = pending.pop(0)
        assert ref_tensor is dispatched
        events.append(("recv", completed_layer_idx, ubatch_idx))
        return ref_tensor + 10 * (completed_layer_idx - int(dense_prefix) + 1)

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)
    forward_context.additional_kwargs["afd_metadata"] = afd_metadata

    class _Runner:
        __call__ = npu_remote_moe.AFDCAMAsyncMoERunner.forward
        _dispatch_cam = npu_remote_moe.AFDCAMAsyncMoERunner._dispatch_cam
        _combine_cam = npu_remote_moe.AFDCAMAsyncMoERunner._combine_cam
        layer_id = npu_remote_moe.AFDCAMAsyncMoERunner.layer_id
        is_internal_router = True

        def __init__(self, layer_idx):
            self.layer_name = f"model.layers.{layer_idx}.mlp.experts"

        def _route_native(self, hidden_states):
            runner_calls.append(self.layer_id)
            return (
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

        def _compute_attention_shared(self, hidden_states):
            events.append(("shared", self.layer_id, 0))
            return 2 * hidden_states

    class _MoE:
        is_sequence_parallel = False

        def __init__(self, layer_idx):
            self.experts = _Runner(layer_idx)

        def __call__(self, hidden_states):
            native_calls.append(self.experts.layer_id)
            return DeepseekV2MoE.forward(self, hidden_states)

    class _MoELayer:
        is_moe_layer = True

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.mlp = _MoE(layer_idx)

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            events.append(("compute", self.layer_idx, 0))
            return hidden_states + 1, residual + 2

    class _DenseLayer:
        is_moe_layer = False

        def __call__(self, positions, hidden_states, residual, llama_4_scaling):
            events.append(("dense", 0))
            return hidden_states + 5, residual + 1

    moe_layers = [_MoELayer(int(dense_prefix) + offset) for offset in range(2)]
    layers: list[_MoELayer | _DenseLayer] = list(moe_layers)
    if dense_prefix:
        layers.insert(0, _DenseLayer())
    model = SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers))
    expected_events: list[tuple[Any, ...]] = [("dense", 0)] if dense_prefix else []
    for layer in moe_layers:
        expected_events.extend(
            (event, layer.layer_idx, 0)
            for event in ("compute", "send", "shared", "recv", "restore")
        )

    for call_idx in range(2):
        events.clear()
        native_calls.clear()
        runner_calls.clear()
        dispatch_layouts.clear()
        restored_layouts.clear()
        hidden_states = torch.full((2, 4), float(2 * call_idx))
        residual = torch.full_like(hidden_states, 7)
        with override_forward_context(forward_context):
            output, output_residual = async_forward.run_attention_gate_afd_forward(
                model,
                hidden_states,
                residual,
                torch.arange(2),
                afd_metadata,
            )

        torch.testing.assert_close(
            output,
            9 * (hidden_states + 5 * int(dense_prefix)) + 62,
        )
        torch.testing.assert_close(output_residual, residual + 4 + int(dense_prefix))
        assert events == expected_events
        assert native_calls == runner_calls == [layer.layer_idx for layer in moe_layers]
        assert forward_context.additional_kwargs[CAM_ASYNC_EXECUTION_KEY] is execution
        assert restored_layouts == dispatch_layouts
        assert not pending
        for layer in moe_layers:
            assert vars(layer.mlp.experts) == {
                "layer_name": f"model.layers.{layer.layer_idx}.mlp.experts"
            }


def test_deepseek_afd_wrapper_keeps_full_model_compile_enabled():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "@native.support_torch_compile\nclass AFDDeepseekV2Model" in source
    assert "from __future__ import annotations" not in source
    assert "self.do_not_compile = True" not in source


def test_deepseek_afd_wrapper_treats_index_topk_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'self.is_v32 = hasattr(config, "index_topk")' in source
    assert "self.is_v32 = config.index_topk is not None" not in source
    assert "topk_tokens = config.index_topk" in source


def test_deepseek_afd_wrapper_treats_llama_4_scaling_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'getattr(self.config, "llama_4_scaling", None)' in source
    assert "self.config.llama_4_scaling" not in source


def test_deepseek_afd_attention_path_can_compute_gate_before_send():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    model_source = source.split("class AFDDeepseekV2Model", 1)[1].split(
        "class AFDDeepseekV2ForCausalLM",
        1,
    )[0]
    model_forward = model_source.split("    def forward(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]
    gate_runner = Path("afd_plugin/model_executor/npu/remote_moe.py").read_text()
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if afd_role == "attention":' in source
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "def _forward_attention(" not in source
    assert "return super().forward(" in model_forward
    assert "deepseek_v2_async_cam_forward.run_model_forward(" in model_forward
    assert "def _route_native(" in gate_runner
    assert "class AFDCAMAsyncMoERunner(AFDRemoteMoERunnerBase):" in gate_runner
    assert "execution.checkpoint(self.layer_id, CAMAsyncPhase.ROUTED)" in gate_runner
    assert (
        "execution.checkpoint(self.layer_id, CAMAsyncPhase.DISPATCHED)" in gate_runner
    )
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert "layer.mlp(hidden_states)" in attention_gate_forward
    assert "layer_done(layer.layer_idx)" in attention_gate_forward
    assert "send_attn_output(" not in executor_source
    assert "recv_ffn_output(" not in executor_source
    assert "prepare_cam_dispatch_payload(" not in executor_source
    assert "restore_cam_dispatch_output(" not in executor_source
    for name in (
        "compute_gate_topk",
        "stage_runners",
        "stage_dispatch_refs",
        "stage_dispatch_layouts",
        "stage_shared_outputs",
        "topk_weights",
        "topk_ids",
    ):
        assert name not in executor_source


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path("afd_plugin/model_executor/npu/remote_moe.py").read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "self.mlp.experts" not in compute_attn_output
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "deepseek_v2_attention_gate" not in compute_attn_output
    helper_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    assert "def compute_attention_gate_topk(" not in helper_source
    assert "force_balanced_topk_ids_enabled" in gate_source
    assert "balanced_topk_ids = torch.arange(" in gate_source
    assert "topk_ids.copy_(" in gate_source
    assert "topk_weights, topk_ids = select_experts(" in (gate_source)
    assert "if force_balanced_topk_ids_enabled():" in gate_source
    assert (
        gate_source.index(
            "topk_weights, topk_ids = select_experts(",
        )
        < gate_source.index("if force_balanced_topk_ids_enabled():")
        < gate_source.index("return topk_weights.to(torch.float32)")
    )


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()

    assert "self.is_moe_layer = is_moe_layer" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in executor_source
    assert (
        "return _ATTENTION_ROLE if compute_gate_on_attention else _FFN_ROLE" in source
    )


def test_deepseek_compute_gate_on_attention_selects_backend_boundary():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'device_type not in ("cuda", "npu")' in source
    assert "self.mlp = AFDDeepseekV2RemoteExpertsMoE(" in source
    assert "GateOnlyRemoteMoE" not in source
    assert "build_attention_moe_runner(" in source
    assert 'prefix=f"{prefix}.mlp"' in source
    assert "compute_gate_topk(" not in source
    assert (
        "# NPU-only: gated MoE FFN compute consumes Attention-side topk payloads."
        in source
    )


@pytest.mark.parametrize(
    "dense_prefix", [False, True], ids=["moe-only", "dense-prefix"]
)
@pytest.mark.parametrize("num_moe_layers", [1, 2])
@pytest.mark.parametrize(
    "use_sequence_parallel", [False, True], ids=["tp", "flashcomm1"]
)
def test_async_moe_pipeline_preserves_stage_order(
    monkeypatch, dense_prefix, cam_scheduler, num_moe_layers, use_sequence_parallel
):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward
    from afd_plugin.model_executor.npu import remote_moe as npu_remote_moe

    events: list[tuple[Any, ...]] = []
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"layer": "full"},
        slot_mapping={},
        additional_kwargs={CAM_ASYNC_SCHEDULER_KEY: cam_scheduler},
    )
    forward_context.ubatch_idx = 0
    forward_context.num_ubatches = 1
    forward_context.num_tokens = 4
    forward_context.pad_size = 0
    forward_context.flash_comm_v1_enabled = use_sequence_parallel
    forward_context.dbo_enabled = False
    native_calls = []
    runner_calls = []
    stage_contexts = {}

    pending = {}
    dispatch_layouts = []
    restored_layouts = []

    def send_attn_output(hidden_states, context, **_kwargs):
        metadata = context.metadata
        assert metadata.seq_lens == [hidden_states.shape[0]]
        assert metadata.stage_idx not in pending
        events.append(("send", metadata.layer_idx, metadata.stage_idx))
        pending[metadata.stage_idx] = (metadata.layer_idx, hidden_states)

    def recv_ffn_output(ref_tensor, ubatch_idx):
        layer_idx, dispatched = pending.pop(ubatch_idx)
        assert ref_tensor is dispatched
        events.append(("recv", layer_idx, ubatch_idx))
        return ref_tensor + 10 * (layer_idx - int(dense_prefix) + 1)

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    parent_metadata = AFDForwardContextMetadata(
        stage_idx=0,
        connector=connector,
        tokens_start_loc=[0],
        requests_start_loc=[0],
        tokens_lens=[4],
        num_stages=1,
    )
    forward_context.additional_kwargs["afd_metadata"] = parent_metadata

    class _Runner:
        __call__ = npu_remote_moe.AFDCAMAsyncMoERunner.forward
        _dispatch_cam = npu_remote_moe.AFDCAMAsyncMoERunner._dispatch_cam
        _combine_cam = npu_remote_moe.AFDCAMAsyncMoERunner._combine_cam
        layer_id = npu_remote_moe.AFDCAMAsyncMoERunner.layer_id
        is_internal_router = True

        def __init__(self, layer_idx):
            self.layer_name = f"model.layers.{layer_idx}.mlp.experts"

        def _route_native(self, hidden_states):
            stage_idx = get_current_forward_context().ubatch_idx
            runner_calls.append((self.layer_id, stage_idx))
            topk = hidden_states[:, :1]
            return topk, topk.to(torch.int32), None

        def _compute_attention_shared(self, hidden_states):
            stage_idx = get_current_forward_context().ubatch_idx
            assert pending[stage_idx][0] == self.layer_id
            assert pending[stage_idx][1] is hidden_states
            events.append(("shared", self.layer_id, stage_idx))
            return hidden_states + self.layer_id - int(dense_prefix) + 1

    class _MoE:
        is_sequence_parallel = False

        def __init__(self, layer_idx):
            self.experts = _Runner(layer_idx)

        def __call__(self, hidden_states):
            native_calls.append(
                (self.experts.layer_id, get_current_forward_context().ubatch_idx)
            )
            return DeepseekV2MoE.forward(self, hidden_states)

    class _MoELayer:
        is_moe_layer = True

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.mlp = _MoE(layer_idx)

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            stage_context = get_current_forward_context()
            stage_idx = stage_context.ubatch_idx
            if stage_context.num_ubatches == 1:
                assert stage_context is forward_context
                assert stage_context.num_tokens == hidden_states.shape[0] * (
                    2 if use_sequence_parallel else 1
                )
                assert positions == "full-positions"
                assert llama_4_scaling == "full-scaling"
                return hidden_states, residual
            assert positions == f"positions-{stage_idx}"
            assert llama_4_scaling == f"scaling-{stage_idx}"
            events.append(
                (
                    "compute",
                    self.layer_idx,
                    stage_idx,
                    stage_context.attn_metadata,
                    stage_context.num_tokens,
                    stage_context.pad_size,
                ),
            )
            stage_contexts[stage_idx] = stage_context
            assert stage_context.dbo_enabled is False
            assert stage_context.num_ubatches == 2
            metadata = stage_context.additional_kwargs["afd_metadata"]
            assert metadata is not parent_metadata
            assert metadata.stage_idx == stage_idx
            assert metadata.num_stages == 2
            assert metadata.tokens_start_loc == [0, actual_counts[0]]
            assert metadata.tokens_lens == list(stage_input_counts)
            return hidden_states, residual

    class _DenseLayer:
        is_moe_layer = False

        def __call__(self, positions, hidden_states, residual, llama_4_scaling):
            assert positions == "full-positions"
            assert llama_4_scaling == "full-scaling"
            events.append(("dense", 0))
            return hidden_states + 3, residual

    def build_stage_inputs(hidden_states, residual, positions, scaling, metadata):
        events.append(("split",))
        assert metadata is execution_plan
        return SimpleNamespace(
            hidden_states=[
                hidden_states[: stage_rows[0]].clone(),
                hidden_states[: stage_rows[1]].clone() + 1,
            ],
            residuals=[None, None],
            positions=["positions-0", "positions-1"],
            llama_4_scaling=["scaling-0", "scaling-1"],
        )

    def restore_stage_outputs(outputs, metadata):
        events.append(("restore-parent",))
        assert metadata is execution_plan
        assert not pending
        return tuple(outputs)

    def prepare_dispatch_payload(
        hidden_states, topk_weights, topk_ids, router_logits, **kwargs
    ):
        assert kwargs["use_sequence_parallel"] is use_sequence_parallel
        layout = object()
        dispatch_layouts.append(layout)
        return SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            layout=layout,
        )

    def restore_dispatch_output(output, layout):
        restored_layouts.append(layout)
        return output

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "build_async_moe_stage_inputs",
        build_stage_inputs,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_async_moe_stage_outputs",
        restore_stage_outputs,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "prepare_cam_dispatch_payload",
        prepare_dispatch_payload,
    )
    monkeypatch.setattr(
        npu_remote_moe,
        "restore_cam_dispatch_output",
        restore_dispatch_output,
    )

    moe_layers = [
        _MoELayer(int(dense_prefix) + offset) for offset in range(num_moe_layers)
    ]
    layers: list[_MoELayer | _DenseLayer] = list(moe_layers)
    if dense_prefix:
        layers.insert(0, _DenseLayer())
    model = SimpleNamespace(start_layer=0, end_layer=len(layers), layers=layers)
    first_layer_idx = int(dense_prefix)
    last_layer_idx = first_layer_idx + 1
    expected_events: list[tuple[Any, ...]] = [("dense", 0)] if dense_prefix else []
    expected_events += [
        ("split",),
        ("compute", first_layer_idx, 0),
        ("send", first_layer_idx, 0),
        ("shared", first_layer_idx, 0),
        ("compute", first_layer_idx, 1),
        ("recv", first_layer_idx, 0),
        ("send", first_layer_idx, 1),
        ("shared", first_layer_idx, 1),
    ]
    if num_moe_layers == 2:
        expected_events += [
            ("compute", last_layer_idx, 0),
            ("recv", first_layer_idx, 1),
            ("send", last_layer_idx, 0),
            ("shared", last_layer_idx, 0),
            ("compute", last_layer_idx, 1),
            ("recv", last_layer_idx, 0),
            ("send", last_layer_idx, 1),
            ("shared", last_layer_idx, 1),
            ("recv", last_layer_idx, 1),
        ]
    else:
        expected_events.append(("recv", first_layer_idx, 1))
    expected_events.append(("restore-parent",))
    gain, offset = {1: (2, 11), 2: (4, 44)}[num_moe_layers]
    worker_threads = None

    for call_idx, stage_input_counts in enumerate(((2, 4), (4, 6))):
        actual_counts = (
            ((2, 2) if call_idx == 0 else (3, 5))
            if use_sequence_parallel
            else stage_input_counts
        )
        stage_rows = tuple(
            count // (2 if use_sequence_parallel else 1) for count in stage_input_counts
        )
        parent_tokens = sum(actual_counts)
        parent_rows = parent_tokens // (2 if use_sequence_parallel else 1)
        forward_context.num_tokens = parent_tokens
        parent_metadata.tokens_lens = [parent_tokens]
        execution_plan = AsyncMoeUbatchMetadata(
            attn_metadata=[
                {"layer": f"stage-{stage}", "request": call_idx} for stage in range(2)
            ],
            stages=[
                AsyncMoeStage(
                    slice(0, 1),
                    slice(0, actual_counts[0]),
                    input_tokens=stage_input_counts[0],
                ),
                AsyncMoeStage(
                    slice(1, 2),
                    slice(actual_counts[0], parent_tokens),
                    input_tokens=stage_input_counts[1],
                ),
            ],
            parent_input_tokens=parent_tokens,
            use_sequence_parallel=use_sequence_parallel,
        )
        events.clear()
        native_calls.clear()
        runner_calls.clear()
        stage_contexts.clear()
        dispatch_layouts.clear()
        restored_layouts.clear()
        with override_forward_context(forward_context):
            output, residual = (
                deepseek_v2_async_cam_forward.run_async_moe_ubatch_afd_forward(
                    model=model,
                    hidden_states=torch.full((parent_rows, 8), float(2 * call_idx)),
                    residual=None,
                    positions="full-positions",
                    afd_metadata=parent_metadata,
                    async_moe_ubatch_metadata=execution_plan,
                    llama_4_scaling="full-scaling",
                )
            )
            assert get_current_forward_context() is forward_context

        assert [event[:3] for event in events] == expected_events
        assert (
            native_calls
            == runner_calls
            == [(layer.layer_idx, stage) for layer in moe_layers for stage in range(2)]
        )
        assert stage_contexts[0] is not stage_contexts[1]
        assert (
            stage_contexts[0].additional_kwargs
            is not stage_contexts[1].additional_kwargs
        )
        assert (
            stage_contexts[0].additional_kwargs["afd_metadata"].tokens_lens
            is not stage_contexts[1].additional_kwargs["afd_metadata"].tokens_lens
        )
        for event in (event for event in events if event[0] == "compute"):
            stage_idx = event[2]
            assert event[3] == {"layer": f"stage-{stage_idx}", "request": call_idx}
            assert event[4] == actual_counts[stage_idx]
            assert event[5] == stage_input_counts[stage_idx] - actual_counts[stage_idx]
        expected = gain * (2 * call_idx + 3 * int(dense_prefix)) + offset
        torch.testing.assert_close(
            output[0], torch.full((stage_rows[0], 8), float(expected))
        )
        torch.testing.assert_close(
            output[1], torch.full((stage_rows[1], 8), float(expected + gain))
        )
        assert restored_layouts == dispatch_layouts
        assert residual is None
        assert not pending
        assert forward_context.attn_metadata == {"layer": "full"}
        assert forward_context.additional_kwargs == {
            "afd_metadata": parent_metadata,
            CAM_ASYNC_SCHEDULER_KEY: cam_scheduler,
        }
        assert parent_metadata.tokens_start_loc == [0]
        assert parent_metadata.tokens_lens == [parent_tokens]
        assert parent_metadata.num_stages == 1
        assert forward_context.ubatch_idx == 0
        assert forward_context.num_ubatches == 1
        assert forward_context.num_tokens == parent_tokens
        assert forward_context.pad_size == 0
        for layer in moe_layers:
            assert vars(layer.mlp.experts) == {
                "layer_name": f"model.layers.{layer.layer_idx}.mlp.experts"
            }

        current_threads = tuple(cam_scheduler._threads)
        assert len(current_threads) == 2
        if worker_threads is None:
            worker_threads = current_threads
        else:
            assert current_threads == worker_threads
        if call_idx == 0:
            native_calls.clear()
            runner_calls.clear()
            execution = CAMAsyncExecutionContext(0, 0, 1, use_sequence_parallel)
            forward_context.additional_kwargs[CAM_ASYNC_EXECUTION_KEY] = execution
            single_hidden = torch.full((3, 8), 7.0)
            forward_context.num_tokens = 3 * (2 if use_sequence_parallel else 1)
            parent_metadata.tokens_lens = [forward_context.num_tokens]
            with override_forward_context(forward_context):
                single_output, single_residual = (
                    deepseek_v2_async_cam_forward.run_attention_gate_afd_forward(
                        model,
                        single_hidden,
                        None,
                        "full-positions",
                        parent_metadata,
                        "full-scaling",
                    )
                )
                assert get_current_forward_context() is forward_context
            torch.testing.assert_close(
                single_output,
                torch.full_like(
                    single_hidden, gain * (7 + 3 * int(dense_prefix)) + offset
                ),
            )
            assert single_residual is None
            assert (
                native_calls
                == runner_calls
                == [(layer.layer_idx, 0) for layer in moe_layers]
            )
            assert not pending
            assert tuple(cam_scheduler._threads) == worker_threads
            assert (
                forward_context.additional_kwargs.pop(CAM_ASYNC_EXECUTION_KEY)
                is execution
            )


def _cam_model_context(scheduler, num_tokens):
    metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0],
        requests_start_loc=[0],
        stage_idx=0,
        connector=SimpleNamespace(),
        tokens_lens=[num_tokens],
        num_stages=1,
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"parent": True},
        slot_mapping={},
        additional_kwargs={
            "afd_metadata": metadata,
            CAM_ASYNC_SCHEDULER_KEY: scheduler,
        },
    )
    context.flash_comm_v1_enabled = False
    context.dbo_enabled = False
    context.ubatch_idx = 0
    context.num_ubatches = 1
    context.num_tokens = num_tokens
    context.pad_size = 0
    return context, metadata


def test_async_pipeline_without_moe_runs_dense_without_starting_workers(
    monkeypatch, cam_scheduler
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as model_forward,
    )

    context, metadata = _cam_model_context(cam_scheduler, 4)
    plan = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 2), 2),
            AsyncMoeStage(slice(1, 2), slice(2, 4), 2),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=False,
    )
    visited = []

    class Dense:
        is_moe_layer = False

        def __init__(self, layer_id):
            self.layer_id = layer_id

        def __call__(self, positions, hidden, residual, scaling):
            assert get_current_forward_context() is context
            visited.append(self.layer_id)
            return hidden + self.layer_id, residual + 1

    def unexpected(*args, **kwargs):
        pytest.fail("A dense-only model must not split inputs or start CAM execution")

    monkeypatch.setattr(model_forward, "build_async_moe_stage_inputs", unexpected)
    monkeypatch.setattr(model_forward, "CAMAsyncRuntimeContext", unexpected)
    layers = [Dense(2), Dense(3)]
    hidden = torch.arange(8).reshape(4, 2).float()
    residual = torch.ones_like(hidden)
    with override_forward_context(context):
        output, actual_residual = model_forward.run_async_moe_ubatch_afd_forward(
            SimpleNamespace(layers=layers, start_layer=0, end_layer=2),
            hidden,
            residual,
            torch.arange(4),
            metadata,
            plan,
        )
        assert get_current_forward_context() is context
    torch.testing.assert_close(output, hidden + 5)
    torch.testing.assert_close(actual_residual, residual + 2)
    assert visited == [2, 3]
    assert cam_scheduler._threads == []
    assert cam_scheduler.quiescent
    assert metadata.tokens_lens == [4]


@pytest.mark.parametrize("failed_stage", [0, 1])
@pytest.mark.parametrize("failed_phase", ["attention", "combine"])
def test_async_pipeline_error_restores_parent_context_without_replay(
    monkeypatch, cam_scheduler, failed_stage, failed_phase
):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as model_forward,
    )

    context, metadata = _cam_model_context(cam_scheduler, 4)
    plan = AsyncMoeUbatchMetadata(
        attn_metadata=[{"stage": 0}, {"stage": 1}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 2), 2),
            AsyncMoeStage(slice(1, 2), slice(2, 4), 2),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=False,
    )
    monkeypatch.setattr(
        model_forward, "get_tensor_model_parallel_world_size", lambda: 2
    )
    original_error = RuntimeError("stage execution failed")
    operations = []

    def record(phase):
        stage_idx = get_current_forward_context().ubatch_idx
        operations.append((phase, stage_idx))
        if (phase, stage_idx) == (failed_phase, failed_stage):
            raise original_error

    def compute_attention(positions, hidden, residual, scaling):
        record("attention")
        return hidden, residual

    def run_moe(hidden):
        execution = require_cam_async_execution_context()
        execution.checkpoint(7, CAMAsyncPhase.ROUTED)
        execution.checkpoint(7, CAMAsyncPhase.DISPATCHED)
        record("combine")
        return hidden + 1

    layer = SimpleNamespace(
        is_moe_layer=True,
        layer_idx=7,
        compute_attn_output=compute_attention,
        mlp=run_moe,
    )
    with override_forward_context(context):
        with pytest.raises(RuntimeError) as caught:
            model_forward.run_async_moe_ubatch_afd_forward(
                SimpleNamespace(layers=[layer], start_layer=0, end_layer=1),
                torch.ones(4, 2),
                None,
                torch.arange(4),
                metadata,
                plan,
            )
        assert caught.value is original_error
        assert get_current_forward_context() is context
    assert operations[-1] == (failed_phase, failed_stage)
    assert operations.count((failed_phase, failed_stage)) == 1
    assert cam_scheduler.quiescent
    assert context.attn_metadata == {"parent": True}
    assert context.additional_kwargs == {
        "afd_metadata": metadata,
        CAM_ASYNC_SCHEDULER_KEY: cam_scheduler,
    }
    assert metadata.tokens_lens == [4]
    assert metadata.num_stages == 1
    assert context.ubatch_idx == 0
    assert context.num_ubatches == 1
    with (
        pytest.raises(RuntimeError, match="FAILED"),
        cam_scheduler.single_stage(use_sequence_parallel=False),
    ):
        pytest.fail("A failed model execution must not be replayed")


def test_same_request_stages_observe_causal_cpu_kv_writes(monkeypatch, cam_scheduler):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as model_forward,
    )

    context, metadata = _cam_model_context(cam_scheduler, 5)
    stages = plan_async_moe_stages(
        [5],
        split="token",
        use_sequence_parallel=False,
        tensor_parallel_size=2,
    )
    assert stages is not None
    assert [stage.request_slice for stage in stages] == [slice(0, 1), slice(0, 1)]
    plan = AsyncMoeUbatchMetadata(
        attn_metadata=[
            {"request_id": "same-request", "slots": torch.arange(3), "seq_len": 3},
            {"request_id": "same-request", "slots": torch.arange(3, 5), "seq_len": 5},
        ],
        stages=stages,
        parent_input_tokens=5,
        use_sequence_parallel=False,
    )
    monkeypatch.setattr(
        model_forward, "get_tensor_model_parallel_world_size", lambda: 2
    )
    hidden = torch.arange(1, 11).reshape(5, 2).float()
    kv_cache = torch.full_like(hidden, torch.nan)
    writes = []

    def causal_attention(positions, stage_hidden, residual, scaling):
        stage_context = get_current_forward_context()
        stage_metadata = stage_context.attn_metadata
        assert stage_metadata["request_id"] == "same-request"
        assert torch.equal(positions, stage_metadata["slots"])
        prefix_len = int(positions[0])
        assert torch.isfinite(kv_cache[:prefix_len]).all()
        assert torch.isnan(kv_cache[prefix_len:]).all()
        kv_cache[positions] = stage_hidden
        writes.append((stage_context.ubatch_idx, tuple(positions.tolist())))
        # This CPU attention stub reads exactly the prefix visible to each token.
        output = torch.stack([kv_cache[: int(pos) + 1].sum(0) for pos in positions])
        assert stage_metadata["seq_len"] == int(positions[-1]) + 1
        return output, residual

    def run_moe(stage_hidden):
        execution = require_cam_async_execution_context()
        execution.checkpoint(7, CAMAsyncPhase.ROUTED)
        execution.checkpoint(7, CAMAsyncPhase.DISPATCHED)
        return stage_hidden + 5

    layer = SimpleNamespace(
        is_moe_layer=True,
        layer_idx=7,
        compute_attn_output=causal_attention,
        mlp=run_moe,
    )
    with override_forward_context(context):
        output, residual = model_forward.run_async_moe_ubatch_afd_forward(
            SimpleNamespace(layers=[layer], start_layer=0, end_layer=1),
            hidden,
            None,
            torch.arange(5),
            metadata,
            plan,
        )
        assert get_current_forward_context() is context
    assert writes == [(0, (0, 1, 2)), (1, (3, 4))]
    torch.testing.assert_close(kv_cache, hidden)
    torch.testing.assert_close(output, hidden.cumsum(0) + 5)
    assert residual is None


def test_deepseek_afd_ffn_path_reuses_ascend_moe_mlp_after_attention_gate():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "compute_attention_gate_moe_ffn(" in compute_ffn_output
    assert "from afd_plugin.model_executor.models.npu import (" in compute_ffn_output
    assert "deepseek_v2_attention_gate," in compute_ffn_output
    assert "AFDF2ATransferPayload(" in compute_moe
    assert "MoEMlpComputeInput(" in compute_moe
    assert "unified_apply_mlp(" in compute_moe
    assert "routed_output, _ = unified_apply_mlp(" in compute_moe
    assert "quant_type == QuantType.W8A8" in compute_moe
    assert 'experts.get_eplb_parameter("w13_weight")' in compute_moe
    assert 'experts.get_eplb_parameter("w2_weight")' in compute_moe
    assert "experts.w13_weight" not in compute_moe
    assert "experts.w2_weight" not in compute_moe
    assert "w13_weight_scale_fp32" in compute_moe
    assert "w13_weight_scale_fp32_list" in compute_moe
    assert "w2_weight_scale_list" in compute_moe
    compute_moe_function = next(
        node
        for node in ast.parse(gate_source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_attention_gate_moe_ffn"
    )
    quant_params_calls = [
        node
        for node in ast.walk(compute_moe_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MoEQuantParams"
    ]
    assert len(quant_params_calls) == 1
    # Check the contract without pinning formatting or optional quant fields.
    assert any(
        keyword.arg == "quant_type"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "quant_type"
        for keyword in quant_params_calls[0].keywords
    )
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert 'getattr(layer.mlp, "swiglu_limit", None)' in compute_moe
    assert "fusion=False" not in compute_moe
    assert "output_dtype=torch.int32" in gate_source
    assert "npu_dequant_swiglu_quant(" in gate_source
    assert "activation_scale=pertoken_scale" in gate_source


@pytest.mark.parametrize(
    ("num_routed_tokens", "num_shared_tokens"),
    [(2, 2), (2, 0), (0, 2), (0, 0)],
)
@pytest.mark.parametrize(
    ("routed_scale_applied_in_topk", "expected_routed_value"),
    [(False, 2.0), (True, 1.0)],
)
def test_deepseek_afd_ffn_skips_empty_rank_local_moe_work(
    monkeypatch,
    num_routed_tokens,
    num_shared_tokens,
    routed_scale_applied_in_topk,
    expected_routed_value,
):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    class FakeQuantType:
        NONE = "none"
        W8A8 = "w8a8"
        W4A8 = "w4a8"

    class KeywordArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    routed_calls = []

    def fake_unified_apply_mlp(*, mlp_compute_input):
        assert mlp_compute_input.quant.quant_type == FakeQuantType.W8A8
        assert mlp_compute_input.quant.is_per_channel_weight is False
        routed_calls.append(mlp_compute_input.hidden_states)
        return (
            torch.ones_like(
                mlp_compute_input.hidden_states,
                dtype=torch.bfloat16,
            ),
            None,
        )

    fake_moe_mlp: Any = ModuleType("vllm_ascend.ops.fused_moe.moe_mlp")
    fake_moe_mlp.unified_apply_mlp = fake_unified_apply_mlp
    fake_stage_contracts: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
    )
    fake_stage_contracts.MoEMlpComputeInput = KeywordArguments
    fake_stage_contracts.MoEWeights = KeywordArguments
    fake_stage_params: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_params",
    )
    fake_stage_params.MoEQuantParams = KeywordArguments
    fake_quant_type: Any = ModuleType("vllm_ascend.quantization.quant_type")
    fake_quant_type.QuantType = FakeQuantType
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_mlp",
        fake_moe_mlp,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
        fake_stage_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_stage_params",
        fake_stage_params,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.quant_type",
        fake_quant_type,
    )

    shared_calls = []

    def fake_compute_shared(
        shared_experts,
        hidden_states,
        dynamic_scales,
        *,
        swiglu_limit,
        output_dtype,
    ):
        shared_calls.append(
            (shared_experts, hidden_states, dynamic_scales, swiglu_limit),
        )
        return torch.zeros_like(hidden_states, dtype=output_dtype)

    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_compute_w8a8_shared_experts_from_int8",
        fake_compute_shared,
    )
    monkeypatch.setattr(
        deepseek_v2_attention_gate,
        "_gmmswigluquant_fusion_enabled",
        lambda: False,
    )

    shared_experts = object()
    experts = SimpleNamespace(
        quant_type=FakeQuantType.W8A8,
        dynamic_eplb=False,
        get_eplb_parameter=lambda name: name,
        activation="silu",
        _shared_experts=shared_experts,
    )
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=experts,
            routed_scaling_factor=2.0,
        ),
    )
    hidden_states = torch.zeros((num_routed_tokens, 4), dtype=torch.int8)
    expand_x_shared = torch.zeros((num_shared_tokens, 4), dtype=torch.int8)

    output = deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.zeros(2, dtype=torch.int64),
        dynamic_scales=torch.ones(num_routed_tokens),
        expand_x_shared=expand_x_shared,
        dynamic_scales_shared=torch.ones(num_shared_tokens),
        topk_scales=None,
        group_list_type=1,
        routed_scale_applied_in_topk=routed_scale_applied_in_topk,
    )

    assert len(routed_calls) == int(num_routed_tokens > 0)
    assert output.routed_output.shape == hidden_states.shape
    assert output.routed_output.dtype == torch.bfloat16
    if num_routed_tokens > 0:
        assert torch.equal(
            output.routed_output,
            torch.full_like(output.routed_output, expected_routed_value),
        )
    assert len(shared_calls) == int(num_shared_tokens > 0)
    if num_shared_tokens > 0:
        assert output.shared_output is not None
        assert output.shared_output.shape == expand_x_shared.shape
        assert shared_calls[0][3] is None
    else:
        assert output.shared_output is None


def test_deepseek_afd_ffn_compute_omits_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "camp2p_stub_io_enabled()" not in source
    assert "_log_ffn_compute_step(" not in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert "_log_ffn_compute_step(" not in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
