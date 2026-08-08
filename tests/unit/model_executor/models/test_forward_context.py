from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.forward_context import (  # noqa: E402
    get_forward_context as get_current_forward_context,
)

from afd_plugin.model_executor.models import (  # noqa: E402
    get_afd_metadata_from_forward_context,
)
from afd_plugin.model_executor.models.npu.async_cam_layout import (  # noqa: E402
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
    get_async_moe_ubatch_metadata_from_forward_context,
)
from afd_plugin.model_executor.npu.async_cam_ubatching import (  # noqa: E402
    AsyncMoeStage,
)


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
    afd_metadata = object()
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

    schedule_calls = []

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


def test_async_cam_profile_forward_skips_real_connector_io(monkeypatch):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    forward_context = SimpleNamespace(
        in_profile_run=True,
        ubatch_idx=0,
    )
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)

    connector_calls = []
    connector = SimpleNamespace(
        send_attn_output=lambda *args, **kwargs: connector_calls.append("send"),
        recv_ffn_output=lambda *args, **kwargs: connector_calls.append("recv"),
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)

    class _ProfileMoELayer:
        is_moe_layer = True
        layer_idx = 0

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            return (
                hidden_states + 1,
                residual,
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

    model = SimpleNamespace(
        layers=[_ProfileMoELayer(), _ProfileMoELayer()],
        start_layer=0,
        end_layer=2,
    )
    hidden_states = torch.zeros((2, 4))

    output, residual = async_forward.run_attention_gate_afd_forward(
        model,
        hidden_states,
        None,
        torch.arange(2),
        afd_metadata,
    )

    assert torch.equal(output, hidden_states + 2)
    assert residual is None
    assert connector_calls == []


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
    gate_proxy = source.split("class GateOnlyRemoteMoE", 1)[1].split(
        "class AFDDeepseekV2RemoteExpertsMoE",
        1,
    )[0]
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if afd_role == "attention":' in source
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "def _forward_attention(" not in source
    assert "return super().forward(" in model_forward
    assert "deepseek_v2_async_cam_forward.run_model_forward(" in model_forward
    assert "compute_gate_topk(" in gate_proxy
    assert "topk_weights=topk_weights" in gate_proxy
    assert "topk_ids=topk_ids" in gate_proxy
    assert "router_logits=router_logits" in gate_proxy
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert "pending_ffn_recv" in attention_gate_forward
    assert "topk_weights" in attention_gate_forward
    assert "topk_ids" in attention_gate_forward


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "compute_attention_gate_topk(" in compute_attn_output
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "from afd_plugin.model_executor.models.npu import (" in compute_attn_output
    assert "deepseek_v2_attention_gate," in compute_attn_output
    assert "force_balanced_topk_ids_enabled" in gate_source
    assert "def _force_balanced_topk_ids(" in gate_source
    assert "topk_ids.copy_(balanced_topk_ids)" in gate_source
    assert "topk_weights, topk_ids = afd_connector.select_experts(" in (gate_source)
    assert "if force_balanced_topk_ids_enabled():" in gate_source
    assert (
        gate_source.index(
            "topk_weights, topk_ids = afd_connector.select_experts(",
        )
        < gate_source.index("if force_balanced_topk_ids_enabled():")
        < gate_source.index("topk_weights = topk_weights.to(torch.float32)")
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
    assert "self.mlp = GateOnlyRemoteMoE(" in source
    assert 'prefix=f"{prefix}.mlp"' in source
    assert (
        "# NPU-only: Attention-side gate/topk is implemented in the NPU helper."
        in source
    )
    assert (
        "# NPU-only: gated MoE FFN compute consumes Attention-side topk payloads."
        in source
    )


def test_async_moe_pipeline_preserves_stage_order(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward

    events = []
    forward_context = SimpleNamespace(
        attn_metadata={"layer": "full"},
        additional_kwargs={},
        ubatch_idx=0,
        num_ubatches=1,
        num_tokens=4,
        pad_size=0,
        flash_comm_v1_enabled=True,
    )

    def send_attn_output(_hidden_states, context, **_kwargs):
        events.append(("send", context.metadata.stage_idx))

    def recv_ffn_output(ref_tensor, ubatch_idx):
        events.append(("recv", ubatch_idx))
        return ref_tensor

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    parent_metadata = SimpleNamespace(
        stage_idx=0,
        connector=connector,
    )
    forward_context.additional_kwargs["afd_metadata"] = parent_metadata
    execution_plan = AsyncMoeUbatchMetadata(
        attn_metadata=[{"layer": "stage-0"}, {"layer": "stage-1"}],
        stages=[
            AsyncMoeStage(
                slice(0, 1),
                slice(0, 2),
                input_tokens=2,
            ),
            AsyncMoeStage(
                slice(1, 2),
                slice(2, 4),
                input_tokens=4,
            ),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=True,
    )
    stage_hidden_states = [torch.zeros((1, 8)), torch.ones((2, 8))]

    def compute_attn_output(
        _positions,
        hidden_states,
        residual,
        _llama_4_scaling,
    ):
        stage_context = get_current_forward_context()
        events.append(
            (
                "compute",
                stage_context.ubatch_idx,
                stage_context.attn_metadata,
                stage_context.num_tokens,
                stage_context.pad_size,
            ),
        )
        topk = hidden_states[:, :1]
        return hidden_states, residual, topk, topk.to(torch.int32), None

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "build_async_moe_stage_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            hidden_states=stage_hidden_states,
            residuals=[None, None],
            positions=["positions-0", "positions-1"],
            llama_4_scaling=["scaling-0", "scaling-1"],
        ),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_async_moe_stage_outputs",
        lambda outputs, _metadata: tuple(outputs),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "prepare_cam_dispatch_payload",
        lambda hidden_states, topk_weights, topk_ids, router_logits, **_kwargs: (
            SimpleNamespace(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                layout=object(),
            )
        ),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_cam_dispatch_output",
        lambda output, _layout: output,
    )

    output, residual = deepseek_v2_async_cam_forward.run_async_moe_ubatch_afd_forward(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=2,
            layers=[
                SimpleNamespace(
                    is_moe_layer=True,
                    layer_idx=layer_idx,
                    compute_attn_output=compute_attn_output,
                )
                for layer_idx in range(2)
            ],
        ),
        hidden_states=torch.zeros((4, 8)),
        residual=None,
        positions="full-positions",
        afd_metadata=parent_metadata,
        async_moe_ubatch_metadata=execution_plan,
        llama_4_scaling="full-scaling",
    )

    assert [event[:2] for event in events] == [
        ("compute", 0),
        ("send", 0),
        ("compute", 1),
        ("recv", 0),
        ("send", 1),
        ("compute", 0),
        ("recv", 1),
        ("send", 0),
        ("compute", 1),
        ("recv", 0),
        ("send", 1),
        ("recv", 1),
    ]
    for event in (event for event in events if event[0] == "compute"):
        stage_idx = event[1]
        assert event[2] == {"layer": f"stage-{stage_idx}"}
        assert event[3] == 2
        assert event[4] == (0, 2)[stage_idx]
    assert all(
        restored is expected
        for restored, expected in zip(output, stage_hidden_states, strict=True)
    )
    assert residual is None
    assert forward_context.attn_metadata == {"layer": "full"}
    assert forward_context.num_tokens == 4
    assert forward_context.pad_size == 0


def test_async_cam_layout_transposes_full_shards_into_stage_shards(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_cam_layout

    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 10), input_tokens=10),
            AsyncMoeStage(slice(0, 1), slice(10, 15), input_tokens=6),
        ],
        parent_input_tokens=16,
        use_sequence_parallel=True,
    )
    global_hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    global_residual = global_hidden + 100
    positions = torch.arange(16)
    scaling = torch.ones(2, 16)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)

    for tp_rank, expected_positions in (
        (0, [[0, 1, 2, 3, 4], [10, 11, 12]]),
        (1, [[5, 6, 7, 8, 9], [13, 14, 0]]),
    ):
        tp_group.rank_in_group = tp_rank
        local_slice = slice(tp_rank * 8, (tp_rank + 1) * 8)

        def all_gather(tensor, token_dim):
            assert token_dim == 0
            assert tensor.shape[1] == 4
            return torch.cat((global_hidden, global_residual), dim=-1)

        monkeypatch.setattr(
            async_cam_layout,
            "tensor_model_parallel_all_gather",
            all_gather,
        )
        stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
            global_hidden[local_slice],
            global_residual[local_slice],
            positions,
            scaling,
            metadata,
        )

        assert [stage.tolist() for stage in stage_inputs.positions] == (
            expected_positions
        )
        assert [int(stage.shape[0]) for stage in stage_inputs.hidden_states] == [
            5,
            3,
        ]
        assert [tuple(stage.shape) for stage in stage_inputs.llama_4_scaling] == [
            (2, 5),
            (2, 3),
        ]
        monkeypatch.setattr(
            async_cam_layout,
            "tensor_model_parallel_all_gather",
            lambda tensor, token_dim: (
                global_hidden[:10]
                if int(tensor.shape[token_dim]) == 5
                else torch.cat(
                    (
                        global_hidden[10:15],
                        global_hidden.new_zeros((1, 2)),
                    ),
                    dim=0,
                )
            ),
        )
        restored = async_cam_layout.restore_async_moe_stage_outputs(
            stage_inputs.hidden_states,
            metadata,
        )
        expected_restored = global_hidden.clone()
        expected_restored[15].zero_()
        assert torch.equal(restored, expected_restored[local_slice])


def test_async_moe_replicated_layout_removes_and_restores_parent_padding():
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_cam_layout

    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 3), input_tokens=3),
            AsyncMoeStage(slice(0, 1), slice(3, 5), input_tokens=2),
        ],
        parent_input_tokens=8,
        use_sequence_parallel=False,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    positions = torch.arange(8)

    stage_inputs = async_cam_layout.build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
    )

    assert [stage[:, 0].tolist() for stage in stage_inputs.hidden_states] == [
        [0.0, 2.0, 4.0],
        [6.0, 8.0],
    ]
    assert [stage.tolist() for stage in stage_inputs.positions] == [
        [0, 1, 2],
        [3, 4],
    ]
    restored = async_cam_layout.restore_async_moe_stage_outputs(
        stage_inputs.hidden_states,
        metadata,
    )
    assert torch.equal(restored[:5], hidden_states[:5])
    assert torch.count_nonzero(restored[5:]) == 0


def test_plain_tp_cam_boundary_shards_and_restores_replicated_tokens(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_cam_layout

    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_cam_layout, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    router_logits = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    padded_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100

    monkeypatch.setattr(
        async_cam_layout,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_output,
    )

    expected_hidden_rows = (
        hidden_states[:3],
        torch.cat((hidden_states[3:], hidden_states.new_zeros((1, 2)))),
    )
    for tp_rank in range(2):
        tp_group.rank_in_group = tp_rank
        payload = async_cam_layout.prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=False,
        )

        assert torch.equal(payload.hidden_states, expected_hidden_rows[tp_rank])
        assert payload.hidden_states.shape[0] == 3
        assert payload.topk_weights.shape[0] == 3
        assert payload.topk_ids.shape[0] == 3
        assert payload.router_logits is not None
        assert payload.router_logits.shape[0] == 3
        assert payload.layout.parent_tokens == 5
        assert payload.layout.padded_tokens == 6
        assert payload.layout.requires_tp_all_gather is True

        local_output = padded_output[tp_rank * 3 : (tp_rank + 1) * 3]
        restored = async_cam_layout.restore_cam_dispatch_output(
            local_output,
            payload.layout,
        )
        assert torch.equal(restored, padded_output[:5])


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
    assert "MoEQuantParams(quant_type=quant_type)" in compute_moe
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert "fusion=False" not in compute_moe


@pytest.mark.parametrize(
    ("num_routed_tokens", "num_shared_tokens"),
    [(2, 2), (2, 0), (0, 2), (0, 0)],
)
def test_deepseek_afd_ffn_skips_empty_rank_local_moe_work(
    monkeypatch,
    num_routed_tokens,
    num_shared_tokens,
):
    from afd_plugin.model_executor.models.npu import deepseek_v2_attention_gate

    class FakeQuantType:
        NONE = "none"
        W8A8 = "w8a8"

    class KeywordArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    routed_calls = []

    def fake_unified_apply_mlp(*, mlp_compute_input):
        routed_calls.append(mlp_compute_input.hidden_states)
        return (
            torch.zeros_like(
                mlp_compute_input.hidden_states,
                dtype=torch.bfloat16,
            ),
            None,
        )

    fake_moe_mlp = ModuleType("vllm_ascend.ops.fused_moe.moe_mlp")
    fake_moe_mlp.unified_apply_mlp = fake_unified_apply_mlp
    fake_stage_contracts = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_contracts",
    )
    fake_stage_contracts.MoEMlpComputeInput = KeywordArguments
    fake_stage_contracts.MoEWeights = KeywordArguments
    fake_stage_params = ModuleType(
        "vllm_ascend.ops.fused_moe.moe_stage_params",
    )
    fake_stage_params.MoEQuantParams = KeywordArguments
    fake_quant_type = ModuleType("vllm_ascend.quantization.quant_type")
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
        output_dtype,
    ):
        shared_calls.append((shared_experts, hidden_states, dynamic_scales))
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
            routed_scaling_factor=1.0,
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
    )

    assert len(routed_calls) == int(num_routed_tokens > 0)
    assert output.routed_output.shape == hidden_states.shape
    assert output.routed_output.dtype == torch.bfloat16
    assert len(shared_calls) == int(num_shared_tokens > 0)
    if num_shared_tokens > 0:
        assert output.shared_output is not None
        assert output.shared_output.shape == expand_x_shared.shape
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
