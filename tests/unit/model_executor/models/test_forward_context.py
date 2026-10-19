# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import (  # noqa: E402
    get_afd_metadata_from_forward_context,
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


def test_async_cam_profile_forward_runs_matched_connector_io(monkeypatch):
    from afd_plugin.model_executor.models.npu import (
        deepseek_v2_async_cam_forward as async_forward,
    )

    forward_context = SimpleNamespace(
        in_profile_run=True,
        ubatch_idx=0,
        flash_comm_v1_enabled=True,
    )
    monkeypatch.setattr(async_forward, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        async_forward,
        "maybe_apply_dbo_yield",
        lambda hidden_states, **_kwargs: hidden_states,
    )

    connector_calls: list[str] = []
    dispatch_layouts: list[object] = []
    restored_layouts: list[object] = []

    def prepare_dispatch_payload(
        hidden_states,
        topk_weights,
        topk_ids,
        router_logits,
        *,
        use_sequence_parallel,
    ):
        assert use_sequence_parallel is False
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
        restored_layouts.append(layout)
        return local_output

    monkeypatch.setattr(
        async_forward,
        "prepare_cam_dispatch_payload",
        prepare_dispatch_payload,
    )
    monkeypatch.setattr(
        async_forward,
        "restore_cam_dispatch_output",
        restore_dispatch_output,
    )

    def send_attn_output(*args, **kwargs):
        connector_calls.append("send")

    def recv_ffn_output(ref_tensor, ubatch_idx, context):
        connector_calls.append("recv")
        return ref_tensor

    connector = SimpleNamespace(
        send_attn_output=send_attn_output,
        recv_ffn_output=recv_ffn_output,
    )
    afd_metadata = SimpleNamespace(connector=connector, stage_idx=0)

    class _ProfileMoELayer:
        is_moe_layer = True
        use_sequence_parallel_moe = False

        layer_idx = 0
        mlp = SimpleNamespace(shared_experts=lambda x: 2 * x)

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
            *,
            already_sequence_parallel=False,
        ):
            return (
                hidden_states + 1,
                residual,
                torch.ones((hidden_states.shape[0], 1)),
                torch.zeros((hidden_states.shape[0], 1), dtype=torch.int32),
                torch.ones((hidden_states.shape[0], 1)),
            )

    model = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=True),
        ),
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

    assert torch.equal(output, (hidden_states + 1) * 9 + 3)
    assert residual is None
    assert connector_calls == ["send", "recv", "send", "recv"]
    assert restored_layouts == dispatch_layouts


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

    routed_calls: list[Tensor] = []

    def fake_apply_moe_mlp(*, mlp_compute_input, quant_method):
        assert mlp_compute_input.quant.quant_type == FakeQuantType.W8A8
        assert mlp_compute_input.quant.is_per_channel_weight is False
        assert quant_method == "scheme"
        assert mlp_compute_input.layer is routed_experts
        routed_calls.append(mlp_compute_input.hidden_states)
        return (
            torch.ones_like(
                mlp_compute_input.hidden_states,
                dtype=torch.bfloat16,
            ),
            None,
        )

    fake_moe_mlp: Any = ModuleType("vllm_ascend.ops.fused_moe.moe_mlp")
    fake_moe_mlp.apply_moe_mlp = fake_apply_moe_mlp
    fake_mlp_contracts: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.dataclass.moe_mlp",
    )
    fake_mlp_contracts.MoEMlpComputeInput = KeywordArguments
    fake_experts_contracts: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts",
    )
    fake_experts_contracts.MoEWeights = KeywordArguments
    fake_quant_contracts: Any = ModuleType(
        "vllm_ascend.ops.fused_moe.dataclass.moe_quant",
    )
    fake_quant_contracts.MoEQuantParams = KeywordArguments
    fake_quant_type: Any = ModuleType("vllm_ascend.quantization.quant_type")
    fake_quant_type.QuantType = FakeQuantType
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_mlp",
        fake_moe_mlp,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.dataclass.moe_mlp",
        fake_mlp_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts",
        fake_experts_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.dataclass.moe_quant",
        fake_quant_contracts,
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
    routed_experts = SimpleNamespace(
        dynamic_eplb=False,
        get_eplb_parameter=lambda name: name,
        activation="silu",
        quant_method=SimpleNamespace(quant_method="scheme"),
    )
    experts = SimpleNamespace(
        quant_type=FakeQuantType.W8A8,
        moe_config=SimpleNamespace(
            swiglu_limit=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
        ),
        shared_experts=SimpleNamespace(_layer=shared_experts),
        routed_experts=routed_experts,
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
