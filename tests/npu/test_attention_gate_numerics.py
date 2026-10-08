# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in gate/router/activation checks using real Ascend device operators.

Run with AFD_RUN_ASCEND_OP_RUNTIME=1 inside an NPU reservation. Only the
single-rank TP metadata is stubbed; weights, loading and all arithmetic use
native implementations. These component checks do not exercise CAM transport.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.npu


@pytest.fixture
def npu_runtime():
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in Ascend runtime")
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    assert enable_custom_op(), "Ascend custom operators are required"
    return torch


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_gate_loading_and_fp32_logits(npu_runtime, monkeypatch, dtype_name, tmp_path):
    torch = npu_runtime
    from transformers import DeepseekV2Config
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear

    from afd_plugin.model_executor.models import deepseek_v2

    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(deepseek_v2.native, "GateLinear", AscendGateLinear)
    dtype = getattr(torch, dtype_name)
    config = DeepseekV2Config(
        hidden_size=64,
        n_routed_experts=8,
        n_shared_experts=0,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
    )
    runtime_config = VllmConfig()
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with set_current_vllm_config(runtime_config), torch.device("npu"):
            moe = deepseek_v2.GateOnlyRemoteMoE(
                config=config,
                layer_idx=1,
                prefix="model.layers.1.mlp",
                vllm_config=runtime_config,
            )
        assert moe.gate.weight.dtype == torch.float32
        assert moe.gate.prefix == "model.layers.1.mlp.gate"
        assert set(dict(moe.named_parameters())) == {
            "gate.weight",
            "gate.e_score_correction_bias",
        }
        generator = torch.Generator().manual_seed(417)
        # FP32 checkpoint values are deliberately not rounded to the model dtype.
        checkpoint = torch.randn(8, 64, generator=generator, dtype=torch.float32)
        moe.gate.weight.weight_loader(moe.gate.weight, checkpoint.npu())
        moe.gate.quant_method.process_weights_after_loading(moe.gate)
        torch.testing.assert_close(moe.gate.weight.cpu(), checkpoint, rtol=0, atol=0)
        hidden = torch.randn(7, 64, generator=generator, dtype=torch.float32).to(dtype)
        with torch.inference_mode():
            actual, _ = moe.gate(hidden.npu())
        expected = torch.nn.functional.linear(hidden.float(), checkpoint)
        assert actual.dtype == torch.float32
        torch.save(
            {
                "input": hidden,
                "checkpoint": checkpoint,
                "actual": actual.cpu(),
                "expected": expected,
            },
            tmp_path / "gate.pt",
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize(
    "scoring_func,renormalize",
    [
        ("softmax", False),
        ("softmax", True),
        ("sigmoid", False),
        ("sigmoid", True),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_cam_router_matches_native_factory(
    npu_runtime, scoring_func, renormalize, with_bias, tmp_path
):
    torch = npu_runtime
    from vllm_ascend.ops.fused_moe.router.router_factory import (
        create_ascend_fused_moe_router,
    )

    from afd_plugin.connectors.npu.async_cam import select_cam_experts

    # The competing groups and experts are close without being exact ties.
    logits = torch.tensor(
        [
            [1.002, 0.998, -3.0, -4.0, 1.001, 0.997, -2.0, -5.0],
            [0.2, 0.199, -0.1, -0.2, 0.201, 0.198, -0.11, -0.21],
        ],
        dtype=torch.float32,
        device="npu",
    )
    bias = (
        torch.tensor([0.0, 0.002, 0.0, 0.0, 0.001, 0.003, 0.0, 0.0], device="npu")
        if with_bias
        else None
    )
    hidden = torch.zeros(2, 64, dtype=torch.bfloat16, device="npu")
    options = dict(
        top_k=2,
        use_grouped_topk=True,
        renormalize=renormalize,
        scoring_func=scoring_func,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )
    native_router = create_ascend_fused_moe_router(global_num_experts=8, **options)
    expected_weights, expected_ids = native_router.select_experts(
        hidden, logits, topk_indices_dtype=torch.int32
    )
    weights, ids = select_cam_experts(
        hidden_states=hidden,
        router_logits=logits,
        mix_placement=False,
        num_logical_experts=8,
        num_shared_experts=0,
        num_experts=8,
        **options,
    )
    torch.save(
        {
            "logits": logits.cpu(),
            "bias": None if bias is None else bias.cpu(),
            "weights": weights.cpu(),
            "ids": ids.cpu(),
            "expected_weights": expected_weights.cpu(),
            "expected_ids": expected_ids.cpu(),
        },
        tmp_path / "router.pt",
    )
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    assert torch.all(ids[:, 0] // 4 == ids[:, 1] // 4)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("clamp_limit", [0.0, 1.0])
def test_native_activation_and_expert_output(
    npu_runtime, dtype_name, clamp_limit, tmp_path
):
    torch = npu_runtime
    import torch_npu
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_ascend.ops.fused_moe.moe_mlp import _unified_apply_activation

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(418)
    hidden = torch.randn(4, 64, generator=generator).to(dtype)
    w13 = (torch.randn(2, 64, 128, generator=generator) * 0.3).to(dtype)
    w2 = (torch.randn(2, 64, 64, generator=generator) * 0.1).to(dtype)
    groups = torch.tensor([2, 4], dtype=torch.int64, device="npu")
    projected = torch_npu.npu_grouped_matmul(
        x=[hidden.npu()],
        weight=[w13.npu()],
        group_list=groups,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]
    contract = SimpleNamespace(activation=MoEActivation.SILU, swiglu_limit=clamp_limit)
    activated = _unified_apply_activation(contract, projected, None)
    actual = torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=[w2.npu()],
        group_list=groups,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]
    reference = []
    for expert in range(2):
        gate_up = (
            hidden[expert * 2 : expert * 2 + 2].float() @ w13[expert].float()
        ).to(dtype)
        gate, up = gate_up.float().chunk(2, dim=-1)
        if clamp_limit:
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(min=-clamp_limit, max=clamp_limit)
        act = (torch.nn.functional.silu(gate) * up).to(dtype)
        reference.append((act.float() @ w2[expert].float()).to(dtype))
    expected = torch.cat(reference)
    torch.save(
        {
            "input": hidden,
            "w13": w13,
            "w2": w2,
            "clamp_limit": clamp_limit,
            "actual": actual.cpu(),
            "expected": expected,
        },
        tmp_path / "expert.pt",
    )
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
