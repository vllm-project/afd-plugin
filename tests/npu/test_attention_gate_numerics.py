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


@pytest.fixture(params=["float16", "bfloat16"])
def gate_case(npu_runtime, monkeypatch, request):
    torch = npu_runtime
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm_ascend import ascend_config
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear

    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(rank_in_group=0, world_size=1),
    )
    dtype = getattr(torch, request.param)
    hidden_size = 64
    num_experts = 8
    runtime_config = VllmConfig()
    generator = torch.Generator().manual_seed(417)
    # Keep unrounded FP32 values so each storage policy is observable.
    checkpoint = torch.randn(
        num_experts,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    hidden = torch.randn(7, hidden_size, generator=generator, dtype=torch.float32).to(
        dtype
    )
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with set_current_vllm_config(runtime_config):
            ascend_config.init_ascend_config(runtime_config)
            yield SimpleNamespace(
                torch=torch,
                dtype=dtype,
                native_gate=AscendGateLinear,
                gate_options={
                    "input_size": hidden_size,
                    "output_size": num_experts,
                    "out_dtype": None,
                    "prefix": "model.layers.1.mlp.gate",
                },
                checkpoint=checkpoint,
                hidden=hidden,
            )
    finally:
        try:
            ascend_config.clear_ascend_config()
        finally:
            torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("fp32_policy", ["params_dtype", "force_fp32_compute"])
def test_native_gate_explicit_fp32_preserves_checkpoint(
    gate_case, fp32_policy, tmp_path
):
    case = gate_case
    torch = case.torch
    options = case.gate_options | {
        fp32_policy: torch.float32 if fp32_policy == "params_dtype" else True
    }
    with torch.device("npu"):
        gate = case.native_gate(**options)
    assert gate.weight.dtype == torch.float32
    gate.weight.weight_loader(gate.weight, case.checkpoint.npu())
    gate.quant_method.process_weights_after_loading(gate)
    torch.testing.assert_close(gate.weight.cpu(), case.checkpoint, rtol=0, atol=0)
    with torch.inference_mode():
        actual, _ = gate(case.hidden.npu())
    expected = torch.nn.functional.linear(case.hidden.float(), case.checkpoint)
    assert actual.dtype == torch.float32
    torch.save(
        {
            "input": case.hidden,
            "checkpoint": case.checkpoint,
            "actual": actual.cpu(),
            "expected": expected,
        },
        tmp_path / "gate.pt",
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)


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
