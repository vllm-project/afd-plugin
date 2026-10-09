# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native MoE entry and complete CAM runner contracts with CPU tensors."""

from __future__ import annotations

import inspect
import math
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (  # noqa: E402
    MoERunner,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE  # noqa: E402

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor import remote_moe  # noqa: E402
from afd_plugin.model_executor.models.npu import async_cam_layout  # noqa: E402
from afd_plugin.model_executor.npu import remote_moe as cam_moe  # noqa: E402
from afd_plugin.model_executor.npu.async_cam_execution import (  # noqa: E402
    CAM_ASYNC_EXECUTION_KEY,
    CAMAsyncExecutionContext,
    CAMAsyncRuntimeContext,
    CAMAsyncUbatchScheduler,
)

pytestmark = pytest.mark.vllm_runtime
ROUTED_SCALE = 2.5
SHARED_DIVISOR = 3.0
TEST_WAIT_SECONDS = 2.0


def _unexpected(*args, **kwargs):
    pytest.fail("CAM runner reached synchronous transport or generic MoE processing")


def _context(connector, execution):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=None,
        slot_mapping={},
        additional_kwargs={
            "afd_metadata": AFDForwardContextMetadata(
                tokens_start_loc=[0, 3],
                requests_start_loc=[0, 1],
                stage_idx=execution.stage_idx,
                connector=connector,
                tokens_lens=[3, 2],
                num_stages=execution.num_stages,
            ),
            CAM_ASYNC_EXECUTION_KEY: execution,
        },
    )


def _make_moe(monkeypatch, events, *, dtype, shared=True):
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.additional_config.update(
        afd={
            "role": "attention",
            "connector": "CAMAsyncAFDConnector",
            "compute_gate_on_attention": True,
        },
    )

    class Gate(nn.Module):
        def forward(self, hidden):
            events.append("gate")
            return hidden[:, :4].float(), None

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.7, dtype=dtype))

        def forward(self, hidden):
            events.append("shared")
            return hidden * self.weight

    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    nn.Module.__init__(moe)
    moe.gate = Gate()
    moe.shared_experts = Shared() if shared else None
    moe.is_sequence_parallel = False
    monkeypatch.setattr(
        remote_moe, "current_platform", SimpleNamespace(device_type="npu")
    )
    monkeypatch.setattr(cam_moe, "validate_remote_moe_config", lambda: None)
    with set_current_vllm_config(config):
        moe.experts = remote_moe.build_attention_moe_runner(
            config,
            gate=moe.gate,
            attention_shared_experts=moe.shared_experts,
            shared_output_divisor_fp16=SHARED_DIVISOR,
            num_shared_experts=2,
            num_experts=4,
            top_k=2,
            hidden_size=7,
            intermediate_size=11,
            params_dtype=dtype,
            prefix="model.layers.3.mlp.experts",
            routed_scaling_factor=ROUTED_SCALE,
        )
    moe.register_forward_pre_hook(lambda *_: events.append("native"))
    for name in (
        "_forward_entry",
        "_forward_impl",
        "_maybe_apply_shared_experts",
        "_maybe_apply_routed_scale_to_output",
        "_maybe_reduce_final_output",
    ):
        monkeypatch.setattr(moe.experts, name, _unexpected)
    monkeypatch.setattr(remote_moe, "remote_ffn_forward", _unexpected)
    monkeypatch.setattr(moe.experts.router, "select_experts", _unexpected)
    return moe


def _selector(monkeypatch, events):
    def select_experts(**kwargs):
        events.append("select")
        hidden = kwargs["hidden_states"]
        assert kwargs["num_shared_experts"] == 2
        assert kwargs["num_logical_experts"] == 4
        weights = torch.tensor([0.2, 0.3], dtype=torch.float32)
        ids = torch.tensor([1, 3], dtype=torch.int32)
        assert kwargs["mix_placement"] is False
        assert kwargs["num_experts"] == 4
        assert kwargs["routed_scaling_factor"] == 1.0
        return weights.repeat(len(hidden), 1), ids.repeat(len(hidden), 1)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        SimpleNamespace(select_experts=select_experts),
    )
    monkeypatch.setattr(cam_moe, "force_balanced_topk_ids_enabled", lambda: False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize(
    "tp_size,tp_rank,use_sp",
    [(1, 0, False), (2, 0, False), (2, 1, False), (2, 1, True)],
)
def test_complete_forward_preserves_numeric_and_layout_contract(
    monkeypatch, dtype, shared, tp_size, tp_rank, use_sp
):
    events: list[str] = []
    moe = _make_moe(monkeypatch, events, dtype=dtype, shared=shared)
    _selector(monkeypatch, events)
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank),
    )
    hidden = torch.arange(1, 36, dtype=dtype).reshape(5, 7) / 7
    sent = []
    coefficient = 0.2 * 2 + 0.3 * 4
    # Characterize the existing AFD dtype contract. FP16 without shared skips
    # routed scaling here, unlike native MoERunner; this is not native parity.
    # V2 FFN currently leaves routed_scale_applied_in_topk at its default False.
    ffn_scale = ROUTED_SCALE if dtype != torch.float16 else 1.0
    routed_output = (hidden.float() * (coefficient * ffn_scale)).to(dtype)

    def send(payload, transfer, **kwargs):
        events.append("dispatch")
        assert transfer.metadata.stage_idx == 1
        assert transfer.metadata.layer_idx == 3
        assert transfer.metadata.seq_lens == [payload.shape[0]]
        assert kwargs["topk_weights"].dtype == torch.float32
        sent.append((payload, kwargs))

    def receive(*, ref_tensor, ubatch_idx):
        events.append("combine")
        assert ubatch_idx == 1
        assert ref_tensor is sent[0][0]
        weights = sent[0][1]["topk_weights"]
        ids = sent[0][1]["topk_ids"]
        weighted = (weights * (ids + 1)).sum(dim=-1, keepdim=True)
        return (ref_tensor.float() * weighted * ffn_scale).to(dtype)

    def gather(local, token_dim):
        events.append("gather")
        assert token_dim == 0
        padded = torch.cat((routed_output, torch.zeros_like(hidden[:1])))
        torch.testing.assert_close(local, padded[tp_rank * 3 : (tp_rank + 1) * 3])
        return padded

    monkeypatch.setattr(async_cam_layout, "tensor_model_parallel_all_gather", gather)
    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    execution = CAMAsyncExecutionContext(4, 1, 2, use_sp)
    monkeypatch.setattr(
        CAMAsyncExecutionContext,
        "checkpoint",
        lambda self, layer, phase: events.append(phase.value),
    )
    state_before = dict(moe.experts.__dict__)
    with override_forward_context(_context(connector, execution)):
        actual = moe(hidden)
    expected = routed_output
    if shared:
        shared_output = hidden * moe.shared_experts.weight
        if dtype == torch.float16:
            shared_output = shared_output / SHARED_DIVISOR
        expected = expected + shared_output
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == hidden.dtype
    assert actual.shape == hidden.shape
    assert events == (
        ["native", "gate", "select", "routed", "dispatch"]
        + (["shared"] if shared else [])
        + ["dispatched", "combine"]
        + (["gather"] if tp_size > 1 and not use_sp else [])
    )
    assert moe.experts.__dict__ == state_before
    assert moe.experts.shared_experts is None
    assert not any("shared" in name for name, _ in moe.experts.named_parameters())
    assert ("shared_experts.weight" in moe.state_dict()) is shared
    assert inspect.signature(cam_moe.AFDCAMAsyncMoERunner.forward) == inspect.signature(
        MoERunner.forward
    )


def test_two_stages_share_one_runner_without_retaining_outputs(monkeypatch):
    events: list[str] = []
    moe = _make_moe(monkeypatch, events, dtype=torch.float32)
    _selector(monkeypatch, events)
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    pending = {}
    transfers = []

    def send(hidden, transfer, **kwargs):
        stage = transfer.metadata.stage_idx
        assert stage not in pending
        pending[stage] = hidden.clone() * 2.0
        transfers.append(("dispatch", stage))

    def receive(*, ref_tensor, ubatch_idx):
        transfers.append(("combine", ubatch_idx))
        return pending.pop(ubatch_idx)

    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    contexts = [
        _context(connector, CAMAsyncExecutionContext(0, i, 2, False)) for i in range(2)
    ]
    runtime = CAMAsyncRuntimeContext(contexts, torch.device("cpu"))
    hidden = [torch.full((rows, 7), value) for rows, value in [(3, 1.2), (2, 3.4)]]
    state_before = dict(moe.experts.__dict__)

    def stage(execution):
        assert get_forward_context() is contexts[execution.stage_idx]
        output = moe(hidden[execution.stage_idx])
        execution.layer_done(3)
        return output

    scheduler = CAMAsyncUbatchScheduler(wait_timeout=TEST_WAIT_SECONDS)
    try:
        with override_forward_context(contexts[0]):
            outputs = scheduler.run(
                stage,
                layer_ids=[3],
                use_sequence_parallel=False,
                activate=runtime.activate,
                thread_context=runtime.thread_context,
            )
    finally:
        scheduler.shutdown()
    assert transfers == [
        ("dispatch", 0),
        ("combine", 0),
        ("dispatch", 1),
        ("combine", 1),
    ]
    for actual, original in zip(outputs, hidden, strict=True):
        torch.testing.assert_close(
            actual, original * 2.0 + original * moe.shared_experts.weight
        )
    assert events.count("native") == events.count("gate") == events.count("select") == 2
    assert not pending
    assert moe.experts.__dict__ == state_before


def test_forward_requires_execution_context_and_rejects_input_ids(monkeypatch):
    events: list[str] = []
    moe = _make_moe(monkeypatch, events, dtype=torch.float32)
    hidden = torch.ones(2, 7)
    context = _context(None, CAMAsyncExecutionContext(0, 0, 1, False))
    context.additional_kwargs.pop(CAM_ASYNC_EXECUTION_KEY)
    with override_forward_context(context):
        with pytest.raises(RuntimeError, match="explicit execution context"):
            moe.experts(hidden, hidden)
        with pytest.raises(NotImplementedError, match="input_ids"):
            moe.experts(hidden, hidden, input_ids=torch.tensor([0, 1]))
    assert events == []


@pytest.mark.ascend_runtime
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_selector_and_ffn_match_independent_reference(
    monkeypatch, fused, shared, dtype
):
    """Check non-FP16 math through real selector/FFN Python paths."""
    selector = pytest.importorskip("vllm_ascend.ops.fused_moe.experts_selector")
    from vllm_ascend.ops.fused_moe import moe_mlp
    from vllm_ascend.quantization.quant_type import QuantType

    from afd_plugin.model_executor.models import deepseek_v2

    events: list[str] = []
    moe = _make_moe(monkeypatch, events, dtype=dtype, shared=shared)
    monkeypatch.setattr(selector, "check_npu_moe_gating_top_k", lambda **_: fused)
    monkeypatch.setattr(cam_moe, "force_balanced_topk_ids_enabled", lambda: False)
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        deepseek_v2.native, "current_platform", SimpleNamespace(device_type="npu")
    )
    kernel_calls = []

    def gating_kernel(logits, *, k, renorm, routed_scaling_factor, **kwargs):
        kernel_calls.append("gate")
        assert routed_scaling_factor == 1.0
        weights, ids = logits.softmax(dim=-1).topk(k)
        if renorm:
            weights /= weights.sum(dim=-1, keepdim=True)
        return weights * routed_scaling_factor, ids.to(torch.int32), None

    monkeypatch.setattr(selector.DeviceOperator, "moe_gating_top_k", gating_kernel)
    hidden = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5]], dtype=dtype)
    experts = SimpleNamespace(
        quant_type=QuantType.NONE,
        moe_config=SimpleNamespace(has_bias=False),
        get_eplb_parameter=lambda _: torch.empty(0),
        _shared_experts=None,
        activation="silu",
        dynamic_eplb=False,
    )
    ffn_layer = SimpleNamespace(
        compute_gate_on_attention=True,
        is_moe_layer=True,
        mlp=SimpleNamespace(experts=experts, routed_scaling_factor=ROUTED_SCALE),
    )
    pending = []

    def send(payload, transfer, **kwargs):
        weights, ids = kwargs["topk_weights"], kwargs["topk_ids"]
        assert weights.shape == ids.shape == (1, 2)
        assert ids.tolist() == [[3, 2]]
        pending.append((weights, ids))

    def mlp_kernel(*, mlp_compute_input):
        kernel_calls.append("mlp")
        # Expert i is the independent linear function E_i(x) = (i + 1) * x.
        ids = pending[0][1]
        return mlp_compute_input.hidden_states * (ids + 1).reshape(-1, 1), None

    monkeypatch.setattr(moe_mlp, "unified_apply_mlp", mlp_kernel)

    def receive(*, ref_tensor, ubatch_idx):
        weights, _ = pending[0]
        payload = deepseek_v2.AFDDeepseekV2DecoderLayer.compute_ffn_output(
            ffn_layer,
            ref_tensor.repeat_interleave(2, dim=0),
            group_list=torch.tensor([0, 0, 1, 2]),
        )
        assert payload.shared_output is None
        return (
            (payload.routed_output * weights.reshape(-1, 1))
            .sum(dim=0, keepdim=True)
            .to(dtype)
        )

    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    execution = CAMAsyncExecutionContext(0, 0, 1, False)
    with override_forward_context(_context(connector, execution)):
        actual = moe(hidden)

    # Logits [0, 1, 2, 3] select experts 3 and 2 with renormalized weights.
    # Derive the answer without reading selector output or its scale arguments.
    high_weight = math.exp(3) / (math.exp(3) + math.exp(2))
    expected = hidden.float() * (4 * high_weight + 3 * (1 - high_weight)) * ROUTED_SCALE
    if shared:
        expected += hidden.float() * float(moe.shared_experts.weight.detach())
    torch.testing.assert_close(actual, expected.to(dtype))
    assert kernel_calls == (["gate"] if fused else []) + ["mlp"]
    assert events == ["native", "gate"] + (["shared"] if shared else [])


@pytest.mark.parametrize("failure", ["gate", "select", "dispatch", "shared", "combine"])
def test_runner_propagates_failure_without_replay(monkeypatch, failure):
    events: list[str] = []
    moe = _make_moe(monkeypatch, events, dtype=torch.float32)
    _selector(monkeypatch, events)
    monkeypatch.setattr(
        async_cam_layout,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )

    def fail(*args, **kwargs):
        events.append(failure)
        raise RuntimeError(f"{failure} failed")

    def send(*args, **kwargs):
        events.append("dispatch")

    def receive(*, ref_tensor, **kwargs):
        events.append("combine")
        return ref_tensor

    connector = SimpleNamespace(send_attn_output=send, recv_ffn_output=receive)
    target, method = {
        "gate": (moe.gate, "forward"),
        "select": (
            sys.modules["vllm_ascend.ops.fused_moe.experts_selector"],
            "select_experts",
        ),
        "dispatch": (connector, "send_attn_output"),
        "shared": (moe.shared_experts, "forward"),
        "combine": (connector, "recv_ffn_output"),
    }[failure]
    monkeypatch.setattr(target, method, fail)
    execution = CAMAsyncExecutionContext(0, 0, 1, False)
    with (
        override_forward_context(_context(connector, execution)),
        pytest.raises(RuntimeError, match=f"{failure} failed"),
    ):
        moe(torch.ones(2, 7))
    order = ["native", "gate", "select", "dispatch", "shared", "combine"]
    assert events == order[: order.index(failure) + 1]
