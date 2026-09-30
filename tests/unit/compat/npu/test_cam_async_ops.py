# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real dispatcher/Meta tests for the Ascend 910C routed-only CAM extension.

Build the extension with SOC_VERSION=910c, then run this file with
SOC_VERSION=910c pytest tests/unit/compat/npu/test_cam_async_ops.py.
No HCCL group or device kernel is launched. These tests do not qualify NPU
communication, numerical correctness, synchronization, or the legacy connector.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import pytest

from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def cam_runtime() -> ModuleType:
    soc = os.environ.get("SOC_VERSION", "").lower()
    if soc != "910c" and not soc.startswith("ascend910_93"):
        pytest.skip("Set SOC_VERSION=910c to test a compiled Ascend 910C CAM extension")
    for module in ("torch", "torch_npu", "afd_plugin._C_ascend"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for compiled CAM Meta tests")
    # Optional runtime imports stay inside the fixture, so CPU test collection
    # remains independent of torch/torch_npu and their process-wide state.
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    ensure_afd_ascend_ops_loaded()
    return torch


def _dispatch_recv(torch: ModuleType, *, dynamic_quant: int = 0, moe_rank_id: int = 2):
    return torch.ops.afd_ascend.afd_async_dispatch_recv(
        torch.empty(1, device="meta", dtype=torch.bfloat16),
        torch.empty(1, device="meta", dtype=torch.float16),
        comm_id=0,
        max_seq_len=64,
        hidden_size=64,
        top_k=2,
        moe_rank_num=2,
        attn_rank_num=2,
        route_expert_num_per_moe=3,
        moe_rank_id=moe_rank_id,
        world_size=4,
        tp_size=2,
        dynamic_quant=dynamic_quant,
        group_name="meta_only",
    )


@pytest.mark.parametrize(
    "name",
    (
        "afd_async_dispatch_send",
        "afd_async_dispatch_recv",
        "afd_async_combine_send",
        "afd_async_combine_recv",
    ),
)
def test_cam_dispatcher_registers_inference_and_meta_kernels(cam_runtime, name: str):
    for dispatch_key in ("PrivateUse1", "AutogradPrivateUse1", "Meta"):
        assert cam_runtime._C._dispatch_has_kernel_for_dispatch_key(
            f"afd_ascend::{name}", dispatch_key
        )
    schema = getattr(cam_runtime.ops.afd_ascend, name).default._schema
    assert all("shared" not in argument.name for argument in schema.arguments)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dynamic_quant", [0, 1])
def test_cam_four_phase_meta_contracts(
    cam_runtime, monkeypatch, dtype_name, dynamic_quant
):
    torch = cam_runtime
    monkeypatch.delenv("BATCH_SIZE_FACTOR", raising=False)
    dtype = getattr(torch, dtype_name)
    x = torch.empty((4, 64), device="meta", dtype=dtype)
    anchor = torch.empty(1, device="meta", dtype=dtype)
    expert_ids = torch.empty((4, 2), device="meta", dtype=torch.int32)
    expert_scales = torch.empty((4, 2), device="meta", dtype=torch.float32)
    comm_args = torch.empty(1, device="meta", dtype=torch.float16)

    sent = torch.ops.afd_ascend.afd_async_dispatch_send(
        x,
        expert_ids,
        comm_args,
        0,
        64,
        4,
        64,
        2,
        2,
        2,
        3,
        0,
        4,
        0,
        2,
        dynamic_quant,
        "meta_only",
    )
    received = torch.ops.afd_ascend.afd_async_dispatch_recv(
        anchor,
        comm_args,
        0,
        64,
        64,
        2,
        2,
        2,
        3,
        2,
        4,
        2,
        dynamic_quant,
        "meta_only",
    )
    assert len(received) == 4
    expanded, scales, batch_info, counts = received
    assert tuple(expanded.shape) == (262144, 64)
    assert expanded.dtype == (torch.int8 if dynamic_quant else dtype)
    assert tuple(scales.shape) == ((262144,) if dynamic_quant else (1,))
    assert scales.dtype == torch.float32
    # compact_v2: [5 header fields][2 TP totals][3 x 2 expert/TP counts].
    assert tuple(batch_info.shape) == (13,)
    assert batch_info.dtype == torch.int64
    assert tuple(counts.shape) == (3,)
    assert counts.dtype == torch.int64

    combined = torch.ops.afd_ascend.afd_async_combine_send(
        expanded.to(dtype),
        comm_args,
        batch_info,
        0,
        64,
        64,
        2,
        2,
        2,
        3,
        2,
        4,
        2,
        "meta_only",
    )
    output = torch.ops.afd_ascend.afd_async_combine_recv(
        anchor,
        expert_ids,
        expert_scales,
        comm_args,
        0,
        4,
        64,
        2,
        2,
        2,
        3,
        0,
        4,
        "meta_only",
    )
    for completion in (sent, combined):
        assert tuple(completion.shape) == (1,)
        assert completion.dtype == torch.int8
    assert tuple(output.shape) == (4, 64)
    assert output.dtype == dtype
    assert all(t.device.type == "meta" for t in (sent, *received, combined, output))


@pytest.mark.parametrize(
    ("factor", "expected_capacity"),
    [("0.5", 131072), ("0.000003814697265625", 1), ("invalid", 262144)],
)
def test_cam_receive_capacity_environment_contract(
    cam_runtime, monkeypatch, factor: str, expected_capacity: int
):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", factor)
    expanded, scales, _, _ = _dispatch_recv(cam_runtime, dynamic_quant=1)

    assert expanded.shape[0] == expected_capacity
    assert scales.shape[0] == expected_capacity


@pytest.mark.parametrize("factor", ["0", "-0.5", "1.5", "nan", "1e-20"])
def test_cam_receive_rejects_invalid_or_empty_capacity(
    cam_runtime, monkeypatch, factor
):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", factor)
    with pytest.raises(RuntimeError, match="BATCH_SIZE_FACTOR"):
        _dispatch_recv(cam_runtime)


def test_cam_receive_rejects_local_moe_rank_id(cam_runtime):
    with pytest.raises(RuntimeError, match="global rank"):
        _dispatch_recv(cam_runtime, moe_rank_id=0)


def test_cam_combine_rejects_noncompact_batch_info_shape(cam_runtime):
    torch = cam_runtime
    with pytest.raises(RuntimeError, match="batch_info must be int64"):
        torch.ops.afd_ascend.afd_async_combine_send(
            torch.empty((8, 64), device="meta", dtype=torch.bfloat16),
            torch.empty(1, device="meta", dtype=torch.float16),
            torch.empty(7, device="meta", dtype=torch.int64),
            0,
            64,
            64,
            2,
            2,
            2,
            3,
            2,
            4,
            2,
            "meta_only",
        )


def test_cam_send_rejects_autograd_inputs(cam_runtime):
    torch = cam_runtime
    with pytest.raises(RuntimeError, match="inference only"):
        torch.ops.afd_ascend.afd_async_dispatch_send(
            torch.empty(
                (4, 64), device="meta", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.empty((4, 2), device="meta", dtype=torch.int32),
            torch.empty(1, device="meta", dtype=torch.float16),
            0,
            64,
            4,
            64,
            2,
            2,
            2,
            3,
            0,
            4,
            0,
            2,
            0,
            "meta_only",
        )
