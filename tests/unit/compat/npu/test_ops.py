# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Meta contracts for the A2E/E2A Ascend operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

if TYPE_CHECKING:
    import torch


def _assert_tensor_spec(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_type: str,
) -> None:
    assert tuple(tensor.shape) == shape
    assert tensor.dtype is dtype
    assert tensor.device.type == device_type


def test_a2e_e2a_meta_contracts() -> None:
    torch = pytest.importorskip("torch")
    try:
        ensure_afd_ascend_ops_loaded()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    batch_size = 4
    hidden_size = 16
    topk = 1
    expert_rank_size = 1
    attention_rank_size = 1

    expert_x = torch.empty(
        (batch_size, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    expert_a2e = torch.ops.afd_ascend.a2e(
        expert_x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        0,
        "meta_group",
        1,
        0,
    )
    assert len(expert_a2e) == 5
    _assert_tensor_spec(
        expert_a2e[0],
        (batch_size, hidden_size),
        torch.bfloat16,
        "meta",
    )
    _assert_tensor_spec(expert_a2e[1], (batch_size, topk), torch.int32, "meta")
    _assert_tensor_spec(expert_a2e[2], (batch_size, topk), torch.float32, "meta")
    _assert_tensor_spec(expert_a2e[3], (1,), torch.int32, "meta")
    _assert_tensor_spec(expert_a2e[4], (batch_size,), torch.bool, "meta")

    attention_x = torch.empty(
        (batch_size, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    attention_a2e = torch.ops.afd_ascend.a2e(
        attention_x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        1,
        "meta_group",
        1,
        0,
    )
    _assert_tensor_spec(attention_a2e[0], (1, 1), torch.bfloat16, "meta")
    _assert_tensor_spec(attention_a2e[1], (1, 1), torch.int32, "meta")
    _assert_tensor_spec(attention_a2e[2], (1, 1), torch.float32, "meta")
    _assert_tensor_spec(attention_a2e[3], (1,), torch.int32, "meta")
    _assert_tensor_spec(attention_a2e[4], (1,), torch.bool, "meta")

    expert_e2a = torch.ops.afd_ascend.e2a(
        expert_a2e[0],
        expert_a2e[3],
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        0,
        "meta_group",
        1,
    )
    _assert_tensor_spec(expert_e2a, (1, 1), torch.bfloat16, "meta")

    attention_e2a = torch.ops.afd_ascend.e2a(
        attention_x,
        attention_a2e[3],
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        1,
        "meta_group",
        1,
    )
    _assert_tensor_spec(
        attention_e2a,
        (batch_size, hidden_size),
        torch.bfloat16,
        "meta",
    )
