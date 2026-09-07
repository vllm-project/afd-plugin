# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from pathlib import Path


def test_dsv4_async_gate_bypasses_native_moe_communicator() -> None:
    """Async CAM must not invoke Ascend's EP/SP selector wrapper.

    That wrapper calls ``forward_context.moe_comm_method`` during DSV4 Hash
    routing.  AFD owns this communication boundary, so the Attention helper
    must invoke the CANN routing operators directly on its local tokens.
    """

    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v4_attention_gate.py",
    ).read_text()

    assert "torch.ops._C_ascend.moe_gating_top_k_hash(" in source
    assert "DeviceOperator.moe_gating_top_k(" in source
    assert "correction_bias = correction_bias.to(router_logits.dtype)" in source
    assert "from vllm_ascend.ops.fused_moe.experts_selector" not in source
    assert "forward_context.moe_comm_method.pad_and_split_input_ids(" not in source
    assert "connector.select_experts" not in source


def test_dsv4_async_gate_validates_local_hash_token_alignment() -> None:
    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v4_attention_gate.py",
    ).read_text()

    assert "DSV4 Hash routing input_ids/token count mismatch on Attention" in source
    assert "input_ids = input_ids.reshape(-1).to(torch.int64)" in source
    assert "forward_context.flash_comm_v1_enabled" in source
    assert "and input_ids.numel() != router_logits.shape[0]" in source
    assert "split_tensor_along_first_dim(" in source
    assert "num_partitions=tp_group.world_size" in source
    assert ")[tp_group.rank_in_group]" in source


def test_dsv4_ffn_does_not_reapply_gate_routed_scale() -> None:
    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v4.py",
    ).read_text()

    assert "routed_scale_applied_in_topk=True" in source
