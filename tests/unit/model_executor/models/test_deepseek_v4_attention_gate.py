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
    """The Hash ids sent to FFN must be sliced like the router logits.

    The alignment logic now lives in ``local_hash_input_ids`` so that the
    send-side selection and this local routing path cannot drift apart. These
    assertions are structural: the helper's numerics are covered by
    ``tests/unit/model_executor/test_deepseek_v4_hash_ids.py``.
    """

    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v4_attention_gate.py",
    ).read_text()

    assert "def local_hash_input_ids(" in source
    assert "DSV4 Hash routing cannot align the ids sent to FFN" in source
    assert "ids = input_ids.reshape(-1).to(torch.int64)" in source
    assert "flash_comm_v1_enabled" in source
    assert "ids.numel() != router_tokens" in source
    assert "split_tensor_along_first_dim(" in source
    assert "num_partitions=group.world_size" in source
    assert ")[group.rank_in_group]" in source
    # The send-side selection must reuse this helper rather than re-deriving a
    # slice; its numerics live in test_deepseek_v4_hash_ids.py.
    assert "def hash_input_ids_from_context(" in source
    assert "return local_hash_input_ids(" in source


def test_dsv4_ffn_does_not_reapply_gate_routed_scale() -> None:
    source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v4.py",
    ).read_text()

    assert "routed_scale_applied_in_topk=True" in source
