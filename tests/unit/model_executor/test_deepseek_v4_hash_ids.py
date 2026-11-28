# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage for CAM token alignment, gate precision and profile routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.ffn.npu.remote import (  # noqa: E402
    _cam_gate_topk,
    local_hash_input_ids,
)


def test_returns_ids_aligned_with_router_tokens():
    ids = torch.tensor([5, 6, 7], dtype=torch.int32)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=3,
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [5, 6, 7]


def test_flattens_multi_dimensional_ids():
    ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)

    result = local_hash_input_ids(
        input_ids=ids,
        router_tokens=4,
    )

    assert result.tolist() == [5, 6, 7, 8]


def test_rejects_missing_ids():
    with pytest.raises(RuntimeError, match="requires input_ids to send"):
        local_hash_input_ids(
            input_ids=None,
            router_tokens=3,
        )


def test_rejects_unalignable_token_count():
    """A count mismatch must fail here, before any cross-role transfer."""
    ids = torch.tensor([5, 6], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="cannot align the ids sent to FFN"):
        local_hash_input_ids(
            input_ids=ids,
            router_tokens=3,
        )


@pytest.mark.usefixtures("registered_routing_ids_op")
@pytest.mark.parametrize(
    ("scoring_func", "expected_norm"),
    [("sqrtsoftplus", 2), ("softmax", 0), ("sigmoid", 1)],
)
@pytest.mark.parametrize("in_profile_run", [False, True])
@pytest.mark.parametrize("precomputed_fp32", [False, True])
def test_dsv4_router_uses_model_gate(
    monkeypatch, scoring_func, expected_norm, in_profile_run, precomputed_fp32
):  # long-test: exercise the CANN selector and profile-only routing in one forward
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(_EXTRA_CTX=SimpleNamespace(in_profile_run=in_profile_run)),
    )
    monkeypatch.setenv("AFD_FORCE_BALANCED_TOPK_IDS", "1")
    hidden = torch.tensor([[1.0, 0.25]], dtype=torch.bfloat16)
    fp32_input = torch.tensor([[1.0, 1.001953125]], dtype=torch.float32)
    gate_input = fp32_input if precomputed_fp32 else hidden
    bias = torch.tensor([0.125, -0.125], dtype=torch.bfloat16)
    gate = SimpleNamespace(
        weight_fp32=torch.eye(2), tid2eid=None, e_score_correction_bias=bias
    )
    experts = SimpleNamespace(
        scoring_func=scoring_func,
        top_k=1,
        topk_group=1,
        num_expert_group=2,
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    def select(logits, *, bias, routed_scaling_factor, **options):
        torch.testing.assert_close(logits, gate_input.float())
        torch.testing.assert_close(bias, gate.e_score_correction_bias.float())
        assert (options["norm_type"], options["renorm"]) == (
            expected_norm,
            int(scoring_func != "sqrtsoftplus"),
        )
        assert (options["k"], options["k_group"], options["group_count"]) == (1, 1, 2)
        ids = logits.argmax(dim=-1, keepdim=True)
        return (
            torch.full_like(ids, routed_scaling_factor, dtype=torch.bfloat16),
            ids,
            None,
        )

    if scoring_func == "sqrtsoftplus":
        monkeypatch.setattr(
            torch.ops._C_ascend,
            "moe_gating_top_k_hash",
            lambda *, x, **kwargs: select(x, **kwargs),
            raising=False,
        )
    else:
        monkeypatch.setitem(
            sys.modules, "torch_npu", SimpleNamespace(npu_moe_gating_top_k=select)
        )

    weights, ids = _cam_gate_topk(experts, gate, gate_input, None)

    assert ids.tolist() == [[int(precomputed_fp32 and not in_profile_run)]]
    torch.testing.assert_close(weights, torch.tensor([[2.5]], dtype=torch.float32))
