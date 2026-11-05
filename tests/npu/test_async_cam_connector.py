# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real Ascend fused-routing behavior for the async CAM connector."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

import torch  # noqa: E402
from vllm_ascend.utils import enable_custom_op  # noqa: E402

from afd_plugin.connectors.npu import async_cam as async_cam_module  # noqa: E402


@pytest.mark.npu
@pytest.mark.parametrize(
    (
        "scoring_func",
        "renormalize",
        "with_bias",
        "num_expert_group",
        "topk_group",
        "expected_ids",
        "expected_scores",
    ),
    [
        ("sigmoid", True, True, 2, 1, [4, 6], [0.60, 0.30]),
        ("sigmoid", True, False, 2, 1, [4, 5], [0.60, 0.59]),
        ("softmax", True, False, 2, 1, [4, 5], [0.60, 0.59]),
        ("softmax", False, False, 2, 1, [4, 5], [0.60, 0.59]),
        ("softmax", True, False, 4, 2, [0, 4], [0.90, 0.60]),
    ],
)
def test_async_select_experts_fused_group_scores(  # long-test: routing oracle
    scoring_func,
    renormalize,
    with_bias,
    num_expert_group,
    topk_group,
    expected_ids,
    expected_scores,
):
    if not torch.npu.is_available():
        pytest.skip("Ascend device required for real fused routing")
    # Match worker initialization: choose the device before loading CANN ops.
    torch.npu.set_device(0)
    assert enable_custom_op()
    scores = torch.tensor([[0.90, 0.08, 0.04, 0.02, 0.60, 0.59, 0.30, 0.01]])
    logits = torch.logit(scores) if scoring_func == "sigmoid" else scores.log()
    bias = (
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0]) if with_bias else None
    )
    weights, ids = async_cam_module.select_cam_experts(
        hidden_states=torch.zeros(1, 16, device="npu"),
        router_logits=logits.npu(),
        top_k=2,
        use_grouped_topk=True,
        renormalize=renormalize,
        scoring_func=scoring_func,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias.npu() if bias is not None else None,
        mix_placement=False,
        num_logical_experts=8,
        num_shared_experts=1,
        num_experts=9,
    )
    # Group selection uses top-2 sums. Bias selects IDs without changing their
    # raw weights. Literal selected scores keep this oracle independent of the
    # router implementation, including when two of four groups are selected.
    expected = torch.tensor([expected_scores])
    if renormalize:
        expected /= expected.sum(dim=-1, keepdim=True)
    else:
        expected /= scores.sum(dim=-1, keepdim=True)
    ids, weights = ids.cpu(), weights.cpu()
    order = ids.argsort(dim=-1)
    assert ids.dtype == torch.int32
    assert torch.equal(
        ids.gather(1, order), torch.tensor([expected_ids], dtype=torch.int32)
    )
    torch.testing.assert_close(weights.gather(1, order), expected * 2.0)
