# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare AFD Attention-side CUDA routing with native and mathematical oracles."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.model_executor.layers.fused_moe.router import fused_topk_router  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.model_layers import GateLinear  # noqa: E402

pytestmark = pytest.mark.gpu

ROUTING_HIDDEN_SIZE = 16
ROUTING_NUM_EXPERTS = 8
ROUTING_TOP_K = 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused top-k needs CUDA")
@pytest.mark.parametrize("renormalize", [True, False])
def test_routing_proxy_matches_native_router(monkeypatch, renormalize):
    """On GPU the proxy's routes equal vLLM's native fused top-k router.

    Why: the FFN applies the weights the Attention side computes; any drift
    from native routing changes the model's output.
    """
    torch.manual_seed(0)
    events: list[tuple] = []
    install_proxy_context(adapter, monkeypatch, events, stage_idx=0)
    gate = (
        GateLinear(ROUTING_HIDDEN_SIZE, ROUTING_NUM_EXPERTS, out_dtype=torch.float32)
        .cuda()
        .to(torch.bfloat16)
    )
    proxy = adapter.RemoteFFNProxy(layer_idx=3)
    proxy.gate = gate
    proxy.top_k = ROUTING_TOP_K
    proxy.renormalize = renormalize
    hidden_states = torch.randn(
        33, ROUTING_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )

    proxy(hidden_states)

    sent = events[0][3]
    router_logits, _ = gate(hidden_states)
    native_weights, native_ids = fused_topk_router.FusedTopKRouter(
        top_k=ROUTING_TOP_K,
        global_num_experts=ROUTING_NUM_EXPERTS,
        renormalize=renormalize,
    )._compute_routing(hidden_states, router_logits, indices_type=None)
    assert torch.equal(sent["topk_ids"], native_ids)
    assert torch.equal(sent["topk_weights"], native_weights)

    reference = torch.softmax(router_logits.float(), dim=-1)
    reference_weights, reference_ids = reference.topk(ROUTING_TOP_K, dim=-1)
    if renormalize:
        reference_weights = reference_weights / reference_weights.sum(-1, keepdim=True)
    assert torch.equal(
        sent["topk_ids"].long().sort(-1).values, reference_ids.sort(-1).values
    )
    torch.testing.assert_close(
        sent["topk_weights"].sort(-1).values,
        reference_weights.sort(-1).values,
        rtol=1e-3,
        atol=1e-3,
    )
