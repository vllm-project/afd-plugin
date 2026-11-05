# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import Tensor  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.router_factory import (  # noqa: E402
    create_fused_moe_router,
)

from afd_plugin.model_executor import routing_simulator as routing  # noqa: E402

pytestmark = pytest.mark.vllm_runtime


def test_routing_cycles_are_deterministic_and_balanced(monkeypatch):
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    monkeypatch.setattr(routing, "get_ep_group", lambda: group)
    hidden, logits = torch.empty((2, 8)), torch.empty((2, 16))
    first = routing.AFDBalancedRoutingStrategy(topn_per_rank=0).route_tokens(
        hidden, logits, 8
    )[1]
    second = routing.AFDBalancedRoutingStrategy(topn_per_rank=0).route_tokens(
        hidden, logits, 8
    )[1]
    assert torch.equal(first, second)
    assert sorted(first.flatten().tolist()) == list(range(16))
    target_ranks = []
    for rank in range(4):
        group.rank_in_group = rank
        _, ids = routing.AFDBalancedRoutingStrategy(topn_per_rank=4).route_tokens(
            torch.empty((1, 8)), torch.empty((1, 32)), 8
        )
        target_ranks.append(ids.flatten().div(8, rounding_mode="floor"))
    assert torch.bincount(torch.cat(target_ranks), minlength=4).tolist() == [8] * 4


@pytest.mark.parametrize("experts,ep_size,topn", [(15, 4, 0), (16, 4, 5), (16, 1, 4)])
def test_invalid_routing_configs_are_rejected(monkeypatch, experts, ep_size, topn):
    monkeypatch.setattr(
        routing,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=ep_size, rank_in_group=0),
    )
    with pytest.raises(ValueError):
        routing.AFDBalancedRoutingStrategy(topn_per_rank=topn).route_tokens(
            torch.empty((1, 8)), torch.empty((1, experts)), 8
        )


def test_strategy_returns_normalized_weights_and_reuses_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        routing,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    strategy = routing.AFDBalancedRoutingStrategy(topn_per_rank=2)

    weights, expert_ids = strategy.route_tokens(
        torch.empty((2, 8)),
        torch.empty((2, 8)),
        top_k=4,
    )
    reused_weights, reused_ids = strategy.route_tokens(
        torch.empty((1, 8)),
        torch.empty((1, 8)),
        top_k=4,
    )

    assert weights.tolist() == [[0.25] * 4] * 2
    assert expert_ids.tolist() == [[0, 1, 4, 5]] * 2
    assert expert_ids.dtype == torch.int32
    assert weights.data_ptr() == reused_weights.data_ptr()
    assert expert_ids.data_ptr() == reused_ids.data_ptr()


def test_strategy_registers_with_vllm_and_preserves_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
        routing.AFD_BALANCED_ROUTING_STRATEGY,
    )
    monkeypatch.setenv("AFD_BENCHMARK_FORCE_LB_TOPN_PER_RANK", "2")
    monkeypatch.setattr(
        routing,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(routing.RoutingSimulator, "_routing_strategies", {})
    routing.register_afd_balanced_routing_strategy()

    router = create_fused_moe_router(top_k=4, global_num_experts=8)
    captured_ids: list[Tensor] = []
    router.set_capture_fn(captured_ids.append)
    _, expert_ids = router.select_experts(
        torch.empty((1, 8)),
        torch.empty((1, 8)),
    )

    assert expert_ids.tolist() == [[0, 1, 4, 5]]
    assert len(captured_ids) == 1
    assert torch.equal(captured_ids[0], expert_ids)
