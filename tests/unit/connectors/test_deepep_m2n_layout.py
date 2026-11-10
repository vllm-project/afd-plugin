# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DeepEP M2N transport planner (``plan_transport_layout``)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu.deepep_m2n.layout import (  # noqa: E402
    RankPlacement,
    TransportLayout,
    plan_transport_layout,
)


def _pods(placement: str) -> tuple[list[RankPlacement], int]:
    """Ranks of ``<a>A<f>F,...`` with DP ranks rising in pod order.

    Each pod is one NVLink domain and lists its Attention ranks first, the way
    a pod that starts Attention on its first GPUs reports them.
    """
    ranks: list[RankPlacement] = []
    next_rank = {"attention": 0, "ffn": 0}
    for pod, entry in enumerate(placement.split(",")):
        attention, ffn = (int(n) for n in entry.removesuffix("F").split("A"))
        for role, count in (("attention", attention), ("ffn", ffn)):
            for _ in range(count):
                ranks.append(RankPlacement(role, next_rank[role], f"host:pod{pod}"))
                next_rank[role] += 1
    return ranks, next_rank["ffn"]


def _plan(placement: str) -> TransportLayout:
    ranks, ffn_size = _pods(placement)
    return plan_transport_layout(ranks, ffn_size)


def _order(layout: TransportLayout) -> str:
    return " ".join(f"{role[0].upper()}{rank}" for role, rank in layout.order)


@pytest.mark.parametrize(
    ("placement", "order", "ffn_ranks_per_domain", "is_identity"),
    [
        ("4A2F", "F0 F1 A0 A1 A2 A3", 0, True),
        ("2A0F,0A2F", "F0 F1 A0 A1", 0, True),
        ("3A2F,5A0F", "F0 F1 A0 A1 A2 A3 A4 A5 A6 A7", 0, True),
        ("0A1F,0A1F,1A0F,1A0F", "F0 F1 A0 A1", 0, True),
        ("1A1F,1A1F", "F0 A0 F1 A1", 1, False),
        ("4A1F,4A1F", "F0 A0 A1 A2 A3 F1 A4 A5 A6 A7", 1, False),
        ("2A2F,2A2F,2A2F", "F0 F1 A0 A1 F2 F3 A2 A3 F4 F5 A4 A5", 2, False),
    ],
)
def test_transport_orders_domains_ffn_first_and_picks_the_afdep_mode(
    placement, order, ffn_ranks_per_domain, is_identity
):
    """Each pod becomes one block with its FFN ranks first. FFN ranks that
    fill the first pods keep the role world (FFN prefix); the same FFN count
    in every pod gives afdEP-comm's per-domain mode.

    Why: DeepEP numbers ranks domain by domain, and afdEP-comm finds the FFN
    ranks either at [0, F) or at the first f ranks of every domain; any other
    order routes rows to the wrong ranks. Layouts that run today must keep the
    role world so they take the unchanged path.
    """
    layout = _plan(placement)

    assert _order(layout) == order
    assert layout.ffn_ranks_per_domain == ffn_ranks_per_domain
    assert layout.is_identity is is_identity


def test_per_domain_ffn_ranks_sit_where_afdep_expects_them():
    """In 3 x 6A2F, FFN rank ``j`` is transport rank ``(j // 2) * 8 + j % 2``.

    Why: an FFN rank loads the experts of its DP (EP) rank, and afdEP-comm
    routes those experts to this transport rank; a mismatch silently computes
    rows with the wrong experts.
    """
    layout = _plan("6A2F,6A2F,6A2F")

    assert (layout.num_domains, layout.domain_size) == (3, 8)
    assert [layout.transport_rank("ffn", j) for j in range(6)] == [
        (j // 2) * 8 + j % 2 for j in range(6)
    ]


def test_role_split_pods_in_any_order_keep_the_ffn_prefix():
    """Attention pods listed before the FFN pod still give FFN-first order.

    Why: domains follow their lowest role-world rank, so the pod order does
    not change the transport the role-split layouts run today.
    """
    layout = _plan("4A0F,0A4F")

    assert _order(layout) == "F0 F1 F2 F3 A0 A1 A2 A3"
    assert layout.is_identity


@pytest.mark.parametrize(
    ("ranks", "ffn_size", "message"),
    [
        # 4A1F,3A2F: the FFN count varies from pod to pod.
        (_pods("4A1F,3A2F")[0], 3, "same number of FFN ranks"),
        # Pods of 1, 1, 2 and 2 GPUs.
        (_pods("0A1F,0A1F,2A0F,2A0F")[0], 2, "same number of ranks"),
        (_pods("0A2F,4A0F")[0], 2, "same number of ranks"),
        # 2A2F,2A2F with FFN DP ranks 0, 2 on pod 0 and 1, 3 on pod 1.
        (
            [
                RankPlacement(role, rank, f"host:pod{pod}")
                for pod, roles in enumerate(
                    [
                        [("ffn", 0), ("ffn", 2), ("attention", 0), ("attention", 1)],
                        [("ffn", 1), ("ffn", 3), ("attention", 2), ("attention", 3)],
                    ]
                )
                for role, rank in roles
            ],
            4,
            "FFN DP ranks must rise in pod order",
        ),
    ],
)
def test_transport_rejects_placements_afdep_cannot_carry(ranks, ffn_size, message):
    """Unequal pods, an FFN count that varies per pod, and FFN DP ranks out
    of pod order raise, naming the placement.

    Why: DeepEP needs equal NVLink domains and afdEP-comm has no mode for the
    rest; failing in the handshake beats a hang in the first dispatch.
    """
    with pytest.raises(RuntimeError, match=message):
        plan_transport_layout(ranks, ffn_size)
