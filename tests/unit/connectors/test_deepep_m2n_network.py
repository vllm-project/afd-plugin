# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for DeepEP M2N topology discovery and network planning."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.gpu.deepep_m2n.network import (  # noqa: E402
    NetworkConfig,
    RankTopology,
    apply_gin_environment,
    plan_network,
)


def _rank(rank: int, host: str, domain: str | None = None) -> RankTopology:
    return RankTopology(
        world_rank=rank, host=host, nvlink_domain=domain or f"host:{host}"
    )


def _two_hosts() -> list[RankTopology]:
    return [_rank(r, f"n{r // 2}") for r in range(4)]


def test_network_config_survives_a_round_trip():
    config = NetworkConfig.from_mapping(
        {"gin": "on", "hybrid": "off", "rdma_traffic_class": 5}
    )

    assert NetworkConfig.from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"gin": "maybe"}, "network.gin"),
        ({"hybrid": "maybe"}, "network.hybrid"),
        ({"rdma_traffic_class": -1}, "rdma_traffic_class"),
        # NICs are NCCL's own choice; the connector plans none.
        ({"nic_policy": "shared"}, "unknown network options"),
        ({"nccl_env": {"NCCL_IB_HCA": "mlx5_0"}}, "unknown network options"),
        ({"bogus": 1}, "unknown network options"),
    ],
)
def test_network_config_rejects_invalid_values(raw, message):
    """Bad ``gin``/``hybrid``/traffic class and unknown keys are rejected.

    Why: bad or removed network settings must fail at parse time.
    """
    with pytest.raises(ValueError, match=message):
        NetworkConfig.from_mapping(raw)


def test_single_host_needs_no_rdma():
    """Four ranks on one host: one domain, GIN off.

    Why: single-node runs must never touch RDMA.
    """
    plan = plan_network([_rank(r, "n0") for r in range(4)], NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (1, 4, False)


def test_ranks_on_several_hosts_use_gin():
    """Two hosts of two ranks: two domains, GIN on.

    Why: GIN (RDMA) is DeepEP's only path between NVLink domains.
    """
    plan = plan_network(_two_hosts(), NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (2, 2, True)


def test_nvl72_is_one_domain_even_with_uneven_hosts():
    # One NVLink fabric spans the hosts, so per-host counts need not match.
    """Four ranks on two trays (3+1) in one NVLink fabric: one domain of 4,
    GIN off.

    Why: NVL72 is one domain; per-host layout rules must not apply.
    """
    ranks = [_rank(r, f"tray{0 if r < 3 else 1}", domain="nvl:abc:1") for r in range(4)]

    plan = plan_network(ranks, NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (1, 4, False)


def test_gin_off_across_nvlink_domains_is_an_error():
    # NVLink load/store cannot reach another domain; GIN is the only path.
    """``gin: off`` with ranks on two hosts raises.

    Why: without GIN DeepEP cannot reach another NVLink domain.
    """
    with pytest.raises(RuntimeError, match="span 2 NVLink domains"):
        plan_network(_two_hosts(), NetworkConfig(gin="off"))


@pytest.mark.parametrize(("gin", "use_gin"), [("off", False), ("on", True)])
def test_forced_gin_on_one_host_is_honored(gin, use_gin):
    """``gin: off``/``on`` with all ranks on one host plans as forced.

    Why: the multi-domain check must not block single-node use.
    """
    plan = plan_network([_rank(r, "n0") for r in range(2)], NetworkConfig(gin=gin))

    assert plan.use_gin is use_gin


def test_gin_environment_follows_the_plan(monkeypatch):
    """``EP_DISABLE_GIN`` is 1 without GIN and absent with it.

    Why: DeepEP reads it when the connector creates its buffers.
    """
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)

    apply_gin_environment(plan_network([_rank(0, "n0")], NetworkConfig()))
    assert os.environ["EP_DISABLE_GIN"] == "1"

    apply_gin_environment(plan_network(_two_hosts(), NetworkConfig()))
    assert "EP_DISABLE_GIN" not in os.environ
