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
    NicInfo,
    RankTopology,
    apply_rank_environment,
    discover_nics,
    plan_network,
)

RAIL_A = "roce:172.22.176.0/21"
RAIL_B = "roce:172.22.136.0/21"
RAIL_C = "roce:172.22.200.0/21"
IB_RAIL = "ib:fe80:0000:0000:0000"


def _rank(rank: int, host: str, local: int, nics=(), domain: str | None = None):
    return RankTopology(
        world_rank=rank,
        host=host,
        nvlink_domain=domain or f"host:{host}",
        local_index=local,
        nics=tuple(nics),
    )


def _nic(name: str, rail: str, distance: int) -> NicInfo:
    return NicInfo(name=name, link_layer="ethernet", rail=rail, distance=distance)


def _isolated_rail_two_nodes() -> list[RankTopology]:
    """Every NIC on its own rail; the nearest NIC differs per rank and host."""
    node0 = [
        [
            _nic("mlx5_0", RAIL_A, 2),
            _nic("mlx5_5", RAIL_B, 6),
            _nic("mlx5_3", RAIL_C, 6),
        ],
        [
            _nic("mlx5_0", RAIL_A, 6),
            _nic("mlx5_5", RAIL_B, 6),
            _nic("mlx5_3", RAIL_C, 2),
        ],
    ]
    node1 = [
        [
            _nic("mlx5_0", RAIL_A, 2),
            _nic("mlx5_5", RAIL_B, 6),
            _nic("mlx5_3", RAIL_C, 6),
        ],
        [
            _nic("mlx5_0", RAIL_A, 6),
            _nic("mlx5_5", RAIL_B, 2),
            _nic("mlx5_3", RAIL_C, 6),
        ],
    ]
    return [
        _rank(0, "n0", 0, node0[0]),
        _rank(1, "n0", 1, node0[1]),
        _rank(2, "n1", 0, node1[0]),
        _rank(3, "n1", 1, node1[1]),
    ]


def test_presets_expand_and_explicit_keys_override():
    """Presets fill in defaults, explicit keys override them, and the config
    survives a round trip.

    Why: presets are defaults, not locks.
    """
    rail_isolated = NetworkConfig.from_mapping({"preset": "roce-rail-isolated"})
    overridden = NetworkConfig.from_mapping({"preset": "ib", "nic_policy": "shared"})

    assert (rail_isolated.nic_policy, rail_isolated.rails_routed) == ("shared", False)
    assert (overridden.nic_policy, overridden.rails_routed) == ("shared", True)
    assert NetworkConfig.from_mapping(rail_isolated.to_mapping()) == rail_isolated


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"gin": "maybe"}, "network.gin"),
        ({"nic_policy": "explicit"}, "needs network.nics"),
        ({"nccl_env": {"PATH": "/tmp"}}, "nccl_env key"),
        ({"nccl_env": {"EP_DISABLE_GIN": "1"}}, "use network.gin"),
        ({"bogus": 1}, "unknown network options"),
    ],
)
def test_network_config_rejects_invalid_values(raw, message):
    """Bad ``gin``, ``explicit`` without NICs, non-NCCL env keys,
    ``EP_DISABLE_GIN`` in ``nccl_env`` and unknown keys are rejected.

    Why: bad or conflicting network settings must fail at parse time.
    """
    with pytest.raises(ValueError, match=message):
        NetworkConfig.from_mapping(raw)


def test_single_host_needs_no_rdma():
    """Four ranks on one host: one domain, GIN off, no NIC pins.

    Why: single-node runs must never touch RDMA.
    """
    plan = plan_network(
        [_rank(r, "n0", r, [_nic("mlx5_0", RAIL_A, 2)]) for r in range(4)],
        NetworkConfig(),
    )

    assert (plan.num_domains, plan.use_gin) == (1, False)
    assert plan.nic_by_rank == (None,) * 4


def test_isolated_rails_put_every_rank_on_one_shared_rail():
    """Hosts whose NICs each sit on their own subnet get the shared policy,
    one rail, and the same NIC for every rank.

    Why: encodes the Phase 4 finding as the automatic choice.
    """
    plan = plan_network(_isolated_rail_two_nodes(), NetworkConfig())

    assert (plan.use_gin, plan.nic_policy, plan.rail) == (True, "shared", RAIL_A)
    assert plan.nic_by_rank == ("mlx5_0",) * 4


def test_shared_rail_uses_each_hosts_own_nic_name():
    """The shared subnet is ``mlx5_0`` on one host and ``mlx5_7`` on the
    other; each rank gets its own host's name.

    Why: NIC names are not consistent across hosts.
    """
    ranks = [
        _rank(0, "n0", 0, [_nic("mlx5_0", RAIL_A, 2)]),
        _rank(1, "n1", 0, [_nic("mlx5_7", RAIL_A, 2), _nic("mlx5_0", RAIL_B, 2)]),
    ]

    plan = plan_network(ranks, NetworkConfig())

    assert plan.nic_by_rank == ("mlx5_0", "mlx5_7")


def test_shared_rail_prefers_the_configured_nic():
    """``nics: [mlx5_5]`` selects that NIC's subnet as the shared rail.

    Why: operators can steer which NIC is shared.
    """
    plan = plan_network(
        _isolated_rail_two_nodes(), NetworkConfig(nic_policy="shared", nics=("mlx5_5",))
    )

    assert (plan.rail, plan.nic_by_rank) == (RAIL_B, ("mlx5_5",) * 4)


@pytest.mark.parametrize(
    "config",
    [NetworkConfig(rails_routed=True), NetworkConfig.from_mapping({"preset": "ib"})],
    ids=["routed-roce", "ib-preset"],
)
def test_routed_fabrics_keep_nccls_nearest_nic(config):
    """``rails_routed`` and the ``ib`` preset keep NCCL's per-GPU NIC choice
    (no pins) with GIN on.

    Why: routed fabrics should use every NIC, not throttle to one.
    """
    plan = plan_network(_isolated_rail_two_nodes(), config)

    assert (plan.use_gin, plan.nic_policy) == (True, "nccl")
    assert plan.nic_by_rank == (None,) * 4


def test_ib_fat_tree_is_one_rail():
    """IB NICs on one subnet prefix get the NCCL policy without pins.

    Why: one rail needs no pinning.
    """
    ib_nic = NicInfo(name="mlx5_0", link_layer="infiniband", rail=IB_RAIL, distance=2)
    plan = plan_network(
        [_rank(r, f"n{r // 2}", r % 2, [ib_nic]) for r in range(4)], NetworkConfig()
    )

    assert (plan.use_gin, plan.nic_policy) == (True, "nccl")


def test_nvl72_is_one_domain_even_with_uneven_hosts():
    # One NVLink fabric spans the hosts, so per-host counts need not match.
    """Four ranks on two trays (3+1) in one NVLink fabric: one domain of 4,
    GIN off.

    Why: NVL72 is one domain; per-host layout rules must not apply.
    """
    ranks = [
        _rank(r, f"tray{0 if r < 3 else 1}", r, domain="nvl:abc:1") for r in range(4)
    ]

    plan = plan_network(ranks, NetworkConfig())

    assert (plan.num_domains, plan.ranks_per_domain, plan.use_gin) == (1, 4, False)


def test_explicit_nics_follow_local_rank():
    """``explicit`` assigns ``nics`` by local rank on every host.

    Why: a manual override for fabrics the planner cannot model.
    """
    plan = plan_network(
        _isolated_rail_two_nodes(),
        NetworkConfig(nic_policy="explicit", nics=("mlx5_0", "mlx5_5")),
    )

    assert plan.nic_by_rank == ("mlx5_0", "mlx5_5", "mlx5_0", "mlx5_5")


def test_no_common_rail_is_an_error():
    """Hosts with no subnet in common raise "no RDMA rail".

    Why: fail at init instead of a QP timeout minutes later.
    """
    ranks = [
        _rank(0, "n0", 0, [_nic("mlx5_0", RAIL_A, 2)]),
        _rank(1, "n1", 0, [_nic("mlx5_0", RAIL_B, 2)]),
    ]

    with pytest.raises(RuntimeError, match="no RDMA rail"):
        plan_network(ranks, NetworkConfig())


@pytest.mark.parametrize(
    "config",
    [NetworkConfig(gin="off"), NetworkConfig.from_mapping({"preset": "single-node"})],
    ids=["gin-off", "single-node-preset"],
)
def test_gin_off_across_nvlink_domains_is_an_error(config):
    # NVLink load/store cannot reach another domain; GIN is the only path.
    """``gin: off`` (or the ``single-node`` preset) with ranks on two hosts
    raises.

    Why: without GIN DeepEP cannot reach another NVLink domain.
    """
    with pytest.raises(RuntimeError, match="span 2 NVLink domains"):
        plan_network(_isolated_rail_two_nodes(), config)


def test_gin_off_on_one_host_is_allowed():
    """``gin: off`` with all ranks on one host plans normally.

    Why: the multi-domain check must not block single-node use.
    """
    plan = plan_network(
        [_rank(r, "n0", r, [_nic("mlx5_0", RAIL_A, 2)]) for r in range(2)],
        NetworkConfig(gin="off"),
    )

    assert not plan.use_gin


def test_forced_gin_on_one_domain_still_plans_nics():
    """``gin: on`` on one host keeps GIN on and pins no NICs.

    Why: a forced setting is honored without inventing NIC pins.
    """
    plan = plan_network(
        [_rank(r, "n0", r, [_nic("mlx5_0", RAIL_A, 2)]) for r in range(2)],
        NetworkConfig(gin="on"),
    )

    assert plan.use_gin
    assert plan.nic_by_rank == (None, None)


def test_rank_environment_pins_nic_and_gin(monkeypatch):
    """Applying a plan sets ``NCCL_IB_HCA``, ``nccl_env`` values and
    ``EP_DISABLE_GIN`` (absent with GIN, 1 without).

    Why: the plan only works if these reach NCCL and DeepEP.
    """
    for key in ("NCCL_IB_HCA", "EP_DISABLE_GIN", "NCCL_IB_TC"):
        monkeypatch.delenv(key, raising=False)
    config = NetworkConfig(nccl_env=(("NCCL_IB_TC", "106"),))
    plan = plan_network(_isolated_rail_two_nodes(), config)

    apply_rank_environment(plan, config, world_rank=3)

    assert os.environ["NCCL_IB_HCA"] == "=mlx5_0"
    assert os.environ["NCCL_IB_TC"] == "106"
    assert "EP_DISABLE_GIN" not in os.environ

    single = plan_network([_rank(0, "n0", 0)], NetworkConfig())
    apply_rank_environment(single, NetworkConfig(), world_rank=0)
    assert os.environ["EP_DISABLE_GIN"] == "1"


def _fake_sysfs(tmp_path):
    """Two RoCE ports (near and far from the GPU) and one IB port."""
    pci_root = tmp_path / "devices"
    near_bridge = pci_root / "pci0000:00" / "0000:00:01.0"
    far_bridge = pci_root / "pci0000:80" / "0000:80:01.0"
    functions = {
        "0000:18:00.0": near_bridge / "0000:18:00.0",
        "0000:19:00.0": near_bridge / "0000:19:00.0",
        "0000:9b:00.0": far_bridge / "0000:9b:00.0",
        "0000:aa:00.0": far_bridge / "0000:aa:00.0",
    }
    sysfs_pci = tmp_path / "bus_pci"
    sysfs_pci.mkdir()
    for bdf, path in functions.items():
        path.mkdir(parents=True)
        (sysfs_pci / bdf).symlink_to(path)
    sysfs_ib = tmp_path / "infiniband"
    ports = {
        "mlx5_0": (
            "0000:19:00.0",
            "Ethernet",
            "0000:0000:0000:0000:0000:ffff:ac16:b021",
        ),
        "mlx5_10": (
            "0000:9b:00.0",
            "Ethernet",
            "0000:0000:0000:0000:0000:ffff:ac16:a021",
        ),
        "mlx5_12": (
            "0000:aa:00.0",
            "InfiniBand",
            "fe80:0000:0000:0000:0e42:a1ff:fe00:0001",
        ),
    }
    for name, (bdf, link_layer, gid) in ports.items():
        port = sysfs_ib / name / "ports" / "1"
        (port / "gids").mkdir(parents=True)
        (port / "state").write_text("4: ACTIVE\n")
        (port / "link_layer").write_text(f"{link_layer}\n")
        (port / "gids" / "0").write_text(
            "fe80:0000:0000:0000:0000:0000:0000:0000\n"
            if link_layer == "Ethernet"
            else gid
        )
        if link_layer == "Ethernet":
            (port / "gids" / "3").write_text(f"{gid}\n")
        (sysfs_ib / name / "device").symlink_to(functions[bdf])
    down = sysfs_ib / "mlx5_6" / "ports" / "1"
    (down / "gids").mkdir(parents=True)
    (down / "state").write_text("1: DOWN\n")
    return sysfs_ib, sysfs_pci


def test_discovery_reads_rails_and_pcie_distance(tmp_path):
    """A fake sysfs tree: active RoCE and IB ports found (down port skipped),
    rails from interface subnets, GID prefix or /32, and PCIe distance
    ordering near before far.

    Why: topology discovery is tested without RDMA hardware.
    """
    sysfs_ib, sysfs_pci = _fake_sysfs(tmp_path)

    nics = {
        nic.name: nic
        for nic in discover_nics(
            "0000:18:00.0",
            sysfs_infiniband=sysfs_ib,
            sysfs_pci=sysfs_pci,
            interface_networks={"172.22.176.33": "172.22.176.0/21"},
        )
    }

    assert set(nics) == {"mlx5_0", "mlx5_10", "mlx5_12"}
    assert nics["mlx5_0"].rail == "roce:172.22.176.0/21"
    # No netmask known for this address: it is its own (isolated) rail.
    assert nics["mlx5_10"].rail == "roce:172.22.160.33/32"
    assert nics["mlx5_12"].rail == IB_RAIL
    assert nics["mlx5_0"].distance < nics["mlx5_10"].distance
