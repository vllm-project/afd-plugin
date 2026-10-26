# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cluster topology discovery and the network plan for DeepEP M2N.

DeepEP V2 moves data inside an NVLink domain with load/stores (LSA) and
between domains with NCCL GIN (RDMA). What that needs depends on the cluster:

- one host, or an NVL72 rack: one NVLink domain, no RDMA (GIN off);
- InfiniBand fat-tree or RoCE with routed rails: GIN on, each GPU uses its
  nearest NIC (NCCL's own choice);
- RoCE whose rails are separate, unrouted subnets: GIN on, and every rank must
  use NICs on one shared rail, because DeepEP's GIN setup connects every pair
  of ranks.

Each rank reports a ``RankTopology`` in the connector's init handshake, and
every rank evaluates the same ``plan_network`` on the gathered list, so the
plan needs no further agreement. Whether rails are routed cannot be seen from
inside a pod; ``rails_routed`` declares it, and ``auto`` assumes they are not
when NICs sit on several rails (the choice that works everywhere, at the cost
of bandwidth).
"""

from __future__ import annotations

import ctypes
import ipaddress
import os
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import psutil
import torch

SYSFS_INFINIBAND = Path("/sys/class/infiniband")
SYSFS_PCI = Path("/sys/bus/pci/devices")
# RoCE v2 GIDs carry the port's IPv4 address as ::ffff:a.b.c.d.
IPV4_MAPPED_GID_PREFIX = "0000:0000:0000:0000:0000:ffff:"
ETHERNET = "ethernet"
INFINIBAND = "infiniband"
UNKNOWN_DISTANCE = 1 << 16

GIN_MODES = ("auto", "on", "off")
NIC_POLICIES = ("auto", "nccl", "shared", "explicit")
HYBRID_MODES = ("auto", "on", "off")
NCCL_ENV_PREFIXES = ("NCCL_", "EP_")
# Set from the plan (``network.gin``); a copy in ``nccl_env`` would fight it.
GIN_DISABLE_ENV = "EP_DISABLE_GIN"
# DeepEP's default RDMA service level / traffic class.
DEFAULT_RDMA_TRAFFIC_CLASS = 3

# Preset -> NetworkConfig defaults; explicit keys in the config override them.
PRESETS: dict[str, dict[str, Any]] = {
    "auto": {},
    "single-node": {"gin": "off", "nic_policy": "nccl"},
    "roce-rail-isolated": {"nic_policy": "shared", "rails_routed": False},
    "roce-routed": {"nic_policy": "nccl", "rails_routed": True},
    "ib": {"nic_policy": "nccl", "rails_routed": True},
    "nvl72": {"nic_policy": "nccl", "rails_routed": True},
}


@dataclass(frozen=True)
class NetworkConfig:
    """The ``network`` section of ``connector_extra_config``.

    Attributes:
        preset: Defaults for a fabric type (see ``PRESETS``).
        gin: ``auto`` enables NCCL GIN only when ranks span several NVLink
            domains; ``on``/``off`` force it (``off`` is rejected when ranks
            span several domains: GIN is the only path between them).
        nic_policy: ``auto`` picks ``nccl`` when rails are routed or all
            NICs share one rail, else ``shared``; ``nccl`` keeps NCCL's
            per-GPU choice; ``shared`` puts every rank on NICs of one rail
            present on every host; ``explicit`` uses ``nics`` by local rank.
        nics: Explicit NIC names: by local rank for ``explicit``, or the NIC
            to prefer (first entry) for ``shared``.
        rails_routed: Whether different rails can reach each other.
            ``None`` means unknown (treated as not routed).
        hybrid: DeepEP hybrid mode (rail-aligned RDMA + NVLink forwarding);
            ``auto`` = on.
        rdma_traffic_class: GIN service level / traffic class.
        nccl_env: Extra ``NCCL_*``/``EP_*`` variables applied before any
            NCCL communicator is created (process-wide).
    """

    preset: str = "auto"
    gin: str = "auto"
    nic_policy: str = "auto"
    nics: tuple[str, ...] = ()
    rails_routed: bool | None = None
    hybrid: str = "auto"
    rdma_traffic_class: int = DEFAULT_RDMA_TRAFFIC_CLASS
    nccl_env: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> NetworkConfig:
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown network options: {unknown}")
        preset = raw.get("preset", "auto")
        if preset not in PRESETS:
            raise ValueError(f"network.preset must be one of {sorted(PRESETS)}")
        merged = {**PRESETS[preset], **raw}
        if "nics" in merged:
            merged["nics"] = tuple(merged["nics"])
        if "nccl_env" in merged:
            merged["nccl_env"] = tuple(
                sorted((str(k), str(v)) for k, v in dict(merged["nccl_env"]).items())
            )
        config = cls(**merged)
        config.validate()
        return config

    def validate(self) -> None:
        if self.gin not in GIN_MODES:
            raise ValueError(f"network.gin must be one of {GIN_MODES}")
        if self.nic_policy not in NIC_POLICIES:
            raise ValueError(f"network.nic_policy must be one of {NIC_POLICIES}")
        if self.hybrid not in HYBRID_MODES:
            raise ValueError(f"network.hybrid must be one of {HYBRID_MODES}")
        if not all(isinstance(nic, str) and nic for nic in self.nics):
            raise ValueError("network.nics must be non-empty NIC names")
        if self.nic_policy == "explicit" and not self.nics:
            raise ValueError("network.nic_policy=explicit needs network.nics")
        if self.rdma_traffic_class < 0:
            raise ValueError("network.rdma_traffic_class must be non-negative")
        for key, _ in self.nccl_env:
            if not key.startswith(NCCL_ENV_PREFIXES):
                raise ValueError(
                    f"network.nccl_env key {key!r} must start with {NCCL_ENV_PREFIXES}"
                )
            if key == GIN_DISABLE_ENV:
                raise ValueError(
                    f"network.nccl_env cannot set {GIN_DISABLE_ENV}; use network.gin"
                )

    def to_mapping(self) -> dict[str, Any]:
        mapping = asdict(self)
        mapping["nics"] = list(self.nics)
        mapping["nccl_env"] = dict(self.nccl_env)
        return mapping


@dataclass(frozen=True)
class NicInfo:
    """One active RDMA port as seen from a rank."""

    name: str
    link_layer: str
    # Ranks can talk RDMA only between NICs on the same rail (unless routed).
    rail: str
    # PCIe hops between the rank's GPU and this NIC (lower is closer).
    distance: int


@dataclass(frozen=True)
class RankTopology:
    world_rank: int
    host: str
    # Ranks with the same id share an NVLink (LSA) domain.
    nvlink_domain: str
    local_index: int
    nics: tuple[NicInfo, ...] = field(default_factory=tuple)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RankTopology:
        return cls(
            world_rank=int(raw["world_rank"]),
            host=str(raw["host"]),
            nvlink_domain=str(raw["nvlink_domain"]),
            local_index=int(raw["local_index"]),
            nics=tuple(NicInfo(**nic) for nic in raw["nics"]),
        )


@dataclass(frozen=True)
class NetworkPlan:
    """What every rank does, derived identically from all ``RankTopology``."""

    num_domains: int
    ranks_per_domain: int
    use_gin: bool
    hybrid: bool
    # world rank -> NIC to pin via NCCL_IB_HCA (None = NCCL's own choice).
    nic_by_rank: tuple[str | None, ...]
    nic_policy: str
    rail: str | None
    rdma_traffic_class: int

    def summary(self) -> str:
        nics = sorted({nic for nic in self.nic_by_rank if nic is not None})
        return (
            f"domains={self.num_domains}x{self.ranks_per_domain} gin={self.use_gin} "
            f"hybrid={self.hybrid} nic_policy={self.nic_policy} rail={self.rail} "
            f"nics={nics or 'nccl'} traffic_class={self.rdma_traffic_class}"
        )


# ---------------------------------------------------------------------------
# Discovery (runs on each rank)
# ---------------------------------------------------------------------------


def discover_local_topology(
    world_rank: int,
    device: torch.device,
    local_index: int,
    *,
    sysfs_infiniband: Path = SYSFS_INFINIBAND,
    sysfs_pci: Path = SYSFS_PCI,
    interface_networks: Mapping[str, str] | None = None,
) -> RankTopology:
    host = socket.gethostname()
    gpu_pci = gpu_pci_address(device)
    return RankTopology(
        world_rank=world_rank,
        host=host,
        nvlink_domain=nvlink_domain_id(gpu_pci) or f"host:{host}",
        local_index=local_index,
        nics=discover_nics(
            gpu_pci,
            sysfs_infiniband=sysfs_infiniband,
            sysfs_pci=sysfs_pci,
            interface_networks=(
                local_interface_networks()
                if interface_networks is None
                else interface_networks
            ),
        ),
    )


def gpu_pci_address(device: torch.device) -> str | None:
    """``dddd:bb:dd.f`` of the CUDA device (CUDA order, not NVML order)."""
    props = torch.cuda.get_device_properties(device)
    return (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )


def nvlink_domain_id(gpu_pci: str | None) -> str | None:
    """Multi-node NVLink (NVL72) clique id from NVML fabric info, if any.

    Returns ``None`` when the GPU is not part of an NVLink fabric (or NVML
    lacks the API), so the caller falls back to "one domain per host".
    """
    if gpu_pci is None:
        return None
    from vllm.third_party import pynvml

    info = pynvml.c_nvmlGpuFabricInfo_t()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(gpu_pci.upper())
        pynvml.nvmlDeviceGetGpuFabricInfo(handle, ctypes.byref(info))
    except pynvml.NVMLError:
        # Not an NVLink-fabric GPU (e.g. HGX H100): NOT_SUPPORTED.
        return None
    # Read the raw UUID bytes: a c_char array field stops at the first zero.
    field_info = type(info).clusterUuid
    cluster = ctypes.string_at(
        ctypes.addressof(info) + field_info.offset, field_info.size
    ).hex()
    if info.state != pynvml.NVML_GPU_FABRIC_STATE_COMPLETED or not int(cluster, 16):
        return None
    return f"nvl:{cluster}:{info.cliqueId}"


def local_interface_networks() -> dict[str, str]:
    """Local IPv4 address -> its network (``a.b.c.d/len``)."""
    networks: dict[str, str] = {}
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family == socket.AF_INET and address.netmask:
                network = ipaddress.ip_network(
                    f"{address.address}/{address.netmask}", strict=False
                )
                networks[address.address] = str(network)
    return networks


def discover_nics(
    gpu_pci: str | None,
    *,
    sysfs_infiniband: Path,
    sysfs_pci: Path,
    interface_networks: Mapping[str, str],
) -> tuple[NicInfo, ...]:
    nics = []
    if not sysfs_infiniband.is_dir():
        return ()
    for device_dir in sorted(sysfs_infiniband.iterdir()):
        port = device_dir / "ports" / "1"
        state = _read(port / "state")
        if state is None or "ACTIVE" not in state:
            continue
        link_layer = (_read(port / "link_layer") or "").strip().lower()
        rail = nic_rail(port, link_layer, interface_networks)
        if rail is None:
            continue
        nics.append(
            NicInfo(
                name=device_dir.name,
                link_layer=link_layer,
                rail=rail,
                distance=pci_distance(
                    gpu_pci, _pci_address(device_dir), sysfs_pci=sysfs_pci
                ),
            )
        )
    return tuple(nics)


def nic_rail(
    port: Path, link_layer: str, interface_networks: Mapping[str, str]
) -> str | None:
    """Rail id: the RoCE port's IPv4 network, or the IB GID subnet prefix."""
    gids_dir = port / "gids"
    if not gids_dir.is_dir():
        return None
    gids = [_read(gid) for gid in sorted(gids_dir.iterdir(), key=_gid_index)]
    if link_layer == ETHERNET:
        for gid in gids:
            if gid and gid.strip().startswith(IPV4_MAPPED_GID_PREFIX):
                ipv4 = _ipv4_from_gid(gid.strip())
                # Without the netmask the port is its own rail (isolated).
                return f"roce:{interface_networks.get(ipv4, ipv4 + '/32')}"
        return None
    if link_layer == INFINIBAND and gids and gids[0]:
        prefix = ":".join(gids[0].strip().split(":")[:4])
        return f"ib:{prefix}"
    return None


def pci_distance(gpu_pci: str | None, nic_pci: str | None, *, sysfs_pci: Path) -> int:
    """PCIe hops between two functions via their common upstream bridge."""
    if gpu_pci is None or nic_pci is None:
        return UNKNOWN_DISTANCE
    try:
        gpu_path = (sysfs_pci / gpu_pci).resolve(strict=True).parts
        nic_path = (sysfs_pci / nic_pci).resolve(strict=True).parts
    except OSError:
        return UNKNOWN_DISTANCE
    common = 0
    for gpu_part, nic_part in zip(gpu_path, nic_path, strict=False):
        if gpu_part != nic_part:
            break
        common += 1
    return (len(gpu_path) - common) + (len(nic_path) - common)


def _pci_address(device_dir: Path) -> str | None:
    try:
        return (device_dir / "device").resolve(strict=True).name
    except OSError:
        return None


def _gid_index(path: Path) -> int:
    return int(path.name) if path.name.isdigit() else 1 << 30


def _ipv4_from_gid(gid: str) -> str:
    high, low = gid[len(IPV4_MAPPED_GID_PREFIX) :].split(":")
    return str(ipaddress.IPv4Address(int(high + low, 16)))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Plan (pure; identical on every rank)
# ---------------------------------------------------------------------------


def plan_network(topologies: list[RankTopology], config: NetworkConfig) -> NetworkPlan:
    topologies = sorted(topologies, key=lambda topology: topology.world_rank)
    domains = _domain_blocks(topologies)
    multi_domain = len(domains) > 1
    if multi_domain and config.gin == "off":
        # Without GIN, DeepEP reaches peers only by NVLink load/store, which
        # cannot cross NVLink domains.
        raise RuntimeError(
            f"network.gin=off, but the ranks span {len(domains)} NVLink domains; "
            "DeepEP needs NCCL GIN (RDMA) between domains"
        )
    use_gin = multi_domain if config.gin == "auto" else config.gin == "on"
    hybrid = config.hybrid != "off"
    policy, rail, nic_by_rank = _choose_nics(topologies, config, use_gin)
    return NetworkPlan(
        num_domains=len(domains),
        ranks_per_domain=len(domains[0]),
        use_gin=use_gin,
        hybrid=hybrid,
        nic_by_rank=nic_by_rank,
        nic_policy=policy,
        rail=rail,
        rdma_traffic_class=config.rdma_traffic_class,
    )


def _domain_blocks(topologies: list[RankTopology]) -> list[list[int]]:
    """Ranks grouped by NVLink domain; DeepEP needs equal, contiguous blocks."""
    blocks: list[tuple[str, list[int]]] = []
    for topology in topologies:
        if blocks and blocks[-1][0] == topology.nvlink_domain:
            blocks[-1][1].append(topology.world_rank)
        else:
            blocks.append((topology.nvlink_domain, [topology.world_rank]))
    names = [name for name, _ in blocks]
    if len(set(names)) != len(names):
        raise RuntimeError(f"ranks of one NVLink domain are not contiguous: {names}")
    sizes = {len(ranks) for _, ranks in blocks}
    if len(sizes) != 1:
        raise RuntimeError(
            "NVLink domains hold different rank counts: "
            f"{[(name, len(ranks)) for name, ranks in blocks]}"
        )
    return [ranks for _, ranks in blocks]


def _choose_nics(
    topologies: list[RankTopology], config: NetworkConfig, use_gin: bool
) -> tuple[str, str | None, tuple[str | None, ...]]:
    none = tuple(None for _ in topologies)
    if not use_gin:
        return "none", None, none
    policy = config.nic_policy
    if policy == "explicit":
        nics = tuple(config.nics[t.local_index % len(config.nics)] for t in topologies)
        return policy, None, nics
    rails = {nic.rail for t in topologies for nic in t.nics}
    if policy == "auto":
        policy = "nccl" if config.rails_routed or len(rails) <= 1 else "shared"
    if policy == "nccl":
        return policy, None, none
    rail = _shared_rail(topologies, config)
    nics = tuple(
        min(
            (nic for nic in t.nics if nic.rail == rail),
            key=lambda nic: (nic.distance, nic.name),
        ).name
        for t in topologies
    )
    return policy, rail, nics


def _shared_rail(topologies: list[RankTopology], config: NetworkConfig) -> str:
    """A rail every rank has a NIC on; prefer ``nics[0]``'s, then the closest."""
    common = set.intersection(*({nic.rail for nic in t.nics} for t in topologies))
    if not common:
        raise RuntimeError("no RDMA rail is reachable from every rank")
    if config.nics:
        preferred = {
            nic.rail
            for t in topologies
            for nic in t.nics
            if nic.name == config.nics[0] and nic.rail in common
        }
        if preferred:
            return min(preferred)
    return min(
        common,
        key=lambda rail: (
            sum(
                min(nic.distance for nic in t.nics if nic.rail == rail)
                for t in topologies
            ),
            rail,
        ),
    )


def apply_rank_environment(
    plan: NetworkPlan, config: NetworkConfig, world_rank: int
) -> None:
    """Process-wide NCCL/DeepEP settings for this rank, before any communicator.

    NCCL reads these when its first communicator in the process is created.
    AFD workers therefore run the connector's
    ``prepare_before_distributed_init`` (which calls this) before vLLM's
    distributed init creates its own NCCL groups; a process that
    already created a communicator keeps the values it started with.
    """
    for key, value in config.nccl_env:
        os.environ[key] = value
    nic = plan.nic_by_rank[world_rank]
    if nic is not None:
        os.environ["NCCL_IB_HCA"] = "=" + nic
    if plan.use_gin:
        os.environ.pop(GIN_DISABLE_ENV, None)
    else:
        os.environ[GIN_DISABLE_ENV] = "1"
