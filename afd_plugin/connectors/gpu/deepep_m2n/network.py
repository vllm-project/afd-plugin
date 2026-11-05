# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cluster topology discovery and the network plan for DeepEP M2N.

DeepEP V2 moves data inside an NVLink domain with load/stores (LSA) and
between domains with NCCL GIN (RDMA):

- one host, or an NVL72 rack: one NVLink domain, no RDMA (GIN off);
- several NVLink domains (InfiniBand, or RoCE with routed rails): GIN on, and
  each GPU uses NCCL's own NIC choice.

DeepEP's GIN setup connects every pair of ranks, so the fabric must route
between the NICs NCCL picks. On RoCE whose rails are separate, unrouted
subnets, pin every rank to NICs on one shared rail through ``NCCL_IB_HCA`` in
the launch environment; the connector does not plan NICs.

Each rank reports a ``RankTopology`` in the connector's init handshake, and
every rank evaluates the same ``plan_network`` on the gathered list, so the
plan needs no further agreement.
"""

from __future__ import annotations

import ctypes
import os
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

GIN_MODES = ("auto", "on", "off")
HYBRID_MODES = ("auto", "on", "off")
# DeepEP reads it when it creates each buffer; set from the plan.
GIN_DISABLE_ENV = "EP_DISABLE_GIN"
# DeepEP's default RDMA service level / traffic class.
DEFAULT_RDMA_TRAFFIC_CLASS = 3


@dataclass(frozen=True)
class NetworkConfig:
    """The ``network`` section of ``connector_extra_config``.

    Attributes:
        gin: ``auto`` enables NCCL GIN only when ranks span several NVLink
            domains; ``on``/``off`` force it (``off`` is rejected when ranks
            span several domains: GIN is the only path between them).
        hybrid: DeepEP hybrid mode (rail-aligned RDMA + NVLink forwarding);
            ``auto`` = on.
        rdma_traffic_class: GIN service level / traffic class.
    """

    gin: str = "auto"
    hybrid: str = "auto"
    rdma_traffic_class: int = DEFAULT_RDMA_TRAFFIC_CLASS

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> NetworkConfig:
        raw = dict(raw or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown network options: {unknown}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.gin not in GIN_MODES:
            raise ValueError(f"network.gin must be one of {GIN_MODES}")
        if self.hybrid not in HYBRID_MODES:
            raise ValueError(f"network.hybrid must be one of {HYBRID_MODES}")
        if self.rdma_traffic_class < 0:
            raise ValueError("network.rdma_traffic_class must be non-negative")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankTopology:
    world_rank: int
    host: str
    # Ranks with the same id share an NVLink (LSA) domain.
    nvlink_domain: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RankTopology:
        return cls(
            world_rank=int(raw["world_rank"]),
            host=str(raw["host"]),
            nvlink_domain=str(raw["nvlink_domain"]),
        )


@dataclass(frozen=True)
class NetworkPlan:
    """What every rank does, derived identically from all ``RankTopology``."""

    num_domains: int
    ranks_per_domain: int
    use_gin: bool
    hybrid: bool
    rdma_traffic_class: int

    def summary(self) -> str:
        return (
            f"domains={self.num_domains}x{self.ranks_per_domain} gin={self.use_gin} "
            f"hybrid={self.hybrid} traffic_class={self.rdma_traffic_class}"
        )


# ---------------------------------------------------------------------------
# Discovery (runs on each rank)
# ---------------------------------------------------------------------------


def discover_local_topology(world_rank: int, device: torch.device) -> RankTopology:
    host = socket.gethostname()
    return RankTopology(
        world_rank=world_rank,
        host=host,
        nvlink_domain=nvlink_domain_id(gpu_pci_address(device)) or f"host:{host}",
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
    return NetworkPlan(
        num_domains=len(domains),
        ranks_per_domain=len(domains[0]),
        use_gin=multi_domain if config.gin == "auto" else config.gin == "on",
        hybrid=config.hybrid != "off",
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


def apply_gin_environment(plan: NetworkPlan) -> None:
    """Set DeepEP's GIN switch for this process from the plan.

    DeepEP reads it when it creates each buffer, so call this before the
    connector creates its buffers. Every rank sets the same value.
    """
    if plan.use_gin:
        os.environ.pop(GIN_DISABLE_ENV, None)
    else:
        os.environ[GIN_DISABLE_ENV] = "1"
