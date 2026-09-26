# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pod layout parsing and AFD rank placement for multi-pod E2E runs.

``plan`` is a pure function: every pod runs it on identical inputs, derives the
identical global plan, and reads its own element. That is what removes the need
for a master process to tell any pod what to do.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass

from tests.e2e.models.deepseek_v4_flash.config import DSV4_ASYNC_CAM_SCENARIO

ATTENTION_ROLE = "attention"
FFN_ROLE = "ffn"
BASELINE_ROLE = "baseline"
ROLE_KINDS = (ATTENTION_ROLE, FFN_ROLE)

ASYNC_AFD_CONNECTOR = "CAMAsyncAFDConnector"
# docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md:88 - the synchronous connectors
# rendezvous at the first FFN rank. The async CAM connector rendezvous at the
# Attention side instead (tools/itask/launch_dsv4_afd_cross_node.sh).
RENDEZVOUS_ROLE_BY_CONNECTOR = {
    "P2pNcclAFDConnector": FFN_ROLE,
    "CAMP2pAFDConnector": FFN_ROLE,
    ASYNC_AFD_CONNECTOR: ATTENTION_ROLE,
}

# Fixed and distinct per role so a pod holding both roles cannot collide.
DP_RPC_PORT_BY_ROLE = {
    ATTENTION_ROLE: 29550,
    FFN_ROLE: 29551,
}

# DSV4 pins its own --data-parallel-address and rejects extra vLLM arguments,
# which conflicts with per-pod slot placement, and has no multi-pod NPU
# validation.
UNSUPPORTED_SCENARIOS = frozenset({DSV4_ASYNC_CAM_SCENARIO})

_POD_SPEC_PATTERN = re.compile(r"\A(?:(\d+)A)?(?:(\d+)F)?\Z")


@dataclass(frozen=True)
class PodSpec:
    """The AFD ranks one pod holds, one entry of ``--pod-layout``."""

    attention: int
    ffn: int

    def ranks(self, role_kind: str) -> int:
        if role_kind == ATTENTION_ROLE:
            return self.attention
        if role_kind == FFN_ROLE:
            return self.ffn
        raise ValueError(f"unknown AFD role {role_kind!r}")

    def __str__(self) -> str:
        return f"{self.attention}A{self.ffn}F"


@dataclass(frozen=True)
class PodLayout:
    """An ordered distribution of AFD ranks into pods."""

    pods: tuple[PodSpec, ...]

    @classmethod
    def parse(cls, text: str) -> PodLayout:
        entries = [entry.strip() for entry in text.split(",")]
        if not text.strip() or any(not entry for entry in entries):
            raise ValueError(f"empty pod layout entry in {text!r}")
        return cls(tuple(_parse_pod_spec(entry) for entry in entries))

    @property
    def num_pods(self) -> int:
        return len(self.pods)

    def total_ranks(self, role_kind: str) -> int:
        return sum(pod.ranks(role_kind) for pod in self.pods)

    def leader_index(self, role_kind: str) -> int | None:
        """Index of the lowest-numbered pod holding ``role_kind``."""
        for index, pod in enumerate(self.pods):
            if pod.ranks(role_kind) > 0:
                return index
        return None

    def canonical(self) -> str:
        return ",".join(str(pod) for pod in self.pods)

    def __str__(self) -> str:
        return self.canonical()


def _parse_pod_spec(entry: str) -> PodSpec:
    match = _POD_SPEC_PATTERN.match(entry.upper())
    if match is None or match.group(0) == "":
        raise ValueError(
            f"invalid pod layout entry {entry!r}; expected <int>A<int>F "
            f"(shorthands <int>A and <int>F are accepted)",
        )
    attention = int(match.group(1) or 0)
    ffn = int(match.group(2) or 0)
    if attention == 0 and ffn == 0:
        raise ValueError(
            f"pod layout entry {entry!r} assigns no ranks; every pod must do work",
        )
    return PodSpec(attention=attention, ffn=ffn)


@dataclass(frozen=True)
class RoleTopology:
    """The logical size of one AFD role, owned by the scenario."""

    ranks: int
    tp_size: int

    @property
    def dp_size(self) -> int:
        return self.ranks // self.tp_size


@dataclass(frozen=True)
class Topology:
    """The logical topology a scenario id fixes, independent of pod layout."""

    attention: RoleTopology
    ffn: RoleTopology
    connector: str
    baseline: bool = False

    def role(self, role_kind: str) -> RoleTopology:
        if role_kind == ATTENTION_ROLE:
            return self.attention
        if role_kind == FFN_ROLE:
            return self.ffn
        raise ValueError(f"unknown AFD role {role_kind!r}")

    @property
    def rendezvous_role(self) -> str | None:
        """Role whose first rank hosts the AFD connector rendezvous."""
        if self.baseline:
            return None
        rendezvous_role = RENDEZVOUS_ROLE_BY_CONNECTOR.get(self.connector)
        if rendezvous_role is None:
            raise ValueError(
                f"no AFD rendezvous role known for connector {self.connector!r}",
            )
        return rendezvous_role

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Topology:
        """Build a topology from a ``configure_scenario``-populated namespace."""
        connector = args.afd_connector or (
            "CAMP2pAFDConnector"
            if args.device_backend == "npu"
            else "P2pNcclAFDConnector"
        )
        return cls(
            attention=RoleTopology(
                ranks=args.num_attention_ranks,
                tp_size=args.attention_tp_size or args.tp_size,
            ),
            ffn=RoleTopology(
                ranks=args.num_ffn_ranks,
                tp_size=args.ffn_tp_size or args.tp_size,
            ),
            connector=connector,
            baseline=args.baseline,
        )


@dataclass(frozen=True)
class RoleSlot:
    """One role's share of one pod: what that pod launches for that role."""

    role: str
    role_kind: str
    dp_size: int
    dp_size_local: int
    dp_start_rank: int
    dp_address: str
    dp_rpc_port: int
    headless: bool
    devices: tuple[str, ...]
    afd_host: str

    @property
    def spans_pods(self) -> bool:
        """Whether this role is split across pods and needs placement flags.

        False reproduces today's single-host argv character for character, which
        is what makes a one-pod layout a true control case.
        """
        return (
            self.dp_size_local != self.dp_size
            or self.dp_start_rank != 0
            or self.headless
        )


@dataclass(frozen=True)
class PodPlan:
    """Everything one pod needs to run its slice of the topology."""

    index: int
    address: str
    slots: tuple[RoleSlot, ...]
    is_evaluator: bool

    def slot(self, role_kind: str) -> RoleSlot | None:
        for slot in self.slots:
            if slot.role_kind == role_kind:
                return slot
        return None

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(device for slot in self.slots for device in slot.devices)


def reject_unsupported_scenario(scenario: str) -> None:
    """Fail fast on a scenario the multi-pod runner cannot place."""
    if scenario in UNSUPPORTED_SCENARIOS:
        raise ValueError(
            f"scenario {scenario} is not supported by the multi-pod runner"
        )


def validate_layout(topology: Topology, layout: PodLayout) -> None:
    """Check a layout against the logical topology the scenario id fixes."""
    for role_kind in ROLE_KINDS:
        role = topology.role(role_kind)
        provided = layout.total_ranks(role_kind)
        if provided != role.ranks:
            raise ValueError(
                f"layout {layout.canonical()} provides "
                f"{layout.total_ranks(ATTENTION_ROLE)}A/"
                f"{layout.total_ranks(FFN_ROLE)}F; scenario requires "
                f"{topology.attention.ranks}A/{topology.ffn.ranks}F",
            )
        if role.tp_size < 1:
            raise ValueError(f"{role_kind} TP size must be positive")
        for index, pod in enumerate(layout.pods):
            if pod.ranks(role_kind) % role.tp_size != 0:
                raise ValueError(
                    f"pod {index} holds {pod.ranks(role_kind)} {role_kind} ranks, "
                    f"which is not divisible by {role_kind} TP size "
                    f"{role.tp_size}; a TP group cannot span pods",
                )
    if topology.baseline and layout.total_ranks(FFN_ROLE) != 0:
        raise ValueError("baseline scenarios cannot place FFN ranks")
    # build_baseline_command has no slot placement flags, so a split baseline
    # would start independent servers instead of one DP group.
    if topology.baseline and layout.num_pods > 1:
        raise ValueError(
            f"baseline scenarios must run on one pod, got layout {layout.canonical()}",
        )


def plan(
    topology: Topology,
    layout: PodLayout,
    addresses: Sequence[str],
) -> list[PodPlan]:
    """Derive every pod's plan. Pure: identical inputs give identical output."""
    validate_layout(topology, layout)
    if len(addresses) != layout.num_pods:
        raise ValueError(
            f"layout {layout.canonical()} needs {layout.num_pods} addresses, "
            f"got {len(addresses)}",
        )

    # Only roles the layout actually places appear here, so a lookup for a role
    # a pod holds is always an int.
    leader_index = {
        role_kind: leader
        for role_kind in ROLE_KINDS
        if (leader := layout.leader_index(role_kind)) is not None
    }
    if ATTENTION_ROLE not in leader_index:
        raise ValueError(f"layout {layout.canonical()} places no Attention ranks")
    attention_leader = leader_index[ATTENTION_ROLE]

    rendezvous_role = topology.rendezvous_role
    if rendezvous_role is None:
        afd_host = ""
    elif rendezvous_role not in leader_index:
        raise ValueError(
            f"connector {topology.connector} rendezvous at the first "
            f"{rendezvous_role} rank, but layout {layout.canonical()} "
            f"places no {rendezvous_role} ranks",
        )
    else:
        afd_host = addresses[leader_index[rendezvous_role]]

    next_dp_rank = dict.fromkeys(ROLE_KINDS, 0)
    plans: list[PodPlan] = []
    for index, pod in enumerate(layout.pods):
        slots: list[RoleSlot] = []
        next_device = 0
        for role_kind in ROLE_KINDS:
            local_ranks = pod.ranks(role_kind)
            devices = tuple(
                str(device) for device in range(next_device, next_device + local_ranks)
            )
            next_device += local_ranks
            if local_ranks == 0:
                continue
            role = topology.role(role_kind)
            local_dp_size = local_ranks // role.tp_size
            slots.append(
                RoleSlot(
                    role=(
                        BASELINE_ROLE
                        if topology.baseline and role_kind == ATTENTION_ROLE
                        else role_kind
                    ),
                    role_kind=role_kind,
                    dp_size=role.dp_size,
                    dp_size_local=local_dp_size,
                    dp_start_rank=next_dp_rank[role_kind],
                    dp_address=addresses[leader_index[role_kind]],
                    dp_rpc_port=DP_RPC_PORT_BY_ROLE[role_kind],
                    headless=index != leader_index[role_kind],
                    devices=devices,
                    afd_host=afd_host,
                ),
            )
            next_dp_rank[role_kind] += local_dp_size
        plans.append(
            PodPlan(
                index=index,
                address=addresses[index],
                slots=tuple(slots),
                is_evaluator=index == attention_leader,
            ),
        )
    return plans
