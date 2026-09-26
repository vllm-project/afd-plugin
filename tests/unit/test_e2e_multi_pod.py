# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Unit coverage for the multi-pod runner's pure logic: no cluster, no device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash.config import DSV4_ASYNC_CAM_SCENARIO
from tests.e2e.multi_pod import identity
from tests.e2e.multi_pod.layout import (
    ATTENTION_ROLE,
    FFN_ROLE,
    PodLayout,
    PodPlan,
    PodSpec,
    RoleSlot,
    RoleTopology,
    Topology,
    plan,
    reject_unsupported_scenario,
    validate_layout,
)

P2P_CONNECTOR = "P2pNcclAFDConnector"
ASYNC_CONNECTOR = "CAMAsyncAFDConnector"


def _topology(
    attention_ranks: int = 2,
    ffn_ranks: int = 2,
    *,
    attention_tp: int = 1,
    ffn_tp: int = 1,
    connector: str = P2P_CONNECTOR,
    baseline: bool = False,
) -> Topology:
    return Topology(
        attention=RoleTopology(ranks=attention_ranks, tp_size=attention_tp),
        ffn=RoleTopology(ranks=ffn_ranks, tp_size=ffn_tp),
        connector=connector,
        baseline=baseline,
    )


def _addresses(count: int) -> list[str]:
    return [f"pod-{index}.svc" for index in range(count)]


def _slot(pod: PodPlan, role_kind: str) -> RoleSlot:
    """Narrow a pod's role slot, failing with the pod that lacked it."""
    slot = pod.slot(role_kind)
    assert slot is not None, f"pod {pod.index} holds no {role_kind} slot"
    return slot


# -- layout parsing ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2A2F", "2A2F"),
        ("2A0F,0A2F", "2A0F,0A2F"),
        ("2A,2F", "2A0F,0A2F"),
        ("1a1f,1A1F", "1A1F,1A1F"),
        (" 2A0F , 0A1F , 0A1F ", "2A0F,0A1F,0A1F"),
    ],
)
def test_layout_parse_normalises_shorthands(text, expected):
    """Shorthand and spacing variants all denote the same canonical layout."""
    assert PodLayout.parse(text).canonical() == expected


@pytest.mark.parametrize(
    "text",
    ["", "0A0F", "2A0F,,0A2F", "2A0F,0A2X", "2", "A2F", "-1A0F", "2A0F,0A0F"],
)
def test_layout_parse_rejects_malformed_entries(text):
    """A layout that cannot describe real work is rejected, not silently accepted."""
    with pytest.raises(ValueError):
        PodLayout.parse(text)


def test_layout_reports_totals_and_leaders():
    """A layout knows its pod count, rank totals, and each role's leading pod."""
    layout = PodLayout.parse("2A0F,0A1F,0A1F")

    assert layout.num_pods == 3
    assert layout.total_ranks(ATTENTION_ROLE) == 2
    assert layout.total_ranks(FFN_ROLE) == 2
    assert layout.leader_index(ATTENTION_ROLE) == 0
    assert layout.leader_index(FFN_ROLE) == 1


def test_layout_leader_index_is_none_without_the_role():
    """A role the layout never places has no leading pod."""
    assert PodLayout.parse("4A0F").leader_index(FFN_ROLE) is None


# -- layout validation ---------------------------------------------------


def test_validate_layout_rejects_a_rank_count_mismatch():
    """A layout not matching the scenario's rank counts is rejected before launch."""
    with pytest.raises(ValueError, match="scenario requires 4A/4F"):
        validate_layout(_topology(4, 4), PodLayout.parse("2A0F,0A2F"))


def test_validate_layout_rejects_a_tp_group_spanning_pods():
    """A tensor-parallel group is never split across pods."""
    with pytest.raises(ValueError, match="TP group cannot span pods"):
        validate_layout(
            _topology(2, 2, ffn_tp=2),
            PodLayout.parse("2A0F,0A1F,0A1F"),
        )


def test_validate_layout_rejects_ffn_ranks_in_a_baseline_scenario():
    """A baseline scenario cannot be given FFN ranks."""
    with pytest.raises(ValueError, match="scenario requires 4A/0F"):
        validate_layout(
            _topology(4, 0, baseline=True),
            PodLayout.parse("2A0F,2A2F"),
        )


def test_validate_layout_rejects_a_baseline_split_over_pods():
    """A native baseline has no cross-pod placement, so it stays on one pod."""
    with pytest.raises(ValueError, match="baseline scenarios must run on one pod"):
        validate_layout(
            _topology(4, 0, baseline=True),
            PodLayout.parse("2A0F,2A0F"),
        )


def test_validate_layout_accepts_a_one_pod_baseline():
    """The one-pod baseline remains the control case."""
    validate_layout(_topology(4, 0, baseline=True), PodLayout.parse("4A0F"))


def test_reject_unsupported_scenario_rejects_dsv4():
    """DSV4 fixes its own placement flags, which multi-pod slots would conflict with."""
    with pytest.raises(ValueError, match="not supported by the multi-pod runner"):
        reject_unsupported_scenario(DSV4_ASYNC_CAM_SCENARIO)


def test_reject_unsupported_scenario_accepts_afd_scenarios():
    reject_unsupported_scenario("afd-graph-2a2f")


# -- placement -----------------------------------------------------------


def test_plan_places_a_role_split_over_two_pods():
    """A role confined to one pod keeps full local DP and no cross-pod placement."""
    pods = plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(2))

    attention = _slot(pods[0], ATTENTION_ROLE)
    ffn = _slot(pods[1], FFN_ROLE)
    assert pods[0].slot(FFN_ROLE) is None
    assert pods[1].slot(ATTENTION_ROLE) is None
    assert (attention.dp_size, attention.dp_size_local) == (2, 2)
    assert (ffn.dp_size, ffn.dp_size_local) == (2, 2)
    assert attention.spans_pods is False
    assert ffn.spans_pods is False
    assert pods[0].is_evaluator is True
    assert pods[1].is_evaluator is False


def test_plan_gives_contiguous_dp_blocks_in_pod_index_order():
    """DP ranks are allocated in contiguous blocks following pod order."""
    pods = plan(_topology(4, 4), PodLayout.parse("2A0F,2A0F,0A2F,0A2F"), _addresses(4))

    attention_starts = [
        _slot(pods[index], ATTENTION_ROLE).dp_start_rank for index in (0, 1)
    ]
    ffn_starts = [_slot(pods[index], FFN_ROLE).dp_start_rank for index in (2, 3)]
    assert attention_starts == [0, 2]
    assert ffn_starts == [0, 2]


def test_plan_marks_exactly_one_leader_per_role():
    """Exactly one pod leads each role; every other holder of it is headless."""
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        leaders = [pod.index for pod in pods if not _slot(pod, role_kind).headless]
        assert leaders == [0]


def test_plan_local_dp_sizes_sum_to_the_global_dp_size():
    """Per-pod DP shares account for the whole role, losing no ranks."""
    pods = plan(_topology(4, 4), PodLayout.parse("2A0F,1A1F,1A1F,0A2F"), _addresses(4))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        slots = [
            _slot(pod, role_kind) for pod in pods if pod.slot(role_kind) is not None
        ]
        assert sum(slot.dp_size_local for slot in slots) == slots[0].dp_size


def test_plan_dp_start_rank_is_monotonic_in_pod_index():
    """DP start ranks rise with pod index and never overlap."""
    pods = plan(_topology(4, 4), PodLayout.parse("1A1F,1A1F,1A1F,1A1F"), _addresses(4))

    for role_kind in (ATTENTION_ROLE, FFN_ROLE):
        starts = [_slot(pod, role_kind).dp_start_rank for pod in pods]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)


@pytest.mark.parametrize(
    ("layout", "expected_afd_host_pod"),
    [
        ("2A2F", 0),
        ("2A0F,0A2F", 1),
        ("1A1F,1A1F", 0),
        ("2A0F,0A1F,0A1F", 1),
    ],
)
def test_plan_points_afd_host_at_the_first_ffn_rank(layout, expected_afd_host_pod):
    """Every pod agrees the AFD rendezvous is where FFN rank 0 lives."""
    parsed = PodLayout.parse(layout)
    addresses = _addresses(parsed.num_pods)

    pods = plan(_topology(), parsed, addresses)

    for pod in pods:
        for slot in pod.slots:
            assert slot.afd_host == addresses[expected_afd_host_pod]


def test_plan_points_the_async_connector_at_the_first_attention_rank():
    """The async CAM connector rendezvouses at Attention rather than FFN."""
    parsed = PodLayout.parse("2A0F,0A2F")
    addresses = _addresses(2)

    pods = plan(_topology(connector=ASYNC_CONNECTOR), parsed, addresses)

    assert _slot(pods[1], FFN_ROLE).afd_host == addresses[0]


def test_plan_gives_a_baseline_scenario_no_afd_host():
    """A baseline scenario runs with no AFD rendezvous at all."""
    pods = plan(_topology(4, 0, baseline=True), PodLayout.parse("4A0F"), _addresses(1))

    slot = _slot(pods[0], ATTENTION_ROLE)
    assert slot.role == "baseline"
    assert slot.afd_host == ""


def test_plan_assigns_disjoint_local_devices_within_a_pod():
    """Roles sharing a pod are given non-overlapping local devices."""
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    for pod in pods:
        devices = pod.devices
        assert devices == ("0", "1")
        assert len(set(devices)) == len(devices)
        assert _slot(pod, ATTENTION_ROLE).devices == ("0",)
        assert _slot(pod, FFN_ROLE).devices == ("1",)


def test_plan_numbers_devices_from_zero_in_an_ffn_only_pod():
    """A pod numbers devices from its own zero, not from a global index."""
    pods = plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(2))

    assert _slot(pods[1], FFN_ROLE).devices == ("0", "1")


def test_plan_splits_ffn_across_pods():
    """A role can span pods with one leader and the rest headless."""
    pods = plan(_topology(), PodLayout.parse("2A0F,0A1F,0A1F"), _addresses(3))

    ffn_pods = [pod.index for pod in pods if pod.slot(FFN_ROLE) is not None]
    assert ffn_pods == [1, 2]
    assert _slot(pods[1], FFN_ROLE).headless is False
    assert _slot(pods[2], FFN_ROLE).headless is True
    assert _slot(pods[2], FFN_ROLE).dp_start_rank == 1
    assert _slot(pods[1], FFN_ROLE).spans_pods is True


def test_plan_uses_distinct_dp_rpc_ports_per_role():
    """Two roles sharing a pod cannot collide on a DP RPC port."""
    pods = plan(_topology(), PodLayout.parse("1A1F,1A1F"), _addresses(2))

    attention_port = _slot(pods[0], ATTENTION_ROLE).dp_rpc_port
    ffn_port = _slot(pods[0], FFN_ROLE).dp_rpc_port
    assert attention_port != ffn_port


def test_plan_rejects_an_address_count_that_does_not_match_the_layout():
    """Planning requires exactly one address per pod."""
    with pytest.raises(ValueError, match="needs 2 addresses"):
        plan(_topology(), PodLayout.parse("2A0F,0A2F"), _addresses(3))


def test_plan_rejects_a_layout_without_attention_ranks():
    """A layout with no Attention ranks has no evaluator and is rejected."""
    with pytest.raises(ValueError, match="places no Attention ranks"):
        plan(_topology(0, 2), PodLayout.parse("0A2F"), _addresses(1))


def test_plan_is_deterministic_regardless_of_evaluation_order():
    """Every pod derives an identical plan, which is what removes the master."""
    topology = _topology(4, 4)
    layout = PodLayout.parse("2A0F,1A1F,1A1F,0A2F")
    addresses = _addresses(4)

    reference = plan(topology, layout, addresses)
    for pod_index in (3, 1, 0, 2):
        assert plan(topology, layout, addresses)[pod_index] == reference[pod_index]


# -- the equivalence property -------------------------------------------


def _single_host_args(scenario: str = "afd-graph-2a2f") -> argparse.Namespace:
    args = argparse.Namespace(
        model="deepseek-ai/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="pod-0.svc",
        afd_port=1239,
        served_model_name_prefix="deepseek-v2-lite-afd",
        scenario=scenario,
        device_backend="gpu",
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        use_decode_bench_connector=False,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        gsm8k_output_path="/tmp/gsm8k",
    )
    runner.configure_scenario(args)
    return args


@pytest.mark.parametrize("role", [ATTENTION_ROLE, FFN_ROLE])
def test_one_pod_layout_reproduces_the_single_host_command(role):
    """A one-pod layout must be a strict generalisation, not a variant."""
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("2A2F"), ["pod-0.svc"])

    expected = runner.build_vllm_command(args, role=role)
    actual = runner.build_vllm_command(args, role=role, slot=_slot(pods[0], role))

    assert actual == expected


def test_a_split_role_adds_exactly_the_five_placement_flags():
    """A role spanning pods gains DP placement flags, with only its leader serving."""
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("1A1F,1A1F"), ["pod-0.svc", "pod-1.svc"])

    leader = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[0], ATTENTION_ROLE),
    )
    follower = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[1], ATTENTION_ROLE),
    )

    for command in (leader, follower):
        assert command[command.index("--data-parallel-size") + 1] == "2"
        assert command[command.index("--data-parallel-size-local") + 1] == "1"
        assert command[command.index("--data-parallel-address") + 1] == "pod-0.svc"
        assert "--data-parallel-rpc-port" in command
    assert leader[leader.index("--data-parallel-start-rank") + 1] == "0"
    assert follower[follower.index("--data-parallel-start-rank") + 1] == "1"
    assert "--headless" not in leader
    assert "--headless" in follower


def test_a_headless_slot_binds_no_api_server():
    """A headless slot reserves no API port it would never use."""
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("1A1F,1A1F"), ["pod-0.svc", "pod-1.svc"])

    follower = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[1], ATTENTION_ROLE),
    )

    assert "--host" not in follower
    assert "--port" not in follower


def test_the_slot_supplies_the_resolved_afd_host():
    """The launched command carries the layout-resolved AFD host, not the default."""
    args = _single_host_args()
    topology = Topology.from_args(args)
    pods = plan(topology, PodLayout.parse("2A0F,0A2F"), ["pod-0.svc", "pod-1.svc"])

    command = runner.build_vllm_command(
        args,
        role=ATTENTION_ROLE,
        slot=_slot(pods[0], ATTENTION_ROLE),
    )
    additional_config = json.loads(command[command.index("--additional-config") + 1])

    assert additional_config["afd"]["host"] == "pod-1.svc"


def test_topology_from_args_tracks_the_scenario():
    """The logical topology, TP sizes and rendezvous role all follow the scenario."""
    topology = Topology.from_args(_single_host_args("afd-v2-graph-tp2"))

    assert topology.attention == RoleTopology(ranks=2, tp_size=2)
    assert topology.ffn == RoleTopology(ranks=2, tp_size=2)
    assert topology.rendezvous_role == FFN_ROLE


def test_build_env_merges_cluster_specific_variables():
    """Cluster variables reach the launched process, leaving device selection intact."""
    args = _single_host_args()

    environment = runner.build_env(
        "0,1",
        args,
        role=ATTENTION_ROLE,
        extra_env={"NCCL_SOCKET_IFNAME": "eth0"},
    )

    assert environment["NCCL_SOCKET_IFNAME"] == "eth0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"


# -- pod identity and pre-flight ----------------------------------------


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"AFD_E2E_POD_INDEX": "2", "JOB_COMPLETION_INDEX": "1"}, 2),
        ({"JOB_COMPLETION_INDEX": "1", "HOSTNAME": "afd-e2e-abc-3"}, 1),
        ({"HOSTNAME": "afd-e2e-abc-3"}, 3),
    ],
)
def test_resolve_pod_index_precedence(environment, expected):
    """A pod takes its identity from the highest-priority source available."""
    assert identity.resolve_pod_index(4, environment=environment) == expected


def test_resolve_pod_index_fails_when_nothing_identifies_the_pod():
    """A pod that cannot identify itself fails before launching anything."""
    with pytest.raises(RuntimeError, match="cannot resolve this pod's index"):
        identity.resolve_pod_index(4, environment={"HOSTNAME": "worker"})


def test_resolve_pod_index_rejects_an_index_outside_the_layout():
    """An identity outside the layout is rejected."""
    with pytest.raises(RuntimeError, match="outside 0..1"):
        identity.resolve_pod_index(2, environment={"AFD_E2E_POD_INDEX": "2"})


def test_find_stale_run_markers_reports_only_other_runs(tmp_path: Path):
    """Leftovers from an earlier run are reported; this run's own processes are not."""

    def _write(pid: str, value: str) -> None:
        entry = tmp_path / pid
        entry.mkdir()
        (entry / "environ").write_bytes(b"PATH=/usr/bin\0" + value.encode() + b"\0")

    _write("101", "AFD_E2E_RUN_ID=old-run-pod0")
    _write("102", "AFD_E2E_RUN_ID=this-run-pod1")
    _write("103", "UNRELATED=1")
    (tmp_path / "self").mkdir()

    survivors = identity.find_stale_run_markers("this-run", proc_root=tmp_path)

    assert survivors == ["pid 101: AFD_E2E_RUN_ID=old-run-pod0"]


def test_find_stale_run_markers_is_empty_without_a_proc_filesystem(tmp_path: Path):
    """The pre-flight stays silent where processes cannot be inspected."""
    assert identity.find_stale_run_markers("run", proc_root=tmp_path / "absent") == []


def test_wait_for_address_returns_once_the_record_appears():
    """Waiting for a peer name tolerates DNS that has not propagated yet."""
    attempts = []

    def resolve(host: str) -> object:
        attempts.append(host)
        if len(attempts) < 3:
            raise OSError("Name or service not known")
        return object()

    identity.wait_for_address(
        "afd-e2e-0.afd-e2e",
        timeout_s=5,
        poll_interval_s=0,
        resolve=resolve,
    )

    assert attempts == ["afd-e2e-0.afd-e2e"] * 3


def test_wait_for_address_reports_the_host_it_could_not_resolve():
    """A name that never resolves fails with the host named."""

    def never(_host: str) -> object:
        raise OSError("Name or service not known")

    with pytest.raises(RuntimeError, match="afd-e2e-0.afd-e2e did not resolve"):
        identity.wait_for_address(
            "afd-e2e-0.afd-e2e",
            timeout_s=0,
            poll_interval_s=0,
            resolve=never,
        )


def test_parse_key_values_rejects_a_bare_token():
    """A malformed KEY=VALUE option is rejected, naming the option at fault."""
    with pytest.raises(ValueError, match="--pod-env expects KEY=VALUE"):
        identity.parse_key_values(["NCCL_DEBUG"], option="--pod-env")


def test_pod_spec_rejects_an_unknown_role():
    """An unknown AFD role is rejected rather than silently counted as zero."""
    with pytest.raises(ValueError, match="unknown AFD role"):
        PodSpec(attention=1, ffn=1).ranks("decode")
