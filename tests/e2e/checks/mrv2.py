# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-side MRV2 probe preparation, per-phase windows and evidence checks."""

from __future__ import annotations

import json
import re
import tempfile
import time
import urllib.request
from pathlib import Path

from tests.e2e.case import (
    Case,
    CaseEnvironment,
    HashInputCheck,
    Mrv2Check,
    Phase,
    PullMegaCheck,
    Role,
)
from tests.e2e.deployment import ServiceDeployment
from tests.e2e.process_utils import (
    close_process_identities,
    find_processes_matching_environment,
    kill_processes_matching_environment,
)

MRV2_EVIDENCE_RPC_TIMEOUT_S = 30
MRV2_EVIDENCE_REPLAY_MIN_COUNT = 2
PROCESS_POLL_INTERVAL_S = 0.2
DBO_NUM_STAGES = 2


def uses_pull_mega(case: Case) -> bool:
    return any(
        isinstance(check, PullMegaCheck)
        for phase in case.phases
        for check in phase.checks
    )


def probe_extension(case: Case) -> str:
    if uses_pull_mega(case):
        return "tests.e2e.models.deepseek_v4_flash.pull_mega_evidence.Worker"
    return "tests.e2e.mrv2_evidence.Worker"


class Mrv2Session:
    """One shared probe per deployment, with independent immutable snapshots."""

    def __init__(self, case: Case, environment: CaseEnvironment, run_id: str) -> None:
        self.case = case
        self.pull_mega = uses_pull_mega(case)
        self.environment = environment
        self.run_id = run_id
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.start_marker: Path | None = None
        self.hash_samples: dict[
            str, dict[tuple[int, int, int, int], set[tuple[int, ...]]]
        ] = {}
        layouts = dict(case.deployment.roles)
        checks = [check for phase in case.phases for check in phase.checks]
        only_pull_mega = all(isinstance(check, PullMegaCheck) for check in checks)
        for check in checks:
            if isinstance(check, PullMegaCheck):
                profile = case.deployment
                if (
                    check.roles != ("attention", "ffn")
                    or profile.backend != "npu"
                    or profile.connector != "CAMAsyncAFDConnector"
                    or dict(profile.connector_extra_config).get("ffn_mega_moe")
                    is not True
                    or profile.cleanup_profile != "strict"
                    or any(role.uses_graph for _, role in profile.roles)
                    or not only_pull_mega
                ):
                    raise ValueError(
                        "Pull Mega evidence requires eager NPU CAM Mega ON "
                        "with strict cleanup"
                    )
                continue
            if not check.roles:
                raise ValueError("MRV2 check requires at least one role")
            for role in check.roles:
                if role not in layouts:
                    raise ValueError(f"MRV2 requests absent role {role}")
                if check.graph and not layouts[role].uses_graph:
                    raise ValueError(f"graph evidence requested for eager role {role}")
                if check.dbo and case.deployment.dbo_for_role(role) is None:
                    raise ValueError(f"DBO evidence requested for non-DBO role {role}")

    def prepare(self) -> None:
        if any(phase.checks for phase in self.case.phases):
            self.directory = tempfile.TemporaryDirectory(prefix="afd-e2e-mrv2-")
            self.start_marker = Path(self.directory.name) / "evaluation-start"

    @property
    def arguments(self) -> tuple[str, ...]:
        if self.start_marker is None:
            return ()
        return ("--worker-extension-cls", probe_extension(self.case))

    @property
    def worker_environment(self) -> dict[str, str]:
        if self.start_marker is None:
            return {}
        env = {
            "VLLM_SERVER_DEV_MODE": "1",
            "AFD_E2E_MRV2_EVIDENCE_START": str(self.start_marker),
        }
        if self.case.deployment.is_baseline:
            env["AFD_E2E_NATIVE_BASELINE"] = "1"
        hash_checks = [
            check
            for phase in self.case.phases
            for check in phase.checks
            if isinstance(check, HashInputCheck)
        ]
        if hash_checks:
            env["AFD_E2E_HASH_INPUT_IDS"] = "1"
            env["AFD_E2E_HASH_REQUIRE_GRAPH"] = (
                "1" if any(check.graph for check in hash_checks) else "0"
            )
        return env

    def rpc(self, method: str, window: str | None = None) -> None:
        env = self.environment
        request = urllib.request.Request(
            f"http://{env.api_host}:{env.api_port_base}/collective_rpc",
            data=json.dumps(
                {
                    "method": method,
                    "timeout": MRV2_EVIDENCE_RPC_TIMEOUT_S,
                    "kwargs": {} if window is None else {"evaluation_window": window},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(
            request, timeout=MRV2_EVIDENCE_RPC_TIMEOUT_S
        ) as reply:
            reply.read()

    def start(self, phase: Phase) -> None:
        if not phase.checks:
            return
        assert self.start_marker is not None
        self.start_marker.with_name("evaluation-complete").unlink(missing_ok=True)
        self.rpc("afd_reset_execution_evidence", phase.name)
        pending = self.start_marker.with_suffix(".pending")
        pending.write_text(phase.name)
        pending.replace(self.start_marker)

    def finish(
        self, phase: Phase, services: ServiceDeployment
    ) -> tuple[tuple[str, str], ...]:
        if not phase.checks:
            return ()
        assert self.start_marker is not None
        self.rpc("afd_log_execution_evidence")
        self.start_marker.with_name("evaluation-complete").write_text(phase.name)
        layouts = dict(self.case.deployment.roles)
        expected = {
            (role, dp, tp)
            for check in phase.checks
            for role in check.roles
            for dp in range(layouts[role].dp)
            for tp in range(layouts[role].tp)
        }
        deadline = time.monotonic() + MRV2_EVIDENCE_RPC_TIMEOUT_S
        while True:
            services.ensure_alive()
            snapshot = tuple(
                (role, line)
                for role, line in services.execution_evidence
                if f"window={phase.name} " in line
            )
            observed = {
                (role, int(match[1]), int(match[2]))
                for role, line in snapshot
                if (match := re.search(r"dp_rank=(\d+) tp_rank=(\d+)", line))
                is not None
            }
            if expected <= observed:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"missing MRV2 ranks in {phase.name}: {sorted(expected - observed)}"
                )
            time.sleep(PROCESS_POLL_INTERVAL_S)
        path = self.environment.output_path / phase.name
        path.mkdir(parents=True, exist_ok=True)
        (path / "mrv2.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "case": self.case.id,
                    "phase": phase.name,
                    "evidence": snapshot,
                },
                indent=2,
            )
            + "\n"
        )
        for check in phase.checks:
            if isinstance(check, PullMegaCheck):
                for role, dp, tp in expected:
                    if not any(
                        recorded_role == role
                        and f"dp_rank={dp} tp_rank={tp} " in line
                        and re.search(
                            r"pull_mega=1 native_completed=[1-9][0-9]*(?: |$)", line
                        )
                        for recorded_role, line in snapshot
                    ):
                        raise RuntimeError(
                            f"{role} rank {(dp, tp)} lacks live "
                            "Pull native Mega evidence"
                        )
            elif isinstance(check, Mrv2Check):
                assert_execution_evidence(
                    list(snapshot),
                    check,
                    role_layouts=layouts,
                    backend=self.case.deployment.backend,
                    asynchronous=self.case.deployment.connector
                    == "CAMAsyncAFDConnector",
                )
            else:
                samples = assert_hash_inputs(
                    list(snapshot),
                    role_layouts=layouts,
                    graph=check.graph,
                    dbo=check.dbo,
                )
                if check.compare_with is not None:
                    assert_hash_freshness(
                        self.hash_samples[check.compare_with],
                        samples,
                        attention_ranks=layouts["attention"].rank_count,
                        dbo=check.dbo,
                    )
                self.hash_samples[phase.name] = samples
        return snapshot

    def close(self) -> None:
        if self.directory is not None:
            self.directory.cleanup()
        if not self.pull_mega:
            return
        expected = set(
            range(sum(role.rank_count for _, role in self.case.deployment.roles))
        )
        observed = set()
        offset = 0
        for role, layout in self.case.deployment.roles:
            path = self.environment.output_path / f"{role}.log"
            if path.exists():
                role_ranks = {
                    int(rank)
                    for rank in re.findall(
                        r"AFD Pull Mega shutdown: rank=(\d+) complete",
                        path.read_text(),
                    )
                }
                observed.update(
                    role_ranks & set(range(offset, offset + layout.rank_count))
                )
            offset += layout.rank_count
        owned_environment = {"AFD_E2E_RUN_ID": self.run_id}
        survivors = find_processes_matching_environment(owned_environment)
        try:
            orphan_pids = [process.pid for process in survivors]
        finally:
            close_process_identities(survivors)
        errors = []
        if orphan_pids:
            errors = kill_processes_matching_environment(
                owned_environment,
                timeout_s=self.case.deployment.termination_timeout_s,
                poll_interval_s=PROCESS_POLL_INTERVAL_S,
                process_name="Pull Mega orphan",
            )
        result = {
            "expected_ranks": sorted(expected),
            "completed_ranks": sorted(observed),
            "orphan_pids": orphan_pids,
            "cleanup_errors": errors,
            "passed": observed == expected and not orphan_pids,
        }
        (self.environment.output_path / "pull-mega-shutdown.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if orphan_pids:
            raise RuntimeError(f"Pull Mega orphan processes: {orphan_pids}; {errors}")
        if observed != expected:
            raise RuntimeError(
                f"Missing Pull Mega shutdown completion: {sorted(expected - observed)}"
            )


def assert_execution_evidence(
    evidence: list[tuple[str, str]],
    check: Mrv2Check,
    *,
    role_layouts: dict[str, Role],
    backend: str,
    asynchronous: bool = False,
) -> None:
    if not check.graph and not check.dbo:
        # Observation-only windows still require every rank in finish().
        return
    metric = (
        "two_stage_graph_replays"
        if check.dbo and check.graph
        else "two_stage_steps"
        if check.dbo
        else "graph_replays"
    )
    minimum = (
        MRV2_EVIDENCE_REPLAY_MIN_COUNT if metric == "two_stage_graph_replays" else 1
    )
    layout_kind = "consecutive_graph" if check.graph else "two_stage"
    common_layouts: set[str] | None = None
    for role in check.roles:
        tp_size = role_layouts[role].tp
        rank_count = role_layouts[role].rank_count
        expected = {
            (dp, tp) for dp in range(rank_count // tp_size) for tp in range(tp_size)
        }
        observed = set()
        layouts: dict[str, set[tuple[int, int]]] = {}
        for recorded_role, line in evidence:
            if recorded_role != role:
                continue
            match = re.search(
                rf"dp_rank=(\d+) tp_rank=(\d+).*\b{metric}=(\d+)(?: |$)", line
            )
            if match is not None and int(match[3]) >= minimum:
                rank = (int(match[1]), int(match[2]))
                observed.add(rank)
                if check.dbo and not asynchronous:
                    payload = json.loads(line.split(" layouts=", 1)[1])
                    for layout in payload[layout_kind]:
                        key = json.dumps(layout, separators=(",", ":"))
                        layouts.setdefault(key, set()).add(rank)
        if missing := expected - observed:
            raise RuntimeError(
                f"{backend.upper()} {role} lacks live "
                f"{metric}>={minimum} evidence on ranks "
                f"{sorted(missing)}"
            )
        if check.dbo and not asynchronous:
            complete_layouts = {
                layout for layout, ranks in layouts.items() if expected <= ranks
            }
            common_layouts = (
                complete_layouts
                if common_layouts is None
                else common_layouts & complete_layouts
            )
    if check.dbo and not asynchronous and not common_layouts:
        raise RuntimeError(
            f"{backend.upper()} DBO lacks matching two-stage layouts "
            "across every participating rank"
            + (" with consecutive graph replays" if check.graph else "")
        )
    print(
        f"[execution-coverage] live {metric}>={minimum} "
        "confirmed on every participating rank"
    )


def assert_hash_inputs(
    evidence: list[tuple[str, str]],
    *,
    role_layouts: dict[str, Role],
    graph: bool,
    dbo: bool,
) -> dict[tuple[int, int, int, int], set[tuple[int, ...]]]:
    """Pair bounded successful live A/F samples in original control/peer order."""
    samples: dict[str, dict[tuple[int, int, int], dict]] = {
        "attention": {},
        "ffn": {},
    }
    observed_ranks: dict[str, set[tuple[int, int]]] = {
        "attention": set(),
        "ffn": set(),
    }
    for role, line in evidence:
        match = re.search(r"dp_rank=(\d+) tp_rank=(\d+)", line)
        assert match is not None
        rank = (int(match[1]), int(match[2]))
        payload = json.loads(line.split(" layouts=", 1)[1])
        for sample in payload["hash_ids"]:
            if not sample["completed"] or sample["control"] <= 0:
                raise RuntimeError(
                    "Hash ID sample did not complete a live control transaction"
                )
            if graph and not sample["replayed"]:
                raise RuntimeError("Hash ID sample lacks completed graph replay")
            key = (sample["control"], sample["stage"], sample["peer"])
            if key in samples[role]:
                raise RuntimeError(f"Duplicate {role} Hash ID sample: {key}")
            if role == "attention" and sample["peer"] != rank[0]:
                raise RuntimeError(
                    f"Attention Hash ID peer does not match DP rank: {key}"
                )
            samples[role][key] = sample
            observed_ranks[role].add(rank)
    for role, ranks in observed_ranks.items():
        rank_count = role_layouts[role].rank_count
        expected = {(dp, 0) for dp in range(rank_count)}
        if ranks != expected:
            raise RuntimeError(
                f"{role} lacks bounded Hash ID samples on ranks {expected - ranks}"
            )
    if samples["attention"].keys() != samples["ffn"].keys():
        raise RuntimeError("A/F Hash ID samples do not pair by control/stage/peer")
    values: dict[tuple[int, int, int, int], set[tuple[int, ...]]] = {}
    covered = set()
    for key, attention in samples["attention"].items():
        ffn = samples["ffn"][key]
        real = attention["real_tokens"]
        bucket = attention["bucket"]
        if (
            ffn["bucket"] != bucket
            or len(ffn["input_ids"]) != bucket
            or len(attention["input_ids"]) != real
            or not 0 <= real <= bucket
            or attention["input_ids"] != ffn["input_ids"][:real]
        ):
            raise RuntimeError(
                f"Native A / actual Hash-layer F input IDs differ: {key}"
            )
        if real:
            _, stage, peer = key
            covered.add((peer, stage))
            values.setdefault((peer, stage, bucket, real), set()).add(
                tuple(attention["input_ids"])
            )
    stages = range(DBO_NUM_STAGES if dbo else 1)
    required = {
        (peer, stage)
        for peer in range(role_layouts["attention"].rank_count)
        for stage in stages
    }
    if missing := required - covered:
        raise RuntimeError(
            f"Hash ID samples lack real tokens for peer/stage {sorted(missing)}"
        )
    print("[hash-inputs] native A and actual Hash-layer F IDs match on every rank")
    return values


def assert_hash_freshness(
    previous: dict[tuple[int, int, int, int], set[tuple[int, ...]]],
    current: dict[tuple[int, int, int, int], set[tuple[int, ...]]],
    *,
    attention_ranks: int,
    dbo: bool,
) -> None:
    """Require new real IDs across rounds at the same padded and real size."""
    changed = {
        (peer, stage)
        for (peer, stage, bucket, real), values in current.items()
        if (peer, stage, bucket, real) in previous
        and values - previous[(peer, stage, bucket, real)]
    }
    stages = range(DBO_NUM_STAGES if dbo else 1)
    required = {(peer, stage) for peer in range(attention_ranks) for stage in stages}
    if missing := required - changed:
        raise RuntimeError(
            "Hash input IDs did not change across concurrent rounds with the same "
            f"bucket and real token count: {sorted(missing)}"
        )
    print("[hash-inputs] both concurrent windows show fresh IDs at the same bucket")
