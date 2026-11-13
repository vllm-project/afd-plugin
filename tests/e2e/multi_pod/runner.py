#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""In-pod driver for a multi-pod AFD E2E run.

Every participating pod runs this same program with the same argv. Identity
comes from the environment, the plan is derived locally by a pure function, and
peers coordinate through a rendezvous store. Nothing outside the pods holds test
state, and each pod tears down only its own children.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from tests.e2e import deployment
from tests.e2e.cancellation import cancellable_run
from tests.e2e.case import Case, CaseEnvironment, JsonValue
from tests.e2e.environment import add_case_environment_arguments, case_environment
from tests.e2e.multi_pod.identity import (
    POD_ADDRESSES_ENV,
    find_stale_run_markers,
    local_address,
    parse_key_values,
    resolve_pod_index,
    wait_for_address,
)
from tests.e2e.multi_pod.layout import (
    ATTENTION_ROLE,
    FFN_ROLE,
    PodLayout,
    PodPlan,
    RoleSlot,
    Topology,
    plan,
    validate_layout,
)
from tests.e2e.multi_pod.rendezvous import (
    DEBUG_VERDICT,
    DEFAULT_STORE_PORT,
    PASS_VERDICT,
    VERDICT_KEY,
    Rendezvous,
)
from tests.e2e.runner import (
    E2E_PROCESS_ROLE_ENV,
    E2E_RUN_ID_ENV,
    LOG_THREAD_JOIN_TIMEOUT_S,
    add_scenario_arguments,
    attention_api_port,
    build_baseline_command,
    build_env,
    build_vllm_command,
    configure_scenario,
    prepare_case,
    print_command,
    process_termination_timeout,
    run_scenario_evaluation,
    start_process,
    stream_output,
    terminate_processes,
    uses_async_connector,
    uses_npu_async_process_cleanup,
)

ADDRESS_BARRIER = "addresses"
LAUNCHED_BARRIER = "launched"
SERVING_BARRIER = "serving"
VERDICT_BARRIER = "verdict"
TEARDOWN_BARRIER = "teardown"

HEALTH_POLL_INTERVAL_S = 2.0
HEALTH_REQUEST_TIMEOUT_S = 5
# On top of a peer's teardown budget: SIGKILL reaping, log thread joins and
# barrier polling skew before it departs the store.
DEPARTURE_GRACE_S = 60
ROLE_LABELS = {ATTENTION_ROLE: "ATTN", FFN_ROLE: "FFN", "baseline": "BASELINE"}


@dataclass(frozen=True)
class PodProcess:
    """A locally launched role process and the label its logs carry."""

    role: str
    label: str
    process: subprocess.Popen[str]


def main() -> int:
    args = parse_args()
    configure_scenario(args)
    # TODO: support MRV2 execution evidence across pods. The single-host runner
    # arms it through Attention RPC and a local marker file the FFN worker
    # polls, which no other pod sees; the pods would have to share the window
    # and the per-rank summaries through the rendezvous store. Until then
    # graph and DBO scenarios pass on accuracy alone, so leave the probes out.
    args.collect_mrv2_execution_evidence = False
    layout = PodLayout.parse(args.pod_layout)
    topology = Topology.from_args(args)
    validate_layout(topology, layout)
    pod_index = resolve_pod_index(layout.num_pods)

    print(
        f"[pod-{pod_index}] scenario={args.scenario} "
        f"layout={layout.canonical()} pods={layout.num_pods} "
        f"run-id={args.run_id}",
        flush=True,
    )

    # Resolve pod 0 before connecting: a DNS record that has not propagated
    # yet would otherwise surface as an unexplained store failure.
    wait_for_address(args.store_host, args.store_dns_timeout)
    rendezvous = Rendezvous(
        host=args.store_host,
        port=args.store_port,
        pod_index=pod_index,
        num_pods=layout.num_pods,
    )
    # Everything after connecting releases the store on the way out, so pod 0
    # outlives every peer's last store call whatever the outcome.
    try:
        rendezvous.verify_agreement(layout.canonical(), args.scenario)
        addresses = resolve_addresses(args, rendezvous, layout.num_pods, pod_index)
        pod_plan = plan(topology, layout, addresses)[pod_index]
        print_plan(pod_plan, addresses)

        stale = find_stale_run_markers(args.run_id)
        if stale:
            message = f"pod {pod_index}: stale processes from an earlier run: {stale}"
            rendezvous.publish_abort(message)
            raise RuntimeError(message)

        processes: list[PodProcess] = []
        log_threads: list[threading.Thread] = []

        def cleanup() -> None:
            try:
                try:
                    terminate_processes(
                        [entry.process for entry in processes],
                        termination_timeout_s=process_termination_timeout(args),
                        deferred_sigkill_pgids=deferred_sigkill_pgids(
                            args,
                            processes,
                        ),
                        force_kill_environment=(
                            {
                                E2E_RUN_ID_ENV: pod_e2e_run_id(args.run_id, pod_index),
                                E2E_PROCESS_ROLE_ENV: FFN_ROLE,
                            }
                            if uses_npu_async_process_cleanup(args)
                            else None
                        ),
                    )
                finally:
                    for thread in log_threads:
                        thread.join(timeout=LOG_THREAD_JOIN_TIMEOUT_S)
            except BaseException as exc:
                publish_quietly(rendezvous, f"phase/{pod_index}/cleanup", str(exc))
                raise
            publish_quietly(rendezvous, f"phase/{pod_index}/cleanup", "ok")

        with cancellable_run(cleanup):
            try:

                def check_alive() -> None:
                    for entry in processes:
                        if entry.process.poll() is not None:
                            raise RuntimeError(
                                f"{entry.label} exited with {entry.process.returncode}"
                            )

                def evaluate() -> str:
                    run_scenario_evaluation(args)
                    return PASS_VERDICT

                run_pod(
                    args,
                    rendezvous,
                    pod_plan,
                    launch_order=(ATTENTION_ROLE, FFN_ROLE)
                    if uses_async_connector(args)
                    else (FFN_ROLE, ATTENTION_ROLE),
                    start_slot=lambda slot: launch_slot(
                        args, pod_plan, slot, processes, log_threads
                    ),
                    check_alive=check_alive,
                    wait_ready=lambda slot, check: wait_for_health(
                        args, slot, args.serving_timeout, on_poll=check
                    ),
                    evaluate=evaluate,
                )
            except BaseException as exc:
                rendezvous.publish_abort(f"pod {pod_index}: {exc}")
                raise

        print(
            f"\n[pod-{pod_index}] E2E SCENARIO {args.scenario} "
            f"LAYOUT {layout.canonical()} PASSED",
            flush=True,
        )
    finally:
        leave_rendezvous(rendezvous, departure_timeout(args))
    return 0


def run_pod(
    args: argparse.Namespace,
    rendezvous: Rendezvous,
    pod_plan: PodPlan,
    *,
    launch_order: tuple[str, ...],
    start_slot: Callable[[RoleSlot], None],
    check_alive: Callable[[], None],
    wait_ready: Callable[[RoleSlot, Callable[[], None]], None],
    evaluate: Callable[[], str],
) -> str:
    """Phases 6-12: ordered launch, readiness, evaluation, verdict, teardown."""
    pod_index = pod_plan.index

    def assert_local_processes_alive() -> None:
        try:
            check_alive()
        except BaseException as exc:
            rendezvous.publish_abort(f"pod {pod_index}: {exc}")
            raise

    # Phases 6-7: the connector's rendezvous role launches first, and every pod
    # waits for it before the other role starts.
    first_kind, second_kind = launch_order
    for kind, barrier in ((first_kind, f"{first_kind}-launched"), (second_kind, None)):
        slot = pod_plan.slot(kind)
        if slot is not None:
            start_slot(slot)
        if barrier is not None:
            rendezvous.barrier(
                barrier,
                args.launch_timeout,
                on_poll=assert_local_processes_alive,
            )

    # Phase 8: "launched" means every pod's children are alive, not serving.
    rendezvous.set(f"phase/{pod_index}/launched", "ok")
    rendezvous.barrier(
        LAUNCHED_BARRIER,
        args.launch_timeout,
        on_poll=assert_local_processes_alive,
    )

    # Phase 9: the Attention leader answering /health transitively proves the
    # whole AFD world formed -- its API server binds only after the connector
    # rendezvous completes, which needs every FFN rank present.
    #
    # Only this one role is polled. An FFN engine core enters a busy loop and
    # never builds an API server (afd_plugin/compat/patches/engine_core.py
    # _run_ffn_busy_loop), and a headless slot starts none by construction, so
    # for both the honest assertion is liveness, published at phase 8.
    serving_slot = pod_plan.slot(ATTENTION_ROLE)
    if serving_slot is not None and not serving_slot.headless:
        wait_ready(serving_slot, assert_local_processes_alive)
    rendezvous.set(f"phase/{pod_index}/serving", "ok")
    rendezvous.barrier(
        SERVING_BARRIER,
        args.serving_timeout,
        on_poll=assert_local_processes_alive,
    )

    # Phase 10: exactly one pod evaluates, against its own local API.
    if pod_plan.is_evaluator:
        try:
            verdict = evaluate()
        except BaseException as exc:
            verdict = f"fail: {exc}"
            rendezvous.set(VERDICT_KEY, verdict)
            rendezvous.publish_abort(verdict)
            raise
        rendezvous.set(VERDICT_KEY, verdict)
    else:
        verdict = rendezvous.wait_for_key(
            VERDICT_KEY,
            args.verdict_timeout,
            on_poll=assert_local_processes_alive,
        )

    # Phase 11: every pod reads the verdict and holds it as its own result.
    rendezvous.barrier(
        VERDICT_BARRIER,
        args.verdict_timeout,
        on_poll=assert_local_processes_alive,
    )
    assert_local_processes_alive()
    print(f"[pod-{pod_index}] verdict: {verdict}", flush=True)
    if verdict not in (PASS_VERDICT, DEBUG_VERDICT):
        raise RuntimeError(f"pod {pod_index} run failed: {verdict}")

    # Phase 12: no pod tears down until every pod has passed its last liveness
    # check. Stopping a local role can kill its remote AFD peers, which a pod
    # still checking would report as a failed run. No on_poll here: from now on
    # a child exiting is expected.
    rendezvous.barrier(TEARDOWN_BARRIER, args.verdict_timeout)
    return verdict


def launch_slot(
    args: argparse.Namespace,
    pod_plan: PodPlan,
    slot: RoleSlot,
    processes: list[PodProcess],
    log_threads: list[threading.Thread],
) -> None:
    """Start this pod's process for one role."""
    command = (
        build_baseline_command(args)
        if slot.role == "baseline"
        else build_vllm_command(args, role=slot.role, slot=slot)
    )
    visible_devices = ",".join(slot.devices)
    label = f"pod-{pod_plan.index}-{ROLE_LABELS[slot.role]}"
    process_env = build_env(
        visible_devices,
        args,
        role=slot.role,
        e2e_run_id=pod_e2e_run_id(args.run_id, pod_plan.index),
        extra_env=parse_key_values(args.pod_env, option="--pod-env"),
    )
    print_command(label, command, args.device_backend, visible_devices)
    process = start_process(slot.role, command, process_env)
    processes.append(PodProcess(role=slot.role, label=label, process=process))
    log_threads.append(stream_output(label, process))
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(f"{label} exited during startup (rc={returncode})")


def wait_for_health(
    args: argparse.Namespace,
    slot: RoleSlot,
    timeout_s: float,
    *,
    on_poll: Callable[[], None],
) -> None:
    """Poll a role leader's own /health until it answers 200.

    /health is preferred over /v1/models because it reports an engine that died
    after binding, which the route table alone does not.
    """
    url = f"http://{args.api_host}:{attention_api_port(args)}/health"
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    while True:
        on_poll()
        try:
            with urllib.request.urlopen(
                url,
                timeout=HEALTH_REQUEST_TIMEOUT_S,
            ) as response:
                if response.status == 200:
                    print(f"{slot.role} API is ready at {url}", flush=True)
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{slot.role} API at {url} did not answer within "
                f"{timeout_s:.0f}s; last error={last_error!r}",
            )
        time.sleep(HEALTH_POLL_INTERVAL_S)


def deferred_sigkill_pgids(
    args: argparse.Namespace,
    processes: list[PodProcess],
) -> tuple[int, ...]:
    """Mirror the single-host runner's NPU async FFN teardown allowance."""
    if not uses_npu_async_process_cleanup(args):
        return ()
    return tuple(entry.process.pid for entry in processes if entry.role == FFN_ROLE)


def pod_e2e_run_id(run_id: str, pod_index: int) -> str:
    """This pod's own E2E run marker, shared by process launch and teardown."""
    return f"{run_id}-pod{pod_index}"


def resolve_addresses(
    args: argparse.Namespace,
    rendezvous: Rendezvous,
    num_pods: int,
    pod_index: int,
) -> list[str]:
    """Resolve every pod's address, deriving it where DNS names are stable."""
    explicit = args.pod_addresses or os.environ.get(POD_ADDRESSES_ENV, "")
    if explicit:
        addresses = [item.strip() for item in explicit.split(",") if item.strip()]
        if len(addresses) != num_pods:
            raise RuntimeError(
                f"{len(addresses)} pod addresses supplied for {num_pods} pods",
            )
        return addresses
    if args.pod_address_template:
        return [args.pod_address_template.format(index=i) for i in range(num_pods)]
    rendezvous.set(f"addr/{pod_index}", local_address())
    rendezvous.barrier(ADDRESS_BARRIER, args.launch_timeout)
    addresses = []
    for index in range(num_pods):
        address = rendezvous.get(f"addr/{index}")
        if address is None:
            raise RuntimeError(f"pod {index} did not publish an address")
        addresses.append(address)
    return addresses


def departure_timeout(args: argparse.Namespace) -> float:
    """Return how long pod 0 keeps the store up for peers to finish cleanup."""
    return process_termination_timeout(args) + DEPARTURE_GRACE_S


def leave_rendezvous(rendezvous: Rendezvous, timeout_s: float) -> None:
    """Release the store: a peer departs, pod 0 waits for every peer first.

    Best effort, like publish_quietly: the exit code already carries the result.
    """
    try:
        if rendezvous.pod_index != 0:
            rendezvous.depart()
            return
        stragglers = rendezvous.wait_for_departures(timeout_s)
        if stragglers:
            print(
                f"[pod-0] closing the store before "
                f"{[f'pod-{index}' for index in stragglers]} departed",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - the exit code carries the result
        print(f"could not leave the rendezvous: {exc}", flush=True)


def publish_quietly(rendezvous: Rendezvous, key: str, value: str) -> None:
    """Best-effort status publication; the store must never fail a teardown."""
    try:
        rendezvous.set(key, value)
    except Exception as exc:  # noqa: BLE001 - the exit code carries the result
        print(f"could not publish {key}: {exc}", flush=True)


def print_plan(pod_plan: PodPlan, addresses: list[str]) -> None:
    print(f"[pod-{pod_plan.index}] address={pod_plan.address}", flush=True)
    print(f"[pod-{pod_plan.index}] peers={addresses}", flush=True)
    print(
        f"[pod-{pod_plan.index}] evaluator={pod_plan.is_evaluator}",
        flush=True,
    )
    for slot in pod_plan.slots:
        print(
            f"[pod-{pod_plan.index}] {slot.role}: devices={','.join(slot.devices)} "
            f"dp={slot.dp_size_local}/{slot.dp_size}@{slot.dp_start_rank} "
            f"dp-address={slot.dp_address}:{slot.dp_rpc_port} "
            f"headless={slot.headless} afd-host={slot.afd_host}",
            flush=True,
        )


def run_case(
    case: Case, environment: CaseEnvironment, args: argparse.Namespace
) -> dict:
    """Coordinate an existing GPU case; no cross-pod execution probes are added."""
    if any(phase.checks for phase in case.phases):
        raise ValueError(
            "multi-pod execution checks require cross-pod evidence support"
        )
    layout = PodLayout.parse(args.pod_layout)
    topology = Topology.from_case(case)
    validate_layout(topology, layout)
    pod_index = resolve_pod_index(layout.num_pods)
    wait_for_address(args.store_host, args.store_dns_timeout)
    rendezvous = Rendezvous(
        host=args.store_host,
        port=args.store_port,
        pod_index=pod_index,
        num_pods=layout.num_pods,
    )
    try:
        rendezvous.verify_agreement(layout.canonical(), case.id)
        addresses = resolve_addresses(args, rendezvous, layout.num_pods, pod_index)
        pod_plan = plan(topology, layout, addresses)[pod_index]
        print_plan(pod_plan, addresses)
        environment = replace(
            environment, devices=environment.devices or pod_plan.devices
        )
        effective, record = prepare_case(case, environment, slots=pod_plan.slots)
        resolved = deployment.resolve(effective, environment, slots=pod_plan.slots)
        record["run_id"] = pod_e2e_run_id(args.run_id, pod_index)
        record["pod"] = {
            **asdict(pod_plan),
            "layout": layout.canonical(),
            "role_devices": dict(resolved.role_devices),
        }
        record["coverage"] = ["task_result", "local_liveness", "local_cleanup"]
        services = deployment.ServiceDeployment(
            resolved,
            run_id=record["run_id"],
            extra_environment=parse_key_values(args.pod_env, option="--pod-env"),
        )
        environment.output_path.mkdir(parents=True, exist_ok=True)

        def cleanup() -> None:
            try:
                services.close()
            except BaseException as exc:
                record["cleanup"] = {"status": "failed", "error": str(exc)}
                publish_quietly(rendezvous, f"phase/{pod_index}/cleanup", str(exc))
                raise
            else:
                record["cleanup"] = {"status": "passed"}
                publish_quietly(rendezvous, f"phase/{pod_index}/cleanup", "ok")
            finally:
                record["processes"] = [
                    {"pid": process.pid, "returncode": process.poll()}
                    for process in services.processes
                ]

        def evaluate() -> str:
            for phase in effective.phases:
                phase_record: dict[str, JsonValue] = {
                    "name": phase.name,
                    "status": "running",
                    "checks": [],
                }
                record["phases"].append(phase_record)
                phase_record["result"] = phase.task.execute(
                    base_url=f"http://{environment.api_host}:{environment.api_port_base}",
                    model_name=f"{case.served_model_name_prefix}-attention",
                    checkpoint=environment.model,
                    output_path=environment.output_path / phase.name,
                )
                phase_record["status"] = "passed"
            # This is workload/configuration result propagation, not per-rank
            # graph/DBO evidence. Every pod records what the evaluator ran.
            rendezvous.set(
                "evaluation/result",
                json.dumps(
                    {
                        "effective": record["effective"],
                        "configuration": record["configuration"],
                        "phases": record["phases"],
                    }
                ),
            )
            return (
                PASS_VERDICT if record["configuration"]["eligible"] else DEBUG_VERDICT
            )

        try:
            with cancellable_run(cleanup):
                record["execution"] = {"status": "running"}
                try:
                    verdict = run_pod(
                        args,
                        rendezvous,
                        pod_plan,
                        launch_order=resolved.launch_order,
                        start_slot=lambda slot: services.start_role(slot.role),
                        check_alive=services.ensure_alive,
                        wait_ready=lambda slot, check: services.wait_ready(
                            path="/health",
                            on_poll=check,
                        ),
                        evaluate=evaluate,
                    )
                    evaluation = json.loads(
                        rendezvous.wait_for_key(
                            "evaluation/result",
                            args.verdict_timeout,
                        )
                    )
                    record.update(evaluation)
                except BaseException as exc:
                    rendezvous.publish_abort(f"pod {pod_index}: {exc}")
                    raise
            record["execution"] = {"status": "passed"}
            record["acceptance_passed"] = verdict == PASS_VERDICT
            return record
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, SystemExit) else "failed"
            record["execution"] = {"status": status, "error": str(exc)}
            for phase_record in record["phases"]:
                if phase_record["status"] == "running":
                    phase_record.update(status=status, error=str(exc))
            raise
        finally:
            (environment.output_path / "run.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            print(
                f"[pod-{pod_index}] CASE {case.id}: {record['execution']['status']}; "
                f"acceptance_passed={record['acceptance_passed']}",
                flush=True,
            )
    except BaseException as exc:
        rendezvous.publish_abort(f"pod {pod_index}: {exc}")
        raise
    finally:
        leave_rendezvous(
            rendezvous, case.deployment.termination_timeout_s + DEPARTURE_GRACE_S
        )


def case_main(
    build_case: Callable[[str, CaseEnvironment], Case],
    case_ids: tuple[str, ...],
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run one pod of a model-declared GPU case"
    )
    add_case_environment_arguments(
        parser,
        case_ids,
        devices_required=False,
        startup_option="--serving-timeout",
    )
    add_pod_arguments(parser)
    parser.set_defaults(startup_timeout=1800)
    args = parser.parse_args(argv)
    args.serving_timeout = args.startup_timeout
    environment = case_environment(args)
    if not environment.devices and os.environ.get("CUDA_VISIBLE_DEVICES"):
        environment = replace(
            environment, devices=tuple(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        )
    case = build_case(args.scenario, environment)
    if args.describe:
        print(
            json.dumps(
                {"canonical": asdict(case), "bindings": asdict(environment)},
                default=str,
                indent=2,
            )
        )
    else:
        run_case(case, environment, args)
    return 0


def add_pod_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pod-layout",
        required=True,
        help=(
            "Comma-separated per-pod AFD rank counts, e.g. '2A0F,0A2F'. "
            "Shorthands '2A' and '2F' are accepted."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier shared by every pod of this run; tags child processes.",
    )
    parser.add_argument(
        "--store-host",
        required=True,
        help="Host of pod 0, which masters the rendezvous store.",
    )
    parser.add_argument("--store-port", type=int, default=DEFAULT_STORE_PORT)
    parser.add_argument("--store-dns-timeout", type=float, default=300)
    parser.add_argument(
        "--pod-addresses",
        default="",
        help="Explicit comma-separated pod addresses, overriding discovery.",
    )
    parser.add_argument(
        "--pod-address-template",
        default="",
        help=(
            "Format string with an {index} field that yields each pod's "
            "address, e.g. 'afd-e2e-abc-{index}.afd-e2e-abc'. Skips the "
            "address exchange barrier."
        ),
    )
    parser.add_argument(
        "--pod-env",
        action="append",
        default=[],
        help="KEY=VALUE added to every launched vLLM process environment.",
    )
    parser.add_argument("--launch-timeout", type=float, default=900)
    parser.add_argument("--verdict-timeout", type=float, default=3600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the retained NPU multi-pod entry.",
    )
    add_scenario_arguments(parser)
    add_pod_arguments(parser)
    parser.add_argument("--serving-timeout", type=float, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
