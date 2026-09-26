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
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from tests.e2e.cancellation import cancellable_run
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
    reject_unsupported_scenario,
    validate_layout,
)
from tests.e2e.multi_pod.rendezvous import (
    DEFAULT_STORE_PORT,
    PASS_VERDICT,
    VERDICT_KEY,
    Rendezvous,
)
from tests.e2e.runner import (
    ASYNC_CAM_SCENARIO,
    E2E_PROCESS_ROLE_ENV,
    E2E_RUN_ID_ENV,
    LOG_THREAD_JOIN_TIMEOUT_S,
    add_scenario_arguments,
    attention_api_port,
    build_baseline_command,
    build_env,
    build_vllm_command,
    configure_scenario,
    print_command,
    run_completion_evaluation,
    run_gsm8k_evaluation,
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

HEALTH_POLL_INTERVAL_S = 2.0
HEALTH_REQUEST_TIMEOUT_S = 5
ROLE_LABELS = {ATTENTION_ROLE: "ATTN", FFN_ROLE: "FFN", "baseline": "BASELINE"}


@dataclass(frozen=True)
class PodProcess:
    """A locally launched role process and the label its logs carry."""

    role: str
    label: str
    process: subprocess.Popen[str]


def main() -> int:
    args = parse_args()
    reject_unsupported_scenario(args.scenario)
    configure_scenario(args)
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
            run_pod(args, rendezvous, pod_plan, processes, log_threads)
        except BaseException as exc:
            rendezvous.publish_abort(f"pod {pod_index}: {exc}")
            raise

    print(
        f"\n[pod-{pod_index}] E2E SCENARIO {args.scenario} "
        f"LAYOUT {layout.canonical()} PASSED",
        flush=True,
    )
    return 0


def run_pod(
    args: argparse.Namespace,
    rendezvous: Rendezvous,
    pod_plan: PodPlan,
    processes: list[PodProcess],
    log_threads: list[threading.Thread],
) -> None:
    """Phases 6-11: ordered launch, readiness, evaluation, verdict."""
    pod_index = pod_plan.index

    def assert_local_processes_alive() -> None:
        for entry in processes:
            returncode = entry.process.poll()
            if returncode is not None:
                message = (
                    f"pod {pod_index} {entry.label} exited "
                    f"(rc={returncode}) during the run"
                )
                rendezvous.publish_abort(message)
                raise RuntimeError(message)

    # Phases 6-7: the connector's rendezvous role launches first, and every pod
    # waits for it before the other role starts.
    first_kind, second_kind = (
        (ATTENTION_ROLE, FFN_ROLE)
        if uses_async_connector(args)
        else (FFN_ROLE, ATTENTION_ROLE)
    )
    for kind, barrier in ((first_kind, f"{first_kind}-launched"), (second_kind, None)):
        slot = pod_plan.slot(kind)
        if slot is not None:
            launch_slot(args, pod_plan, slot, processes, log_threads)
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
        wait_for_health(
            args,
            serving_slot,
            args.serving_timeout,
            on_poll=assert_local_processes_alive,
        )
    rendezvous.set(f"phase/{pod_index}/serving", "ok")
    rendezvous.barrier(
        SERVING_BARRIER,
        args.serving_timeout,
        on_poll=assert_local_processes_alive,
    )

    # Phase 10: exactly one pod evaluates, against its own local API.
    if pod_plan.is_evaluator:
        try:
            if args.scenario == ASYNC_CAM_SCENARIO:
                run_completion_evaluation(args)
            else:
                run_gsm8k_evaluation(args)
        except BaseException as exc:
            verdict = f"fail: {exc}"
            rendezvous.set(VERDICT_KEY, verdict)
            rendezvous.publish_abort(verdict)
            raise
        rendezvous.set(VERDICT_KEY, PASS_VERDICT)
        verdict = PASS_VERDICT
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
    if verdict != PASS_VERDICT:
        raise RuntimeError(f"pod {pod_index} run failed: {verdict}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pod's slice of a multi-pod AFD E2E scenario.",
    )
    add_scenario_arguments(parser)
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
    parser.add_argument("--serving-timeout", type=float, default=1800)
    parser.add_argument("--verdict-timeout", type=float, default=3600)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
