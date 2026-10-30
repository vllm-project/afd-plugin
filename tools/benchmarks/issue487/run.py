#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Launch the fixed-byte issue #487 experiments, once on each participating host.

The planner and summary command need only Python's standard library. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORLD_SIZE = 10
SENDERS = 8
TARGETS = 2
MAX_AIV = 48
POLL_SECONDS = 0.2
STOP_TIMEOUT_SECONDS = 5
CONTROL_MESSAGE_BYTES = 4096
ROOT = Path(__file__).resolve().parents[3]


def integer_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def stop_launcher(signum, _frame):
    # Raising lets run_case's finally terminate only its own process groups.
    raise SystemExit(128 + signum)


def placement(mode: str, location: str, sender_devices, target_devices):
    sender_host, target_host = ("S", "S") if location == "same" else ("S", "R")
    if location == "cross-fixed-receiver":
        sender_host, target_host = "R", "S"
    sender_start, target_start = (2, 0) if mode == "cs" else (0, 8)
    ranks = [
        {"rank": sender_start + i, "host": sender_host, "device": device}
        for i, device in enumerate(sender_devices)
    ] + [
        {"rank": target_start + i, "host": target_host, "device": device}
        for i, device in enumerate(target_devices)
    ]
    return sorted(ranks, key=lambda entry: entry["rank"])


def make_cases(args):
    cases = []
    for mode, location, aiv, fan_in, gmm in itertools.product(
        args.modes, args.placements, args.aivs, args.fan_ins, args.gmm
    ):
        if location == "cross-fixed-receiver" and mode != "dr":
            continue
        ranks = placement(mode, location, args.sender_devices, args.target_devices)
        case = {
            "mode": mode,
            "placement": location,
            "aiv": aiv,
            "fan_in": fan_in,
            "gmm": gmm == "on",
            "gmm_aic": args.gmm_aic,
            "rounds": args.rounds,
            "samples": args.samples,
            "profile": args.profile,
            "readback": args.readback,
            "ranks": ranks,
            "master_addr": {"S": args.sender_ip, "R": args.receiver_ip}[
                ranks[0]["host"]
            ],
            "master_port": args.base_port + len(cases),
            "group_timeout": args.group_timeout,
        }
        case["id"] = (
            f"{len(cases):03d}-{mode}-{location}-aiv{aiv}-fan{fan_in}"
            f"-gmm{args.gmm_aic if case['gmm'] else 0}"
        )
        case["oversubscribed"] = case["gmm"] and aiv + 2 * args.gmm_aic > MAX_AIV
        case["plan_sha256"] = hashlib.sha256(
            json.dumps(case, sort_keys=True).encode()
        ).hexdigest()
        cases.append(case)
    return cases


def resolve_native(args):
    if args.extension:
        extension = args.extension.resolve(strict=True)
    else:
        candidates = list((ROOT / "afd_plugin").glob("_C_ascend*.so"))
        if len(candidates) != 1:
            raise ValueError("Pass --extension /absolute/path/to/_C_ascend*.so")
        extension = candidates[0]
    vendor = args.vendor or extension.parent / "_cann_ops_custom/vendors/afd-plugin"
    vendor = vendor.resolve(strict=True)
    if not (vendor / "op_api/lib/libcust_opapi.so").is_file():
        raise ValueError(f"Missing native libcust_opapi.so in {vendor}")
    return extension, vendor


def connect_launchers(args):
    """Keep R from timing out in a cross case while S runs same-host cases."""
    if args.node == "S":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.sender_ip, args.control_port))
            listener.listen(1)
            listener.settimeout(args.case_timeout)
            connection, _ = listener.accept()
    else:
        deadline = time.monotonic() + args.case_timeout
        while True:
            try:
                connection = socket.create_connection(
                    (args.sender_ip, args.control_port), timeout=STOP_TIMEOUT_SECONDS
                )
                break
            except (ConnectionRefusedError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(POLL_SECONDS)
    connection.settimeout(args.case_timeout)
    return connection


def coordinate(connection, reader, case, phase: str):
    message = f"{case['plan_sha256']} {phase}\n".encode()
    connection.sendall(message)
    peer = reader.readline(CONTROL_MESSAGE_BYTES)
    if peer != message:
        raise RuntimeError(f"Peer disconnected or S/R plans differ: {peer!r}")


def run_case(args, case, extension: Path, vendor: Path):
    local = [entry for entry in case["ranks"] if entry["host"] == args.node]
    if not local:
        print(f"{case['id']}: no ranks on {args.node}", flush=True)
        return
    directory = args.output.resolve() / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    spec = directory / "case.json"
    spec.write_text(json.dumps(case, indent=2) + "\n")
    processes, logs = [], []
    try:
        for entry in local:
            rank = entry["rank"]
            target_helper = case["mode"] == "cs" and rank < TARGETS
            selected_vendor = args.cs_helper.resolve() if target_helper else vendor
            library = selected_vendor / "op_api/lib/libcust_opapi.so"
            if not library.is_file():
                raise ValueError(f"Missing custom-op library: {library}")
            environment = os.environ.copy()
            environment.update(
                ASCEND_RT_VISIBLE_DEVICES=str(entry["device"]),
                HCCL_IF_IP={"S": args.sender_ip, "R": args.receiver_ip}[args.node],
                GLOO_SOCKET_IFNAME=args.interface,
                HCCL_SOCKET_IFNAME=args.interface,
                OMP_NUM_THREADS="2",
                HCCL_CONNECT_TIMEOUT="120",
                HCCL_EXEC_TIMEOUT="120",
                HCCL_BUFFSIZE="512",
                BATCH_SIZE_FACTOR="0.0625",
                RANK=str(rank),
                LOCAL_RANK="0",
                WORLD_SIZE=str(WORLD_SIZE),
                AFD_CUST_OPAPI_LIB_PATH=str(library),
                # Exactly one AFD vendor per process; never mix the CS helper
                # and production kernel packages in the same worker.
                ASCEND_CUSTOM_OPP_PATH=str(selected_vendor),
                LD_LIBRARY_PATH=str(library.parent)
                + os.pathsep
                + environment.get("LD_LIBRARY_PATH", ""),
                PYTHONUNBUFFERED="1",
            )
            log = (directory / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        args.python,
                        str(Path(__file__).with_name("worker.py")),
                        "--case",
                        str(spec),
                        "--rank",
                        str(rank),
                        "--extension",
                        str(extension),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        print(f"{case['id']}: started ranks {[r['rank'] for r in local]}", flush=True)
        deadline = time.monotonic() + args.case_timeout
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Worker failed: {codes}; see {directory}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Case exceeded {args.case_timeout}s: {directory}")
            time.sleep(POLL_SECONDS)
    finally:
        # Only this launcher's children/process groups are terminated.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        for process in processes:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        for log in logs:
            log.close()


def summarize(directories):
    cases = {}
    for directory in directories:
        for path in directory.glob("*/rank*.json"):
            report = json.loads(path.read_text())
            key = report["case"]["plan_sha256"]
            ranks = cases.setdefault(key, {})
            rank = report["rank"]
            if rank in ranks:
                raise ValueError(f"Duplicate rank {rank}: {path}")
            ranks[rank] = report
    if not cases:
        raise ValueError("No rank reports found")
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "case",
            "mode",
            "placement",
            "aiv",
            "fan_in",
            "gmm_aic",
            "gmm_aiv",
            "oversubscribed",
            "profile",
            "readback",
            "mean_rank_us",
            "rank0_us",
            "min_rank_us",
            "max_rank_us",
            "measured_ranks",
            "correctness",
        ]
    )
    for ranks in sorted(
        cases.values(), key=lambda ranks: next(iter(ranks.values()))["case"]["id"]
    ):
        if set(ranks) != set(range(WORLD_SIZE)):
            raise ValueError(f"Incomplete case: have ranks {sorted(ranks)}; need 0..9")
        case = ranks[0]["case"]
        means = {
            rank: statistics.mean(report["elapsed_us"])
            for rank, report in ranks.items()
            if report["elapsed_us"]
        }
        expected = (
            {8, 9}
            if case["mode"] == "dr"
            else set(range(2, 2 + case["fan_in"]))
            if case["mode"] == "cs"
            else set(range(case["fan_in"]))
        )
        if set(means) != expected or not all(r["correctness"] for r in ranks.values()):
            raise ValueError(f"Invalid/incomplete result for {case['id']}")
        expected_samples = case["rounds"] * (
            case["samples"] * case["fan_in"] if case["mode"] == "dr" else 1
        )
        for rank in expected:
            samples = ranks[rank]["elapsed_us"]
            if len(samples) != expected_samples or any(
                not math.isfinite(value) or value <= 0 for value in samples
            ):
                raise ValueError(f"Invalid event samples for {case['id']} rank {rank}")
        writer.writerow(
            [
                case["id"],
                case["mode"],
                case["placement"],
                case["aiv"],
                case["fan_in"],
                case["gmm_aic"] if case["gmm"] else 0,
                2 * case["gmm_aic"] if case["gmm"] else 0,
                case["oversubscribed"],
                case["profile"],
                case["readback"],
                statistics.mean(means.values()),
                means.get(0, ""),
                min(means.values()),
                max(means.values()),
                len(means),
                "pass",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--node", choices=("S", "R"))
    parser.add_argument("--sender-ip")
    parser.add_argument("--receiver-ip")
    parser.add_argument(
        "--interface", help="Local interface, e.g. eth0 (may differ per host)"
    )
    parser.add_argument(
        "--sender-devices", type=integer_list, default=list(range(8, 16))
    )
    parser.add_argument("--target-devices", type=integer_list, default=[0, 1])
    parser.add_argument(
        "--modes", nargs="+", choices=("cs", "ds", "dr"), default=["cs", "ds", "dr"]
    )
    parser.add_argument(
        "--placements",
        nargs="+",
        choices=("same", "cross", "cross-fixed-receiver"),
        default=["same", "cross"],
    )
    parser.add_argument("--aivs", type=integer_list, default=[4])
    parser.add_argument("--fan-ins", type=integer_list, default=[1])
    parser.add_argument(
        "--gmm", nargs="+", choices=("off", "on"), default=["off", "on"]
    )
    parser.add_argument("--gmm-aic", type=int, choices=(20, 22), default=22)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--samples", type=int, default=16, help="DR ready groups per round"
    )
    parser.add_argument(
        "--readback", choices=("immediate", "deferred"), default="immediate"
    )
    parser.add_argument("--profile", choices=("none", "pipe", "l2"), default="none")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument(
        "--cs-helper", type=Path, help="Target-only helper vendor directory"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-port", type=int, default=29500)
    parser.add_argument("--control-port", type=int, default=29499)
    parser.add_argument("--group-timeout", type=int, default=300)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("issue487-results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.sender_ip or not args.receiver_ip:
        parser.error("--sender-ip and --receiver-ip are required")
    if args.sender_ip == args.receiver_ip:
        parser.error("S and R must be distinct hosts")
    if (
        len(args.sender_devices) != SENDERS
        or len(args.target_devices) != TARGETS
        or len(set(args.sender_devices + args.target_devices)) != WORLD_SIZE
        or min(args.sender_devices + args.target_devices) < 0
    ):
        parser.error(
            "Need eight sender and two target ordinals, all distinct/nonnegative"
        )
    if not args.aivs or set(args.aivs) - {4, 8, 48}:
        parser.error("--aivs must be drawn from 4,8,48")
    if not args.fan_ins or set(args.fan_ins) - {1, 2, 4, 8}:
        parser.error("--fan-ins must be drawn from 1,2,4,8")
    if min(args.rounds, args.samples, args.group_timeout, args.case_timeout) < 1:
        parser.error("Rounds, samples and timeouts must be positive")
    cases = make_cases(args)
    if not cases or args.base_port < 1024 or args.base_port + len(cases) > 65536:
        parser.error("Empty matrix or invalid rendezvous port range")
    if not 1024 <= args.control_port <= 65535 or (
        args.base_port <= args.control_port < args.base_port + len(cases)
    ):
        parser.error("Control port must be valid and outside the rendezvous port range")
    if args.dry_run:
        print(json.dumps(cases, indent=2))
        return
    if sys.platform != "linux":
        parser.error("Native runs require Linux; use --dry-run on other hosts")
    if not args.node or not args.interface:
        parser.error("--node and --interface are required for execution")
    if "cs" in args.modes and args.cs_helper is None:
        parser.error("CS requires --cs-helper; build it with build_cs_helper.py")
    extension, vendor = resolve_native(args)
    signal.signal(signal.SIGTERM, stop_launcher)
    needs_peer = any(case["placement"] != "same" for case in cases)
    if needs_peer:
        print("Waiting for both S/R launchers on the control port", flush=True)
        with connect_launchers(args) as connection, connection.makefile("rb") as reader:
            for case in cases:
                coordinate(connection, reader, case, "start")
                run_case(args, case, extension, vendor)
                coordinate(connection, reader, case, "done")
    else:
        for case in cases:
            run_case(args, case, extension, vendor)


if __name__ == "__main__":
    main()
