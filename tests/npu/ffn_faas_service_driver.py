# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded, serial launcher for real FFN service/client acceptance cases."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def status(port: int) -> dict | None:
    try:
        with urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
            return json.load(response)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attention-ranks", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=156)
    parser.add_argument("--tp-rows-aligned", action="store_true")
    parser.add_argument(
        "--case",
        choices=[
            "normal",
            "idle-stop",
            "inflight-stop",
            "killed-peer",
            "lost-store",
            "signal-stop",
        ],
        default="normal",
    )
    parser.add_argument("--port", type=int, default=29640)
    parser.add_argument("--status-port", type=int, default=29740)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        ["npu-smi", "info"], stdout=(out / "health-pre.log").open("w"), check=True
    )
    shared = [
        "--attention-ranks",
        str(args.attention_ranks),
        "--attention-tp",
        "2",
        "--ffn-ranks",
        "2",
        "--port",
        str(args.port),
    ]
    ffn_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "afd_plugin.ffn_faas",
        "serve",
        str(args.model),
        *shared,
        "--devices",
        f"{args.attention_ranks},{args.attention_ranks + 1}",
        "--capacity",
        "8",
        "--slots",
        "2",
        "--timeout",
        "10",
        "--startup-timeout",
        "120",
        "--status-port",
        str(args.status_port),
        "--status-file",
        str(out / "ffn.json"),
    ]
    attention_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        *(["--monitor-interval=30"] if args.case == "lost-store" else []),
        f"--nproc-per-node={args.attention_ranks}",
        "-m",
        "tests.npu.ffn_faas_service",
        "--model",
        str(args.model),
        *shared,
        "--rounds",
        str(args.rounds),
        "--mode",
        args.case,
        "--output-dir",
        str(out / "attention"),
    ]
    report = {
        "ffn_command": ffn_command,
        "attention_command": attention_command,
        "case": args.case,
        "management_observations": [],
        "tp_rows_aligned": args.tp_rows_aligned,
    }
    if args.tp_rows_aligned:
        attention_command.append("--tp-rows-aligned")
    processes: list[subprocess.Popen] = []
    started = time.monotonic()
    passed = False
    try:
        with (
            (out / "ffn.log").open("w") as ffn_log,
            (out / "attention.log").open("w") as attention_log,
        ):
            ffn = subprocess.Popen(
                ffn_command,
                stdout=ffn_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(ffn)
            attention = subprocess.Popen(
                attention_command,
                stdout=attention_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(attention)
            report.update(
                ffn_launcher_pid=ffn.pid, attention_launcher_pid=attention.pid
            )
            deadline = time.monotonic() + 150
            triggered = args.case == "normal"
            last_phases = None
            while attention.poll() is None:
                observations = [status(args.status_port + rank) for rank in range(2)]
                phases = [s["phase"] if s else None for s in observations]
                if phases != last_phases:
                    report["management_observations"].append(
                        {"elapsed": time.monotonic() - started, "states": observations}
                    )
                    last_phases = phases
                armed = all(
                    (out / "attention" / f"armed-{rank}").exists()
                    for rank in range(args.attention_ranks)
                )
                if not triggered and armed:
                    assert phases == ["ready", "ready"], phases
                    target = observations[0]
                    assert target is not None
                    fault = {
                        "case": args.case,
                        "target_ffn_rank": 0,
                        "target_pid": target["pid"],
                        "requested_monotonic": time.monotonic(),
                    }
                    if args.case == "lost-store":
                        victim = json.loads((out / "attention" / "armed-0").read_text())
                        fault.update(
                            target_role="attention-store-owner",
                            target_rank=0,
                            target_pid=victim["pid"],
                        )
                        fault.pop("target_ffn_rank")
                        os.kill(int(victim["pid"]), signal.SIGKILL)
                    elif args.case == "killed-peer":
                        os.kill(int(target["pid"]), signal.SIGKILL)
                    elif args.case == "signal-stop":
                        os.kill(int(target["pid"]), signal.SIGTERM)
                    else:
                        request = Request(
                            f"http://127.0.0.1:{args.status_port}/stop", method="POST"
                        )
                        with urlopen(request, timeout=2) as response:
                            assert response.status == 202
                    fault["confirmed_monotonic"] = time.monotonic()
                    trigger_file = out / "fault-triggered.pending"
                    trigger_file.write_text(json.dumps(fault, indent=2) + "\n")
                    trigger_file.replace(out / "fault-triggered.json")
                    report["fault_trigger_seconds"] = time.monotonic() - started
                    triggered = True
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "service acceptance exceeded its bounded deadline"
                    )
                time.sleep(0.05)
            report["attention_exit_code"] = attention.returncode
            report["ffn_exit_code"] = ffn.wait(timeout=30)
            assert triggered
            report_ranks = (
                range(1, args.attention_ranks)
                if args.case == "lost-store"
                else range(args.attention_ranks)
            )
            a_reports = [
                json.loads((out / "attention" / f"rank-{rank}.json").read_text())
                for rank in report_ranks
            ]
            if args.case == "normal":
                assert attention.returncode == 0 and ffn.returncode == 0, report
                assert all(
                    r["result"] == "passed" and r["resources_released"]
                    for r in a_reports
                )
                f_reports = [
                    json.loads((out / f"ffn-rank-{rank}.json").read_text())
                    for rank in range(2)
                ]
                assert all(
                    r["phase"] == "stopped"
                    and r["resources_released"]
                    and r["completed_work"] > 2
                    for r in f_reports
                )
                # One work per agreed chunk/source, including paired warmup.
                # Count partial final device batches even on DRAINED.
                warmup_work = min(2, len(a_reports[0]["spec"]["layer_indices"]))
                attention_tp = a_reports[0]["spec"]["attention_tp"]
                expected_work = sum(
                    warmup_work
                    + sum(c["transport_chunks"] for c in entry["comparisons"])
                    for entry in a_reports
                    if entry["rank"] % attention_tp == 0
                )
                assert all(r["completed_work"] == expected_work for r in f_reports)
                for r in a_reports:
                    samples = r["memory_samples"]
                    assert len(samples) >= 2
                    # Caching allocation can stabilize above its first sample;
                    # actual live tensors must not grow with generations.
                    assert (
                        samples[-1]["allocated"]
                        <= samples[0]["allocated"] + 2 * 1024 * 1024
                    )
                    assert (
                        samples[-1]["store_keys"]
                        <= samples[0]["store_keys"] + args.attention_ranks + 2
                    )
            else:
                assert (
                    attention.returncode != 0
                    if args.case == "lost-store"
                    else attention.returncode == 0
                ) and ffn.returncode != 0, report
                assert all(r["result"] == "expected_failure" for r in a_reports)
            passed = True
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        subprocess.run(
            ["npu-smi", "info"], stdout=(out / "health-post.log").open("w"), check=True
        )
        report.update(passed=passed, elapsed_seconds=time.monotonic() - started)
        (out / "driver.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
