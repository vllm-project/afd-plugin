# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build the experimental device binary on CPU; never initialize a device."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

CPU_WINDOW_SECONDS = 600
BUILD_JOBS = 2
KILL_GRACE_SECONDS = 5
MAX_BUILD_SECONDS = CPU_WINDOW_SECONDS - 2 * KILL_GRACE_SECONDS
POLL_SECONDS = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--soc-version",
        choices=("Ascend910_9392", "Ascend910_9382"),
        default="Ascend910_9392",
    )
    parser.add_argument(
        "--toolkit", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest")
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    budget_path = build / "budget.json"
    budget = (
        json.loads(budget_path.read_text())
        if budget_path.exists()
        else {"started_monotonic": time.monotonic(), "attempts": []}
    )
    deadline = budget["started_monotonic"] + MAX_BUILD_SECONDS
    attempt = len(budget["attempts"]) + 1
    # The SDK preprocessing/link steps rewrite intermediate objects in place.
    # A completed artifact is not a reusable compiler input on the next build.
    artifacts = build / f"attempt-{attempt}-objects"
    record: dict = {
        "attempt": attempt,
        "soc_version": args.soc_version,
        "artifact_directory": str(artifacts),
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.is_file()
        },
        "steps": [],
        "device_execution": False,
    }
    budget["attempts"].append(record)
    budget_path.write_text(json.dumps(budget, indent=2) + "\n")
    env = {
        **os.environ,
        "CMAKE_BUILD_PARALLEL_LEVEL": str(BUILD_JOBS),
        "MAX_JOBS": str(BUILD_JOBS),
    }
    pending_signal: int | None = None

    def cancel(signum, frame):
        nonlocal pending_signal
        pending_signal = signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancel)
    commands = (
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(artifacts),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DSOC_VERSION={args.soc_version}",
            f"-DASCEND_CANN_PACKAGE_PATH={args.toolkit}",
        ],
        [
            "cmake",
            "--build",
            str(artifacts),
            "--parallel",
            str(BUILD_JOBS),
            "--target",
            "ffn_faas_mixed_probe",
        ],
        [
            "cmake",
            "--build",
            str(artifacts),
            "--parallel",
            str(BUILD_JOBS),
            "--target",
            "probe_tiling",
        ],
        [str(artifacts / "probe_tiling"), str(artifacts / "tiling.bin")],
    )
    for index, command in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the cumulative CPU build budget is exhausted")
        log = build / f"attempt-{attempt}-step-{index}.log"
        step: dict = {"command": command, "log": str(log), "started": time.monotonic()}
        record["steps"].append(step)
        try:
            with log.open("w") as output:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                step["pid"] = process.pid
                budget_path.write_text(json.dumps(budget, indent=2) + "\n")
                try:
                    while True:
                        if pending_signal is not None:
                            raise SystemExit(128 + pending_signal)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                "the cumulative CPU build deadline expired"
                            )
                        try:
                            step["exit_code"] = process.wait(
                                timeout=min(POLL_SECONDS, remaining)
                            )
                            break
                        except subprocess.TimeoutExpired:
                            continue
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=KILL_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        finally:
            step["finished"] = time.monotonic()
            budget_path.write_text(json.dumps(budget, indent=2) + "\n")
        if step["exit_code"] != 0:
            raise RuntimeError(f"CPU build failed; inspect {log}")
    print(
        json.dumps(
            {"compiled": True, "device_execution": False, "build": str(artifacts)}
        )
    )


if __name__ == "__main__":
    main()
