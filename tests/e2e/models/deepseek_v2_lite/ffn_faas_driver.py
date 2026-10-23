# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Serial full-model baseline/FaaS/failure acceptance with owned process groups."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def ffn_status(port: int) -> dict | None:
    try:
        with urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
            return json.load(response)
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["baseline", "normal", "killed-peer"], required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29660)
    parser.add_argument("--status-port", type=int, default=29760)
    parser.add_argument(
        "--boundaries",
        action="store_true",
        help="diagnostic first-prefill hooks; excludes performance evidence",
    )
    parser.add_argument(
        "--boundary-prompt-index", type=int, choices=range(4), default=None
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        ["npu-smi", "info"], stdout=(out / "health-pre.log").open("w"), check=True
    )
    env = os.environ.copy()
    env.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        AFD_FORCE_SPAWN_MULTIPROCESSING="1",
        VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="90",
        PYTHONUNBUFFERED="1",
    )
    env["VLLM_PLUGINS"] = "ascend" if args.case == "baseline" else "ascend,afd"
    env.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    marker = out / "inference-started.json"
    attention_command = [
        sys.executable,
        "-m",
        "tests.e2e.models.deepseek_v2_lite.ffn_faas_inference",
        "baseline" if args.case == "baseline" else "attention",
        "--model",
        str(args.model),
        "--port",
        str(args.port),
        "--output",
        str(out / "inference.json"),
        "--inference-marker",
        str(marker),
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
        "--attention-ranks",
        "2",
        "--attention-tp",
        "2",
        "--ffn-ranks",
        "2",
        "--devices",
        "2,3",
        "--port",
        str(args.port),
        "--capacity",
        "8",
        "--slots",
        "2",
        "--timeout",
        "30",
        "--startup-timeout",
        "180",
        "--status-port",
        str(args.status_port),
        "--status-file",
        str(out / "ffn.json"),
    ]
    if args.boundaries:
        attention_command.extend(["--boundary-directory", str(out / "boundaries")])
        if args.boundary_prompt_index is not None:
            attention_command.extend(
                ["--boundary-prompt-index", str(args.boundary_prompt_index)]
            )
    processes: list[subprocess.Popen] = []
    report = {
        "case": args.case,
        "attention_command": attention_command,
        "ffn_command": ffn_command if args.case != "baseline" else None,
        "environment": {
            k: env[k]
            for k in (
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "AFD_FORCE_SPAWN_MULTIPROCESSING",
                "VLLM_PLUGINS",
                "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
            )
        },
    }
    started = time.monotonic()
    passed = False
    try:
        with (
            (out / "ffn.log").open("w") as ffn_log,
            (out / "attention.log").open("w") as attention_log,
        ):
            ffn = None
            if args.case != "baseline":
                ffn = subprocess.Popen(
                    ffn_command,
                    env=env,
                    stdout=ffn_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(ffn)
            attention = subprocess.Popen(
                attention_command,
                env=env,
                stdout=attention_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(attention)
            deadline = time.monotonic() + 480
            initial_business_work = None
            triggered = False
            while attention.poll() is None:
                if args.case == "killed-peer" and not triggered and marker.exists():
                    observation = ffn_status(args.status_port)
                    assert (
                        observation is not None and observation["phase"] == "ready"
                    ), observation
                    if initial_business_work is None:
                        initial_business_work = observation["completed_work"]
                    elif observation["completed_work"] > initial_business_work:
                        fault = {
                            "target_pid": observation["pid"],
                            "target_ffn_rank": 0,
                            "monotonic": time.monotonic(),
                            "ffn_status": observation,
                            "inference_marker": json.loads(marker.read_text()),
                        }
                        os.kill(int(observation["pid"]), signal.SIGKILL)
                        (out / "fault-triggered.json").write_text(
                            json.dumps(fault, indent=2) + "\n"
                        )
                        report["fault"] = fault
                        triggered = True
                        deadline = time.monotonic() + 100
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "full-model process did not finish within its deadline"
                    )
                time.sleep(0.05)
            report["attention_exit_code"] = attention.returncode
            if ffn is not None:
                report["ffn_exit_code"] = ffn.wait(timeout=40)
            if args.case == "killed-peer":
                assert ffn is not None
                assert (
                    triggered and attention.returncode != 0 and ffn.returncode != 0
                ), report
                text = (out / "attention.log").read_text()
                assert any(
                    message in text
                    for message in (
                        "SessionPeerFailureError",
                        "NativeDeviceError",
                        "FFN FaaS worker exiting without unsafe destructors",
                    )
                ), text[-4000:]
                report["bounded_vllm_worker_failure"] = True
            else:
                assert attention.returncode == 0, report
                assert (
                    len(json.loads((out / "inference.json").read_text())["records"])
                    == 4
                )
                if ffn is not None:
                    assert ffn.returncode == 0, report
                    for rank in range(2):
                        state = json.loads((out / f"ffn-rank-{rank}.json").read_text())
                        assert (
                            state["phase"] == "stopped"
                            and state["resources_released"]
                            and state["completed_work"] > 2
                        )
            passed = True
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        subprocess.run(
            ["npu-smi", "info"], stdout=(out / "health-post.log").open("w"), check=True
        )
        report.update(passed=passed, elapsed_seconds=time.monotonic() - started)
        (out / "driver.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
