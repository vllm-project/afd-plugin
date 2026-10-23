# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prepare/run one fixed BF16/eager V2-Lite mode; --plan-only starts nothing."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from tests.e2e.models.deepseek_v2_lite.ffn_faas_inference import (
    PROMPTS,
    inference_afd_config,
)
from tests.e2e.process_utils import (
    kill_processes_matching_environment,
    terminate_process_groups,
)

WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 7
TERMINATION_TIMEOUT_SECONDS = 40
PROCESS_REAP_SECONDS = 10
ORPHAN_CLEANUP_SECONDS = 120
POLL_SECONDS = 0.2
RUN_ID_ENV = "AFD_FFN_BENCHMARK_RUN_ID"
REPO_ROOT = Path(__file__).resolve().parents[4]


def visible_devices(value: str) -> tuple[str, ...]:
    devices = tuple(value.split(","))
    if (
        len(devices) != 2
        or len(set(devices)) != 2
        or not all(device.isdecimal() for device in devices)
    ):
        raise argparse.ArgumentTypeError("supply exactly two distinct numeric IDs")
    return devices


def launch_plan(args: argparse.Namespace) -> dict:
    inference_mode = {
        "legacy": "legacy-attention",
        "faas": "attention",
        "native": "baseline",
    }[args.mode]
    common = [
        "--trust-remote-code",
        "--dtype=bfloat16",
        "--enforce-eager",
        "--enable-expert-parallel",
        "--max-model-len=256",
        "--max-num-seqs=2",
        "--max-num-batched-tokens=32",
        "--gpu-memory-utilization=0.6",
        "--no-enable-prefix-caching",
        "--seed=0",
        "--distributed-executor-backend=mp",
        "--disable-log-stats",
    ]
    attention_command = [
        sys.executable,
        "-m",
        "tests.e2e.models.deepseek_v2_lite.ffn_faas_inference",
        inference_mode,
        "--model",
        str(args.model),
        "--port",
        str(args.port),
        "--output",
        str(args.output_dir / "inference.json"),
        "--samples-output",
        str(args.output_dir / "samples.jsonl"),
        "--warmups",
        str(WARMUP_ITERATIONS),
        "--repeats",
        str(MEASURED_ITERATIONS),
    ]
    if args.mode == "legacy":
        ffn_command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(args.model),
            "--data-parallel-size=2",
            "--tensor-parallel-size=1",
            f"--port={args.ffn_api_port}",
            "--host=127.0.0.1",
            "--shutdown-timeout=10",
            "--additional-config",
            json.dumps(inference_afd_config(inference_mode, args.port, role="ffn")),
            *common,
        ]
    elif args.mode == "faas":
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
            "--attention-ranks=2",
            "--attention-tp=2",
            "--ffn-ranks=2",
            "--devices=0,1",
            f"--port={args.port}",
            "--capacity=8",
            "--slots=2",
            "--window-mb=64",
            "--timeout=30",
            "--startup-timeout=180",
            f"--status-port={args.status_port}",
            "--status-file",
            str(args.output_dir / "ffn.json"),
        ]
    else:
        ffn_command = None
    overrides = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "90",
        "VLLM_PLUGINS": "ascend" if args.mode == "native" else "ascend,afd",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "AFD_ASYNC_MOE_LAYOUT_LOG": "0",
        "AFD_CAM_OP_IO_LOG": "0",
        "AFD_FORCE_BALANCED_TOPK_IDS": "0",
        "AFD_ASYNC_CAM_LAYERED_GMM": "0",
        "HCCL_BUFFSIZE": "64",
        "PYTHONUNBUFFERED": "1",
        RUN_ID_ENV: str(uuid.uuid4()),
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")))
        ),
    }
    return {
        "mode": args.mode,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "model": str(args.model),
        "attention_command": attention_command,
        "ffn_command": ffn_command,
        "environment_overrides": overrides,
        "environment_unset": [
            "ASCEND_LAUNCH_BLOCKING",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1",
            "VLLM_SERVER_DEV_MODE",
            "AFD_PLUGIN_EARLY_ENGINE_PATCH",
        ],
        "attention_visible_devices": ",".join(args.attention_visible_devices),
        "ffn_visible_devices": (
            None if args.mode == "native" else ",".join(args.ffn_visible_devices)
        ),
        "prompts": PROMPTS,
        "warmups": WARMUP_ITERATIONS,
        "repeats": MEASURED_ITERATIONS,
        "topology": (
            "native TP2/EP2"
            if ffn_command is None
            else (
                "A DP1/TP2; F vLLM DP2/TP1 with EP2"
                if args.mode == "legacy"
                else "A DP1/TP2; F standalone EP2/TP1"
            )
        ),
        "performance_acceptance": "pending; no numeric threshold selected",
        "timing": "synchronous LLM.generate only; all warmup/repeat samples retained",
    }


def execute(plan: dict, output_dir: Path) -> None:
    processes: list[subprocess.Popen[str]] = []
    report: dict = {"measurements_complete": False, "cleanup_failures": []}
    cleanup_in_progress = False

    def cancel(signum, frame):
        if not cleanup_in_progress:
            raise SystemExit(128 + signum)

    handlers = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with ExitStack() as stack:
            for role in ("ffn", "attention"):
                command = plan[f"{role}_command"]
                if command is None:
                    continue
                env = os.environ.copy()
                env.update(plan["environment_overrides"])
                for name in plan["environment_unset"]:
                    env.pop(name, None)
                env["ASCEND_RT_VISIBLE_DEVICES"] = plan[f"{role}_visible_devices"]
                log = stack.enter_context((output_dir / f"{role}.log").open("w"))
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
                )
            attention = processes[-1]
            while attention.poll() is None:
                if (
                    len(processes) == 2
                    and processes[0].poll() is not None
                    and (
                        processes[0].returncode != 0
                        or not (
                            plan["mode"] == "faas"
                            and (output_dir / "inference.json").exists()
                        )
                    )
                ):
                    raise RuntimeError("FFN process exited before successful A drain")
                time.sleep(POLL_SECONDS)
            if attention.returncode != 0:
                raise RuntimeError(f"Attention exited with {attention.returncode}")
            if plan["mode"] == "faas":
                if processes[0].wait(timeout=TERMINATION_TIMEOUT_SECONDS) != 0:
                    raise RuntimeError("FaaS failed normal drain")
                for rank in range(2):
                    state = json.loads(
                        (output_dir / f"ffn-rank-{rank}.json").read_text()
                    )
                    if state["phase"] != "stopped" or not state["resources_released"]:
                        raise RuntimeError(
                            f"FaaS rank {rank} did not release resources"
                        )
            inference = json.loads((output_dir / "inference.json").read_text())
            samples = inference["samples"]
            measured = [
                s["generate_seconds"] for s in samples if s["phase"] == "measured"
            ]
            if (
                len(samples) != WARMUP_ITERATIONS + MEASURED_ITERATIONS
                or len(measured) != MEASURED_ITERATIONS
            ):
                raise RuntimeError("missing fixed warmup/repeat samples")
            if any(sample["output_tokens"] != 64 for sample in samples):
                raise RuntimeError(
                    "original four-prompt/64-output-token workload changed"
                )
            first_ids = [record["token_ids"] for record in samples[0]["records"]]
            if any(
                [record["token_ids"] for record in sample["records"]] != first_ids
                for sample in samples
            ):
                raise RuntimeError("greedy outputs changed between samples")
            report.update(
                measurements_complete=True,
                seconds_min=min(measured),
                seconds_max=max(measured),
                seconds_mean=statistics.mean(measured),
                seconds_median=statistics.median(measured),
                seconds_stdev=statistics.stdev(measured),
                measured_seconds=measured,
            )
    finally:
        cleanup_in_progress = True
        try:
            report["cleanup_failures"] = terminate_process_groups(
                processes,
                termination_timeout_s=TERMINATION_TIMEOUT_SECONDS,
                poll_interval_s=POLL_SECONDS,
                reap_timeout_s=PROCESS_REAP_SECONDS,
                deferred_sigkill_pgids=tuple(p.pid for p in processes),
                process_name="fixed FFN benchmark",
            )
            report["cleanup_failures"].extend(
                kill_processes_matching_environment(
                    {RUN_ID_ENV: plan["environment_overrides"][RUN_ID_ENV]},
                    timeout_s=ORPHAN_CLEANUP_SECONDS,
                    poll_interval_s=POLL_SECONDS,
                    process_name="fixed FFN benchmark",
                )
            )
        finally:
            for sig, previous in handlers.items():
                signal.signal(sig, signal.SIG_DFL if previous is None else previous)
            report["exit_codes"] = [p.poll() for p in processes]
            (output_dir / "driver.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["cleanup_failures"]:
        raise RuntimeError(report["cleanup_failures"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "faas", "native"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attention-visible-devices", type=visible_devices, required=True
    )
    parser.add_argument("--ffn-visible-devices", type=visible_devices)
    parser.add_argument("--port", type=int, default=29660)
    parser.add_argument("--ffn-api-port", type=int, default=29860)
    parser.add_argument("--status-port", type=int, default=29760)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.mode != "native" and args.ffn_visible_devices is None:
        parser.error("AFD modes require --ffn-visible-devices")
    if args.mode == "native" and args.ffn_visible_devices is not None:
        parser.error("native TP2 uses only --attention-visible-devices")
    selected = (*args.attention_visible_devices, *(args.ffn_visible_devices or ()))
    if len(set(selected)) != len(selected):
        parser.error("A and F must use disjoint devices")
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if inherited and not set(selected).issubset(inherited.split(",")):
        parser.error(
            "selected runtime IDs must belong to the inherited allocation mask"
        )
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.plan_only:
        if not args.model.is_dir():
            parser.error("model must be an existing local BF16 V2-Lite checkpoint")
        if (args.model / "quant_model_description.json").exists():
            parser.error("the fixed comparison requires the unquantized checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = launch_plan(args)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.plan_only:
        execute(plan, args.output_dir)


if __name__ == "__main__":
    main()
