#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run one isolated cross-node connector benchmark in an existing Slurm job.

Attention and FFN each use two allocated GPUs. Extra arguments after -- are
passed to prefill_skew.py. Server commands, logs and raw results are retained.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    connectors = {
        "async": "GpuAsyncAFDConnector",
        "m2n": "DeepEPM2NAFDConnector",
        "p2p": "P2pNcclAFDConnector",
    }
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=connectors)
    p.add_argument("--output", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--attention-host", required=True)
    p.add_argument("--ffn-host", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-port", type=int, default=18507)
    p.add_argument("--rendezvous-port", type=int, default=6351)
    p.add_argument(
        "--nccl-root",
        type=Path,
        default=Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/nvidia/nccl",
    )
    p.add_argument("--log-level", choices=("INFO", "WARNING"), default="WARNING")
    p.add_argument("--dbo", action="store_true")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--token-cap", type=int, default=4096)
    args, client_arguments = p.parse_known_args()
    args.client = client_arguments
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    nccl = args.nccl_root.resolve()
    hosts = {"attention": args.attention_host, "ffn": args.ffn_host}
    python = Path(sys.executable).absolute()
    env_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    env = dict(
        os.environ,
        PATH=str(python.parent) + ":" + os.environ["PATH"],
        PYTHONPATH=str(root),
        OMP_NUM_THREADS="4",
        MAX_JOBS="4",
        CUDA_HOME=env_cuda_home,
        VLLM_NCCL_SO_PATH=str(nccl / "lib/libnccl.so.2"),
        AFDEP_NCCL_ROOT=str(nccl),
        NVSHMEM_REMOTE_TRANSPORT="ibrc",
        NVSHMEM_SYMMETRIC_SIZE="4G",
        VLLM_PLUGINS="afd",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_LOGGING_LEVEL=args.log_level,
        PYTHONFAULTHANDLER="1",
        PYTHONUNBUFFERED="1",
    )
    for key in (
        "ASYNC_VALIDATION_MARKER",
        "AFD_E2E_MRV2_EVIDENCE_START",
        "AFD_ASYNC_DEBUG",
        "VLLM_TORCH_PROFILER_DIR",
    ):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("AFD_GPU_ATTENTION_PROFILER_", "AFD_GPU_FFN_PROFILER_")):
            env.pop(key)
    common = [
        str(python),
        "-X",
        "faulthandler",
        str(python.parent / "vllm"),
        "serve",
        args.model,
        "--host",
        "0.0.0.0",
        "--served-model-name",
        "prefill-bench",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--data-parallel-size",
        "2",
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--enable-ep-weight-filter",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        str(args.token_cap),
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.5",
        "--api-server-count",
        "1",
        "--shutdown-timeout",
        "120",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--uvicorn-log-level",
        "warning",
    ]
    if args.eager:
        common += ["--enforce-eager"]
    else:
        common += [
            '--compilation-config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}',
            "--cudagraph-capture-sizes",
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
        ]
    if args.dbo:
        common += [
            "--enable-dbo",
            "--dbo-decode-token-threshold",
            "2",
            "--dbo-prefill-token-threshold",
            "12",
        ]
    procs = []
    logs = []
    started = time.time()
    forced = False
    try:
        for role in ("attention", "ffn"):
            port = args.api_port if role == "attention" else args.api_port + 1
            config = {
                "afd": dict(
                    role=role,
                    connector=connectors[args.arm],
                    **{"async": args.arm == "async"},
                    compute_gate_on_attention=args.arm != "p2p",
                    num_attention_ranks=2,
                    num_ffn_ranks=2,
                    host=hosts["attention" if args.arm == "async" else "ffn"],
                    port=args.rendezvous_port,
                )
            }
            if args.arm == "m2n":
                config["afd"]["connector_extra_config"] = {
                    "expert_backend": "deepgemm",
                    "payload_format": "bf16",
                }
            pidfile = output / (role + ".pid")
            pidfile.unlink(missing_ok=True)
            argv = [
                "srun",
                "--jobid=" + args.job_id,
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=20",
                "--gres=gpu:2",
                "--nodelist=" + hosts[role],
                "bash",
                "-c",
                'ulimit -c 0; echo $$ > "$1"; shift; exec "$@"',
                "afd-benchmark",
                str(pidfile),
                *common,
                "--port",
                str(port),
                "--additional-config",
                json.dumps(config),
            ]
            (output / (role + "-command.json")).write_text(json.dumps(argv, indent=2))
            log = (output / (role + ".log")).open("w")
            logs.append(log)
            print("LAUNCH", role, " ".join(argv), flush=True)
            procs.append(
                subprocess.Popen(
                    argv,
                    env=env,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 900
        while True:
            if any(proc.poll() is not None for proc in procs):
                raise RuntimeError("Server exited during startup")
            try:
                with urllib.request.urlopen(
                    f"http://{args.attention_host}:{args.api_port}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Health readiness timeout")
            time.sleep(2)
        print("READY", args.arm, time.time() - started, flush=True)
        client = args.client[1:] if args.client[:1] == ["--"] else args.client
        command = [
            "srun",
            "--jobid=" + args.job_id,
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--gres=gpu:2",
            "--nodelist=" + hosts["attention"],
            str(python),
            str(root / "tools/benchmarks/prefill_skew.py"),
            "--output",
            str(output / "metrics"),
            "--url",
            f"http://127.0.0.1:{args.api_port}",
            *client,
        ]
        (output / "benchmark-command.json").write_text(json.dumps(command, indent=2))
        with (output / "client.log").open("w") as clientlog:
            clientproc = subprocess.Popen(
                command,
                env=env,
                cwd=root,
                stdout=clientlog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while clientproc.poll() is None:
                if any(proc.poll() is not None for proc in procs):
                    os.killpg(clientproc.pid, signal.SIGTERM)
                    raise RuntimeError("Server exited during measurement")
                time.sleep(1)
            if clientproc.returncode:
                raise RuntimeError("Benchmark client failed")
        print("MEASUREMENT_DONE", args.arm, flush=True)
    finally:
        for role, proc in zip(("attention", "ffn"), procs, strict=False):
            pidfile = output / (role + ".pid")
            if proc.poll() is None and pidfile.exists():
                pid = int(pidfile.read_text())
                subprocess.run(
                    [
                        "srun",
                        "--jobid=" + args.job_id,
                        "--overlap",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=2",
                        "--gres=gpu:2",
                        "--nodelist=" + hosts[role],
                        str(python),
                        "-c",
                        "import os,signal,sys; os.kill(int(sys.argv[1]),signal.SIGINT)",
                        str(pid),
                    ],
                    env=env,
                    timeout=30,
                    check=False,
                    stdout=subprocess.DEVNULL,
                )
        for proc in procs:
            try:
                proc.wait(timeout=150)
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
        for log in logs:
            log.close()
        (output / "cleanup.json").write_text(
            json.dumps(
                dict(forced=forced, launcher_exit_codes=[p.returncode for p in procs]),
                indent=2,
            )
        )
        if forced:
            raise RuntimeError("Forced cleanup required")


if __name__ == "__main__":
    main()
