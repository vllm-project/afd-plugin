# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded 4A/4F MRV2 generation regression: layered eager versus FFN FULL.

Functional coverage only: test observers fence eager transactions. Do not use
this fixture for performance comparisons. Inference and process cleanup are
recorded separately.

This has its own fixed DBO-disabled profile, separate from the existing
16-die eager Attention DBO case. It reuses common command/cleanup code.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import httpx

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash.config import additional_config
from tests.e2e.process_utils import terminate_process_groups

CAPACITY = 128
GENERATED_TOKENS = 8
WORKER_SHUTDOWN_TIMEOUT = 60
SERVICE_SHUTDOWN_TIMEOUT = 90
SHUTDOWN_TIMEOUT = 120


class LaunchPlan(TypedDict):
    command: list[str]
    environment: dict[str, str]


def profile(args) -> SimpleNamespace:
    return SimpleNamespace(
        scenario="afd-dsv4-flash-mrv2-ffn-graph",
        model=args.model,
        vllm_bin=args.vllm_bin,
        device_backend="npu",
        baseline=False,
        num_attention_ranks=4,
        num_ffn_ranks=4,
        attention_tp_size=4,
        ffn_tp_size=1,
        tp_size=1,
        afd_connector="CAMAsyncAFDConnector",
        afd_async=True,
        compute_gate_on_attention=True,
        afd_host=args.afd_host,
        afd_port=args.afd_port,
        api_host="127.0.0.1",
        api_port_base=args.api_port,
        served_model_name_prefix="afd449-p4",
        use_v2_model_runner=True,
        cuda_graph_full_decode_only=False,
        enable_dbo=False,
        use_decode_bench_connector=False,
        afd_connector_extra_config=[
            json.dumps(
                {
                    "dynamicQuant": 1,
                    "attn_ranks_per_dp": 4,
                    "async_moe_ubatching": False,
                    "hccl_buffer_size": 256,
                }
            )
        ],
        common_vllm_arg=[
            "--api-server-count",
            "1",
            "--seed",
            "1024",
            "--max-model-len",
            "1024",
            "--max-num-batched-tokens",
            str(CAPACITY),
            "--max-num-seqs",
            "2",
            "--block-size",
            "128",
            "--gpu-memory-utilization",
            "0.7",
            "--quantization",
            "ascend",
            "--attention_config.indexer_kv_dtype",
            "int8",
            "--tokenizer-mode",
            "deepseek_v4",
            "--trust-remote-code",
            "--enable-chunked-prefill",
            "--model-loader-extra-config",
            '{"enable_multithread_load":true,"num_threads":16}',
            "--worker-extension-cls",
            "tests.e2e.models.deepseek_v4_flash.ffn_graph_evidence.FFNGraphEvidence",
        ],
        attention_vllm_arg=[
            "--all2all-backend",
            "allgather_reducescatter",
            "--data-parallel-address",
            args.afd_host,
            "--no-disable-hybrid-kv-cache-manager",
        ],
        ffn_vllm_arg=["--all2all-backend", "flashinfer_all2allv"],
    )


def command_for(args, settings, role: str) -> list[str]:
    command = runner.build_vllm_command(settings, role=role)
    command[command.index("--shutdown-timeout") + 1] = str(SERVICE_SHUTDOWN_TIMEOUT)
    offset = command.index("--additional-config") + 1
    config = json.loads(command[offset])
    config.update(additional_config(role, async_cam=True))
    command[offset] = json.dumps(config)
    if role == "ffn" and args.mode == "full":
        command.remove("--enforce-eager")
        command.extend(["--compilation-config", '{"cudagraph_mode":"FULL"}'])
    return command


def environment_for(args, settings, role: str) -> dict[str, str]:
    devices = args.devices.split(",")
    visible = ",".join(devices[:4] if role == "attention" else devices[4:])
    env = runner.build_env(
        visible, settings, role=role, e2e_run_id=args.output_dir.name
    )
    env.update(
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_PLUGINS": (
                "ascend,afd,ascend_model,ascend_model_loader,ascend_kv_connector"
            ),
            "AFD_ASYNC_CAM_LAYERED_GMM": "1" if role == "ffn" else "0",
            "BATCH_SIZE_FACTOR": str(CAPACITY / 262144),
            "OMP_NUM_THREADS": "4",
            "OMP_PROC_BIND": "false",
            "HCCL_BUFFSIZE": "256",
            "HCCL_CONNECT_TIMEOUT": "120",
            "HCCL_EXEC_TIMEOUT": "300",
            "VLLM_RPC_TIMEOUT": "300000",
            "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": str(WORKER_SHUTDOWN_TIMEOUT),
            "HCCL_IF_IP": args.afd_host,
            "HCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",
            "TP_SOCKET_IFNAME": "eth0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "AFD_FFN_GRAPH_EVIDENCE_DIR": str(args.output_dir / "workers"),
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
            "SOC_VERSION": "910c",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "TASK_QUEUE_ENABLE": "1",
        }
    )
    # CANN caches its operator registry when the device is initialized. The
    # package paths must be inherited by CLI/EngineCore/worker processes, before
    # model loading performs any NPU operation (the P3 fixture did this early).
    vendor = runner.REPO_ROOT / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    for key, path in (
        ("ASCEND_CUSTOM_OPP_PATH", vendor),
        ("LD_LIBRARY_PATH", vendor / "op_api/lib"),
    ):
        entries = [str(path), *env.get(key, "").split(":")]
        env[key] = ":".join(dict.fromkeys(entry for entry in entries if entry))
    env["AFD_CUST_OPAPI_LIB_PATH"] = str(vendor / "op_api/lib/libcust_opapi.so")
    return env


def saved_environment(env: dict[str, str]) -> dict[str, str]:
    prefixes = (
        "AFD_",
        "VLLM_",
        "ASCEND_",
        "HCCL_",
        "OMP_",
        "PYTORCH_",
        "TORCH",
        "GLOO_",
        "TP_",
        "BATCH_",
        "TASK_QUEUE",
        "SOC_",
    )
    return {
        k: v
        for k, v in env.items()
        if k != "VLLM_API_KEY"
        and (
            k.startswith(prefixes)
            or k in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")
        )
    }


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def post(client, url: str, data: dict):
    response = client.post(url, json=data)
    response.raise_for_status()
    return response.json()


def make_requests(client, base: str, model: str) -> list[dict]:
    text = "The quick brown fox jumps over the lazy dog. " * 80
    tokens = post(client, base + "/tokenize", {"model": model, "prompt": text})[
        "tokens"
    ]
    if len(tokens) < 257:
        raise RuntimeError("Fixed prompt does not contain enough tokens")
    cases = []
    for size in (16, 128, 129, 257):
        cases.append({"name": f"tokens-{size}", "prompt": tokens[:size]})
    cases.append({"name": "concurrent-two", "prompt": [tokens[:31], tokens[5:70]]})
    for item in cases:
        item["request"] = {
            "model": model,
            "prompt": item.pop("prompt"),
            "max_tokens": GENERATED_TOKENS,
            "temperature": 0,
            "seed": 1024,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    return cases


def run_requests(client, base: str, cases: list[dict], output_dir: Path) -> list[dict]:
    results = []
    for item in cases:
        result = post(client, base + "/v1/completions", item["request"])
        save(output_dir / f"response-{item['name']}.json", result)
        choices = sorted(result["choices"], key=lambda c: c["index"])
        expected_count = 2 if isinstance(item["request"]["prompt"][0], list) else 1
        if len(choices) != expected_count or any(
            len(c.get("token_ids") or []) != GENERATED_TOKENS for c in choices
        ):
            raise RuntimeError(
                f"Incomplete token response for {item['name']}: {result}"
            )
        results.append(
            {
                "name": item["name"],
                "response": result,
                "token_ids": [c["token_ids"] for c in choices],
            }
        )
        save(output_dir / "responses.json", results)
    return results


def snapshots(client, args) -> dict:
    attention = post(
        client,
        f"http://127.0.0.1:{args.api_port}/collective_rpc",
        {"method": "afd_ffn_graph_snapshot", "timeout": 30},
    )
    # FFN EngineCore deliberately runs a connector-only busy loop. Its existing
    # periodic health RPC writes test-only host snapshots; it has no HTTP API.
    floor = time.monotonic_ns()
    deadline = time.monotonic() + 10
    while True:
        records = []
        for rank in range(4, 8):
            path = args.output_dir / "workers" / f"latest-ffn-{rank}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["observed_ns"] >= floor:
                    records.append(record)
        if len(records) == 4:
            return {"attention": attention, "ffn": {"results": records}}
        if time.monotonic() >= deadline:
            raise TimeoutError("Missing fresh FFN health snapshots")
        time.sleep(0.1)


def audit_execution(before: dict, after: dict, mode: str) -> None:
    attention = after["attention"]["results"]
    attention_before = {r["rank"]: r for r in before["attention"]["results"]}
    ffn_before = {r["rank"]: r for r in before["ffn"]["results"]}
    ffn_after = after["ffn"]["results"]
    if {r["rank"] for r in attention} != set(range(4)):
        raise AssertionError("Missing Attention rank evidence")
    if {r["rank"] for r in ffn_after} != set(range(4, 8)):
        raise AssertionError("Missing FFN rank evidence")
    for record in attention:
        if record["runner"] != "AFDNPUAttentionModelRunnerV2":
            raise AssertionError("Actual Attention runner is not MRV2")
        previous = attention_before[record["rank"]]["attention"]
        live_before = previous.get("phases", {}).get("live", 0)
        if record["attention"].get("phases", {}).get("live", 0) <= live_before:
            raise AssertionError("No continuous live Attention execution")
    for record in ffn_after:
        baseline = ffn_before[record["rank"]]
        if record["runner"] != "AsyncNPUFFNModelRunner" or record["mode"] != mode:
            raise AssertionError("FFN runner or effective execution mode changed")
        layer_ids = record["layer_ids"]
        if len(layer_ids) != 43:
            raise AssertionError("Expected all 43 checkpoint layers")
        # Eager observes every transaction; FULL samples only the final header
        # of each replay batch, which cannot establish per-layer coverage.
        if mode == "eager":
            for layer in map(str, layer_ids):
                count = record["completion"]["layers"].get(layer, 0)
                initial = baseline["completion"].get("layers", {}).get(layer, 0)
                if count <= initial:
                    raise AssertionError(f"No live FFN completion for layer {layer}")
        if mode == "full":
            if record["capture_count"] != 1:
                raise AssertionError("FFN FULL must capture exactly once")
            if record["replay_count"] <= baseline["replay_count"]:
                raise AssertionError("No request-time FFN graph replay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("eager", "full"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--afd-host", required=True)
    parser.add_argument("--afd-port", type=int, default=29880)
    parser.add_argument("--api-port", type=int, default=19280)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.devices.split(","))) != 8:
        parser.error("exactly eight distinct devices required")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "full" and args.reference_dir is None:
        parser.error("FULL requires the completed layered eager reference")
    settings = profile(args)
    plans: dict[str, LaunchPlan] = {
        role: {
            "command": command_for(args, settings, role),
            "environment": environment_for(args, settings, role),
        }
        for role in ("attention", "ffn")
    }
    save(
        args.output_dir / "launch.json",
        {
            role: {
                "command": data["command"],
                "environment": saved_environment(data["environment"]),
            }
            for role, data in plans.items()
        },
    )
    if args.plan_only:
        return
    if args.reference_dir:
        accepted = json.loads((args.reference_dir / "passed.json").read_text())
        if accepted != {"mode": "eager", "passed": True}:
            raise ValueError("Reference must be a completed layered eager run")

    received_signal: int | None = None
    cleanup_in_progress = False
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {sig: signal.getsignal(sig) for sig in handled_signals}

    def interrupt_run(signum, _frame):
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        if not cleanup_in_progress:
            raise SystemExit(128 + signum)

    for sig in handled_signals:
        signal.signal(sig, interrupt_run)
    processes = {}
    logs = []
    try:
        for role, data in plans.items():
            log = (args.output_dir / f"{role}.log").open("w")
            logs.append(log)
            processes[role] = subprocess.Popen(
                data["command"],
                env=data["environment"],
                cwd=runner.REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        save(
            args.output_dir / "pids.json",
            {role: process.pid for role, process in processes.items()},
        )
        base = f"http://127.0.0.1:{args.api_port}"
        with httpx.Client(timeout=300, trust_env=False) as client:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                runner.ensure_processes_alive(list(processes.values()))
                try:
                    ready = client.get(base + "/v1/models", timeout=2).is_success
                except httpx.HTTPError:
                    ready = False
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("A/F services did not become ready")
                time.sleep(2)
            before = snapshots(client, args)
            save(args.output_dir / "before.json", before)
            cases = (
                json.loads((args.reference_dir / "requests.json").read_text())
                if args.reference_dir
                else make_requests(client, base, "afd449-p4-attention")
            )
            save(args.output_dir / "requests.json", cases)
            results = run_requests(client, base, cases, args.output_dir)
            after = snapshots(client, args)
            save(args.output_dir / "after.json", after)
            audit_execution(before, after, args.mode)
            if args.reference_dir:
                reference = json.loads(
                    (args.reference_dir / "responses.json").read_text()
                )
                actual_tokens = [(r["name"], r["token_ids"]) for r in results]
                expected_tokens = [(r["name"], r["token_ids"]) for r in reference]
                save(
                    args.output_dir / "comparison.json",
                    {"token_ids_exact": actual_tokens == expected_tokens},
                )
                if actual_tokens != expected_tokens:
                    raise AssertionError(
                        "FULL generated tokens differ from layered eager"
                    )
        save(
            args.output_dir / "inference-passed.json",
            {"mode": args.mode, "passed": True},
        )
    finally:
        cleanup_in_progress = True
        cleanup_error: BaseException | None = None
        try:
            # Original DR is not cancellable. The fixture may terminate the owned
            # processes; this is cleanup, not a graceful-shutdown qualification.
            failures = terminate_process_groups(
                list(processes.values()),
                termination_timeout_s=SHUTDOWN_TIMEOUT,
                poll_interval_s=0.2,
                reap_timeout_s=5,
                process_name="AFD A/F",
            )
            forced = [
                item for item in failures if item.startswith("forced SIGKILL for ")
            ]
            failures = [item for item in failures if item not in forced]
            for log in logs:
                log.close()
            save(
                args.output_dir / "cleanup.json",
                {
                    "failures": failures,
                    "forced_cleanup": forced,
                    "graceful_shutdown_qualified": False,
                    "exit_codes": {
                        role: process.poll() for role, process in processes.items()
                    },
                },
            )
            if failures:
                print("Cleanup failed: " + repr(failures), file=sys.stderr)
                if sys.exc_info()[0] is None:
                    raise RuntimeError("Service cleanup failed")
        except BaseException as exc:
            cleanup_error = exc
        finally:
            for sig, previous_handler in previous_handlers.items():
                signal.signal(
                    sig,
                    signal.SIG_DFL if previous_handler is None else previous_handler,
                )
        if received_signal is not None:
            raise SystemExit(128 + received_signal) from cleanup_error
        if cleanup_error is not None:
            raise cleanup_error

    save(args.output_dir / "passed.json", {"mode": args.mode, "passed": True})


if __name__ == "__main__":
    main()
