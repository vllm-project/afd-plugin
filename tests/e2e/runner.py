#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execute model-declared Cases and retain the existing NPU CLI callers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from tests.e2e import deployment
from tests.e2e.accuracy.gsm8k import (
    _extract_gsm8k_accuracy,
    _extract_gsm8k_sample_count,
    _run_lm_eval,
)
from tests.e2e.cancellation import cancellable_run
from tests.e2e.case import (
    NPU_ASYNC_TERMINATION_TIMEOUT_S,
    NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S,
    NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S,
    WORKER_SHUTDOWN_TIMEOUT_ENV,
    Case,
    CaseEnvironment,
    JsonValue,
    Mrv2Check,
    Role,
)
from tests.e2e.checks.mrv2 import (
    Mrv2Session,
    assert_execution_evidence,
)
from tests.e2e.environment import parse_case_environment
from tests.e2e.models.deepseek_v4_flash import config as dsv4_config
from tests.e2e.models.deepseek_v4_flash.completions import evaluate_completions
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_ASYNC_CAM_SCENARIO,
    DSV4_ATTENTION_RANKS,
    DSV4_ATTENTION_TP_SIZE,
    DSV4_FFN_RANKS,
    DSV4_PROCESS_TERMINATION_TIMEOUT_S,
    DSV4_SCENARIOS,
    DSV4_SYNC_CAMP2P_SCENARIOS,
    DSV4_SYNC_SHAPES,
    sync_shape,
)
from tests.e2e.multi_pod.layout import RoleSlot
from tests.e2e.process_utils import (
    kill_processes_matching_environment,
    terminate_process_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASYNC_AFD_CONNECTOR = "CAMAsyncAFDConnector"
NPU_EAGER_DBO_SCENARIO = "afd-eager-dbo-2a2f"
NPU_EAGER_DBO_DECODE_THRESHOLD = 32
NPU_EAGER_DBO_PREFILL_THRESHOLD = 512
# Graph capture default for the scenarios that do not carry their own launch
# profile; the DSV4 synchronous profiles override it.
DEFAULT_CUDAGRAPH_CAPTURE_SIZE = 8
E2E_RUN_ID_ENV = "AFD_E2E_RUN_ID"
E2E_PROCESS_ROLE_ENV = "AFD_E2E_PROCESS_ROLE"
PROCESS_TERMINATION_TIMEOUT_S = 20
PROCESS_POLL_INTERVAL_S = 0.2
PROCESS_REAP_TIMEOUT_S = 5
# NPU async teardown: workers blocked in uninterruptible driver/HCCL teardown
# take tens of seconds to disappear after SIGKILL (measured up to ~45s on A3),
# so the NPU async path waits much longer before reporting survivors.
NPU_ASYNC_PROCESS_KILL_TIMEOUT_S = 120
LOG_THREAD_JOIN_TIMEOUT_S = 2
VLLM_SHUTDOWN_TIMEOUT_S = 10
DEFAULT_GSM8K_SAMPLE_LIMIT = 7
GSM8K_NUM_FEWSHOT = 8
GSM8K_FULL_SAMPLE_COUNT = 1319
FULL_GSM8K_TIMEOUT_S = 8 * 60 * 60
GSM8K_LIMIT_ENV = "AFD_GSM8K_LIMIT"
GSM8K_THRESHOLD_ENV = "AFD_GSM8K_THRESHOLD"
DEFAULT_GSM8K_THRESHOLD = 0.27
DBO_EVAL_NUM_CONCURRENT = 12
DBO_EVAL_MIN_SAMPLES = 2 * DBO_EVAL_NUM_CONCURRENT
DBO_EVAL_NUM_UBATCHES = 2
MRV2_EVIDENCE_RPC_TIMEOUT_S = 30
MRV2_EVIDENCE_REPLAY_MIN_COUNT = 2
MRV2_EVIDENCE_PREFIX = "AFD E2E execution: "


def main() -> int:
    args = parse_args()
    configure_scenario(args)
    attention_devices = parse_csv(args.attention_devices)
    ffn_devices = parse_csv(args.ffn_devices)
    validate_topology(args, attention_devices, ffn_devices)
    use_npu_async_process_cleanup = uses_npu_async_process_cleanup(args)
    e2e_run_id = (
        f"{os.getpid()}-{time.monotonic_ns()}"
        if use_npu_async_process_cleanup
        else None
    )

    processes: list[subprocess.Popen[str]] = []
    processes_by_role: dict[str, subprocess.Popen[str]] = {}
    log_threads: list[threading.Thread] = []
    mrv2_execution_evidence: list[tuple[str, str]] = []
    mrv2_evidence_dir: tempfile.TemporaryDirectory[str] | None = None
    mrv2_evidence_start: Path | None = None

    def cleanup() -> None:
        ffn_process = processes_by_role.get("ffn")
        deferred_sigkill_pgids = (
            (ffn_process.pid,)
            if use_npu_async_process_cleanup and ffn_process is not None
            else ()
        )
        try:
            terminate_processes(
                processes,
                termination_timeout_s=process_termination_timeout(args),
                deferred_sigkill_pgids=deferred_sigkill_pgids,
                force_kill_environment=(
                    {
                        E2E_RUN_ID_ENV: e2e_run_id,
                        E2E_PROCESS_ROLE_ENV: "ffn",
                    }
                    if e2e_run_id is not None
                    else None
                ),
            )
        finally:
            try:
                for thread in log_threads:
                    thread.join(timeout=LOG_THREAD_JOIN_TIMEOUT_S)
            finally:
                if mrv2_evidence_dir is not None:
                    mrv2_evidence_dir.cleanup()

    launch_order: tuple[tuple[str, str], ...]
    with cancellable_run(cleanup):
        if needs_mrv2_execution_evidence(args):
            mrv2_evidence_dir = tempfile.TemporaryDirectory(prefix="afd-e2e-mrv2-")
            mrv2_evidence_start = Path(mrv2_evidence_dir.name) / "evaluation-start"
        if args.baseline:
            role_devices = {"baseline": attention_devices}
            launch_order = (("baseline", "BASELINE"),)
        else:
            role_devices = {
                "attention": attention_devices,
                "ffn": ffn_devices,
            }
            launch_order = (
                ("attention", "ATTN"),
                ("ffn", "FFN"),
            )
            if not uses_async_connector(args):
                launch_order = tuple(reversed(launch_order))

        for role, label in launch_order:
            command = (
                build_baseline_command(args)
                if role == "baseline"
                else build_vllm_command(args, role=role)
            )
            visible_devices = ",".join(role_devices[role])
            process_env = build_env(
                visible_devices,
                args,
                role=role,
                e2e_run_id=e2e_run_id,
                mrv2_evidence_start=mrv2_evidence_start,
            )
            print_command(
                label,
                command,
                args.device_backend,
                visible_devices,
            )
            process = start_process(
                role,
                command,
                process_env,
            )
            processes.append(process)
            processes_by_role[role] = process
            log_threads.append(stream_output(role, process, mrv2_execution_evidence))
            ensure_alive(process, f"{label} process exited during startup")

        wait_for_openai_api(args, processes)
        ensure_processes_alive(processes)

        start_evaluation_window(args, mrv2_evidence_start, "evaluation")
        run_scenario_evaluation(args)
        finish_evaluation_window(
            args,
            mrv2_evidence_start,
            "evaluation",
            mrv2_execution_evidence,
            processes,
        )

        ensure_processes_alive(processes)

    print(f"\nE2E SCENARIO {args.scenario} PASSED")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed baseline or AFD E2E scenario.",
    )
    add_scenario_arguments(parser)
    parser.add_argument(
        "--attention-devices",
        default="0",
        help=(
            "Comma-separated device IDs for the Attention serve process. "
            "The number of devices must match Attention DP times TP."
        ),
    )
    parser.add_argument(
        "--ffn-devices",
        default="",
        help=(
            "Comma-separated device IDs for the FFN serve process. "
            "The number of devices must match FFN DP times TP."
        ),
    )
    return parser.parse_args()


def add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the scenario options every runner shares.

    Device selection is deliberately excluded: the single-host runner takes it
    from the caller, while the multi-pod runner derives it from the pod layout.
    """
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "baseline-graph",
            "afd-eager-2a1f",
            "afd-graph-2a1f",
            "afd-graph-dbo-2a1f",
            "afd-eager-2a2f",
            NPU_EAGER_DBO_SCENARIO,
            "afd-graph-2a2f",
            "afd-graph-dbo-2a2f",
            *DSV4_SCENARIOS,
        ],
        required=True,
        help="Fixed E2E scenario to run.",
    )
    parser.add_argument(
        "--completion-output-path",
        help="JSON file containing concurrent DSV4 requests and responses.",
    )
    parser.add_argument(
        "--gsm8k-output-path",
        help="Directory or file path where lm-eval writes GSM8K results.",
    )
    parser.add_argument(
        "--vllm-bin",
        default="vllm",
        help="vLLM executable to run. Defaults to 'vllm'.",
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument(
        "--api-port-base",
        type=int,
        # Shared hosts often run unrelated services on the default vLLM
        # port; E2E drivers can shift the whole port range via env.
        default=int(os.environ.get("AFD_E2E_API_PORT_BASE", "8000")),
    )
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--served-model-name-prefix",
        default="deepseek-v2-lite-afd",
        help="Prefix used for role-specific served model names.",
    )
    parser.add_argument(
        "--use-decode-bench-connector",
        action="store_true",
        help="Pass an AFDDecodeBenchConnector kv-transfer-config to Attention.",
    )
    parser.add_argument(
        "--afd-connector",
        default=None,
        help=(
            "AFD connector name. Defaults to P2pNcclAFDConnector for GPU and "
            "CAMP2pAFDConnector for NPU."
        ),
    )
    parser.add_argument(
        "--afd-async",
        action="store_true",
        help="Set additional_config['afd']['async']=true.",
    )
    parser.add_argument(
        "--compute-gate-on-attention",
        action="store_true",
        help="Set additional_config['afd']['compute_gate_on_attention']=true.",
    )
    parser.add_argument(
        "--afd-connector-extra-config",
        action="append",
        default=[],
        help=(
            "JSON object merged into "
            "additional_config['afd']['connector_extra_config']."
        ),
    )
    parser.add_argument(
        "--device-backend",
        choices=["gpu", "npu"],
        default="gpu",
        help="Device backend. 'gpu' uses CUDA workers, 'npu' uses Ascend workers.",
    )
    parser.add_argument(
        "--common-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added to all processes.",
    )
    parser.add_argument(
        "--attention-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to Attention processes.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to FFN processes.",
    )


def configure_scenario(args: argparse.Namespace) -> None:
    """Configure retained NPU callers; GPU cases are model-owned declarations."""
    settings = {
        "baseline-graph": (True, True, False, 4, 0),
        "afd-eager-2a1f": (False, False, False, 2, 1),
        "afd-graph-2a1f": (False, True, False, 2, 1),
        "afd-graph-dbo-2a1f": (False, True, True, 2, 1),
        "afd-eager-2a2f": (False, False, False, 2, 2),
        NPU_EAGER_DBO_SCENARIO: (False, False, True, 2, 2),
        "afd-graph-2a2f": (False, True, False, 2, 2),
        "afd-graph-dbo-2a2f": (False, True, True, 2, 2),
        DSV4_ASYNC_CAM_SCENARIO: (
            False,
            False,
            False,
            DSV4_ATTENTION_RANKS,
            DSV4_FFN_RANKS,
        ),
    }
    for scenario, profile in DSV4_SYNC_SHAPES.items():
        # The A5 host profile keeps native DBO off: split batches are the
        # current suspect for its DSA operator tiling failure.
        settings[scenario] = (
            False,
            dsv4_config.sync_use_graph(profile),
            False,
            profile.attention_ranks,
            profile.ffn_ranks,
        )
    baseline, graph, dbo, attention_ranks, ffn_ranks = settings[args.scenario]
    args.baseline = baseline
    args.cuda_graph_full_decode_only = graph
    args.enable_dbo = dbo
    args.num_attention_ranks = attention_ranks
    args.num_ffn_ranks = ffn_ranks
    args.attention_data_parallel_address = None
    args.collect_mrv2_execution_evidence = True
    args.tp_size = 1
    profile = sync_shape(args.scenario)
    args.attention_tp_size = (
        DSV4_ATTENTION_TP_SIZE
        if args.scenario == DSV4_ASYNC_CAM_SCENARIO
        else profile.attention_tp_size
        if profile is not None
        else 1
    )
    args.ffn_tp_size = profile.ffn_tp_size if profile is not None else 1
    if args.scenario not in DSV4_SCENARIOS and args.gsm8k_output_path is None:
        raise ValueError("--gsm8k-output-path is required for GSM8K scenarios")
    if args.scenario == DSV4_ASYNC_CAM_SCENARIO:
        dsv4_config.configure_scenario(args)
    elif args.scenario in DSV4_SYNC_CAMP2P_SCENARIOS:
        dsv4_config.configure_sync_camp2p_scenario(args)
    if graph:
        args.cudagraph_capture_size = (
            profile.cudagraph_capture_size
            if profile is not None
            else DEFAULT_CUDAGRAPH_CAPTURE_SIZE
        )
    if args.scenario == NPU_EAGER_DBO_SCENARIO:
        args.dbo_decode_token_threshold = NPU_EAGER_DBO_DECODE_THRESHOLD
        args.dbo_prefill_token_threshold = NPU_EAGER_DBO_PREFILL_THRESHOLD
    elif dbo:
        args.dbo_decode_token_threshold = 2
        args.dbo_prefill_token_threshold = 8


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_topology(
    args: argparse.Namespace,
    attention_devices: list[str],
    ffn_devices: list[str],
) -> None:
    if len(attention_devices) != len(set(attention_devices)):
        raise ValueError("Attention devices must be unique")
    if not args.baseline and len(ffn_devices) != len(set(ffn_devices)):
        raise ValueError("FFN devices must be unique")
    if not args.baseline and set(attention_devices) & set(ffn_devices):
        raise ValueError("Attention and FFN devices must not overlap")
    if args.num_attention_ranks != len(attention_devices):
        raise ValueError(
            f"--attention-devices must contain exactly "
            f"{args.num_attention_ranks} devices",
        )
    if not args.baseline and args.num_ffn_ranks != len(ffn_devices):
        raise ValueError(
            f"--ffn-devices must contain exactly {args.num_ffn_ranks} device",
        )
    if args.baseline:
        baseline_ranks = 4
        if args.num_attention_ranks != baseline_ranks or args.num_ffn_ranks != 0:
            raise ValueError(
                f"baseline E2E requires {baseline_ranks} Attention ranks "
                "and no FFN ranks",
            )
        if role_tp_size(args, "attention") != 1:
            raise ValueError("baseline E2E requires Attention TP=1")
        return
    if args.scenario == NPU_EAGER_DBO_SCENARIO and args.device_backend != "npu":
        raise ValueError("Lite eager DBO scenario requires NPU")
    if args.scenario in DSV4_SCENARIOS and args.device_backend != "npu":
        raise ValueError("DSV4 scenarios require NPU")
    for role, rank_count in (
        ("attention", args.num_attention_ranks),
        ("ffn", args.num_ffn_ranks),
    ):
        tp_size = role_tp_size(args, role)
        if tp_size < 1:
            raise ValueError(f"{role} TP size must be positive")
        if rank_count % tp_size != 0:
            raise ValueError(
                f"{role} rank count must be divisible by TP size "
                f"(ranks={rank_count}, tp={tp_size})",
            )


def build_baseline_command(args: argparse.Namespace) -> list[str]:
    """Build the native baseline command without AFD config."""
    if any(
        arg == "--additional-config" or arg.startswith("--additional-config=")
        for arg in args.common_vllm_arg
    ):
        raise ValueError(
            "baseline --common-vllm-arg cannot inject --additional-config",
        )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--shutdown-timeout",
        str(VLLM_SHUTDOWN_TIMEOUT_S),
        "--served-model-name",
        served_model_name(args, "baseline"),
        "--data-parallel-size",
        str(args.num_attention_ranks),
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
    ]
    if needs_mrv2_execution_evidence(args):
        cmd.extend(["--worker-extension-cls", "tests.e2e.mrv2_evidence.Worker"])
    cmd.extend(dbo_vllm_arguments(args))
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
            ],
        )
        cmd.extend(
            [
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")
    cmd.extend(["--host", args.api_host, "--port", str(attention_api_port(args))])
    cmd.extend(args.common_vllm_arg)
    return cmd


def build_vllm_command(
    args: argparse.Namespace,
    *,
    role: str,
    slot: RoleSlot | None = None,
) -> list[str]:
    """Build one role's ``vllm serve`` command.

    ``slot`` carries this pod's share of a role in a multi-pod run. A role that
    lives entirely in one pod produces the single-host command unchanged.
    """
    tp_size = role_tp_size(args, role)
    role_total_ranks = (
        args.num_attention_ranks if role == "attention" else args.num_ffn_ranks
    )
    role_dp_size = max(1, role_total_ranks // tp_size)
    is_npu = args.device_backend == "npu"
    sync_profile = sync_shape(args.scenario)
    connector = args.afd_connector or (
        "CAMP2pAFDConnector" if is_npu else "P2pNcclAFDConnector"
    )

    afd_config: dict[str, Any] = {
        "afd": {
            "role": role,
            "connector": connector,
            "host": args.afd_host if slot is None else slot.afd_host,
            "port": args.afd_port,
            "num_attention_ranks": args.num_attention_ranks,
            "num_ffn_ranks": args.num_ffn_ranks,
        },
    }
    if args.afd_async:
        afd_config["afd"]["async"] = True
    if args.compute_gate_on_attention:
        afd_config["afd"]["compute_gate_on_attention"] = True
    connector_extra_config = parse_afd_connector_extra_config(
        args.afd_connector_extra_config,
    )
    if connector_extra_config:
        afd_config["afd"]["connector_extra_config"] = connector_extra_config
    if args.scenario in DSV4_SCENARIOS:
        afd_config.update(
            dsv4_config.additional_config(
                role, async_cam=args.scenario == DSV4_ASYNC_CAM_SCENARIO
            )
        )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--shutdown-timeout",
        str(
            NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S
            if is_npu and args.afd_async
            else VLLM_SHUTDOWN_TIMEOUT_S
        ),
        "--served-model-name",
        served_model_name(args, role),
        "--data-parallel-size",
        str(role_dp_size),
        "--tensor-parallel-size",
        str(tp_size),
    ]
    if sync_profile is None or sync_profile.enable_expert_parallel:
        # Non-DSV4 AFD scenarios always run expert parallel. The A5 DSV4 sync
        # profile follows its launch script, which runs Attention DP2/TP1 and
        # FFN DP2/TP1 with expert parallelism; the A3 profile uses Attention
        # DP2/TP4 and FFN DP8/TP1 with expert parallelism.
        cmd.append("--enable-expert-parallel")
        # Skip non-local routed-expert weights before they are read; the
        # Attention role discards routed experts, FFN keeps only its EP shard.
        cmd.append("--enable-ep-weight-filter")
    cmd.extend(
        [
            "--additional-config",
            json.dumps(afd_config, separators=(",", ":")),
        ],
    )
    if slot is not None and slot.spans_pods:
        cmd.extend(
            [
                "--data-parallel-size-local",
                str(slot.dp_size_local),
                "--data-parallel-address",
                slot.dp_address,
                "--data-parallel-rpc-port",
                str(slot.dp_rpc_port),
            ],
        )
        # The leader always starts at DP rank 0, so it omits the flag: vLLM
        # treats any explicit start rank on a non-headless engine as hybrid LB
        # (external LB when its local size is 1), and then rejects the
        # headless peers (vllm/engine/arg_utils.py:2262).
        if slot.headless:
            cmd.extend(
                ["--data-parallel-start-rank", str(slot.dp_start_rank), "--headless"],
            )
    elif role == "attention" and args.attention_data_parallel_address is not None:
        # A scenario-pinned Attention DP address is emitted exactly once; in a
        # multi-pod run only the slot knows the pod that leads the role.
        cmd.extend(
            [
                "--data-parallel-address",
                (
                    args.attention_data_parallel_address
                    if slot is None
                    else slot.dp_address
                ),
            ],
        )
    if needs_mrv2_execution_evidence(args):
        cmd.extend(["--worker-extension-cls", "tests.e2e.mrv2_evidence.Worker"])
    profile_compilation_config = (
        None
        if sync_profile is None
        else dsv4_config.sync_compilation_config(sync_profile)
    )
    if profile_compilation_config is not None:
        # A profile that carries its host script's compilation config passes it
        # verbatim, so the runner adds none of its own capture-size flags.
        cmd.extend(
            [
                "--compilation-config",
                json.dumps(profile_compilation_config, separators=(",", ":")),
            ],
        )
    elif args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        # A scenario that fixes its own `--max-num-seqs` (the DSV4 launch
        # profiles do) keeps it: the capture size only has to cover it.
        max_num_seqs_args = (
            []
            if any(arg == "--max-num-seqs" for arg in args.common_vllm_arg)
            else ["--max-num-seqs", capture_size]
        )
        cmd.extend(
            [
                *max_num_seqs_args,
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")

    cmd.extend(dbo_vllm_arguments(args))

    # A headless slot starts no API server (vllm/entrypoints/cli/serve.py:61),
    # so binding one would only reserve a port it never uses.
    serves_api = slot is None or not slot.headless
    if role == "attention":
        if serves_api:
            cmd.extend(
                ["--host", args.api_host, "--port", str(attention_api_port(args))],
            )
        if args.use_decode_bench_connector:
            cmd.extend(["--kv-transfer-config", decode_bench_connector_config()])
        cmd.extend(args.attention_vllm_arg)
    else:
        if serves_api:
            cmd.extend(
                ["--host", args.api_host, "--port", str(ffn_api_port(args))],
            )
        cmd.extend(args.ffn_vllm_arg)
    cmd.extend(args.common_vllm_arg)
    return cmd


def dbo_vllm_arguments(args: argparse.Namespace) -> list[str]:
    """Keep native and AFD qualification on identical DBO thresholds."""
    if not args.enable_dbo:
        return []
    prefill_threshold = (
        args.dbo_prefill_token_threshold
        if args.dbo_prefill_token_threshold is not None
        else args.cudagraph_capture_size
    )
    return [
        "--enable-dbo",
        "--dbo-decode-token-threshold",
        str(args.dbo_decode_token_threshold),
        "--dbo-prefill-token-threshold",
        str(prefill_threshold),
    ]


def role_tp_size(args: argparse.Namespace, role: str) -> int:
    if role in ("attention", "baseline"):
        return args.attention_tp_size or args.tp_size
    if role == "ffn":
        return args.ffn_tp_size or args.tp_size
    raise ValueError(f"unknown AFD role {role!r}")


def parse_afd_connector_extra_config(values: list[str]) -> dict[str, Any]:
    connector_extra_config: dict[str, Any] = {}
    for raw_value in values:
        value = json.loads(raw_value)
        if not isinstance(value, dict):
            raise ValueError("--afd-connector-extra-config must be a JSON object")
        connector_extra_config.update(value)
    return connector_extra_config


def uses_async_connector(args: argparse.Namespace) -> bool:
    return args.afd_connector == ASYNC_AFD_CONNECTOR


def uses_npu_async_process_cleanup(args: argparse.Namespace) -> bool:
    """Return whether E2E teardown must find and kill every FFN process."""
    return args.device_backend == "npu" and args.scenario == DSV4_ASYNC_CAM_SCENARIO


def process_termination_timeout(args: argparse.Namespace) -> float:
    """Return the scenario's teardown budget, shared by every runner."""
    if args.device_backend == "npu" and args.afd_async and not args.baseline:
        return NPU_ASYNC_TERMINATION_TIMEOUT_S
    if args.scenario in DSV4_SCENARIOS:
        return DSV4_PROCESS_TERMINATION_TIMEOUT_S
    return PROCESS_TERMINATION_TIMEOUT_S


def run_scenario_evaluation(args: argparse.Namespace) -> None:
    """Run the scenario's acceptance check, shared by every runner."""
    if args.scenario in DSV4_SCENARIOS:
        run_concurrent_completion_evaluation(args)
    else:
        run_gsm8k_evaluation(args)


def decode_bench_connector_config() -> str:
    return json.dumps(
        {
            "kv_connector": "AFDDecodeBenchConnector",
            "kv_connector_module_path": "tools.benchmarks.decode_bench",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0,
            },
        },
        separators=(",", ":"),
    )


def served_model_name(args: argparse.Namespace, role: str) -> str:
    return f"{args.served_model_name_prefix}-{role}"


def attention_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base


def ffn_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base + 1


def run_gsm8k_evaluation(args: argparse.Namespace) -> None:
    """Run the configured GSM8K workload against the scenario's public API."""
    if args.gsm8k_output_path is None:
        raise RuntimeError("--gsm8k-output-path is required for GSM8K scenarios")
    configured_limit = os.environ.get(
        GSM8K_LIMIT_ENV,
        str(DEFAULT_GSM8K_SAMPLE_LIMIT),
    )
    sample_limit = None if configured_limit == "all" else int(configured_limit)
    if args.enable_dbo and sample_limit is not None:
        sample_limit = max(sample_limit, DBO_EVAL_MIN_SAMPLES)
    expected_sample_count = (
        GSM8K_FULL_SAMPLE_COUNT if sample_limit is None else sample_limit
    )
    full_run_options = (
        {"timeout_s": FULL_GSM8K_TIMEOUT_S} if sample_limit is None else {}
    )
    scenario_options = {}
    if args.enable_dbo:
        scenario_options["num_concurrent"] = DBO_EVAL_NUM_CONCURRENT
    role = "baseline" if args.baseline else "attention"
    results = _run_lm_eval(
        f"http://{args.api_host}:{attention_api_port(args)}",
        served_model_name(args, role),
        output_path=args.gsm8k_output_path,
        num_fewshot=GSM8K_NUM_FEWSHOT,
        tokenizer=args.model,
        limit=sample_limit,
        **scenario_options,
        **full_run_options,
    )
    sample_count = _extract_gsm8k_sample_count(results)
    if sample_count != expected_sample_count:
        raise RuntimeError(
            f"GSM8K evaluated {sample_count} samples; expected {expected_sample_count}",
        )
    minimum_accuracy = float(
        os.environ.get(GSM8K_THRESHOLD_ENV, str(DEFAULT_GSM8K_THRESHOLD)),
    )
    accuracy = _extract_gsm8k_accuracy(results)
    if not accuracy >= minimum_accuracy:
        raise RuntimeError(
            f"GSM8K accuracy {accuracy:.4f} is below the required "
            f"threshold {minimum_accuracy:.4f}",
        )


def uses_dbo_execution(args: argparse.Namespace) -> bool:
    return args.enable_dbo


def execution_evidence_roles(args: argparse.Namespace) -> tuple[str, ...]:
    if args.baseline:
        return ("baseline",)
    # Async CAM stages belong to Attention; FFN consumes independent native
    # work items. Sync graph/DBO must prove the matching path on both roles.
    return ("attention",) if uses_async_connector(args) else ("attention", "ffn")


def needs_mrv2_execution_evidence(args: argparse.Namespace) -> bool:
    return (
        not args.baseline
        and (args.cuda_graph_full_decode_only or uses_dbo_execution(args))
        and args.collect_mrv2_execution_evidence
    )


def start_evaluation_window(
    args: argparse.Namespace, start_marker: Path | None, window: str
) -> None:
    if not needs_mrv2_execution_evidence(args):
        return
    assert start_marker is not None
    start_marker.with_name("evaluation-complete").unlink(missing_ok=True)
    mrv2_execution_evidence_rpc(args, "afd_reset_execution_evidence", window)
    pending = start_marker.with_suffix(".pending")
    pending.write_text(window)
    pending.replace(start_marker)


def finish_evaluation_window(
    args: argparse.Namespace,
    start_marker: Path | None,
    window: str,
    evidence: list[tuple[str, str]],
    processes: list[subprocess.Popen[str]],
) -> dict[tuple[int, int, int, int], set[tuple[int, ...]]]:
    if not needs_mrv2_execution_evidence(args):
        return {}
    assert start_marker is not None
    mrv2_execution_evidence_rpc(args, "afd_log_execution_evidence")
    start_marker.with_name("evaluation-complete").write_text(window)
    wait_for_mrv2_execution_summaries(evidence, args, processes, window=window)
    scoped = [(role, line) for role, line in evidence if f"window={window} " in line]
    if args.cuda_graph_full_decode_only or uses_dbo_execution(args):
        assert_mrv2_execution_evidence(scoped, args)
    return {}


def mrv2_execution_evidence_rpc(
    args: argparse.Namespace, method: str, window: str | None = None
) -> None:
    """Broadcast on Attention; its DP/TP workers log their own summaries.

    Pinned vLLM's DP load balancer returns only the first DP engine's result,
    so the final assertion consumes the per-worker summary lines instead.
    FFN's daemon reports through its existing periodic worker health RPC.
    """
    request = urllib.request.Request(
        f"http://{args.api_host}:{attention_api_port(args)}/collective_rpc",
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
    ) as response:
        response.read()


def wait_for_mrv2_execution_summaries(
    evidence: list[tuple[str, str]],
    args: argparse.Namespace,
    processes: list[subprocess.Popen[str]],
    *,
    window: str | None = None,
) -> None:
    expected: set[tuple[str, int, int]] = set()
    for role in execution_evidence_roles(args):
        tp_size = role_tp_size(args, role)
        rank_count = args.num_ffn_ranks if role == "ffn" else args.num_attention_ranks
        expected.update(
            (role, dp, tp)
            for dp in range(rank_count // tp_size)
            for tp in range(tp_size)
        )
    deadline = time.monotonic() + MRV2_EVIDENCE_RPC_TIMEOUT_S
    while True:
        ensure_processes_alive(processes)
        observed = {
            (role, int(match[1]), int(match[2]))
            for role, line in evidence
            if window is None or f"window={window} " in line
            if (match := re.search(r"dp_rank=(\d+) tp_rank=(\d+)", line)) is not None
        }
        if expected <= observed:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for MRV2 execution summaries: "
                f"window={window} {sorted(expected - observed)}"
            )
        time.sleep(PROCESS_POLL_INTERVAL_S)


def assert_mrv2_execution_evidence(
    evidence: list[tuple[str, str]], args: argparse.Namespace
) -> None:
    roles = execution_evidence_roles(args)
    layouts = {
        role: Role(
            dp=(args.num_ffn_ranks if role == "ffn" else args.num_attention_ranks)
            // role_tp_size(args, role),
            tp=role_tp_size(args, role),
        )
        for role in roles
    }
    assert_execution_evidence(
        evidence,
        Mrv2Check(
            roles, graph=args.cuda_graph_full_decode_only, dbo=uses_dbo_execution(args)
        ),
        role_layouts=layouts,
        backend=args.device_backend,
        asynchronous=uses_async_connector(args),
    )


def run_concurrent_completion_evaluation(args: argparse.Namespace) -> None:
    profile = sync_shape(args.scenario)
    role = "baseline" if args.baseline else "attention"
    evaluate_completions(
        url=f"http://{args.api_host}:{attention_api_port(args)}/v1/chat/completions",
        model=served_model_name(args, role),
        output_path=Path(args.completion_output_path),
        # Preserve the NPU profile's answer/plumbing acceptance policy.
        check_answer=True if profile is None else profile.check_answer,
    )


def build_env(
    visible_devices: str,
    args: argparse.Namespace,
    *,
    role: str | None = None,
    e2e_run_id: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    mrv2_evidence_start: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "18000")
    env[visible_devices_env_name(args.device_backend)] = visible_devices
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env.pop("AFD_E2E_HASH_INPUT_IDS", None)
    env.pop("AFD_E2E_HASH_REQUIRE_GRAPH", None)
    if needs_mrv2_execution_evidence(args):
        env["VLLM_SERVER_DEV_MODE"] = "1"
        if args.baseline:
            env["AFD_E2E_NATIVE_BASELINE"] = "1"
        else:
            env.pop("AFD_E2E_NATIVE_BASELINE", None)
        if mrv2_evidence_start is not None:
            env["AFD_E2E_MRV2_EVIDENCE_START"] = str(mrv2_evidence_start)
    if args.baseline:
        env["VLLM_PLUGINS"] = "ascend" if args.device_backend == "npu" else ""
    else:
        env["VLLM_PLUGINS"] = "ascend,afd" if args.device_backend == "npu" else "afd"
    if args.enable_dbo:
        env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    env["PYTHONUNBUFFERED"] = "1"
    if e2e_run_id is not None:
        if role is None:
            raise ValueError("role is required when setting an E2E run id")
        env[E2E_RUN_ID_ENV] = e2e_run_id
        env[E2E_PROCESS_ROLE_ENV] = role
    if args.device_backend == "npu":
        env.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
        if args.afd_async and not args.baseline:
            env.setdefault(
                WORKER_SHUTDOWN_TIMEOUT_ENV,
                str(NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S),
            )
    env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not current_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
    )
    return env


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def stream_output(
    name: str,
    process: subprocess.Popen[str],
    mrv2_execution_evidence: list[tuple[str, str]] | None = None,
) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")
            if mrv2_execution_evidence is not None and MRV2_EVIDENCE_PREFIX in line:
                mrv2_execution_evidence.append((name, line.strip()))

    thread = threading.Thread(target=worker, name=f"{name}-log-stream", daemon=True)
    thread.start()
    return thread


def wait_for_openai_api(
    args: argparse.Namespace,
    processes: list[subprocess.Popen[str]],
) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/models"
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        for process in processes:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"vLLM process exited before Attention API was ready "
                    f"(returncode={returncode}, command={process.args!r})",
                )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"\nAttention API is ready at {url}")
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for Attention API at {url}; last error={last_error!r}",
    )


def ensure_alive(process: subprocess.Popen[str], message: str) -> None:
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(f"{message} (returncode={returncode})")


def ensure_processes_alive(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM process exited unexpectedly (returncode={returncode})",
            )


def terminate_processes(
    processes: list[subprocess.Popen[str]],
    *,
    termination_timeout_s: float = PROCESS_TERMINATION_TIMEOUT_S,
    deferred_sigkill_pgids: tuple[int, ...] = (),
    force_kill_environment: dict[str, str] | None = None,
) -> None:
    failures = terminate_process_groups(
        processes,
        termination_timeout_s=termination_timeout_s,
        poll_interval_s=PROCESS_POLL_INTERVAL_S,
        reap_timeout_s=PROCESS_REAP_TIMEOUT_S,
        deferred_sigkill_pgids=deferred_sigkill_pgids,
    )
    if force_kill_environment is not None:
        failures.extend(
            kill_processes_matching_environment(
                force_kill_environment,
                timeout_s=NPU_ASYNC_PROCESS_KILL_TIMEOUT_S,
                poll_interval_s=PROCESS_POLL_INTERVAL_S,
                process_name="FFN",
            ),
        )
    if failures:
        raise RuntimeError("; ".join(failures))


def visible_devices_env_name(device_backend: str) -> str:
    return (
        "ASCEND_RT_VISIBLE_DEVICES"
        if device_backend == "npu"
        else "CUDA_VISIBLE_DEVICES"
    )


def print_command(
    name: str,
    command: list[str],
    device_backend: str,
    visible_devices: str,
) -> None:
    printable = " ".join(shell_quote(token) for token in command)
    env_name = visible_devices_env_name(device_backend)
    print(f"\n=== Starting {name} ({env_name}={visible_devices}) ===")
    print(printable)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def prepare_case(
    case: Case, environment: CaseEnvironment, *, slots: tuple[RoleSlot, ...] = ()
) -> tuple[Case, dict]:
    """Resolve documented overrides without mutating the canonical declaration."""
    effective_phases = []
    reasons: list[str] = []
    for phase in case.phases:
        task, phase_reasons = phase.task.resolve(os.environ)
        effective_phases.append(replace(phase, task=task))
        reasons.extend(f"{phase.name}: {reason}" for reason in phase_reasons)
    profile = case.deployment
    override_names: tuple[str, ...] = (GSM8K_LIMIT_ENV, GSM8K_THRESHOLD_ENV)
    if profile.worker_shutdown_timeout_s is not None:
        override_names += (WORKER_SHUTDOWN_TIMEOUT_ENV,)
        worker_timeout = os.environ.get(WORKER_SHUTDOWN_TIMEOUT_ENV)
        if worker_timeout is not None:
            profile = replace(profile, worker_shutdown_timeout_s=int(worker_timeout))
    effective = replace(case, deployment=profile, phases=tuple(effective_phases))
    resolved = deployment.resolve(effective, environment, slots=slots)
    probe_arguments = (
        ("--worker-extension-cls", "tests.e2e.mrv2_evidence.Worker")
        if any(phase.checks for phase in effective.phases)
        else ()
    )
    record = {
        "case_id": case.id,
        "run_id": uuid.uuid4().hex,
        "canonical": asdict(case),
        "effective": asdict(effective),
        "bindings": {
            **asdict(environment),
            "output_path": str(environment.output_path),
        },
        "overrides": {
            name: os.environ[name] for name in override_names if name in os.environ
        },
        "configuration": {
            "eligible": not reasons,
            "classification": "debug" if reasons else "qualified",
            "reasons": reasons,
        },
        "commands": {
            role: resolved.command(role, probe_arguments)
            for role, _ in resolved.role_devices
        },
        "execution": {"status": "not_started"},
        "cleanup": {"status": "not_started"},
        "phases": [],
        "acceptance_passed": False,
    }
    return effective, record


def run_case(case: Case, environment: CaseEnvironment) -> dict:
    """Execute one deployment; success follows task, evidence and cleanup."""
    effective, record = prepare_case(case, environment)
    resolved = deployment.resolve(effective, environment)
    probe = Mrv2Session(effective, environment, record["run_id"])
    services = deployment.ServiceDeployment(resolved, run_id=record["run_id"])
    environment.output_path.mkdir(parents=True, exist_ok=True)

    def cleanup() -> None:
        failures = []
        try:
            services.close()
        except BaseException as exc:
            failures.append(str(exc))
        finally:
            try:
                probe.close()
            except BaseException as exc:
                failures.append(str(exc))
            record["processes"] = [
                {"pid": process.pid, "returncode": process.poll()}
                for process in services.processes
            ]
        if failures:
            record["cleanup"] = {"status": "failed", "errors": failures}
            raise RuntimeError("; ".join(failures))
        record["cleanup"] = {"status": "passed"}

    try:
        with cancellable_run(cleanup):
            probe.prepare()
            services.probe_arguments = probe.arguments
            services.probe_environment = probe.worker_environment
            services.start()
            services.ensure_alive()
            record["execution"] = {"status": "running"}
            for phase in effective.phases:
                phase_record: dict[str, JsonValue] = {
                    "name": phase.name,
                    "status": "running",
                    "checks": [asdict(check) for check in phase.checks],
                }
                record["phases"].append(phase_record)
                probe.start(phase)
                role = "baseline" if effective.deployment.is_baseline else "attention"
                phase_record["result"] = phase.task.execute(
                    base_url=f"http://{environment.api_host}:{environment.api_port_base}",
                    model_name=f"{case.served_model_name_prefix}-{role}",
                    checkpoint=environment.model,
                    output_path=environment.output_path / phase.name,
                )
                snapshot = probe.finish(phase, services)
                phase_record["evidence_records"] = len(snapshot)
                phase_record["status"] = "passed"
            services.ensure_alive()
        record["execution"] = {"status": "passed"}
        record["acceptance_passed"] = record["configuration"]["eligible"]
        return record
    except BaseException as exc:
        record["execution"] = {
            "status": "cancelled" if isinstance(exc, SystemExit) else "failed",
            "error": str(exc),
        }
        for phase_record in record["phases"]:
            if phase_record["status"] == "running":
                phase_record["status"] = record["execution"]["status"]
                phase_record["error"] = str(exc)
        raise
    finally:
        (environment.output_path / "run.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(
            f"\nE2E CASE {case.id}: {record['execution']['status']}; "
            f"configuration={record['configuration']['classification']}; "
            f"acceptance_passed={record['acceptance_passed']}",
            flush=True,
        )


def case_main(
    build_case: Callable[[str, CaseEnvironment], Case],
    case_ids: tuple[str, ...],
    argv: list[str] | None = None,
) -> int:
    """The model injects its builder; shared code never imports the model."""
    case_id, environment, describe = parse_case_environment(case_ids, argv)
    case = build_case(case_id, environment)
    if describe:
        _, record = prepare_case(case, environment)
        print(json.dumps(record, indent=2))
    else:
        run_case(case, environment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
