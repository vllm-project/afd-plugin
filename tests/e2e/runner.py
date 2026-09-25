#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run fixed baseline and AFD E2E scenarios on real hardware."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tests.e2e.accuracy.gsm8k import (
    _extract_gsm8k_accuracy,
    _extract_gsm8k_sample_count,
    _run_lm_eval,
)
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
from tests.e2e.process_utils import (
    kill_processes_matching_environment,
    terminate_process_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASYNC_AFD_CONNECTOR = "CAMAsyncAFDConnector"
ASYNC_CAM_SCENARIO = "afd-eager-async-cam"
ASYNC_CAM_ATTENTION_RANKS = 2
ASYNC_CAM_FFN_RANKS = 2
ASYNC_CAM_ATTENTION_TP_SIZE = 2
ASYNC_UBATCH_SCENARIO = "afd-async-ubatch"
ASYNC_UBATCH_ATTENTION_RANKS = 2
ASYNC_UBATCH_FFN_RANKS = 1
ASYNC_UBATCH_ATTENTION_TP_SIZE = 2
ASYNC_UBATCH_NUM_STAGES = 2
ASYNC_UBATCH_BATCH_SIZE = 2
V2_SYNC_CONNECTOR = "P2pNcclAFDConnector"
# Graph capture default for the scenarios that do not carry their own launch
# profile; the DSV4 synchronous profiles override it.
DEFAULT_CUDAGRAPH_CAPTURE_SIZE = 8
V2_SCENARIOS = (
    "afd-v2-eager-1a1f",
    "afd-v2-eager-dp2",
    "afd-v2-eager-tp2",
    "afd-v2-graph-1a1f",
    "afd-v2-graph-dp2",
    "afd-v2-graph-tp2",
)
V2_SINGLE_RANK_SCENARIOS = frozenset(
    ("afd-v2-eager-1a1f", "afd-v2-graph-1a1f"),
)
V2_TENSOR_PARALLEL_SCENARIOS = frozenset(
    ("afd-v2-eager-tp2", "afd-v2-graph-tp2"),
)
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
COMPLETION_REQUEST_TIMEOUT_S = 120
COMPLETION_MAX_TOKENS = 32
COMPLETION_TEMPERATURE = 0
DBO_EVAL_NUM_CONCURRENT = 12
DBO_EVAL_MIN_SAMPLES = 2 * DBO_EVAL_NUM_CONCURRENT
DBO_EVAL_NUM_UBATCHES = 2
# The engine logs one DEBUG line per executed step carrying its ubatch slice
# list; a step line containing UBatchSlice entries is a live two-ubatch run.
DBO_SPLIT_EVIDENCE_ENTRY = "UBatchSlice("
ACCOUNTING_PROMPT = (
    "<|im_start|>system\n"
    "You are a professional accountant. Answer questions using accounting "
    "knowledge, output only the option letter (A/B/C/D).<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: A company's balance sheet as of December 31, 2023 shows:\n"
    "  Current assets: Cash and equivalents 5 million yuan, Accounts "
    "receivable 8 million yuan, Inventory 6 million yuan\n"
    "  Non-current assets: Net fixed assets 12 million yuan\n"
    "  Current liabilities: Short-term loans 4 million yuan, Accounts "
    "payable 3 million yuan\n"
    "  Non-current liabilities: Long-term loans 9 million yuan\n"
    "  Owner's equity: Paid-in capital 10 million yuan, Retained earnings ?\n"
    "Requirement: Calculate the company's Asset-Liability Ratio and Current "
    "Ratio (round to two decimal places).\n"
    "Options:\n"
    "A. Asset-Liability Ratio=58.33%, Current Ratio=1.90\n"
    "B. Asset-Liability Ratio=62.50%, Current Ratio=2.17\n"
    "C. Asset-Liability Ratio=65.22%, Current Ratio=1.75\n"
    "D. Asset-Liability Ratio=68.00%, Current Ratio=2.50<|im_end|>\n"
    "<|im_start|>assistant\n"
)


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
    dbo_split_steps: list[float] = []
    dbo_eval_started_at: float | None = None
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    received_signal: int | None = None
    cleanup_in_progress = False

    def exit_after_cleanup(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        if not cleanup_in_progress:
            raise SystemExit(128 + signum)

    for signum in handled_signals:
        signal.signal(signum, exit_after_cleanup)

    launch_order: tuple[tuple[str, str], ...]
    try:
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
            log_threads.append(stream_output(role, process, dbo_split_steps))
            ensure_alive(process, f"{label} process exited during startup")

        wait_for_openai_api(args, processes)
        ensure_processes_alive(processes)

        if args.scenario == ASYNC_CAM_SCENARIO:
            run_completion_evaluation(args)
        elif args.scenario in DSV4_SCENARIOS:
            run_concurrent_completion_evaluation(args)
        else:
            if args.enable_dbo:
                dbo_eval_started_at = time.time()
            run_gsm8k_evaluation(args)
        if args.enable_dbo:
            assert_dbo_live_split_coverage(
                dbo_split_steps,
                dbo_eval_started_at,
                args,
            )

        ensure_processes_alive(processes)
    finally:
        body_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        cleanup_in_progress = True
        ffn_process = processes_by_role.get("ffn")
        deferred_sigkill_pgids = (
            (ffn_process.pid,)
            if use_npu_async_process_cleanup and ffn_process is not None
            else ()
        )
        try:
            try:
                try:
                    terminate_processes(
                        processes,
                        termination_timeout_s=(
                            DSV4_PROCESS_TERMINATION_TIMEOUT_S
                            if args.scenario in DSV4_SCENARIOS
                            else PROCESS_TERMINATION_TIMEOUT_S
                        ),
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
                        for signum, previous_handler in previous_handlers.items():
                            # Preloaded native libraries can install handlers
                            # unknown to Python (getsignal returns None). Python
                            # cannot restore those; reset to the OS default.
                            signal.signal(
                                signum,
                                signal.SIG_DFL
                                if previous_handler is None
                                else previous_handler,
                            )
            except BaseException as exc:
                cleanup_error = exc
        finally:
            cleanup_in_progress = False

        if received_signal is not None:
            signal_error = SystemExit(128 + received_signal)
            if cleanup_error is not None:
                raise signal_error from cleanup_error
            raise signal_error
        if cleanup_error is not None:
            if body_error is not None:
                raise body_error from cleanup_error
            raise cleanup_error

    print(f"\nE2E SCENARIO {args.scenario} PASSED")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed baseline or AFD E2E scenario.",
    )
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
            "afd-graph-2a2f",
            "afd-graph-dbo-2a2f",
            ASYNC_CAM_SCENARIO,
            ASYNC_UBATCH_SCENARIO,
            *DSV4_SCENARIOS,
            *V2_SCENARIOS,
        ],
        required=True,
        help="Fixed E2E scenario to run.",
    )
    parser.add_argument(
        "--completion-output-path",
        help="JSON file containing the ten concurrent DSV4 requests and responses.",
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
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port-base", type=int, default=8000)
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
    return parser.parse_args()


def configure_scenario(args: argparse.Namespace) -> None:
    """Set topology and features for the selected fixed scenario."""
    is_async_cam = args.scenario == ASYNC_CAM_SCENARIO
    is_async_ubatch = args.scenario == ASYNC_UBATCH_SCENARIO
    is_dsv4 = args.scenario in DSV4_SCENARIOS
    scenario_settings = {
        "baseline-graph": (True, True, False, 4, 0),
        "afd-eager-2a1f": (False, False, False, 2, 1),
        "afd-graph-2a1f": (False, True, False, 2, 1),
        "afd-graph-dbo-2a1f": (False, True, True, 2, 1),
        "afd-eager-2a2f": (False, False, False, 2, 2),
        "afd-graph-2a2f": (False, True, False, 2, 2),
        "afd-graph-dbo-2a2f": (False, True, True, 2, 2),
        ASYNC_CAM_SCENARIO: (
            False,
            False,
            False,
            ASYNC_CAM_ATTENTION_RANKS,
            ASYNC_CAM_FFN_RANKS,
        ),
        ASYNC_UBATCH_SCENARIO: (
            False,
            False,
            False,
            ASYNC_UBATCH_ATTENTION_RANKS,
            ASYNC_UBATCH_FFN_RANKS,
        ),
        DSV4_ASYNC_CAM_SCENARIO: (
            False,
            False,
            False,
            DSV4_ATTENTION_RANKS,
            DSV4_FFN_RANKS,
        ),
        "afd-v2-eager-1a1f": (False, False, False, 1, 1),
        "afd-v2-eager-dp2": (False, False, False, 2, 2),
        "afd-v2-eager-tp2": (False, False, False, 2, 2),
        "afd-v2-graph-1a1f": (False, True, False, 1, 1),
        "afd-v2-graph-dp2": (False, True, False, 2, 2),
        "afd-v2-graph-tp2": (False, True, False, 2, 2),
    }
    for sync_scenario, sync_profile in DSV4_SYNC_SHAPES.items():
        # The A5 script's native DBO stays off for these scenarios: a split batch
        # is the current suspect for the DSA operator tiling failure on that
        # host.
        scenario_settings[sync_scenario] = (
            False,
            dsv4_config.sync_use_graph(sync_profile),
            False,
            sync_profile.attention_ranks,
            sync_profile.ffn_ranks,
        )
    baseline, use_graph, enable_dbo, attention_ranks, ffn_ranks = scenario_settings[
        args.scenario
    ]
    args.baseline = baseline
    args.cuda_graph_full_decode_only = use_graph
    args.enable_dbo = enable_dbo
    args.num_attention_ranks = attention_ranks
    args.num_ffn_ranks = ffn_ranks
    args.tp_size = 1
    active_sync_profile = sync_shape(args.scenario)
    if args.scenario == DSV4_ASYNC_CAM_SCENARIO:
        args.attention_tp_size = DSV4_ATTENTION_TP_SIZE
    elif active_sync_profile is not None:
        args.attention_tp_size = active_sync_profile.attention_tp_size
    elif is_async_cam:
        args.attention_tp_size = ASYNC_CAM_ATTENTION_TP_SIZE
    elif is_async_ubatch:
        args.attention_tp_size = ASYNC_UBATCH_ATTENTION_TP_SIZE
    elif args.scenario in V2_TENSOR_PARALLEL_SCENARIOS:
        args.attention_tp_size = 2
    else:
        args.attention_tp_size = 1
    if args.scenario in V2_TENSOR_PARALLEL_SCENARIOS:
        args.ffn_tp_size = 2
    elif active_sync_profile is not None:
        # A synchronous DSV4 profile sizes the FFN side exactly like its
        # Attention side: A5 shards by data parallel with expert parallelism,
        # A3 by tensor parallel.
        args.ffn_tp_size = active_sync_profile.ffn_tp_size
    else:
        args.ffn_tp_size = 1
    args.use_v2_model_runner = args.scenario in V2_SCENARIOS
    if args.use_v2_model_runner:
        if args.afd_async or args.afd_connector == ASYNC_AFD_CONNECTOR:
            raise ValueError("ModelRunnerV2 E2E scenarios require synchronous AFD")
        if args.compute_gate_on_attention:
            raise ValueError(
                "ModelRunnerV2 E2E scenarios require compute_gate_on_attention=false",
            )
        if args.afd_connector not in (None, V2_SYNC_CONNECTOR):
            raise ValueError(
                "ModelRunnerV2 E2E scenarios require P2pNcclAFDConnector",
            )
        if args.afd_connector_extra_config:
            raise ValueError(
                "ModelRunnerV2 E2E scenarios do not support connector extra config",
            )
        if args.use_decode_bench_connector:
            raise ValueError(
                "ModelRunnerV2 E2E scenarios do not support decode-bench connector",
            )
    if not is_async_cam and not is_dsv4 and args.gsm8k_output_path is None:
        raise ValueError("--gsm8k-output-path is required for GSM8K scenarios")
    if is_async_cam:
        args.afd_connector = ASYNC_AFD_CONNECTOR
        args.afd_async = True
        args.compute_gate_on_attention = True
        extra_config = parse_afd_connector_extra_config(
            args.afd_connector_extra_config,
        )
        extra_config["attn_ranks_per_dp"] = ASYNC_CAM_ATTENTION_TP_SIZE
        args.afd_connector_extra_config = [
            json.dumps(extra_config, separators=(",", ":")),
        ]
    if is_async_ubatch:
        args.afd_connector = ASYNC_AFD_CONNECTOR
        args.afd_async = True
        args.compute_gate_on_attention = True
        extra_config = parse_afd_connector_extra_config(
            args.afd_connector_extra_config,
        )
        # This fixed scenario must exercise the TP/SP token-split path. Replace
        # conflicting caller values instead of silently changing its contract.
        extra_config["attn_ranks_per_dp"] = ASYNC_UBATCH_ATTENTION_TP_SIZE
        extra_config["async_moe_ubatching"] = True
        extra_config["async_moe_num_ubatches"] = ASYNC_UBATCH_NUM_STAGES
        extra_config["async_moe_split"] = "token"
        args.afd_connector_extra_config = [
            json.dumps(extra_config, separators=(",", ":")),
        ]
        # CAM dispatch allocates its HCCL communication pool (~8.6 GB) outside
        # the torch memory pool; the default 0.92 GPU memory utilization leaves
        # too little room and fails with EL0004. 0.8 leaves enough headroom.
        if not any(
            arg == "--gpu-memory-utilization"
            or arg.startswith("--gpu-memory-utilization=")
            for arg in args.common_vllm_arg
        ):
            args.common_vllm_arg.extend(["--gpu-memory-utilization", "0.8"])
    if args.scenario == DSV4_ASYNC_CAM_SCENARIO:
        dsv4_config.configure_scenario(args)
    elif args.scenario in DSV4_SYNC_CAMP2P_SCENARIOS:
        dsv4_config.configure_sync_camp2p_scenario(args)
    if use_graph:
        args.cudagraph_capture_size = (
            active_sync_profile.cudagraph_capture_size
            if active_sync_profile is not None
            else DEFAULT_CUDAGRAPH_CAPTURE_SIZE
        )
    if enable_dbo:
        args.dbo_decode_token_threshold = 1
        args.dbo_prefill_token_threshold = 8
        if not any(
            arg == "--no-enable-chunked-prefill" for arg in args.common_vllm_arg
        ):
            args.common_vllm_arg.append("--no-enable-chunked-prefill")


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
        if args.num_attention_ranks != 4 or args.num_ffn_ranks != 0:
            raise ValueError(
                "baseline E2E requires four Attention ranks and no FFN ranks",
            )
        if role_tp_size(args, "attention") != 1:
            raise ValueError("baseline E2E requires Attention TP=1")
        return
    if args.use_v2_model_runner and args.device_backend != "gpu":
        raise ValueError("ModelRunnerV2 E2E scenarios require GPU")
    if (
        args.scenario in (ASYNC_CAM_SCENARIO, ASYNC_UBATCH_SCENARIO)
        and args.device_backend != "npu"
    ):
        raise ValueError("async CAM scenarios require NPU")
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
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
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
) -> list[str]:
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
            "host": args.afd_host,
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
        afd_config.update(dsv4_config.additional_config())
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--shutdown-timeout",
        str(VLLM_SHUTDOWN_TIMEOUT_S),
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
        # FFN DP2/TP1 with expert parallelism; the A3 profile shards by tensor
        # parallel and keeps the expert-parallel world at one.
        cmd.append("--enable-expert-parallel")
    cmd.extend(
        [
            "--additional-config",
            json.dumps(afd_config, separators=(",", ":")),
        ],
    )
    if args.use_v2_model_runner:
        cmd.extend(
            [
                "--no-enable-prefix-caching",
                "--no-enable-chunked-prefill",
                "--no-async-scheduling",
            ],
        )
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

    if args.enable_dbo:
        prefill_threshold = (
            args.dbo_prefill_token_threshold
            if args.dbo_prefill_token_threshold is not None
            else args.cudagraph_capture_size
        )
        cmd.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                str(args.dbo_decode_token_threshold),
                "--dbo-prefill-token-threshold",
                str(prefill_threshold),
            ],
        )

    if role == "attention":
        cmd.extend(
            ["--host", args.api_host, "--port", str(attention_api_port(args))],
        )
        if args.use_decode_bench_connector:
            cmd.extend(["--kv-transfer-config", decode_bench_connector_config()])
        cmd.extend(args.attention_vllm_arg)
    else:
        cmd.extend(
            ["--host", args.api_host, "--port", str(ffn_api_port(args))],
        )
        cmd.extend(args.ffn_vllm_arg)
    cmd.extend(args.common_vllm_arg)
    return cmd


def role_tp_size(args: argparse.Namespace, role: str) -> int:
    if role == "attention":
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
    return args.device_backend == "npu" and args.scenario in (
        ASYNC_CAM_SCENARIO,
        ASYNC_UBATCH_SCENARIO,
        DSV4_ASYNC_CAM_SCENARIO,
    )


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
    scenario_options = (
        {"batch_size": ASYNC_UBATCH_BATCH_SIZE}
        if args.scenario == ASYNC_UBATCH_SCENARIO
        else {}
    )
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


def assert_dbo_live_split_coverage(
    split_step_times: list[float],
    eval_started_at: float | None,
    args: argparse.Namespace,
) -> None:
    live_split_steps = sum(
        1
        for received_at in split_step_times
        if eval_started_at is None or received_at >= eval_started_at
    )
    if live_split_steps:
        print(
            "\n[dbo-coverage] live DBO split coverage confirmed: "
            f"{live_split_steps} two-ubatch step(s) recorded in the "
            f"evaluation window",
        )
        return
    raise RuntimeError(
        "DBO was enabled but no live request was ever split into "
        f"{DBO_EVAL_NUM_UBATCHES} ubatches: 0 two-ubatch steps were recorded "
        "inside the evaluation window (warmup/capture-only execution does "
        f"not count). Client concurrency: {DBO_EVAL_NUM_CONCURRENT}, sample "
        f"floor: {DBO_EVAL_MIN_SAMPLES}. Thresholds: "
        f"dbo_decode_token_threshold={args.dbo_decode_token_threshold}, "
        f"dbo_prefill_token_threshold={args.dbo_prefill_token_threshold}. "
        "Increase the client concurrency until both attention ranks hold "
        "enough real tokens for two non-empty ubatches.",
    )


def run_completion_evaluation(args: argparse.Namespace) -> None:
    """Send the async CAM smoke request and require one returned choice."""
    payload = json.dumps(
        {
            "model": served_model_name(args, "attention"),
            "prompt": ACCOUNTING_PROMPT,
            "max_tokens": COMPLETION_MAX_TOKENS,
            "temperature": COMPLETION_TEMPERATURE,
        },
    ).encode()
    request = urllib.request.Request(
        f"http://{args.api_host}:{attention_api_port(args)}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=COMPLETION_REQUEST_TIMEOUT_S,
        ) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"completion request failed with HTTP {exc.code}: {body}",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"completion request failed: {exc}") from exc

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("completion response contains no choices")
    choice = choices[0]
    text = choice.get("text") if isinstance(choice, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("completion response contains no text")
    print(f"Completion response: {text}")


def run_concurrent_completion_evaluation(args: argparse.Namespace) -> None:
    profile = sync_shape(args.scenario)
    evaluate_completions(
        url=f"http://{args.api_host}:{attention_api_port(args)}/v1/chat/completions",
        model=served_model_name(args, "attention"),
        output_path=Path(args.completion_output_path),
        # A synchronous profile may state that its host does not answer reliably
        # yet, and then the oracle checks the concurrent plumbing only.
        check_answer=True if profile is None else profile.check_answer,
    )


def build_env(
    visible_devices: str,
    args: argparse.Namespace,
    *,
    role: str | None = None,
    e2e_run_id: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "18000")
    env[visible_devices_env_name(args.device_backend)] = visible_devices
    if args.device_backend != "npu":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.use_v2_model_runner else "0"
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
    if (
        args.device_backend == "npu"
        and role in ("attention", "ffn")
        and role_tp_size(args, role) <= 1
    ):
        env.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
    if args.scenario == DSV4_ASYNC_CAM_SCENARIO:
        env.update(dsv4_config.role_environment(role))
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
    dbo_split_steps: list[float] | None = None,
) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")
            if (
                dbo_split_steps is not None
                and name == "attention"
                and DBO_SPLIT_EVIDENCE_ENTRY in line
            ):
                dbo_split_steps.append(time.time())

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


if __name__ == "__main__":
    sys.exit(main())
