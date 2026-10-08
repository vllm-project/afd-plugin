# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed DSV4 Flash async CAM deployment and acceptance parameters."""

from __future__ import annotations

import argparse
import json

DSV4_ASYNC_CAM_SCENARIO = "afd-dsv4-flash-async-cam-dp2tp4-ep8"
DSV4_ATTENTION_RANKS = 8
DSV4_FFN_RANKS = 8
DSV4_ATTENTION_TP_SIZE = 4
DSV4_CONCURRENT_REQUESTS = 10
DSV4_REQUEST_TIMEOUT_S = 300
DSV4_COMPLETION_MAX_TOKENS = 256
DSV4_PROMPT_FIRST_OPERAND = 12
DSV4_PROMPT_SECOND_OPERAND = 7
# Sixteen NPU workers take longer than the small cases to destroy HCCL
# resources; the observed launcher shutdown alone exceeded 20 seconds.
DSV4_PROCESS_TERMINATION_TIMEOUT_S = 60


def configure_scenario(args: argparse.Namespace) -> None:
    if args.completion_output_path is None:
        raise ValueError("--completion-output-path is required for DSV4")
    args.afd_connector = "CAMAsyncAFDConnector"
    args.afd_async = True
    args.compute_gate_on_attention = True
    args.afd_connector_extra_config = [
        json.dumps(
            {
                "dynamicQuant": 1,
                "attn_ranks_per_dp": DSV4_ATTENTION_TP_SIZE,
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": 2,
                "async_moe_split": "token",
            }
        )
    ]
    # Keep this local 16-NPU case aligned with the DSV4 prefill scripts.
    # Reject ad-hoc overrides so its case ID denotes one fixed deployment.
    if args.common_vllm_arg or args.attention_vllm_arg or args.ffn_vllm_arg:
        raise ValueError("DSV4 scenario does not accept extra vLLM arguments")
    if args.use_decode_bench_connector:
        raise ValueError("DSV4 scenario runs without a KV transfer connector")
    args.common_vllm_arg = [
        "--api-server-count",
        "1",
        "--seed",
        "1024",
        "--max-model-len",
        "1048576",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "16",
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
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 128}),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
    ]
    args.ffn_vllm_arg = ["--all2all-backend", "flashinfer_all2allv"]
    args.attention_vllm_arg = [
        "--all2all-backend",
        "allgather_reducescatter",
        "--data-parallel-address",
        args.afd_host,
        "--no-disable-hybrid-kv-cache-manager",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v4",
    ]


def additional_config(role: str) -> dict[str, bool]:
    return {
        "enable_flashcomm1": role == "attention",
        "enable_cpu_binding": True,
        "enable_dsa_cp": False,
        "multistream_dsv4_dsa_overlap": False,
    }
