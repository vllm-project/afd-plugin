# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA DeepSeek-V4-Flash qualification profiles, separate from Ascend CAM."""

from __future__ import annotations

import argparse
from types import MappingProxyType

DSV4_GPU_ROLE_RANKS = 2
DSV4_GPU_DEVICE_COUNT = 2 * DSV4_GPU_ROLE_RANKS
# With ten/twelve concurrent requests across DP2, a 16-token graph leaves
# the second eight-token child empty. Bucket 8 lets live decode exercise both
# four-token children, as the DBO execution oracle requires.
DSV4_GPU_CAPTURE_SIZE = 8
DSV4_GPU_MAX_MODEL_LEN = 4096
DSV4_GPU_MAX_BATCHED_TOKENS = 1024
DSV4_GPU_MEMORY_UTILIZATION = 0.9
DSV4_GPU_SEED = 1024
DSV4_GPU_CONCURRENT_ROUNDS = 2
DSV4_GPU_HASH_WINDOWS = tuple(
    f"concurrent-{index + 1}" for index in range(DSV4_GPU_CONCURRENT_ROUNDS)
)
DSV4_GPU_ANSWER_REPETITIONS = 8

# (native baseline, full-decode graph, DBO, Attention ranks, FFN ranks).
# Native DP2/TP1/EP2 matches the per-role parallelism of AFD 2A2F.
DSV4_GPU_SCENARIO_SETTINGS = MappingProxyType(
    {
        "baseline-dsv4-flash-eager": (True, False, False, DSV4_GPU_ROLE_RANKS, 0),
        "baseline-dsv4-flash-graph": (True, True, False, DSV4_GPU_ROLE_RANKS, 0),
        "afd-dsv4-flash-eager-2a2f": (
            False,
            False,
            False,
            DSV4_GPU_ROLE_RANKS,
            DSV4_GPU_ROLE_RANKS,
        ),
        "afd-dsv4-flash-graph-2a2f": (
            False,
            True,
            False,
            DSV4_GPU_ROLE_RANKS,
            DSV4_GPU_ROLE_RANKS,
        ),
        "baseline-dsv4-flash-eager-dbo": (True, False, True, DSV4_GPU_ROLE_RANKS, 0),
        "baseline-dsv4-flash-graph-dbo": (True, True, True, DSV4_GPU_ROLE_RANKS, 0),
        "afd-dsv4-flash-eager-dbo-2a2f": (
            False,
            False,
            True,
            DSV4_GPU_ROLE_RANKS,
            DSV4_GPU_ROLE_RANKS,
        ),
        "afd-dsv4-flash-graph-dbo-2a2f": (
            False,
            True,
            True,
            DSV4_GPU_ROLE_RANKS,
            DSV4_GPU_ROLE_RANKS,
        ),
    }
)
DSV4_GPU_SCENARIOS = tuple(DSV4_GPU_SCENARIO_SETTINGS)
DSV4_GPU_BASIC_SCENARIOS = tuple(
    name for name, settings in DSV4_GPU_SCENARIO_SETTINGS.items() if not settings[2]
)
DSV4_GPU_DBO_SCENARIOS = tuple(
    name for name, settings in DSV4_GPU_SCENARIO_SETTINGS.items() if settings[2]
)


def configure_scenario(args: argparse.Namespace) -> None:
    """Fix the synchronous text-only CUDA contract for native and AFD controls."""
    if args.device_backend != "gpu":
        raise ValueError("CUDA DSV4 scenarios require --device-backend=gpu")
    if args.completion_output_path is None or args.gsm8k_output_path is None:
        raise ValueError("CUDA DSV4 requires completion and GSM8K output paths")
    if args.common_vllm_arg or args.attention_vllm_arg or args.ffn_vllm_arg:
        raise ValueError("CUDA DSV4 profiles do not accept extra vLLM arguments")
    if args.use_decode_bench_connector:
        raise ValueError("CUDA DSV4 qualification runs without KV transfer")
    args.afd_connector = "P2pNcclAFDConnector"
    args.afd_async = False
    args.compute_gate_on_attention = False
    args.afd_connector_extra_config = []
    # Let the checkpoint and the exact target's device-aware native oracle
    # select quantization/MoE kernels on both paths. Do not copy Ascend W4A8,
    # HCCL, or MegaMoE settings into this CUDA profile.
    args.common_vllm_arg = [
        "--max-model-len",
        str(DSV4_GPU_MAX_MODEL_LEN),
        "--max-num-seqs",
        str(DSV4_GPU_CAPTURE_SIZE),
        "--max-num-batched-tokens",
        str(DSV4_GPU_MAX_BATCHED_TOKENS),
        "--gpu-memory-utilization",
        str(DSV4_GPU_MEMORY_UTILIZATION),
        "--tokenizer-mode",
        "deepseek_v4",
        "--kv-cache-dtype",
        "fp8",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--seed",
        str(DSV4_GPU_SEED),
        "--trust-remote-code",
    ]
