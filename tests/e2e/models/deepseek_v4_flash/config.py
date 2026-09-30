# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed DSV4 Flash deployment and acceptance parameters.

Two transports are covered: the asynchronous CAM connector, and the
synchronous CAMP2P connector, which needs no CAM vendor package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

DSV4_ASYNC_CAM_SCENARIO = "afd-dsv4-flash-async-cam-dp2tp4-ep8"
DSV4_SYNC_CAMP2P_A5_SCENARIO = "afd-dsv4-flash-sync-camp2p-2a2f"
DSV4_SYNC_CAMP2P_A3_SCENARIO = "afd-dsv4-flash-sync-camp2p-8a8f"
DSV4_SYNC_CAMP2P_SCENARIOS = (
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
)
DSV4_SCENARIOS = (DSV4_ASYNC_CAM_SCENARIO, *DSV4_SYNC_CAMP2P_SCENARIOS)
DSV4_ATTENTION_RANKS = 8
DSV4_FFN_RANKS = 8
DSV4_ATTENTION_TP_SIZE = 4


DSV4_SYNC_CAMP2P_CONNECTOR = "CAMP2pAFDConnector"
# CAMP2P sizes its AFD HCCL domains through this override; the A5 and A3 runs
# used 2048 MB. quant_mode stays 0, the only mode the runtime accepts today.
DSV4_SYNC_HCCL_BUFFER_SIZE_MB = 2048
DSV4_SYNC_QUANT_MODE = 0
DSV4_ASCEND_QUANTIZATION = "ascend"
# Context, batch, and memory budget. The asynchronous case keeps the 16-die
# budget it was validated with; the synchronous cases use the smaller profile
# their launch scripts record.
DSV4_MAX_MODEL_LEN = "1048576"
DSV4_MAX_NUM_BATCHED_TOKENS = "8192"
DSV4_MAX_NUM_SEQS = "16"
DSV4_MEMORY_UTILIZATION = "0.7"
# The block the DSV4 DSA, compressor, and indexer caches are laid out for; the
# case's own deployment pins it, and a verbatim profile records it only where
# its host script leaves a default the case cannot run correctly with.
DSV4_BLOCK_SIZE = "128"
DSV4_SYNC_MAX_MODEL_LEN = "8192"
DSV4_SYNC_MAX_NUM_BATCHED_TOKENS = "1024"
DSV4_CONCURRENT_REQUESTS = 10
DSV4_REQUEST_TIMEOUT_S = 300
DSV4_COMPLETION_MAX_TOKENS = 256
DSV4_PROMPT_FIRST_OPERAND = 12
DSV4_PROMPT_SECOND_OPERAND = 7
# Sixteen NPU workers take longer than the small cases to destroy HCCL
# resources; the observed launcher shutdown alone exceeded 20 seconds.
DSV4_PROCESS_TERMINATION_TIMEOUT_S = 60
# The A5 recorded launch script runs a 4096 context with ACL graph capture and
# native DBO instead of the eager 8192/1024 deployment the A3 profile records.
# The case keeps the graph capture and the context but runs without DBO: its
# split path is the current suspect for the DSA attention tiling failure on A5.
# The script's thresholds stay recorded on the profile below, so re-enabling DBO
# is a one-field change.
DSV4_SYNC_A5_MAX_MODEL_LEN = "4096"
DSV4_SYNC_A5_CUDAGRAPH_CAPTURE_SIZE = 16
DSV4_SYNC_ALLOC_CONF_EXPANDABLE = "expandable_segments:True"
DSV4_SYNC_ALLOC_CONF_PLAIN = "expandable_segments:False"
# The A5 script runs both roles on one host and announces the loopback address.
DSV4_SYNC_LOCAL_AFD_HOST = "127.0.0.1"
# The A3 profile sizes its CAMP2P domains per domain instead of through the
# caller's global HCCL_BUFFSIZE.
DSV4_SYNC_CONNECTOR_EXTRA_CONFIG = {
    "hccl_buffer_size": DSV4_SYNC_HCCL_BUFFER_SIZE_MB,
    "quant_mode": DSV4_SYNC_QUANT_MODE,
}


# DeepSeek V4 does not fit on one Attention or one FFN die, so each host runs the
# smallest shape its recorded launch script works with, and the two differ in
# more than rank count:
#
# - A5 runs Attention DP2/TP1 and FFN DP2/TP1 with expert parallelism, ACL graph
#   capture over 16 decodes, and the script's 4096 context, but without the
#   script's native DBO, with prefix caching off, and with the case's 128-token
#   block. It omits `--quantization`, `connector_extra_config`,
#   and the multithread loader, and exports HCCL_BUFFSIZE=2048 itself.
# - A3 runs 8A8F on sixteen dies with the parallelism its recorded deployment
#   uses — Attention DP2/TP4 and FFN DP8/TP1 with expert parallelism — eager,
#   with the 8192/1024 budget. Eight dies do not fit the checkpoint there.
class DSV4SyncShape(NamedTuple):
    """Fixed synchronous CAMP2P deployment profile for one host class."""

    attention_ranks: int
    ffn_ranks: int
    # Process environment that host's launch script relies on. AFD forces spawn
    # multiprocessing so workers re-initialize the device in a fresh process, and
    # the A5 script relies on the platform default instead.
    force_spawn: bool = True
    # CAMP2P sizes its own AFD domains, so an inherited global HCCL_BUFFSIZE must
    # not leak in; A5 exports one instead of sizing per domain. The recorded A3
    # rendezvous uses the caller's address and so requires the NIC variables,
    # while the A5 script starts both roles on 127.0.0.1.
    keep_hccl_buffsize: bool = False
    npu_alloc_conf: str = DSV4_SYNC_ALLOC_CONF_EXPANDABLE
    nic_env_required: bool = True
    # Tensor-parallel size of each role; the remaining ranks of a role shard by
    # data parallel, with expert parallelism where the host enables it.
    attention_tp_size: int = 1
    ffn_tp_size: int = 1
    enable_expert_parallel: bool = False
    use_graph: bool = False
    cudagraph_capture_size: int = 0
    max_model_len: str = DSV4_SYNC_MAX_MODEL_LEN
    # None omits the flag entirely so vLLM's own default applies.
    max_num_batched_tokens: str | None = DSV4_SYNC_MAX_NUM_BATCHED_TOKENS
    max_num_seqs: str | None = DSV4_MAX_NUM_SEQS
    memory_utilization: str | None = DSV4_MEMORY_UTILIZATION
    multithread_load: bool = True
    connector_extra_config: dict[str, int] | None = DSV4_SYNC_CONNECTOR_EXTRA_CONFIG
    quantization_from_checkpoint: bool = True
    # A profile whose host does not return reliable answers yet sets this false:
    # the concurrent oracle then checks that the ten requests were served
    # together and finished, without comparing the answer. The exact check stays
    # on for the asynchronous case, which is the one host that answers it.
    check_answer: bool = True
    # A profile whose host launch script is the validated deployment emits only
    # the flags that script passes, instead of the case's extra deployment
    # defaults (API server count, seed, block size, prefix caching, chunked
    # prefill, and the Attention data-parallel address).
    verbatim_launch: bool = False
    # Cache layout a verbatim profile records when its script leaves the default:
    # a None block size keeps the script's own, and prefix caching stays on
    # unless the profile turns it off.
    disable_prefix_caching: bool = False
    block_size: str | None = None
    # Exact `--compilation-config` JSON when the script passes one; the runner
    # then omits its own capture-size flags.
    compilation_config: dict[str, object] | None = None

    @property
    def device_count(self) -> int:
        return self.attention_ranks + self.ffn_ranks


DSV4_SYNC_SHAPES = {
    DSV4_SYNC_CAMP2P_A5_SCENARIO: DSV4SyncShape(
        attention_ranks=2,
        ffn_ranks=2,
        enable_expert_parallel=True,
        use_graph=True,
        cudagraph_capture_size=DSV4_SYNC_A5_CUDAGRAPH_CAPTURE_SIZE,
        max_model_len=DSV4_SYNC_A5_MAX_MODEL_LEN,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        memory_utilization=None,
        multithread_load=False,
        connector_extra_config=None,
        quantization_from_checkpoint=False,
        verbatim_launch=True,
        # The script leaves prefix caching and the block size at vLLM's defaults.
        # Its ten concurrent chat requests share one template prefix, and that
        # reuse is the current suspect for the corrupted answers this profile
        # produces on A5; the DSA caches are laid out for the 128-token block the
        # case pins on both hosts.
        disable_prefix_caching=True,
        block_size=DSV4_BLOCK_SIZE,
        compilation_config={
            "cudagraph_capture_sizes": [DSV4_SYNC_A5_CUDAGRAPH_CAPTURE_SIZE],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        # This host does not answer the concurrent oracle reliably yet (see the
        # sync CAMP2P blocker in tests/e2e/README.md), so the case covers the
        # concurrent plumbing — ten requests served and finished — and leaves
        # the sum to the hosts that answer it.
        check_answer=False,
        # The A5 script runs both roles on one host, keeps the platform's
        # multiprocessing start method, and exports its own HCCL_BUFFSIZE.
        nic_env_required=False,
        force_spawn=False,
        keep_hccl_buffsize=True,
        npu_alloc_conf=DSV4_SYNC_ALLOC_CONF_PLAIN,
    ),
    DSV4_SYNC_CAMP2P_A3_SCENARIO: DSV4SyncShape(
        # Sixteen dies: eight do not fit the checkpoint on this host. Attention
        # shards DP2/TP4 and FFN DP8/TP1 with expert parallelism, the shape its
        # recorded deployment uses.
        attention_ranks=8,
        ffn_ranks=8,
        attention_tp_size=4,
        enable_expert_parallel=True,
        # Smoke coverage, like A5: this host has not been validated against the
        # answer oracle, so the case checks the concurrent plumbing only.
        check_answer=False,
    ),
}


def sync_use_graph(shape: DSV4SyncShape) -> bool:
    """Return whether the profile captures ACL graphs instead of running eager."""
    return shape.use_graph


def sync_compilation_config(shape: DSV4SyncShape) -> dict[str, object] | None:
    """Return the profile's exact `--compilation-config`, or None when eager."""
    if not sync_use_graph(shape):
        return None
    return shape.compilation_config


def sync_shape(scenario: str) -> DSV4SyncShape | None:
    """Return the synchronous launch profile of a scenario, if it has one."""
    return DSV4_SYNC_SHAPES.get(scenario)


def _declared_quant_method(model: str) -> str | None:
    """Return the quantization method the checkpoint config declares, if any."""
    try:
        config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return None
    method = quantization_config.get("quant_method")
    return method if isinstance(method, str) and method else None


def sync_camp2p_quantization(model: str, shape: DSV4SyncShape) -> str | None:
    """Resolve the `--quantization` value a synchronous case should pass.

    The A5 profile follows its launch script, which passes no `--quantization`
    at all and lets the FP8/W4A8 checkpoint decide. The A3 int8 W8A8 checkpoint
    is loaded through the Ascend method, so that profile resolves `ascend` from
    the checkpoint's own declaration and omits the flag when the checkpoint
    declares something else.
    """
    if not shape.quantization_from_checkpoint:
        return None
    declared = _declared_quant_method(model)
    if declared is not None and declared != DSV4_ASCEND_QUANTIZATION:
        return None
    return DSV4_ASCEND_QUANTIZATION


def _configure_dsv4_arguments(
    args: argparse.Namespace,
    quantization: str | None,
    *,
    max_model_len: str = DSV4_MAX_MODEL_LEN,
    max_num_batched_tokens: str | None = DSV4_MAX_NUM_BATCHED_TOKENS,
    max_num_seqs: str | None = DSV4_MAX_NUM_SEQS,
    memory_utilization: str | None = DSV4_MEMORY_UTILIZATION,
    multithread_load: bool = True,
    verbatim_launch: bool = False,
    disable_prefix_caching: bool = False,
    block_size: str | None = None,
) -> None:
    """Apply the fixed model arguments shared by every DSV4 scenario.

    `quantization` is the `--quantization` value to pass, or None to omit the
    flag so the checkpoint's own configuration decides. The remaining keyword
    arguments default to the asynchronous case's fixed deployment; the
    synchronous cases resolve them per host, because the recorded A5 and A3
    deployments differ from the 16-die asynchronous case. A None budget value
    omits that flag entirely so vLLM's own default applies, and
    `multithread_load` selects the scripted multithreaded weight loader.

    `verbatim_launch` emits only what a recorded `vllm serve` script passes,
    because for that host the script is the validated deployment: no API server
    count, seed, batch or memory budget, block size, prefix-caching, or
    chunked-prefill flag. Tokenizer mode and the remote-code flag stay, since the
    concurrent oracle needs the model's chat template and tokenizer. A verbatim
    profile may still record the cache layout it needs (`block_size`,
    `disable_prefix_caching`) for a host whose script leaves a default the case
    cannot run correctly with.
    """
    if args.completion_output_path is None:
        raise ValueError("--completion-output-path is required for DSV4")
    # Keep these local cases aligned with the DSV4 prefill scripts.
    # Reject ad-hoc overrides so a case ID denotes one fixed deployment.
    if args.common_vllm_arg or args.attention_vllm_arg or args.ffn_vllm_arg:
        raise ValueError("DSV4 scenario does not accept extra vLLM arguments")
    if args.use_decode_bench_connector:
        raise ValueError("DSV4 scenario runs without a KV transfer connector")
    deployment_defaults = (
        [] if verbatim_launch else ["--api-server-count", "1", "--seed", "1024"]
    )
    # The case's own deployment pins the 128-token block the DSA caches are laid
    # out for and disables prefix caching; a verbatim profile takes its script's
    # default unless it records one of the two.
    block_size_flags = (
        ["--block-size", block_size]
        if block_size is not None
        else ([] if verbatim_launch else ["--block-size", DSV4_BLOCK_SIZE])
    )
    cache_flags = (
        ["--no-enable-prefix-caching"]
        if disable_prefix_caching or not verbatim_launch
        else []
    )
    if not verbatim_launch:
        cache_flags.append("--enable-chunked-prefill")
    args.common_vllm_arg = [
        *deployment_defaults,
        "--max-model-len",
        max_model_len,
        *(
            ["--max-num-batched-tokens", max_num_batched_tokens]
            if max_num_batched_tokens is not None
            else []
        ),
        *(["--max-num-seqs", max_num_seqs] if max_num_seqs is not None else []),
        *block_size_flags,
        *(
            ["--gpu-memory-utilization", memory_utilization]
            if memory_utilization is not None
            else []
        ),
        *(["--quantization", quantization] if quantization is not None else []),
        "--tokenizer-mode",
        "deepseek_v4",
        *(
            [
                "--model-loader-extra-config",
                json.dumps({"enable_multithread_load": True, "num_threads": 128}),
            ]
            if multithread_load
            else []
        ),
        "--trust-remote-code",
        *cache_flags,
    ]
    args.attention_vllm_arg = [
        *(
            []
            if verbatim_launch
            else [
                "--data-parallel-address",
                args.afd_host,
                "--no-disable-hybrid-kv-cache-manager",
            ]
        ),
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v4",
    ]


def configure_scenario(args: argparse.Namespace) -> None:
    """Configure the 16-NPU asynchronous CAM deployment."""
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
    _configure_dsv4_arguments(args, DSV4_ASCEND_QUANTIZATION)


def configure_sync_camp2p_scenario(args: argparse.Namespace) -> None:
    """Configure a synchronous CAMP2P deployment from its host profile.

    The caller selects the A5 (2A2F) or A3 (8A8F) host through the scenario;
    both keep the gate on FFN, because CAMP2P carries the Hash-layer token ids
    over the a2e ids channel, and neither needs a CAM vendor package. The rest
    of the deployment follows that host's recorded launch script: its rank
    layout, per-role tensor parallelism and expert parallelism, graph capture,
    the context and batch budget, the weight loader, and whether the case or the
    caller's shell sizes the CAMP2P HCCL domains. The A5 script's native DBO is
    the one recorded setting the profile carries but leaves off, while its split
    path is root-caused.
    """
    shape = DSV4_SYNC_SHAPES[args.scenario]
    args.afd_connector = DSV4_SYNC_CAMP2P_CONNECTOR
    args.afd_async = False
    args.compute_gate_on_attention = False
    args.afd_connector_extra_config = (
        []
        if shape.connector_extra_config is None
        else [json.dumps(shape.connector_extra_config, separators=(",", ":"))]
    )
    _configure_dsv4_arguments(
        args,
        sync_camp2p_quantization(args.model, shape),
        max_model_len=shape.max_model_len,
        max_num_batched_tokens=shape.max_num_batched_tokens,
        max_num_seqs=shape.max_num_seqs,
        memory_utilization=shape.memory_utilization,
        multithread_load=shape.multithread_load,
        verbatim_launch=shape.verbatim_launch,
        disable_prefix_caching=shape.disable_prefix_caching,
        block_size=shape.block_size,
    )


def additional_config() -> dict[str, bool]:
    """Return the DSV4 model-path switches every DSV4 case pins.

    These are not deployment preferences. The pinned Ascend runtime defaults
    `multistream_dsv4_dsa_overlap` to True (`vllm_ascend/ascend_config.py`), and
    that path drives the DSA RoPE through `inplace_partial_rotary_mul`, whose
    tiling function rejects the shapes A5 hands it. The case therefore keeps the
    switch off, alongside the DSA context-parallel and shared-compressor paths
    it does not cover.
    """
    return {
        "enable_cpu_binding": True,
        "enable_force_load_balance": False,
        "enable_dsa_cp": False,
        "multistream_dsv4_dsa_overlap": False,
        "enable_dsv4_shared_compressor_workspace": False,
    }


def role_environment(role: str | None) -> dict[str, str]:
    return {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1" if role == "attention" else "0"}
