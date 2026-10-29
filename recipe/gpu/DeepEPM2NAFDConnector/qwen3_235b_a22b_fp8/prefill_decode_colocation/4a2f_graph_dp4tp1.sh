#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-235B-A22B-FP8 4A2F prefill/decode colocation on one node with 6 GPUs:
# Attention DP4/TP1 (or DP1/TP4 with ATTN_TP=4) and FFN DP2/TP1 (expert
# parallel), FULL_DECODE_ONLY CUDA graphs. Every knob is an environment
# variable (default in parentheses):
#
#   MODEL_PATH               weights dir or HF id (/path/model_weights/Qwen3-235B-A22B-FP8)
#   AFD_CONNECTOR            DeepEPM2NAFDConnector | P2pNcclAFDConnector
#                            (DeepEPM2NAFDConnector)
#   M2N_PAYLOAD_FORMAT       DeepEPM2NAFDConnector dispatch format (fp8_block128_fp32)
#   M2N_EXTRA_CONFIG         extra connector_extra_config JSON members for
#                            DeepEPM2NAFDConnector, e.g. "expert_num_sms": 130 (empty)
#   KV_CONNECTOR             none | decode_bench (none); see the README
#   KV_TRANSFER_CONFIG       raw --kv-transfer-config JSON for Attention;
#                            overrides KV_CONNECTOR (unset)
#   ENABLE_PREFIX_CACHING    1 | 0 (1)
#   ENABLE_DBO               1 | 0 (0)
#   DBO_DECODE_TOKEN_THRESHOLD / DBO_PREFILL_TOKEN_THRESHOLD (2 / 8)
#   MAX_NUM_SEQS             per Attention DP rank (128)
#   MAX_NUM_BATCHED_TOKENS   prefill chunk per step (3072)
#   CUDAGRAPH_CAPTURE_SIZES  comma-separated decode graph sizes (8,16,32,64,96,128)
#   MAX_MODEL_LEN            (4096)
#   GPU_MEMORY_UTILIZATION   (unset: vLLM default)
#   ATTN_DEVICES / FFN_DEVICES (0,1,2,3 / 4,5)
#   ATTN_TP                  Attention tensor-parallel size; Attention DP is the
#                            number of ATTN_DEVICES divided by it (1)
#   AFD_HOST / AFD_PORT      AFD connector rendezvous (127.0.0.1 / 6269)
#   AFD_PROCESS_GROUP_TIMEOUT_S  AFD join timeout; the roles can finish
#                            loading minutes apart (7200)
#   ATTN_PORT / FFN_PORT     HTTP ports (18305 / 18306)
#   EXTRA_ARGS               extra `vllm serve` args for both roles (empty)
#   ATTN_EXTRA_ARGS / FFN_EXTRA_ARGS  extra `vllm serve` args for one role (empty)
#
# KV_CONNECTOR=decode_bench only (tools/benchmarks/decode_bench.py):
#   DECODE_BENCH_NUM_PROMPTS     pool prompts per Attention rank; 0 = dummy KV (500)
#   DECODE_BENCH_PROMPT_LEN      pool prompt length (1025)
#   DECODE_BENCH_MAX_OUTPUT_LEN  longest measured output; sizes the KV check (256)
#   DECODE_BENCH_SEED            pool token seed (0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
M2N_PAYLOAD_FORMAT=${M2N_PAYLOAD_FORMAT:-fp8_block128_fp32}
M2N_EXTRA_CONFIG=${M2N_EXTRA_CONFIG:-}
KV_CONNECTOR=${KV_CONNECTOR:-none}
KV_TRANSFER_CONFIG=${KV_TRANSFER_CONFIG:-}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-2}
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-3072}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64,96,128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
ATTN_DEVICES=${ATTN_DEVICES:-0,1,2,3}
FFN_DEVICES=${FFN_DEVICES:-4,5}
ATTN_TP=${ATTN_TP:-1}
AFD_HOST=${AFD_HOST:-127.0.0.1}
AFD_PORT=${AFD_PORT:-6269}
AFD_PROCESS_GROUP_TIMEOUT_S=${AFD_PROCESS_GROUP_TIMEOUT_S:-7200}
ATTN_PORT=${ATTN_PORT:-18305}
FFN_PORT=${FFN_PORT:-18306}
EXTRA_ARGS=${EXTRA_ARGS:-}
ATTN_EXTRA_ARGS=${ATTN_EXTRA_ARGS:-}
FFN_EXTRA_ARGS=${FFN_EXTRA_ARGS:-}
DECODE_BENCH_NUM_PROMPTS=${DECODE_BENCH_NUM_PROMPTS:-500}
DECODE_BENCH_PROMPT_LEN=${DECODE_BENCH_PROMPT_LEN:-1025}
DECODE_BENCH_MAX_OUTPUT_LEN=${DECODE_BENCH_MAX_OUTPUT_LEN:-256}
DECODE_BENCH_SEED=${DECODE_BENCH_SEED:-0}

export VLLM_USE_V2_MODEL_RUNNER=1

NUM_ATTN_RANKS=$(tr ',' '\n' <<<"$ATTN_DEVICES" | wc -l | tr -d ' ')
NUM_FFN_RANKS=$(tr ',' '\n' <<<"$FFN_DEVICES" | wc -l | tr -d ' ')
if ((NUM_ATTN_RANKS % ATTN_TP)); then
    echo "ATTN_TP=${ATTN_TP} does not divide the ${NUM_ATTN_RANKS} Attention devices" >&2
    exit 1
fi
ATTN_DP=$((NUM_ATTN_RANKS / ATTN_TP))

case "$AFD_CONNECTOR" in
    # DeepEP M2N routes on Attention and sends FP8 tokens to the FFN experts.
    DeepEPM2NAFDConnector)
        COMPUTE_GATE_ON_ATTENTION=true
        CONNECTOR_EXTRA_CONFIG=", \"connector_extra_config\": {\"payload_format\": \"${M2N_PAYLOAD_FORMAT}\"${M2N_EXTRA_CONFIG:+, ${M2N_EXTRA_CONFIG}}}"
        ;;
    # P2P routes on the FFN and sends BF16 hidden states. Do not run its FFN
    # with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the first step hangs.
    P2pNcclAFDConnector)
        COMPUTE_GATE_ON_ATTENTION=false
        CONNECTOR_EXTRA_CONFIG=""
        ;;
    *)
        echo "unsupported AFD_CONNECTOR=${AFD_CONNECTOR}" >&2
        exit 1
        ;;
esac

if [[ -z "$KV_TRANSFER_CONFIG" ]]; then
    case "$KV_CONNECTOR" in
        none) ;;
        decode_bench)
            KV_TRANSFER_CONFIG=$(cat <<JSON
{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"num_prompts":${DECODE_BENCH_NUM_PROMPTS},"prompt_len":${DECODE_BENCH_PROMPT_LEN},"seed":${DECODE_BENCH_SEED},"max_output_len":${DECODE_BENCH_MAX_OUTPUT_LEN}}}
JSON
)
            ;;
        *)
            echo "unsupported KV_CONNECTOR=${KV_CONNECTOR}" >&2
            exit 1
            ;;
    esac
fi
if [[ "$KV_TRANSFER_CONFIG" == *tools.benchmarks.* ]]; then
    # tools/benchmarks is not in the wheel; the vLLM scheduler and worker
    # subprocesses import it from the repository root.
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

afd_config() {
    cat <<JSON
{"afd": {"role": "$1", "connector": "${AFD_CONNECTOR}", "host": "${AFD_HOST}", "port": ${AFD_PORT}, "num_attention_ranks": ${NUM_ATTN_RANKS}, "num_ffn_ranks": ${NUM_FFN_RANKS}, "compute_gate_on_attention": ${COMPUTE_GATE_ON_ATTENTION}, "afd_process_group_timeout_s": ${AFD_PROCESS_GROUP_TIMEOUT_S}${CONNECTOR_EXTRA_CONFIG}}}
JSON
}

MAX_CUDAGRAPH_CAPTURE_SIZE=$(tr ',' '\n' <<<"$CUDAGRAPH_CAPTURE_SIZES" | sort -n | tail -n 1)
COMMON_ARGS=(
    --enable-expert-parallel
    --enable-ep-weight-filter
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
    # A cold load of the 235B checkpoint can outlast vLLM's 1800 s gloo timeout
    # between DP peers.
    --cpu-distributed-timeout-seconds 7200
    --trust-remote-code
)
if [[ "$ENABLE_DBO" == 1 ]]; then
    COMMON_ARGS+=(
        --enable-dbo
        --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
        --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    COMMON_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
# shellcheck disable=SC2206
COMMON_ARGS+=($EXTRA_ARGS)

ATTN_ARGS=(--data-parallel-size "$ATTN_DP" --tensor-parallel-size "$ATTN_TP")
if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
    ATTN_ARGS+=(--enable-prefix-caching)
else
    ATTN_ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
    ATTN_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi
# shellcheck disable=SC2206
ATTN_ARGS+=($ATTN_EXTRA_ARGS)

FFN_ARGS=(--data-parallel-size "$NUM_FFN_RANKS" --tensor-parallel-size 1)
# shellcheck disable=SC2206
FFN_ARGS+=($FFN_EXTRA_ARGS)

CUDA_VISIBLE_DEVICES=$FFN_DEVICES uv run vllm serve "$MODEL_PATH" \
    --additional-config "$(afd_config ffn)" \
    "${COMMON_ARGS[@]}" \
    "${FFN_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$FFN_PORT" > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=$ATTN_DEVICES uv run vllm serve "$MODEL_PATH" \
    --additional-config "$(afd_config attention)" \
    "${COMMON_ARGS[@]}" \
    "${ATTN_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$ATTN_PORT" > attn.log 2>&1 &

wait
