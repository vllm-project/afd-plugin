#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B 2A2F prefill/decode colocation on one 4-GPU node: Attention
# DP2/TP1 and FFN DP2/TP1 (expert parallel), FULL_DECODE_ONLY CUDA graphs.
# Every knob is an environment variable (default in parentheses):
#
#   MODEL_PATH               weights dir or HF id (/path/model_weights/Qwen3-30B-A3B)
#   AFD_CONNECTOR            DeepEPM2NAFDConnector | P2pNcclAFDConnector
#                            (DeepEPM2NAFDConnector)
#   KV_CONNECTOR             none | decode_bench (none); see the README
#   KV_TRANSFER_CONFIG       raw --kv-transfer-config JSON for Attention;
#                            overrides KV_CONNECTOR (unset)
#   ENABLE_PREFIX_CACHING    1 | 0 (1)
#   ENABLE_DBO               1 | 0 (0)
#   DBO_DECODE_TOKEN_THRESHOLD / DBO_PREFILL_TOKEN_THRESHOLD (2 / 12)
#   MAX_NUM_SEQS             (64)
#   MAX_NUM_BATCHED_TOKENS   prefill chunk per step (4096)
#   CUDAGRAPH_CAPTURE_SIZES  comma-separated decode graph sizes (8,16,32,64)
#   MAX_MODEL_LEN            (4096)
#   GPU_MEMORY_UTILIZATION   (unset: vLLM default)
#   ATTN_DEVICES / FFN_DEVICES (0,1 / 2,3)
#   AFD_HOST / AFD_PORT      AFD connector rendezvous (127.0.0.1 / 6239)
#   ATTN_PORT / FFN_PORT     HTTP ports (18000 / 18001)
#   EXTRA_ARGS               extra `vllm serve` args for both roles (empty)
#
# KV_CONNECTOR=decode_bench only (tools/benchmarks/decode_bench.py):
#   DECODE_BENCH_NUM_PROMPTS     pool prompts per Attention rank; 0 = dummy KV (1000)
#   DECODE_BENCH_PROMPT_LEN      pool prompt length (1025)
#   DECODE_BENCH_MAX_OUTPUT_LEN  longest measured output; sizes the KV check (1024)
#   DECODE_BENCH_SEED            pool token seed (0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
KV_CONNECTOR=${KV_CONNECTOR:-none}
KV_TRANSFER_CONFIG=${KV_TRANSFER_CONFIG:-}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-2}
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-12}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
ATTN_DEVICES=${ATTN_DEVICES:-0,1}
FFN_DEVICES=${FFN_DEVICES:-2,3}
AFD_HOST=${AFD_HOST:-127.0.0.1}
AFD_PORT=${AFD_PORT:-6239}
ATTN_PORT=${ATTN_PORT:-18000}
FFN_PORT=${FFN_PORT:-18001}
EXTRA_ARGS=${EXTRA_ARGS:-}
DECODE_BENCH_NUM_PROMPTS=${DECODE_BENCH_NUM_PROMPTS:-1000}
DECODE_BENCH_PROMPT_LEN=${DECODE_BENCH_PROMPT_LEN:-1025}
DECODE_BENCH_MAX_OUTPUT_LEN=${DECODE_BENCH_MAX_OUTPUT_LEN:-1024}
DECODE_BENCH_SEED=${DECODE_BENCH_SEED:-0}

export VLLM_USE_V2_MODEL_RUNNER=1

case "$AFD_CONNECTOR" in
    # DeepEP M2N routes on Attention: the gate and top-k run there.
    DeepEPM2NAFDConnector) COMPUTE_GATE_ON_ATTENTION=true ;;
    P2pNcclAFDConnector) COMPUTE_GATE_ON_ATTENTION=false ;;
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
{"afd": {"role": "$1", "connector": "${AFD_CONNECTOR}", "host": "${AFD_HOST}", "port": ${AFD_PORT}, "num_attention_ranks": 2, "num_ffn_ranks": 2, "compute_gate_on_attention": ${COMPUTE_GATE_ON_ATTENTION}}}
JSON
}

MAX_CUDAGRAPH_CAPTURE_SIZE=$(tr ',' '\n' <<<"$CUDAGRAPH_CAPTURE_SIZES" | sort -n | tail -n 1)
COMMON_ARGS=(
    --data-parallel-size 2
    --tensor-parallel-size 1
    --enable-expert-parallel
    --enable-ep-weight-filter
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
    --trust-remote-code
)
if [[ "$ENABLE_DBO" == 1 ]]; then
    COMMON_ARGS+=(
        --enable-dbo
        --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
        --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
    COMMON_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    COMMON_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
# shellcheck disable=SC2206
COMMON_ARGS+=($EXTRA_ARGS)

ATTN_ARGS=()
if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
    ATTN_ARGS+=(--enable-prefix-caching)
else
    ATTN_ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
    ATTN_ARGS+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi

CUDA_VISIBLE_DEVICES=$FFN_DEVICES vllm serve "$MODEL_PATH" \
    --additional-config "$(afd_config ffn)" \
    "${COMMON_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$FFN_PORT" > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=$ATTN_DEVICES vllm serve "$MODEL_PATH" \
    --additional-config "$(afd_config attention)" \
    "${COMMON_ARGS[@]}" \
    "${ATTN_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$ATTN_PORT" > attn.log 2>&1 &

wait
