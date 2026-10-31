#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-235B-A22B-FP8 vanilla vLLM baseline on one 4-GPU node.
#
# This is the no-AFD counterpart to
# recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh.
# It launches a single vLLM process with tensor parallelism 4 and no AFD
# connector, AFD additional config, or separate Attention/FFN roles.
#
# Every knob is an environment variable:
#
#   MODEL_PATH               weights dir or HF id (/path/model_weights/Qwen3-235B-A22B-FP8)
#   TENSOR_PARALLEL_SIZE     tensor-parallel size (4)
#   CUDA_VISIBLE_DEVICES     GPUs to use (0,1,2,3)
#   HOST / PORT              HTTP bind address and port (127.0.0.1 / 18000)
#   MAX_NUM_SEQS             maximum sequences (128)
#   MAX_NUM_BATCHED_TOKENS   prefill chunk per step (3072)
#   MAX_MODEL_LEN            maximum model length (34816)
#   CUDAGRAPH_CAPTURE_SIZES  comma-separated decode graph sizes
#                            (8,16,32,48,64,96,112,128)
#   ENABLE_PREFIX_CACHING    1 | 0 (1)
#   ENABLE_DBO               1 | 0 (0)
#   DBO_DECODE_TOKEN_THRESHOLD / DBO_PREFILL_TOKEN_THRESHOLD (2 / 2)
#   ALL2ALL_BACKEND          MoE all-to-all backend (allgather_reducescatter);
#                            DBO requires DeepEP or NIXL instead
#   GPU_MEMORY_UTILIZATION   optional GPU memory utilization fraction (unset)
#   EXTRA_ARGS               extra `vllm serve` arguments (empty)

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-3072}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34816}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,48,64,96,112,128}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-2}
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-2}
ALL2ALL_BACKEND=${ALL2ALL_BACKEND:-allgather_reducescatter}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES

MAX_CUDAGRAPH_CAPTURE_SIZE=$(tr ',' '\n' <<<"$CUDAGRAPH_CAPTURE_SIZES" | sort -n | tail -n 1)

ARGS=(
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --enable-expert-parallel
    --enable-ep-weight-filter
    --all2all-backend "$ALL2ALL_BACKEND"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
    --cpu-distributed-timeout-seconds 3600
    --trust-remote-code
)

if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
    ARGS+=(--enable-prefix-caching)
else
    ARGS+=(--no-enable-prefix-caching)
fi

if [[ "$ENABLE_DBO" == 1 ]]; then
    ARGS+=(
        --enable-dbo
        --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
        --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
fi

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

exec vllm serve "$MODEL_PATH" \
    "${ARGS[@]}" \
    --host "$HOST" \
    --port "$PORT"
