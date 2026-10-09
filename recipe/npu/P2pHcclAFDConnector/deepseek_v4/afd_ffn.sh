#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ROLE=ffn
MODEL_PATH="${MODEL_PATH:-}"
NIC_NAME="${NIC_NAME:-}"
HCCL_IF_IP="${HCCL_IF_IP:-}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
ATTENTION_DEVICES="${ATTENTION_DEVICES:-0,1,2,3}"
FFN_DEVICES="${FFN_DEVICES:-4,5}"
ATTENTION_RANKS="${ATTENTION_RANKS:-4}"
FFN_RANKS="${FFN_RANKS:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
U_BATCHES="${U_BATCHES:-1}"

API_HOST="${FFN_API_HOST:-127.0.0.1}"
API_PORT="${FFN_API_PORT:-8911}"
ROLE_DEVICES="$FFN_DEVICES"
ROLE_RANKS="$FFN_RANKS"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${FFN_MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HCCL_IF_BASE_PORT="${FFN_HCCL_IF_BASE_PORT:-52000}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

preflight_role

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"ffn","connector":"P2pHcclAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

run_role_service "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd-ffn \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$FFN_RANKS" \
  --tensor-parallel-size 1 \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --shutdown-timeout 20 \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --block-size 32 \
  --kv-cache-dtype auto \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${SCHEDULING_ARGS[@]}" \
  "${UBATCH_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
