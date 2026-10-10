#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ROLE=attention
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

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8910}"
ROLE_DEVICES="$ATTENTION_DEVICES"
ROLE_RANKS="$ATTENTION_RANKS"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${ATTENTION_MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

preflight_role

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"attention","connector":"P2pHcclAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"
PD_ARGS=()
if [[ "${ENABLE_PD:-0}" == 1 ]]; then
  : "${PREFILL_DP_SIZE:?Set PREFILL_DP_SIZE for Mooncake PD}"
  PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
  : "${DECODE_ENGINE_ID:?Set DECODE_ENGINE_ID for Mooncake PD}"
  : "${DECODE_KV_PORT:?Set DECODE_KV_PORT for Mooncake PD}"
  KV_CONFIG="$(build_a5_pd_kv_config kv_consumer "$DECODE_ENGINE_ID" "$DECODE_KV_PORT" "$ATTENTION_DEVICES")"
  printf '[dsv4-afd] A5 Attention Mooncake KV config=%s\n' "$KV_CONFIG"
  PD_ARGS=(--kv-transfer-config "$KV_CONFIG")
fi
DSPARK_ARGS=()
if [[ "${ENABLE_DSPARK:-0}" == 1 ]]; then
  DSPARK_BLOCK_SIZE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dspark_block_size"])' "$MODEL_PATH/config.json")"
  [[ "$DSPARK_BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || afd_die "MODEL_PATH must be a DSpark checkpoint with dspark_block_size"
  # The pinned vLLM 0.23 schema exposes DSpark through the MTP method. The
  # Ascend runtime selects AscendDSparkProposer from the checkpoint marker.
  DSPARK_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"greedy","enforce_eager":true}' "$DSPARK_BLOCK_SIZE")"
  DSPARK_ARGS=(--speculative-config "$DSPARK_CONFIG")
fi

run_role_service "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$ATTENTION_RANKS" \
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
  "${PD_ARGS[@]}" \
  "${DSPARK_ARGS[@]}" \
  "${SCHEDULING_ARGS[@]}" \
  "${UBATCH_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
