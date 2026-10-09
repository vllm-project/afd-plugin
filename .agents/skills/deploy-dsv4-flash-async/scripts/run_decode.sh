#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

require_common_network
: "${DSV4_MODEL:?Set DSV4_MODEL to the checkpoint directory}"
configure_network "${D_NODE_IP}"

LOG_FILE=${DECODE_LOG_FILE:-${LOG_DIR}/decode.log}
PID_FILE=${DECODE_PID_FILE:-${PID_DIR}/decode.pid}

KV_TRANSFER_CONFIG=$(printf '%s' "{
  \"kv_connector\": \"MooncakeHybridConnector\",
  \"kv_role\": \"kv_consumer\",
  \"kv_port\": \"${PD_KV_DECODE_PORT}\",
  \"engine_id\": \"1\",
  \"kv_connector_extra_config\": {
    \"prefill\": {\"dp_size\": ${PREFILL_DP_SIZE}, \"tp_size\": ${PREFILL_TP_SIZE}},
    \"decode\": {\"dp_size\": ${DECODE_DP_SIZE}, \"tp_size\": ${DECODE_TP_SIZE}}
  }
}")

DECODE_ADDITIONAL_CONFIG=$(printf '%s' '{
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": false
  },
  "enable_cpu_binding": true,
  "multistream_overlap_shared_expert": true,
  "recompute_scheduler_enable": false
}')

echo "Starting DSV4 D: DP${DECODE_DP_SIZE}TP${DECODE_TP_SIZE}+EP, devices=${DECODE_VISIBLE_DEVICES}"
ASCEND_RT_VISIBLE_DEVICES="${DECODE_VISIBLE_DEVICES}" \
VLLM_ASCEND_APPLY_DSV4_PATCH=${VLLM_ASCEND_APPLY_DSV4_PATCH:-1} \
nohup "${VLLM_CLI}" serve "${DSV4_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${DECODE_PORT}" \
  --api-server-count 1 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --data-parallel-size "${DECODE_DP_SIZE}" \
  --data-parallel-address "${D_NODE_IP}" \
  --data-parallel-rpc-port "${DECODE_DP_RPC_PORT}" \
  --tensor-parallel-size "${DECODE_TP_SIZE}" \
  --enable-expert-parallel \
  --seed 1024 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${DECODE_MAX_NUM_SEQS}" \
  --async-scheduling \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${DECODE_GPU_MEMORY_UTILIZATION}" \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${MODEL_LOADER_THREADS}}" \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  --additional-config "${DECODE_ADDITIONAL_CONFIG}" > "${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "Started D pid=$(<"${PID_FILE}") log=${LOG_FILE}"
