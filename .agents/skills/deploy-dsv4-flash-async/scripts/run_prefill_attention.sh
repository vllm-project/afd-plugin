#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

require_prefill_network
validate_topology
configure_network "${LOCAL_NODE_IP}"

LOG_FILE=${PREFILL_ATTN_LOG_FILE:-${LOG_DIR}/prefill_attention_node${PREFILL_NODE_ID}.log}
PID_FILE=${PREFILL_ATTN_PID_FILE:-${PID_DIR}/prefill_attention_node${PREFILL_NODE_ID}.pid}

ENABLE_CPU_BINDING=${ENABLE_CPU_BINDING:-true}
case "${ENABLE_CPU_BINDING}" in
  true | false) ;;
  *)
    echo "ENABLE_CPU_BINDING must be true or false" >&2
    exit 2
    ;;
esac

AFD_ADDITIONAL_CONFIG=$(printf '%s' "{
  \"enable_cpu_binding\": ${ENABLE_CPU_BINDING},
  \"enable_force_load_balance\": false,
  \"enable_dsa_cp\": false,
  \"multistream_dsv4_dsa_overlap\": false,
  \"enable_dsv4_shared_compressor_workspace\": true,
  \"afd\": {
    \"role\": \"attention\",
    \"connector\": \"CAMAsyncAFDConnector\",
    \"async\": true,
    \"host\": \"${AFD_HOST}\",
    \"port\": ${AFD_PORT},
    \"num_attention_ranks\": ${NUM_ATTENTION_RANKS},
    \"num_ffn_ranks\": ${NUM_FFN_RANKS},
    \"compute_gate_on_attention\": true,
    \"connector_extra_config\": {
      \"dynamicQuant\": 1,
      \"attn_ranks_per_dp\": ${ATTN_RANKS_PER_DP},
      \"async_moe_ubatching\": true,
      \"async_moe_num_ubatches\": 2,
      \"async_moe_split\": \"token\"
    }
  }
}")

KV_TRANSFER_ARGS=()
case "${PREFILL_ENABLE_KV_CONNECTOR}" in
  1)
    KV_TRANSFER_CONFIG=$(printf '%s' "{
      \"kv_connector\": \"MooncakeHybridConnector\",
      \"kv_role\": \"kv_producer\",
      \"kv_port\": \"${PD_KV_PREFILL_PORT}\",
      \"engine_id\": \"0\",
      \"kv_connector_extra_config\": {
        \"prefill\": {\"dp_size\": ${PREFILL_DP_SIZE}, \"tp_size\": ${PREFILL_TP_SIZE}},
        \"decode\": {\"dp_size\": ${DECODE_DP_SIZE}, \"tp_size\": ${DECODE_TP_SIZE}}
      }
    }")
    KV_TRANSFER_ARGS=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
    ;;
  0) ;;
  *)
    echo "PREFILL_ENABLE_KV_CONNECTOR must be 0 or 1" >&2
    exit 1
    ;;
esac

DP_ARGS=(
  --data-parallel-size "${PREFILL_DP_SIZE}"
  --data-parallel-address "${P_NODE_IP}"
  --data-parallel-rpc-port "${PREFILL_DP_RPC_PORT}"
)
if [[ -n "${PREFILL_DP_SIZE_LOCAL}" ]]; then
  DP_ARGS+=(--data-parallel-size-local "${PREFILL_DP_SIZE_LOCAL}")
fi
if (( PREFILL_DP_START_RANK > 0 )); then
  DP_ARGS+=(--data-parallel-start-rank "${PREFILL_DP_START_RANK}")
fi
case "${PREFILL_HEADLESS}" in
  0) DP_ARGS+=(--api-server-count 1) ;;
  1) DP_ARGS+=(--headless) ;;
  *)
    echo "PREFILL_HEADLESS must be 0 or 1" >&2
    exit 1
    ;;
esac

PROFILER_ARGS=()
if [[ -n "${PREFILL_ATTN_PROFILER_CONFIG:-}" ]]; then
  PROFILER_ARGS=(--profiler-config "${PREFILL_ATTN_PROFILER_CONFIG}")
fi

echo "Starting DSV4 P/Attention: topology=${PREFILL_TOPOLOGY} node=${PREFILL_NODE_ID} global=DP${PREFILL_DP_SIZE}TP${PREFILL_TP_SIZE} local_dp=${PREFILL_DP_SIZE_LOCAL:-${PREFILL_DP_SIZE}} start_rank=${PREFILL_DP_START_RANK} headless=${PREFILL_HEADLESS} devices=${PREFILL_ATTN_VISIBLE_DEVICES} afd=${AFD_HOST}:${AFD_PORT} chunk=${PREFILL_MAX_NUM_BATCHED_TOKENS}"
ASCEND_RT_VISIBLE_DEVICES="${PREFILL_ATTN_VISIBLE_DEVICES}" \
HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE}" \
VLLM_ASCEND_ENABLE_FLASHCOMM1="${PREFILL_FLASHCOMM1:-1}" \
nohup "${VLLM_CLI}" serve "${DSV4_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${PREFILL_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  "${DP_ARGS[@]}" \
  --tensor-parallel-size "${PREFILL_TP_SIZE}" \
  --enable-expert-parallel \
  --enforce-eager \
  --seed 1024 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${MODEL_LOADER_THREADS}}" \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --enable-chunked-prefill \
  ${KV_TRANSFER_ARGS[@]+"${KV_TRANSFER_ARGS[@]}"} \
  --additional-config "${AFD_ADDITIONAL_CONFIG}" \
  ${PROFILER_ARGS[@]+"${PROFILER_ARGS[@]}"} > "${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "Started P/Attention pid=$(<"${PID_FILE}") log=${LOG_FILE}"
