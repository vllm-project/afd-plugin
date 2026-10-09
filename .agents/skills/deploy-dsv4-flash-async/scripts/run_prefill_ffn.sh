#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

require_prefill_network
validate_topology
configure_network "${LOCAL_NODE_IP}"

LOG_FILE=${PREFILL_FFN_LOG_FILE:-${LOG_DIR}/prefill_ffn_node${PREFILL_NODE_ID}.log}
PID_FILE=${PREFILL_FFN_PID_FILE:-${PID_DIR}/prefill_ffn_node${PREFILL_NODE_ID}.pid}

ENABLE_CPU_BINDING=${ENABLE_CPU_BINDING:-true}
case "${ENABLE_CPU_BINDING}" in
  true | false) ;;
  *)
    echo "ENABLE_CPU_BINDING must be true or false" >&2
    exit 2
    ;;
esac

# Keep the established eager launch as the default. FULL is reserved for the
# Async CAM layered FFN graph and is enabled only for this role.
PREFILL_FFN_GRAPH_MODE=${PREFILL_FFN_GRAPH_MODE:-EAGER}
case "${PREFILL_FFN_GRAPH_MODE}" in
  EAGER) FFN_GRAPH_ARGS=(--enforce-eager) ;;
  FULL)
    FFN_GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL"}')
    ;;
  *)
    echo "PREFILL_FFN_GRAPH_MODE must be EAGER or FULL" >&2
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
    \"role\": \"ffn\",
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

# FlashComm1/SP requires TP > 1. This split FFN role is TP1; SP runs on the
# TP4 Attention role instead.
echo "Starting DSV4 P/FFN: topology=${PREFILL_TOPOLOGY} node=${PREFILL_NODE_ID} EP${FFN_DP_SIZE} devices=${PREFILL_FFN_VISIBLE_DEVICES} afd=${AFD_HOST}:${AFD_PORT} chunk=${PREFILL_MAX_NUM_BATCHED_TOKENS} graph=${PREFILL_FFN_GRAPH_MODE}"
ASCEND_RT_VISIBLE_DEVICES="${PREFILL_FFN_VISIBLE_DEVICES}" \
HCCL_BUFFSIZE="${FFN_HCCL_BUFFSIZE}" \
VLLM_ASCEND_ENABLE_FLASHCOMM1=0 \
nohup "${VLLM_CLI}" serve "${DSV4_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${FFN_PORT}" \
  --api-server-count 1 \
  --served-model-name "${SERVED_MODEL_NAME}-ffn" \
  --data-parallel-size "${FFN_DP_SIZE}" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  "${FFN_GRAPH_ARGS[@]}" \
  --seed 1024 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${FFN_MAX_NUM_SEQS}" \
  --block-size "${BLOCK_SIZE}" \
  --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${MODEL_LOADER_THREADS}}" \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --additional-config "${AFD_ADDITIONAL_CONFIG}" > "${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "Started P/FFN pid=$(<"${PID_FILE}") log=${LOG_FILE}"
