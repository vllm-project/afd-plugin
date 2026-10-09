#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

: "${FULL_DP_SIZE:?FULL_DP_SIZE is required}"
: "${FULL_DP_SIZE_LOCAL:?FULL_DP_SIZE_LOCAL is required}"
: "${FULL_DP_START_RANK:?FULL_DP_START_RANK is required}"
: "${FULL_HEADLESS:?FULL_HEADLESS is required}"
: "${FULL_TP_SIZE:?FULL_TP_SIZE is required}"
: "${FULL_VISIBLE_DEVICES:?FULL_VISIBLE_DEVICES is required}"

require_prefill_network
configure_network "${LOCAL_NODE_IP}"

# The baseline must not load the AFD plugin or its compatibility patches, but
# it must preserve the task image's CANN Python/TBE paths (notably the ``acl``
# module). Filter only the AFD source tree from the inherited path.
BASELINE_PYTHONPATH="${PYTHON_VLLM_ASCEND_PATH}:${PYTHON_VLLM_PATH}"
IFS=: read -r -a INHERITED_PYTHONPATH_ENTRIES <<< "${PYTHONPATH:-}"
for path_entry in "${INHERITED_PYTHONPATH_ENTRIES[@]}"; do
  [[ -n "${path_entry}" ]] || continue
  case "${path_entry}" in
    "${PYTHON_AFD_PLUGIN_PATH}" | "${PYTHON_VLLM_ASCEND_PATH}" | "${PYTHON_VLLM_PATH}")
      continue
      ;;
  esac
  BASELINE_PYTHONPATH+=":${path_entry}"
done
if [[ -n "${BASELINE_EXTRA_PYTHONPATH:-}" ]]; then
  BASELINE_PYTHONPATH+=":${BASELINE_EXTRA_PYTHONPATH}"
fi
export PYTHONPATH="${BASELINE_PYTHONPATH}"
export VLLM_PLUGINS=${BASELINE_VLLM_PLUGINS:-ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling}

KV_TRANSFER_ARGS=()
case "${PREFILL_ENABLE_KV_CONNECTOR}" in
  1)
    KV_TRANSFER_CONFIG=$(printf '%s' "{
      \"kv_connector\": \"MooncakeHybridConnector\",
      \"kv_role\": \"kv_producer\",
      \"kv_port\": \"${PD_KV_PREFILL_PORT}\",
      \"engine_id\": \"0\",
      \"kv_connector_extra_config\": {
        \"prefill\": {\"dp_size\": ${FULL_DP_SIZE}, \"tp_size\": ${FULL_TP_SIZE}},
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
  --data-parallel-size "${FULL_DP_SIZE}"
  --data-parallel-size-local "${FULL_DP_SIZE_LOCAL}"
  --data-parallel-address "${P_NODE_IP}"
  --data-parallel-rpc-port "${PREFILL_DP_RPC_PORT}"
)
if (( FULL_DP_START_RANK > 0 )); then
  DP_ARGS+=(--data-parallel-start-rank "${FULL_DP_START_RANK}")
fi
case "${FULL_HEADLESS}" in
  0) DP_ARGS+=(--api-server-count 1) ;;
  1) DP_ARGS+=(--headless) ;;
  *)
    echo "FULL_HEADLESS must be 0 or 1" >&2
    exit 2
    ;;
esac

ENABLE_DSV4_SHARED_COMPRESSOR_WORKSPACE=${ENABLE_DSV4_SHARED_COMPRESSOR_WORKSPACE:-true}
case "${ENABLE_DSV4_SHARED_COMPRESSOR_WORKSPACE}" in
  true | false) ;;
  *)
    echo "ENABLE_DSV4_SHARED_COMPRESSOR_WORKSPACE must be true or false" >&2
    exit 2
    ;;
esac

ENABLE_SHARED_EXPERT_DP=${ENABLE_SHARED_EXPERT_DP:-true}
case "${ENABLE_SHARED_EXPERT_DP}" in
  true | false) ;;
  *)
    echo "ENABLE_SHARED_EXPERT_DP must be true or false" >&2
    exit 2
    ;;
esac

ENABLE_CPU_BINDING=${ENABLE_CPU_BINDING:-true}
case "${ENABLE_CPU_BINDING}" in
  true | false) ;;
  *)
    echo "ENABLE_CPU_BINDING must be true or false" >&2
    exit 2
    ;;
esac

ADDITIONAL_CONFIG=$(printf '{
  "enable_cpu_binding": %s,
  "enable_shared_expert_dp": %s,
  "enable_dsa_cp": false,
  "multistream_dsv4_dsa_overlap": false,
  "enable_dsv4_shared_compressor_workspace": %s
}' "${ENABLE_CPU_BINDING}" "${ENABLE_SHARED_EXPERT_DP}" "${ENABLE_DSV4_SHARED_COMPRESSOR_WORKSPACE}")

exec env \
  ASCEND_RT_VISIBLE_DEVICES="${FULL_VISIBLE_DEVICES}" \
  HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE}" \
  VLLM_ASCEND_ENABLE_FLASHCOMM1="${PREFILL_FLASHCOMM1:-1}" \
  "${VLLM_CLI}" serve "${DSV4_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${PREFILL_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  "${DP_ARGS[@]}" \
  --tensor-parallel-size "${FULL_TP_SIZE}" \
  --enable-expert-parallel \
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
  --enforce-eager \
  "${KV_TRANSFER_ARGS[@]}" \
  --additional-config "${ADDITIONAL_CONFIG}"
