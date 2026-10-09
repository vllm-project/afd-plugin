#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ROLE=prefill
MODEL_PATH="${MODEL_PATH:-}"
NIC_NAME="${NIC_NAME:-}"
HCCL_IF_IP="${HCCL_IF_IP:-}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${PREFILL_API_PORT:-8100}"
PREFILL_DEVICES="${PREFILL_DEVICES:-0,1}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-1}"
ATTENTION_RANKS="${ATTENTION_RANKS:-4}"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HCCL_IF_BASE_PORT="${PREFILL_HCCL_IF_BASE_PORT:-50000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
: "${PREFILL_ENGINE_ID:?Set PREFILL_ENGINE_ID for Mooncake PD}"
: "${PREFILL_KV_PORT:?Set PREFILL_KV_PORT for Mooncake PD}"

: "${MODEL_PATH:?Set MODEL_PATH to the native A5 DeepSeek-V4 checkpoint}"
: "${NIC_NAME:?Set NIC_NAME to the local network interface}"
: "${HCCL_IF_IP:?Set HCCL_IF_IP to the local IPv4 address}"
[[ "$PREFILL_TP_SIZE" == 1 ]] || afd_die "this delivery fixes Prefill TP1"
IFS=',' read -r -a prefill_devices_array <<<"$PREFILL_DEVICES"
((${#prefill_devices_array[@]} == PREFILL_DP_SIZE)) \
  || afd_die "PREFILL_DEVICES must contain $PREFILL_DP_SIZE devices"
declare -A prefill_seen=()
for device in "${prefill_devices_array[@]}"; do
  [[ "$device" =~ ^[0-7]$ ]] || afd_die "invalid Prefill device ID: $device"
  [[ -z "${prefill_seen[$device]:-}" ]] \
    || afd_die "Prefill device list repeats device $device"
  prefill_seen[$device]=1
done
validate_model_config
export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_DEVICES"
export VLLM_HOST_IP="$HCCL_IF_IP"
export HCCL_IF_IP HCCL_IF_BASE_PORT
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd}"
export VLLM_USE_V1=1
KV_CONFIG="$(build_a5_pd_kv_config kv_producer "$PREFILL_ENGINE_ID" "$PREFILL_KV_PORT" "$PREFILL_DEVICES")"
printf '[dsv4-afd] A5 Prefill Mooncake KV config=%s\n' "$KV_CONFIG"

run_role_service "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$PREFILL_DP_SIZE" \
  --tensor-parallel-size "$PREFILL_TP_SIZE" \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --block-size 32 \
  --enforce-eager \
  --kv-transfer-config "$KV_CONFIG"
