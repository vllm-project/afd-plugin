#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

usage() {
  echo "Usage: MODEL_PATH=<path> LOCAL_IP=<ip> NIC_NAME=<nic> $0 <attention|ffn>" >&2
  echo "Launch each role in a separate terminal." >&2
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi
ROLE="$1"

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash W8A8 checkpoint}"
: "${LOCAL_IP:?Set LOCAL_IP to this node communication IP}"
: "${NIC_NAME:?Set NIC_NAME to the NPU network interface}"

AFD_HOST="${AFD_HOST:-$LOCAL_IP}"
AFD_PORT="${AFD_PORT:-1239}"
ATTENTION_PORT="${ATTENTION_PORT:-8000}"
FFN_PORT="${FFN_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-70000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
HCCL_BUFFER_SIZE_MB="${HCCL_BUFFER_SIZE_MB:-4096}"
ATTENTION_DEVICES="${ATTENTION_DEVICES:-0,1,2,3,4,5,6,7}"
FFN_DEVICES="${FFN_DEVICES:-8,9,10,11,12,13,14,15}"
CAM_VENDOR_PATH="${CAM_VENDOR_PATH:-/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "DeepSeek-V4 config not found: $MODEL_PATH/config.json" >&2
  exit 1
fi

case "$ROLE" in
  attention)
    API_PORT="$ATTENTION_PORT"
    VISIBLE_DEVICES="$ATTENTION_DEVICES"
    FLASHCOMM1=1
    WORKER_CLASS=afd_plugin.v1.worker.npu.AFDNPUAttentionWorker
    PARALLEL_ARGS=(
      --data-parallel-size 4
      --tensor-parallel-size 2
    )
    ;;
  ffn)
    API_PORT="$FFN_PORT"
    VISIBLE_DEVICES="$FFN_DEVICES"
    FLASHCOMM1=0
    WORKER_CLASS=afd_plugin.v1.worker.npu.AFDNPUFFNWorker
    PARALLEL_ARGS=(
      --data-parallel-size 8
      --tensor-parallel-size 1
    )
    ;;
  *)
    echo "ROLE must be attention or ffn, got: $ROLE" >&2
    exit 2
    ;;
esac

CAM_OPAPI_DIR="$CAM_VENDOR_PATH/op_api/lib"
if [[ -f "$CAM_OPAPI_DIR/libopapi.so" ]]; then
  CAM_OPAPI="$CAM_OPAPI_DIR/libopapi.so"
  CAM_OPAPI_LOAD_PATH="$CAM_OPAPI_DIR"
elif [[ -f "$CAM_OPAPI_DIR/libcust_opapi.so" ]]; then
  # CAM 209.x may ship only libcust_opapi.so, while UMDK resolves the vendor
  # implementation by the libopapi.so name. Provide a process-local alias.
  CAM_OPAPI_COMPAT_DIR="${CAM_OPAPI_COMPAT_DIR:-${TMPDIR:-/tmp}/afd-cam-opapi}"
  mkdir -p "$CAM_OPAPI_COMPAT_DIR"
  ln -sfn "$CAM_OPAPI_DIR/libcust_opapi.so" "$CAM_OPAPI_COMPAT_DIR/libopapi.so"
  CAM_OPAPI="$CAM_OPAPI_COMPAT_DIR/libopapi.so"
  CAM_OPAPI_LOAD_PATH="$CAM_OPAPI_COMPAT_DIR:$CAM_OPAPI_DIR"
else
  echo "CAM op-api library not found under $CAM_OPAPI_DIR" >&2
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export ASCEND_CUSTOM_OPP_PATH="$CAM_VENDOR_PATH:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$CAM_OPAPI_LOAD_PATH:$CAM_VENDOR_PATH/op_api:${LD_LIBRARY_PATH:-}"
export CAM_CUST_OPAPI_LIB_PATH="$CAM_OPAPI"
export LD_PRELOAD="$CAM_OPAPI${LD_PRELOAD:+:$LD_PRELOAD}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export AFD_FORCE_SPAWN_MULTIPROCESSING="${AFD_FORCE_SPAWN_MULTIPROCESSING:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# The recorded DP4TP2/EP8 run required 4096 MB for the mbt=65536 CAM domain.
# Keep the process-wide fallback aligned with the connector-scoped value.
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-$HCCL_BUFFER_SIZE_MB}"
# FlashComm1/SP is Attention-local. FFN TP1 rejects it.
export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
export AFD_FORCE_BALANCED_TOPK_IDS=0

ADDITIONAL_CONFIG="$(
  printf '%s' "{
    \"enable_force_load_balance\": false,
    \"multistream_dsv4_dsa_overlap\": false,
    \"enable_dsv4_shared_compressor_workspace\": true,
    \"afd\": {
      \"role\": \"$ROLE\",
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"host\": \"$AFD_HOST\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 8,
      \"num_ffn_ranks\": 8,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"attn_ranks_per_dp\": 2,
        \"async_moe_ubatching\": true,
        \"async_moe_num_ubatches\": 2,
        \"async_moe_split\": \"token\",
        \"hccl_buffer_size\": $HCCL_BUFFER_SIZE_MB
      }
    }
  }"
)"

exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --served-model-name deepseek_v4_flash \
  --worker-cls "$WORKER_CLASS" \
  "${PARALLEL_ARGS[@]}" \
  --enable-expert-parallel \
  --enforce-eager \
  --quantization ascend \
  --tokenizer-mode deepseek_v4 \
  --block-size 128 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --seed 1024 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --additional-config "$ADDITIONAL_CONFIG"
