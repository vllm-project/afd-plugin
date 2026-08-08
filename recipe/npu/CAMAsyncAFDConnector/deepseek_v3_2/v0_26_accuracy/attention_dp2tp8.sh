#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the full DeepSeek-V3.2 W8A8 model directory}"
: "${LOCAL_IP:?Set LOCAL_IP to the communication IP of the Attention node}"
: "${AFD_HOST:?Set AFD_HOST to the Attention node IP used for CAM rendezvous}"
: "${NIC_NAME:?Set NIC_NAME to the NPU network interface}"

API_PORT="${API_PORT:-8000}"
AFD_PORT="${AFD_PORT:-1239}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEFAULT_VISIBLE_DEVICES}"

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export AFD_FORCE_BALANCED_TOPK_IDS=0
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-4096}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,afd}"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"

ADDITIONAL_CONFIG="$(
  printf '%s' "{
    \"enable_force_load_balance\": false,
    \"afd\": {
      \"role\": \"attention\",
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"host\": \"$AFD_HOST\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 16,
      \"num_ffn_ranks\": 16,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"attn_ranks_per_dp\": 8,
        \"async_moe_ubatching\": true,
        \"async_moe_num_ubatches\": 2,
        \"async_moe_split\": \"token\"
      }
    }
  }"
)"

exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --served-model-name deepseek_v3_2 \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --seed 1024 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --additional-config "$ADDITIONAL_CONFIG"
