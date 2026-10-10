#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

DSV4_FLASH_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${DSV4_FLASH_SCRIPT_DIR}/../../../.." && pwd)}

export PYTHON=${PYTHON:-python3}
VLLM_CLI=${VLLM_CLI:-vllm}
DSV4_CODE_ROOT=${DSV4_CODE_ROOT:-/a3_inference/itask/workdir/jcz02615514/jcz-afd1}
export PYTHON_VLLM_PATH=${PYTHON_VLLM_PATH:-${DSV4_CODE_ROOT}/vllm}
export PYTHON_VLLM_ASCEND_PATH=${PYTHON_VLLM_ASCEND_PATH:-${DSV4_CODE_ROOT}/vllm-ascend}
export PYTHON_AFD_PLUGIN_PATH=${PYTHON_AFD_PLUGIN_PATH:-${DSV4_CODE_ROOT}/afd-plugin}
export PYTHONPATH="${PYTHON_AFD_PLUGIN_PATH}:${PYTHON_VLLM_ASCEND_PATH}:${PYTHON_VLLM_PATH}:${PYTHONPATH:-}"

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export AFD_FORCE_SPAWN_MULTIPROCESSING=${AFD_FORCE_SPAWN_MULTIPROCESSING:-1}

DSV4_MODEL=${DSV4_MODEL:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv4}

# P_NODE_IP is both the P-side communication address and the CamAsync
# rendezvous address. It must be reachable from the D node and all P ranks.
# P_NODE_IP is the primary prefill node, vLLM DP coordinator, and default AFD
# rendezvous host. P_SECONDARY_NODE_IP is required by the two-node benchmark
# topologies. LOCAL_NODE_IP is derived from PREFILL_NODE_ID unless explicitly
# overridden.
P_NODE_IP=${P_NODE_IP:-}
P_SECONDARY_NODE_IP=${P_SECONDARY_NODE_IP:-}
D_NODE_IP=${D_NODE_IP:-}
NIC_NAME=${NIC_NAME:-}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
PROXY_HOST=${PROXY_HOST:-0.0.0.0}

PREFILL_PORT=${PREFILL_PORT:-7100}
FFN_PORT=${FFN_PORT:-7101}
DECODE_PORT=${DECODE_PORT:-7100}
PROXY_PORT=${PROXY_PORT:-8000}
AFD_PORT=${AFD_PORT:-1239}
AFD_HOST=${AFD_HOST:-${P_NODE_IP}}
PREFILL_DP_RPC_PORT=${PREFILL_DP_RPC_PORT:-12321}
DECODE_DP_RPC_PORT=${DECODE_DP_RPC_PORT:-12321}
PD_KV_PREFILL_PORT=${PD_KV_PREFILL_PORT:-30000}
PD_KV_DECODE_PORT=${PD_KV_DECODE_PORT:-30100}

# AFD benchmark layouts use two 16-NPU 910C nodes. Node 0 owns 16 Attention
# ranks in the original layouts; TP2 layouts reserve unused devices idle.
# Node 1 owns all 8 FFN ranks, normally on devices 8-15.
# afd_dp3tp4_ep8 uses Attention-only node 0 and FFN-only node 1 (devices 0-7).
# Keep "legacy" for the original single-node DP2TP4 + EP8 layout.
PREFILL_TOPOLOGY=${PREFILL_TOPOLOGY:-legacy}
PREFILL_NODE_ID=${PREFILL_NODE_ID:-0}
case "${PREFILL_TOPOLOGY}" in
  afd_dp12tp2 | afd_dp10tp2 | afd_dp8tp2 | afd_dp6tp2)
    topology_dp=${PREFILL_TOPOLOGY#afd_dp}
    topology_dp=${topology_dp%tp2}
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-${topology_dp}}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-2}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-$((topology_dp * 2))}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-2}
    ;;
  afd_dp6tp4)
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-6}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-4}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-24}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-4}
    ;;
  afd_dp3tp8)
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-3}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-8}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-24}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-8}
    ;;
  afd_dp4tp2)
    # Single-node AFD layout: Attention DP4xTP2 on devices 0-7 and
    # FFN TP1/EP8 on devices 8-15.
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-4}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-2}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-8}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-2}
    ;;
  afd_dp3tp4_ep8)
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-3}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-4}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-12}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-4}
    ;;
  legacy)
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-2}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-4}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-8}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-4}
    ;;
  ep16 | ep16_dp4tp4 | ep16_dp8tp2 | ep16_dp2tp8 | ep32)
    # Share the full-model global layout with the PD consumer.
    case "${PREFILL_TOPOLOGY}" in
      ep16 | ep16_dp4tp4) topology_dp=4; topology_tp=4 ;;
      ep16_dp8tp2) topology_dp=8; topology_tp=2 ;;
      ep16_dp2tp8) topology_dp=2; topology_tp=8 ;;
      ep32) topology_dp=4; topology_tp=8 ;;
    esac
    PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-${topology_dp}}
    PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-${topology_tp}}
    NUM_ATTENTION_RANKS=${NUM_ATTENTION_RANKS:-1}
    ATTN_RANKS_PER_DP=${ATTN_RANKS_PER_DP:-1}
    ;;
  *)
    echo "Unknown PREFILL_TOPOLOGY=${PREFILL_TOPOLOGY}" >&2
    # A sourced script returns; direct execution falls back to exit.
    # shellcheck disable=SC2317
    return 2 2>/dev/null || exit 2
    ;;
esac

FFN_DP_SIZE=${FFN_DP_SIZE:-8}
NUM_FFN_RANKS=${NUM_FFN_RANKS:-8}

case "${PREFILL_NODE_ID}" in
  0)
    LOCAL_NODE_IP=${LOCAL_NODE_IP:-${P_NODE_IP}}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_HEADLESS=${PREFILL_HEADLESS:-0}
    ;;
  1)
    LOCAL_NODE_IP=${LOCAL_NODE_IP:-${P_SECONDARY_NODE_IP}}
    PREFILL_HEADLESS=${PREFILL_HEADLESS:-1}
    ;;
  *)
    echo "PREFILL_NODE_ID must be 0 or 1" >&2
    # A sourced script returns; direct execution falls back to exit.
    # shellcheck disable=SC2317
    return 2 2>/dev/null || exit 2
    ;;
esac

case "${PREFILL_TOPOLOGY}:${PREFILL_NODE_ID}" in
  afd_dp3tp4_ep8:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_START_ATTENTION=${PREFILL_START_ATTENTION:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-3}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}
    ;;
  afd_dp3tp4_ep8:1)
    # This node owns only FFN; all three Attention DP replicas are on node 0.
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_START_ATTENTION=${PREFILL_START_ATTENTION:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-0}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_FFN_VISIBLE_DEVICES=${PREFILL_FFN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp12tp2:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-8}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    ;;
  afd_dp12tp2:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-8}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp10tp2:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-6}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}
    ;;
  afd_dp10tp2:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-6}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp8tp2:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp8tp2:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-4}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp6tp2:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-0}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp6tp2:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-2}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-4}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3}
    ;;
  afd_dp6tp4:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    ;;
  afd_dp6tp4:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-2}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-4}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp3tp8:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-0}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-2}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    ;;
  afd_dp3tp8:1)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-1}
    PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK:-2}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp4tp2:0)
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-4}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    ;;
  afd_dp4tp2:1)
    echo "afd_dp4tp2 is a single-node topology; launch only PREFILL_NODE_ID=0" >&2
    # A sourced script returns; direct execution falls back to exit.
    # shellcheck disable=SC2317
    return 2 2>/dev/null || exit 2
    ;;
  *)
    PREFILL_DP_SIZE_LOCAL=${PREFILL_DP_SIZE_LOCAL:-}
    PREFILL_ATTN_VISIBLE_DEVICES=${PREFILL_ATTN_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    PREFILL_START_FFN=${PREFILL_START_FFN:-1}
    ;;
esac
PREFILL_START_ATTENTION=${PREFILL_START_ATTENTION:-1}
PREFILL_FFN_VISIBLE_DEVICES=${PREFILL_FFN_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
PREFILL_ENABLE_KV_CONNECTOR=${PREFILL_ENABLE_KV_CONNECTOR:-0}

# The D topology follows the vllm-ascend DSV4 Flash 910C PD example.
DECODE_DP_SIZE=${DECODE_DP_SIZE:-16}
DECODE_TP_SIZE=${DECODE_TP_SIZE:-1}
DECODE_VISIBLE_DEVICES=${DECODE_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

case "${PREFILL_TOPOLOGY}" in
  afd_dp3tp4_ep8 | afd_dp12tp2 | afd_dp10tp2 | afd_dp8tp2 | afd_dp6tp2 | afd_dp6tp4 | afd_dp3tp8 | afd_dp4tp2 | ep16 | ep16_dp4tp4 | ep16_dp8tp2 | ep16_dp2tp8 | ep32) DEFAULT_MAX_MODEL_LEN=65536 ;;
  *) DEFAULT_MAX_MODEL_LEN=1048576 ;;
esac
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}
PREFILL_MAX_NUM_BATCHED_TOKENS=${PREFILL_MAX_NUM_BATCHED_TOKENS:-8192}
DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:-120}
PREFILL_MAX_NUM_SEQS=${PREFILL_MAX_NUM_SEQS:-16}
FFN_MAX_NUM_SEQS=${FFN_MAX_NUM_SEQS:-16}
DECODE_MAX_NUM_SEQS=${DECODE_MAX_NUM_SEQS:-60}
BLOCK_SIZE=${BLOCK_SIZE:-128}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.7}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.9}
MODEL_LOADER_THREADS=${MODEL_LOADER_THREADS:-128}

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-4096}
PREFILL_HCCL_BUFFSIZE=${PREFILL_HCCL_BUFFSIZE:-4096}
FFN_HCCL_BUFFSIZE=${FFN_HCCL_BUFFSIZE:-${PREFILL_HCCL_BUFFSIZE}}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2000}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-3600000}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export AFD_FORCE_BALANCED_TOPK_IDS=${AFD_FORCE_BALANCED_TOPK_IDS:-0}
export AFD_CAM_OP_IO_LOG=${AFD_CAM_OP_IO_LOG:-0}
export AFD_ASYNC_MOE_LAYOUT_LOG=${AFD_ASYNC_MOE_LAYOUT_LOG:-0}

# Source-built Ascend operators are installed with afd-plugin. The runtime
# loader configures the packaged vendor and loads afd_plugin._C_ascend.
# No external CAM vendor paths or library preloading are needed here.

LOG_DIR=${LOG_DIR:-${DSV4_FLASH_SCRIPT_DIR}/logs}
PID_DIR=${PID_DIR:-${LOG_DIR}/pids}
mkdir -p "${LOG_DIR}" "${PID_DIR}"

require_common_network() {
  : "${P_NODE_IP:?Set P_NODE_IP to the P node communication IP}"
  : "${D_NODE_IP:?Set D_NODE_IP to the D node communication IP}"
  : "${NIC_NAME:?Set NIC_NAME to the HCCL/Gloo network interface}"
}

require_prefill_network() {
  : "${DSV4_MODEL:?Set DSV4_MODEL to the checkpoint directory}"
  : "${P_NODE_IP:?Set P_NODE_IP to the primary P node communication IP}"
  : "${LOCAL_NODE_IP:?Set LOCAL_NODE_IP or the node IP for PREFILL_NODE_ID}"
  : "${NIC_NAME:?Set NIC_NAME to the HCCL/Gloo network interface}"
  if [[ "${PREFILL_ENABLE_KV_CONNECTOR}" == 1 ]]; then
    : "${D_NODE_IP:?Set D_NODE_IP when PREFILL_ENABLE_KV_CONNECTOR=1}"
  fi
}

configure_network() {
  local local_ip=$1
  export HCCL_IF_IP="${local_ip}"
  export HCCL_SOCKET_IFNAME="${NIC_NAME}"
  export GLOO_SOCKET_IFNAME="${NIC_NAME}"
  export TP_SOCKET_IFNAME="${NIC_NAME}"
}

validate_topology() {
  case "${PREFILL_MAX_NUM_BATCHED_TOKENS}" in
    4096 | 8192 | 16384 | 32768 | 49152 | 65536) ;;
    *)
      echo "PREFILL_MAX_NUM_BATCHED_TOKENS must be one of 4096, 8192, 16384, 32768, 49152, 65536" >&2
      return 1
      ;;
  esac
  if (( PREFILL_DP_SIZE * PREFILL_TP_SIZE != NUM_ATTENTION_RANKS )); then
    echo "PREFILL_DP_SIZE * PREFILL_TP_SIZE must equal NUM_ATTENTION_RANKS" >&2
    return 1
  fi
  if (( FFN_DP_SIZE != NUM_FFN_RANKS )); then
    echo "FFN_DP_SIZE must equal NUM_FFN_RANKS for TP1/EP FFN" >&2
    return 1
  fi
  if (( PREFILL_TP_SIZE != ATTN_RANKS_PER_DP )); then
    echo "PREFILL_TP_SIZE must equal ATTN_RANKS_PER_DP" >&2
    return 1
  fi
  if [[ -n "${PREFILL_DP_SIZE_LOCAL}" ]] &&
    (( PREFILL_DP_START_RANK + PREFILL_DP_SIZE_LOCAL > PREFILL_DP_SIZE )); then
    echo "PREFILL_DP_START_RANK + PREFILL_DP_SIZE_LOCAL exceeds global PREFILL_DP_SIZE" >&2
    return 1
  fi
}

# Resolve the bundled workload only for benchmark entrypoints, not deployment.
prepare_bench_dataset() {
  if [[ -z "${DSV4_BENCH_DATASET_PATH:-}" ]]; then
    local cache_dir=${DSV4_BENCH_DATASET_CACHE_DIR:-${REPO_ROOT}/bench_results/dsv4-flash/datasets}
    DSV4_BENCH_DATASET_PATH=$("${PYTHON}" "${DSV4_FLASH_SCRIPT_DIR}/prepare_bench_dataset.py" \
      --output "${cache_dir}/formal_0_1_2_vllm_bench.jsonl")
  fi
  export DSV4_BENCH_DATASET_PATH
}

verify_bench_dataset() {
  "${PYTHON}" - "$1" "$2" <<'PYTHON'
import hashlib
import pathlib
import sys

CHUNK_BYTES = 1024 * 1024
path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"Dataset SHA-256 mismatch: expected {expected}, got {actual}")
PYTHON
}
