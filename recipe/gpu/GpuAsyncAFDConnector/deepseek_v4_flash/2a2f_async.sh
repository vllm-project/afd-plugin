#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Candidate MRV2 recipe: two Attention and two FFN GPUs on one NVLink node.
# GPU correctness, CUDA Graph and DBO qualification are pending.
# Launch inside an allocation, for example:
# srun --partition=h20 --gres=gpu:4 --cpus-per-task=16 --mem=128G \
#   --time=02:00:00 bash recipe/gpu/GpuAsyncAFDConnector/deepseek_v4_flash/2a2f_async.sh
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to an existing DeepSeek-V4-Flash checkpoint}"
: "${CUDA_VISIBLE_DEVICES:?Launch inside a Slurm allocation with four GPUs}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#DEVICES[@]} < 4 )); then
    echo 'This recipe requires four allocated GPUs on one NVLink node.' >&2
    exit 1
fi

read -r -a VLLM_CMD <<< "${VLLM_CMD:-vllm}"
LOG_DIR=${LOG_DIR:-/tmp/afd-async-gpu}
AFD_PORT=${AFD_PORT:-6271}
API_PORT=${API_PORT:-18307}
FFN_API_PORT=${FFN_API_PORT:-$((API_PORT + 1))}
mkdir -p "$LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=1
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

COMMON_ARGS=(
    --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel
    --max-num-seqs "${MAX_NUM_SEQS:-16}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-model-len "${MAX_MODEL_LEN:-16384}"
    --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
    --api-server-count 1 --host 127.0.0.1 --trust-remote-code
)
DBO_ARGS=()
if [[ ${ENABLE_DBO:-0} == 1 ]]; then
    DBO_ARGS=(--enable-dbo
        --dbo-decode-token-threshold "${DBO_DECODE_THRESHOLD:-2}"
        --dbo-prefill-token-threshold "${DBO_PREFILL_THRESHOLD:-12}")
fi
GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
ATTN_GRAPH_ARGS=(--enforce-eager)
FFN_GRAPH_ARGS=(--enforce-eager)
if [[ ${ATTN_EAGER:-1} == 0 ]]; then
    ATTN_GRAPH_ARGS=("${GRAPH_ARGS[@]}")
fi
if [[ ${FFN_EAGER:-1} == 0 ]]; then
    FFN_GRAPH_ARGS=("${GRAPH_ARGS[@]}")
fi
AFD_CONFIG='{"afd":{"role":"attention","connector":"GpuAsyncAFDConnector",'
AFD_CONFIG+='"async":true,"compute_gate_on_attention":true,"host":"127.0.0.1",'
AFD_CONFIG+='"num_attention_ranks":2,"num_ffn_ranks":2,"port":'"$AFD_PORT"'}}'
FFN_CONFIG=${AFD_CONFIG/\"role\":\"attention\"/\"role\":\"ffn\"}

PIDS=()
cleanup() {
    if (( ${#PIDS[@]} )); then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES="${DEVICES[0]},${DEVICES[1]}" "${VLLM_CMD[@]}" serve "$MODEL_PATH" \
    "${COMMON_ARGS[@]}" "${DBO_ARGS[@]}" "${ATTN_GRAPH_ARGS[@]}" \
    --additional-config "$AFD_CONFIG" --port "$API_PORT" > "$LOG_DIR/attn.log" 2>&1 &
PIDS+=("$!")
CUDA_VISIBLE_DEVICES="${DEVICES[2]},${DEVICES[3]}" "${VLLM_CMD[@]}" serve "$MODEL_PATH" \
    "${COMMON_ARGS[@]}" "${DBO_ARGS[@]}" "${FFN_GRAPH_ARGS[@]}" \
    --additional-config "$FFN_CONFIG" --port "$FFN_API_PORT" > "$LOG_DIR/ffn.log" 2>&1 &
PIDS+=("$!")
echo "Attention endpoint port: $API_PORT; logs: $LOG_DIR/{attn,ffn}.log"
wait -n "${PIDS[@]}"
