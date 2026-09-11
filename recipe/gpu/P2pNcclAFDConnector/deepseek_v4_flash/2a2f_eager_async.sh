#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# 2A2F DeepSeek-V4-Flash with the synchronous P2pNcclAFDConnector, eager.
#
# The AFD DeepSeek-V4 adapter's first release requires this connector and
# rejects compute_gate_on_attention (V4's native router runs on the FFN side;
# token-aligned input_ids cross the Attention-to-FFN boundary instead).
#
# Launch under a GPU reservation:
#   gpu run --gpus 4 -- bash recipe/gpu/P2pNcclAFDConnector/deepseek_v4_flash/2a2f_eager_async.sh
set -u

MODEL_PATH=${MODEL_PATH:-/data/boao/deepseek-v4-flash}
VLLM_CMD=${VLLM_CMD:-vllm}
LOG_DIR=${LOG_DIR:-.}
mkdir -p "$LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#DEVICES[@]}" -lt 4 ]; then
    echo "need 4 visible GPUs, got ${#DEVICES[@]}: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi
ATTN_DEVICES="${DEVICES[0]},${DEVICES[1]}"
FFN_DEVICES="${DEVICES[2]},${DEVICES[3]}"
echo "attention on ${ATTN_DEVICES}, ffn on ${FFN_DEVICES}"

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_MODEL_LEN_ARG=()
[ -n "$MAX_MODEL_LEN" ] && MAX_MODEL_LEN_ARG=(--max-model-len "$MAX_MODEL_LEN")
AFD_PORT=${AFD_PORT:-6271}
API_PORT=${API_PORT:-18307}
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_ARGS=()
if [ "$ENABLE_DBO" = 1 ]; then
    DBO_ARGS=(
        --enable-dbo
        --dbo-decode-token-threshold "${DBO_DECODE_THRESHOLD:-2}"
        --dbo-prefill-token-threshold "${DBO_PREFILL_THRESHOLD:-12}"
    )
fi

ROLE_ARGS=(serve "$MODEL_PATH"
    --data-parallel-size 2
    --tensor-parallel-size 1
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    "${MAX_MODEL_LEN_ARG[@]}"
    --tokenizer-mode deepseek_v4
    --kv-cache-dtype fp8
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enforce-eager
    "${DBO_ARGS[@]}"
    --api-server-count 1
    --host 127.0.0.1
    --trust-remote-code)

CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" $VLLM_CMD "${ROLE_ARGS[@]}" \
    --served-model-name deepseek-v4-flash-afd-attention \
    --additional-config "{
        \"afd\": {
            \"role\": \"attention\",
            \"connector\": \"P2pNcclAFDConnector\",
            \"host\": \"127.0.0.1\",
            \"port\": $AFD_PORT,
            \"num_attention_ranks\": 2,
            \"num_ffn_ranks\": 2
        }
    }" \
    --port "$API_PORT" > "$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES="$FFN_DEVICES" $VLLM_CMD "${ROLE_ARGS[@]}" \
    --served-model-name deepseek-v4-flash-afd-ffn \
    --additional-config "{
        \"afd\": {
            \"role\": \"ffn\",
            \"connector\": \"P2pNcclAFDConnector\",
            \"host\": \"127.0.0.1\",
            \"port\": $AFD_PORT,
            \"num_attention_ranks\": 2,
            \"num_ffn_ranks\": 2
        }
    }" \
    --port "$((API_PORT + 1))" > "$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!

# shellcheck disable=SC2317,SC2329  # invoked by the EXIT trap below.
# (SC2329 on shellcheck >= 0.11, SC2317 on older ones; CI runs an older one.)
cleanup() {
    kill "$ATTN_PID" "$FFN_PID" 2>/dev/null
    wait "$ATTN_PID" "$FFN_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 "${READY_TIMEOUT:-900}"); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "server ready on http://127.0.0.1:$API_PORT"
        if [ -n "${SMOKE:-}" ]; then
            curl -s "http://127.0.0.1:$API_PORT/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"deepseek-v4-flash-afd-attention\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}"
            echo
            exit 0
        fi
        wait "$ATTN_PID" "$FFN_PID"
        exit 0
    fi
    if ! kill -0 "$ATTN_PID" 2>/dev/null || ! kill -0 "$FFN_PID" 2>/dev/null; then
        echo "a server exited early; see $LOG_DIR/attn.log and $LOG_DIR/ffn.log" >&2
        exit 1
    fi
    sleep 1
done
echo "timed out waiting for the server" >&2
exit 1
