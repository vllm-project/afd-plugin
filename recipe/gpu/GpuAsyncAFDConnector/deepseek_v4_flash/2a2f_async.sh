#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# 2A2F DeepSeek-V4-Flash on the async GPU connector.
#
# FFN_EAGER picks the FFN side's run mode; it defaults to eager, which is what
# measured fastest on this model -- see the note on it below.
#
# Launch under a GPU reservation, which sets CUDA_VISIBLE_DEVICES:
#   gpu run --gpus 4 -- \
#     bash recipe/gpu/GpuAsyncAFDConnector/deepseek_v4_flash/2a2f_async.sh
#
# The two roles are separate vllm serve processes: the AFD process group hosts
# its own TCPStore, which cannot be created under a single torchrun/torchelastic
# launcher.
set -u

MODEL_PATH=${MODEL_PATH:-/path/model_weights/deepseek-v4-flash}
# How to invoke vLLM. `uv run vllm` is right from a synced checkout; override
# to point at an interpreter that actually has the plugin installed.
read -r -a VLLM_CMD <<< "${VLLM_CMD:-uv run vllm}"
# The FFN experts run eagerly by default. Their padded graphs hold ~19 GiB
# and buy nothing on V4: the FFN is compute-bound, so the per-item launch
# saving is negligible, while a replay costs its whole captured bucket.
# Measured pure prefill, 2A2F on 4x L20X, 128x1024 tokens: 17.9 s eager
# against 19.0 s replaying every item. Set FFN_EAGER=0 to capture them anyway.
FFN_EAGER=${FFN_EAGER:-1}
FFN_GRAPH_ARGS=()
[ "$FFN_EAGER" = 1 ] && FFN_GRAPH_ARGS=(--enforce-eager)
# Attention-side run mode:
#   1 -- eager (the default)
#   0 -- FULL_DECODE_ONLY graphs
#
# Decode is where the Attention graphs pay, and they pay a lot: measured on
# DeepSeek-V2-Lite 1A1F decode (64 req x 256 output tokens, conc 16), eager
# 36.6 s against 28.3 s with graphs -- 22.6% off the wall clock, with the
# wrapper reporting a 98% replay share. Prefill is the opposite: those graphs
# are captured and never replayed there, and the same flags measured dead
# parity (2.123 s vs 2.128 s). Leave it eager for a prefill-only run; turn it
# on for anything that decodes.
ATTN_EAGER=${ATTN_EAGER:-1}
LOG_DIR=${LOG_DIR:-.}
mkdir -p "$LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0
# Single node over NVLink: skip the IB transport probe.
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
# Two servers on one box spawn a lot of threads; the HF tokenizer's rayon pool
# is the first thing to fail when thread creation gets refused.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Split the reserved devices in half: first two Attention, last two FFN.
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#DEVICES[@]}" -lt 4 ]; then
    echo "need 4 visible GPUs, got ${#DEVICES[@]}: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi
ATTN_DEVICES="${DEVICES[0]},${DEVICES[1]}"
FFN_DEVICES="${DEVICES[2]},${DEVICES[3]}"
echo "attention on ${ATTN_DEVICES}, ffn on ${FFN_DEVICES}"

# Lower this when sharing a box: vLLM refuses to start if the desired
# fraction exceeds what is actually free.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# Prefill batch size drives whether each MoE call clears the compute-bound
# inflection point, so it is the knob to raise when benchmarking.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
# The checkpoint declares a 1M-token context; sizing KV against that leaves
# nothing for the experts.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
# V4 ships its own tokenizer, and its native attention wants an fp8 KV cache.
TOKENIZER_MODE=${TOKENIZER_MODE:-deepseek_v4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
# Free-form passthrough, e.g. EXTRA_ARGS="--no-enable-prefix-caching".
read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS:-}"
AFD_PORT=${AFD_PORT:-6271}
API_PORT=${API_PORT:-18307}
# The FFN server never takes HTTP -- its EngineCore is a connector daemon --
# but it still starts an API server, and both roles racing for one port means
# whichever loses exits and takes its role down with it. Give it its own.
FFN_API_PORT=${FFN_API_PORT:-$((API_PORT + 1))}

# One capture bucket per decode size 1..MAX_NUM_SEQS. A single bucket at the
# max would pad every decode batch up to it, which wastes the compute the
# graphs just saved.
ATTN_GRAPH_ARGS=(--enforce-eager)
if [ "$ATTN_EAGER" != 1 ]; then
    DECODE_SIZES=()
    for size in $(seq 1 "$MAX_NUM_SEQS"); do
        DECODE_SIZES+=("$size")
    done
    ATTN_GRAPH_ARGS=(
        --max-cudagraph-capture-size "$MAX_NUM_SEQS"
        --cudagraph-capture-sizes "${DECODE_SIZES[@]}"
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
fi

AFD_CONFIG_ATTN='{
    "afd": {
        "role": "attention",
        "connector": "GpuAsyncAFDConnector",
        "async": true,
        "compute_gate_on_attention": true,
        "host": "127.0.0.1",
        "port": '"$AFD_PORT"',
        "num_attention_ranks": 2,
        "num_ffn_ranks": 2
    }
}'
AFD_CONFIG_FFN=${AFD_CONFIG_ATTN/\"role\": \"attention\"/\"role\": \"ffn\"}

CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" "${VLLM_CMD[@]}" serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config "$AFD_CONFIG_ATTN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tokenizer-mode "$TOKENIZER_MODE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${ATTN_GRAPH_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --trust-remote-code > "$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES="$FFN_DEVICES" "${VLLM_CMD[@]}" serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config "$AFD_CONFIG_FFN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tokenizer-mode "$TOKENIZER_MODE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${FFN_GRAPH_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --host 127.0.0.1 \
    --port "$FFN_API_PORT" \
    --trust-remote-code > "$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!

# shellcheck disable=SC2317,SC2329  # invoked by the EXIT trap below.
# (SC2329 on shellcheck >= 0.11, SC2317 on older ones; CI runs an older one.)
cleanup() {
    kill "$ATTN_PID" "$FFN_PID" 2>/dev/null
    wait "$ATTN_PID" "$FFN_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 "${READY_TIMEOUT:-600}"); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "server ready on http://127.0.0.1:$API_PORT"
        echo
        echo "curl -s http://127.0.0.1:$API_PORT/v1/completions \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'"
        echo
        if [ -n "${SMOKE:-}" ]; then
            curl -s "http://127.0.0.1:$API_PORT/v1/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is",
                     "max_tokens":16,"temperature":0,
                     "skip_special_tokens":false,"logprobs":2}'
            echo
            exit 0
        fi
        # Stay up so the servers can take requests; Ctrl-C tears both down.
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
