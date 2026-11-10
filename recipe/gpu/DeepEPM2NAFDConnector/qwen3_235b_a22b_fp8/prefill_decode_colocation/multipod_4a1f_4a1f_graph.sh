# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Qwen3-235B-A22B-FP8 with DeepEPM2NAFDConnector, 2 x 4A1F, FULL_DECODE_ONLY
# CUDA graphs: 2 pods of 5 GPUs, each with FFN on GPU 0 and Attention DP4 x TP1
# on GPUs 1-4 (Attention DP8, FFN DP2/EP2 in total). Send traffic to port
# 18000 of the POD=AFD_0 pod.
#
# Use .agents/skills/deploy-afd-k8s/SKILL.md to deploy this recipe
# on 2 pods using the setup below.
# For pod 1: POD=AFD_0
# For pod 2: POD=AFD_1
#
# NUM_PODS x (ATTN_PER_POD Attention + FFN_PER_POD FFN) sets any symmetric
# layout (2 x 1A1F, 3 x 6A2F, ...); give every pod the same values. HEAD_HOST
# points the AFD rendezvous and both DP leaders at pod 0 (runs without k8s).

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
NUM_PODS=${NUM_PODS:-2}
ATTN_PER_POD=${ATTN_PER_POD:-4}
FFN_PER_POD=${FFN_PER_POD:-1}
PAYLOAD_FORMAT=${PAYLOAD_FORMAT:-fp8_block128_fp32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}            # per Attention DP rank
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-3072}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
CAPTURE_SIZES=${CAPTURE_SIZES:-8,16,32,64,96,128,192,256}   # ascending
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-2}
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-2}
export VLLM_USE_V2_MODEL_RUNNER=1

K=${POD#AFD_} A=$ATTN_PER_POD F=$FFN_PER_POD

# role_dp PER_POD ADDRESS RPC_PORT -> DP: this pod's DP flags for one role.
# Pod 0 leads and omits its start rank: vLLM reads an explicit start rank on a
# non-headless engine as hybrid load balancing and rejects the headless peers.
role_dp() {
    DP=(--data-parallel-size $((NUM_PODS * $1)) --data-parallel-size-local "$1"
        --data-parallel-address "$2" --data-parallel-rpc-port "$3")
    if ((K > 0)); then DP+=(--headless --data-parallel-start-rank $((K * $1))); fi
}

afd() {
    cat <<JSON
{"afd": {"role": "$1", "connector": "DeepEPM2NAFDConnector",
 "host": "${HEAD_HOST:-vllm-ffn-p2p-service}", "port": 6239,
 "num_attention_ranks": $((NUM_PODS * A)), "num_ffn_ranks": $((NUM_PODS * F)),
 "compute_gate_on_attention": true, "afd_process_group_timeout_s": 3600,
 "connector_extra_config": {"payload_format": "${PAYLOAD_FORMAT}"}}}
JSON
}

COMMON=(
    --tensor-parallel-size 1
    --enable-expert-parallel
    --enable-ep-weight-filter
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-cudagraph-capture-size "${CAPTURE_SIZES##*,}"
    --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CAPTURE_SIZES}]}"
    --cpu-distributed-timeout-seconds 3600   # pods may finish loading minutes apart
    --host 127.0.0.1
)
if ((ENABLE_DBO)); then
    COMMON+=(--enable-dbo
             --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
             --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD")
fi

# FFN on local GPUs [0, F), Attention on [F, F + A).
role_dp "$F" "${HEAD_HOST:-vllm-ffn-dp-service}" 13355
CUDA_VISIBLE_DEVICES=$(seq 0 $((F - 1)) | paste -sd, -) vllm serve "$MODEL_PATH" \
    --additional-config "$(afd ffn)" "${COMMON[@]}" "${DP[@]}" \
    --port 18001 > ffn.log 2>&1 &

role_dp "$A" "${HEAD_HOST:-vllm-attn-dp-service}" 13345
CUDA_VISIBLE_DEVICES=$(seq "$F" $((F + A - 1)) | paste -sd, -) vllm serve "$MODEL_PATH" \
    --additional-config "$(afd attention)" "${COMMON[@]}" "${DP[@]}" \
    --port 18000 > attn.log 2>&1 &

wait
