#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B (BF16) on DeepEPM2NAFDConnector, FULL_DECODE_ONLY graphs + DBO.
# Placement: one pod per comma-separated LAYOUT entry (default 2A2F, one pod of
# 4 GPUs). <a>A<f>F puts f FFN ranks on the pod's first f GPUs and a Attention
# ranks on the next a GPUs, e.g. LAYOUT=2A0F,0A2F or LAYOUT=1A1F,1A1F.
# For pod k (k = 0, 1, ...): POD=AFD_<k>, with the same LAYOUT on every pod.
# Settings: recipe/gpu/DeepEPM2NAFDConnector/README.md#recipe-settings

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
LAYOUT=${LAYOUT:-2A2F}
TUNING=(--max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096)
CONNECTOR_CONFIG=${M2N_EXTRA_CONFIG:-}   # connector_extra_config members; BF16 payload
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64}   # ascending
MODE=(--max-cudagraph-capture-size "${CUDAGRAPH_CAPTURE_SIZES##*,}"
      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}"
      --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)

# ---- The rest is the same in every DeepEPM2NAFDConnector recipe. ----
export VLLM_USE_V2_MODEL_RUNNER=1
ATTN_TP=${ATTN_TP:-1}
AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector} GATE_ON_ATTENTION=true
# P2P routes on the FFN and takes no connector_extra_config.
[[ $AFD_CONNECTOR == P2pNcclAFDConnector ]] && GATE_ON_ATTENTION=false CONNECTOR_CONFIG=
[[ -n $CONNECTOR_CONFIG ]] && CONNECTOR_CONFIG=", \"connector_extra_config\": {${CONNECTOR_CONFIG}}"
IFS=, read -ra PODS <<<"$LAYOUT"
K=${POD:-AFD_0} && K=${K#AFD_}
if [[ ! $LAYOUT =~ ^[0-9]+A[0-9]+F(,[0-9]+A[0-9]+F)*$ || ! $K =~ ^[0-9]+$ ]] || ((K >= ${#PODS[@]})); then
    echo "need LAYOUT=<a>A<f>F[,...] and POD=AFD_<k> in it: LAYOUT=$LAYOUT POD=${POD:-}" >&2
    exit 1
fi
NUM_A=() NUM_F=()
for p in "${PODS[@]}"; do NUM_A+=("${p%%A*}") && p=${p#*A} && NUM_F+=("${p%F}"); done
((${#PODS[@]} > 1)) || HEAD_HOST=${HEAD_HOST:-127.0.0.1}

# Extra `vllm serve` flags go last, so they override the defaults.
read -ra ARGS <<<"${VLLM_ARGS:-}" && ARGS+=("$@")
read -ra ATTN_EXTRA <<<"${ATTN_ARGS:-}"
read -ra FFN_EXTRA <<<"${FFN_ARGS:-}"
if [[ -n ${DECODE_BENCH:-} ]]; then  # tools/benchmarks is imported from the repository
    ATTN_EXTRA=(--kv-transfer-config "{\"kv_connector\": \"AFDDecodeBenchConnector\", \"kv_connector_module_path\": \"tools.benchmarks.decode_bench\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": ${DECODE_BENCH}}" "${ATTN_EXTRA[@]}")
    export PYTHONPATH=${AFD_REPO:-$PWD}${PYTHONPATH:+:$PYTHONPATH}
fi

# role_dp TP DP_HOST DP_RPC_PORT <ranks per pod>... -> TOTAL and this pod's DP
# flags. A role spread over pods is led by its first pod; the others run
# headless from their start rank (the leader must omit it, or vLLM selects
# hybrid load balancing and rejects the headless peers).
role_dp() {
    local tp=$1 host=$2 port=$3 i pods=0 start=0 lead=-1
    shift 3
    local n=("$@")
    TOTAL=0
    for i in "${!n[@]}"; do
        ((n[i])) || continue
        ((n[i] % tp)) && { echo "TP $tp does not divide ${n[i]} ranks on pod $i" >&2; exit 1; }
        ((lead < 0)) && lead=$i
        ((i < K)) && start=$((start + n[i]))
        pods=$((pods + 1)) TOTAL=$((TOTAL + n[i]))
    done
    DP=(--tensor-parallel-size "$tp" --data-parallel-size $((TOTAL / tp)))
    ((pods > 1)) && DP+=(--data-parallel-size-local $((n[K] / tp))
        --data-parallel-address "$host" --data-parallel-rpc-port "$port")
    ((pods > 1 && K != lead)) && DP+=(--headless --data-parallel-start-rank $((start / tp)))
}
role_dp 1 "${FFN_HEAD_HOST:-${HEAD_HOST:-vllm-ffn-dp-service}}" 13355 "${NUM_F[@]}"
FFN_DP=("${DP[@]}") NUM_FFN=$TOTAL
role_dp "$ATTN_TP" "${ATTN_HEAD_HOST:-${HEAD_HOST:-vllm-attn-dp-service}}" 13345 "${NUM_A[@]}"
ATTN_DP=("${DP[@]}") NUM_ATTN=$TOTAL

afd() {  # The AFD rendezvous runs on the pod that holds FFN rank 0.
    cat <<JSON
{"afd": {"role": "$1", "connector": "${AFD_CONNECTOR}",
 "host": "${FFN_HEAD_HOST:-${HEAD_HOST:-vllm-ffn-p2p-service}}", "port": 6239,
 "num_attention_ranks": ${NUM_ATTN}, "num_ffn_ranks": ${NUM_FFN},
 "compute_gate_on_attention": ${GATE_ON_ATTENTION}, "afd_process_group_timeout_s": 3600${CONNECTOR_CONFIG}}}
JSON
}

# The roles can finish loading minutes apart, hence the 3600 s timeouts.
COMMON=(--enable-expert-parallel --enable-ep-weight-filter --trust-remote-code
        --cpu-distributed-timeout-seconds 3600 "${TUNING[@]}" "${MODE[@]}" --host 127.0.0.1)

# FFN on this pod's GPUs [0, F), Attention on [F, F + A).
F=${NUM_F[K]} A=${NUM_A[K]}
if ((F)); then
    CUDA_VISIBLE_DEVICES=$(seq 0 $((F - 1)) | paste -sd, -) vllm serve "$MODEL_PATH" \
        --additional-config "$(afd ffn)" "${COMMON[@]}" "${FFN_DP[@]}" \
        "${ARGS[@]}" "${FFN_EXTRA[@]}" --port 18001 > ffn.log 2>&1 &
fi
if ((A)); then
    CUDA_VISIBLE_DEVICES=$(seq "$F" $((F + A - 1)) | paste -sd, -) vllm serve "$MODEL_PATH" \
        --additional-config "$(afd attention)" "${COMMON[@]}" "${ATTN_DP[@]}" \
        "${ARGS[@]}" "${ATTN_EXTRA[@]}" --port 18000 > attn.log 2>&1 &
fi
wait
