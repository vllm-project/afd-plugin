# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Shared launcher of the GPU prefill_decode_colocation recipes: placement
# (LAYOUT/POD -> devices and DP flags), the AFD config and the two `vllm serve`
# processes. A recipe sets the variables below, sources this file and calls
# afd_colocation_serve "$@". See recipe/gpu/common/README.md.
#
# Set by the recipe:
#   MODEL_PATH, AFD_CONNECTOR, LAYOUT   model, connector class, default layout
#   MODE                                graph (FULL_DECODE_ONLY CUDA graphs of
#                                       CUDAGRAPH_CAPTURE_SIZES) or eager
#   SERVE_ARGS                          array of model/tuning flags, both roles
#   CONNECTOR_CONFIG                    optional connector_extra_config members
# Run settings (environment): POD, ATTN_TP, FFN_TP, VLLM_ARGS, ATTN_ARGS,
#   FFN_ARGS, M2N_EXTRA_CONFIG, DECODE_BENCH, AFD_REPO, HEAD_HOST,
#   ATTN_HEAD_HOST, FFN_HEAD_HOST, SERVE_HOST, AFD_DRY_RUN.

ATTN_PORT=18000 FFN_PORT=18001 AFD_PORT=6239 ATTN_DP_RPC_PORT=13345 FFN_DP_RPC_PORT=13355

# role_dp TP DP_HOST DP_RPC_PORT <ranks per pod>... -> TOTAL and this pod's DP
# flags in DP. A role spread over pods is led by its first pod; the others run
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
    return 0
}

afd_json() {  # The AFD rendezvous runs on the pod that holds FFN rank 0.
    cat <<JSON
{"afd": {"role": "$1", "connector": "${AFD_CONNECTOR}",
 "host": "${FFN_HEAD_HOST:-${HEAD_HOST:-vllm-ffn-p2p-service}}", "port": ${AFD_PORT},
 "num_attention_ranks": ${NUM_ATTN}, "num_ffn_ranks": ${NUM_FFN},
 "compute_gate_on_attention": ${GATE_ON_ATTENTION}, "afd_process_group_timeout_s": 3600${CONNECTOR_CONFIG}}}
JSON
}

# launch LOG FIRST_GPU NUM_GPUS <vllm serve args>...
launch() {
    local log=$1 devices
    devices=$(seq "$2" $(($2 + $3 - 1)) | paste -sd, -)
    shift 3
    if [[ -n ${AFD_DRY_RUN:-} ]]; then
        echo "CUDA_VISIBLE_DEVICES=$devices vllm serve $(printf '%q ' "$@")> $log"
    else
        CUDA_VISIBLE_DEVICES=$devices vllm serve "$@" > "$log" 2>&1 &
    fi
}

afd_colocation_serve() {
    export VLLM_USE_V2_MODEL_RUNNER=1
    local p
    case ${MODE:-} in
        graph) MODE_ARGS=(--max-cudagraph-capture-size "${CUDAGRAPH_CAPTURE_SIZES##*,}"
                          --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [${CUDAGRAPH_CAPTURE_SIZES}]}") ;;
        eager) MODE_ARGS=(--enforce-eager) ;;
        *) echo "MODE must be graph or eager, got '${MODE:-}'" >&2; exit 1 ;;
    esac

    GATE_ON_ATTENTION=true
    CONNECTOR_CONFIG=${CONNECTOR_CONFIG:-}${M2N_EXTRA_CONFIG:+${CONNECTOR_CONFIG:+, }$M2N_EXTRA_CONFIG}
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

    role_dp "${FFN_TP:-1}" "${FFN_HEAD_HOST:-${HEAD_HOST:-vllm-ffn-dp-service}}" "$FFN_DP_RPC_PORT" "${NUM_F[@]}"
    FFN_DP=("${DP[@]}") NUM_FFN=$TOTAL
    role_dp "${ATTN_TP:-1}" "${ATTN_HEAD_HOST:-${HEAD_HOST:-vllm-attn-dp-service}}" "$ATTN_DP_RPC_PORT" "${NUM_A[@]}"
    ATTN_DP=("${DP[@]}") NUM_ATTN=$TOTAL

    # The roles can finish loading minutes apart, hence the 3600 s timeouts.
    COMMON=(--enable-expert-parallel --trust-remote-code --cpu-distributed-timeout-seconds 3600
            "${SERVE_ARGS[@]}" "${MODE_ARGS[@]}" --host "${SERVE_HOST:-127.0.0.1}")

    # FFN on this pod's GPUs [0, F), Attention on [F, F + A).
    local F=${NUM_F[K]} A=${NUM_A[K]}
    ((F)) && launch ffn.log 0 "$F" "$MODEL_PATH" --additional-config "$(afd_json ffn)" \
        "${COMMON[@]}" "${FFN_DP[@]}" "${ARGS[@]}" "${FFN_EXTRA[@]}" --port "$FFN_PORT"
    ((A)) && launch attn.log "$F" "$A" "$MODEL_PATH" --additional-config "$(afd_json attention)" \
        "${COMMON[@]}" "${ATTN_DP[@]}" "${ARGS[@]}" "${ATTN_EXTRA[@]}" --port "$ATTN_PORT"
    wait
}
