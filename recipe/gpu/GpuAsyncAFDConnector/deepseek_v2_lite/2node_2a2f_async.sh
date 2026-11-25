#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Run once per node with two visible GPUs and the async CUDA extension installed.
# Set MODEL_PATH to the local checkpoint and AFD_HOST to the Attention node's
# reachable address on both nodes.
# Attention node: bash 2node_2a2f_async.sh attention
# FFN node:       bash 2node_2a2f_async.sh ffn (defaults to FFN_LAUNCH=faas;
#                 set FFN_LAUNCH=worker to run through vllm serve instead)
set -euo pipefail

case "${1:-}" in
    attention)
        AFD_MODEL=${ATTENTION_MODEL_PATH:-${MODEL_PATH:?Set MODEL_PATH or ATTENTION_MODEL_PATH}}
        AFD_API_PORT=${ATTENTION_API_PORT:-8000}
        ;;
    ffn)
        AFD_MODEL=${FFN_MODEL_PATH:-${MODEL_PATH:?Set MODEL_PATH or FFN_MODEL_PATH}}
        AFD_API_PORT=${FFN_API_PORT:-8001}
        FFN_LAUNCH=${FFN_LAUNCH:-faas}
        ;;
    *)
        echo "Usage: bash $0 {attention|ffn} [engine options...]" >&2
        exit 2
        ;;
esac
AFD_ROLE=$1
shift
: "${AFD_HOST:?Set AFD_HOST to the reachable address of the Attention node}"
AFD_PORT=${AFD_PORT:-6281}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-ibrc}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-1G}"
export VLLM_PLUGINS=afd VLLM_USE_V2_MODEL_RUNNER=1

COMMON_ARGS=(
    --trust-remote-code --dtype bfloat16 --seed 0
    --data-parallel-size 2 --tensor-parallel-size 1
    --enable-expert-parallel --enable-ep-weight-filter
    --max-model-len 4096 --max-num-batched-tokens 1024 --max-num-seqs 8
    --gpu-memory-utilization 0.5
    --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --cudagraph-capture-sizes 2 4 8
    --additional-config "{\"afd\":{\"role\":\"$AFD_ROLE\",\"connector\":\"GpuAsyncAFDConnector\",\"async\":true,\"compute_gate_on_attention\":true,\"num_attention_ranks\":2,\"num_ffn_ranks\":2,\"host\":\"$AFD_HOST\",\"port\":$AFD_PORT}}"
)

if [[ $AFD_ROLE == ffn ]]; then
    case "$FFN_LAUNCH" in
        faas)
            exec "${TORCHRUN_BIN:-torchrun}" --nproc-per-node=2 \
                --master-addr="${FFN_MASTER_ADDR:-127.0.0.1}" \
                --master-port="${FFN_MASTER_PORT:-29500}" \
                -m afd_plugin.ffn_faas serve "$AFD_MODEL" --backend gpu-async \
                --status-host "${FFN_STATUS_HOST:-127.0.0.1}" \
                --status-port "${FFN_STATUS_PORT:-29700}" \
                --receiver-shutdown-timeout "${FFN_SHUTDOWN_TIMEOUT:-120}" \
                "${COMMON_ARGS[@]}" \
                "$@"
            ;;
        worker)
            ;;
        *)
            echo "Unsupported FFN_LAUNCH=$FFN_LAUNCH (expected faas or worker)" >&2
            exit 2
            ;;
    esac
fi

exec "${VLLM_BIN:-vllm}" serve "$AFD_MODEL" \
    --host 0.0.0.0 --port "$AFD_API_PORT" --served-model-name deepseek-v2-lite \
    --api-server-count 1 --shutdown-timeout 120 \
    "${COMMON_ARGS[@]}" \
    "$@"
