#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common_env.sh
# Runtime path is relative to this script; syntax/behavior are checked together.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

DATASET_PATH=${DSV4_BENCH_DATASET_PATH:-}
DATASET_SHA256=${DSV4_BENCH_DATASET_SHA256:-1ebccbd149bc8f28568d3d5eced3911d2a9473fb015bdb829b898a26abf63d08}
BENCH_HOST=${DSV4_BENCH_HOST:-127.0.0.1}
BENCH_PORT=${DSV4_BENCH_PORT:-${PREFILL_PORT}}
REQUEST_RATE=${DSV4_BENCH_REQUEST_RATE:-10}
MAX_CONCURRENCY=${DSV4_BENCH_MAX_CONCURRENCY-}
NUM_WARMUPS=${DSV4_BENCH_NUM_WARMUPS:-16}
TOPOLOGY=${DSV4_BENCH_TOPOLOGY:-${PREFILL_TOPOLOGY}}
CHUNK_SIZE=${DSV4_BENCH_CHUNK_SIZE:-${PREFILL_MAX_NUM_BATCHED_TOKENS}}
REPEAT=${DSV4_BENCH_REPEAT:-1}
RESULT_ROOT=${DSV4_BENCH_RESULT_ROOT:-${REPO_ROOT}/bench_results/dsv4-flash}
RESULT_FILENAME=${DSV4_BENCH_RESULT_FILENAME:-result.json}
OVERWRITE=${DSV4_BENCH_OVERWRITE:-0}

usage() {
  cat <<'EOF'
Usage: bash run_bench.sh [options] [vllm bench options]

Options:
  --topology NAME          afd_dp3tp4_ep8, afd_dp12tp2, afd_dp10tp2, afd_dp8tp2, afd_dp6tp2,
                           afd_dp6tp4, afd_dp3tp8, afd_dp4tp2,
                           ep16[_dp4tp4|_dp8tp2|
                           _dp2tp8], ep32, or dual_ep16_router.
  --chunk-size TOKENS      4096, 8192, 16384, 32768, 49152, or 65536.
  --request-rate RATE      Offered requests/s; accepts a positive number or inf.
  --repeat N               Repeat index used in the result path.
  --max-concurrency N      Optional client cap. Omit or use 0 for open-loop.
  --num-warmups N          Warmup requests (default: 16).
  -h, --help               Show this help message.

Unrecognized options are forwarded to vllm bench serve.
EOF
}

EXTRA_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --topology)
      TOPOLOGY=${2:?--topology requires a value}
      shift 2
      ;;
    --topology=*)
      TOPOLOGY=${1#*=}
      shift
      ;;
    --chunk-size)
      CHUNK_SIZE=${2:?--chunk-size requires a value}
      shift 2
      ;;
    --chunk-size=*)
      CHUNK_SIZE=${1#*=}
      shift
      ;;
    --request-rate)
      REQUEST_RATE=${2:?--request-rate requires a value}
      shift 2
      ;;
    --request-rate=*)
      REQUEST_RATE=${1#*=}
      shift
      ;;
    --repeat)
      REPEAT=${2:?--repeat requires a value}
      shift 2
      ;;
    --repeat=*)
      REPEAT=${1#*=}
      shift
      ;;
    --max-concurrency)
      MAX_CONCURRENCY=${2:?--max-concurrency requires a value}
      shift 2
      ;;
    --max-concurrency=*)
      MAX_CONCURRENCY=${1#*=}
      shift
      ;;
    --num-warmups)
      NUM_WARMUPS=${2:?--num-warmups requires a value}
      shift 2
      ;;
    --num-warmups=*)
      NUM_WARMUPS=${1#*=}
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

: "${DSV4_MODEL:?Set DSV4_MODEL to the checkpoint directory}"

case "${TOPOLOGY}" in
  afd_dp3tp4_ep8 | afd_dp12tp2 | afd_dp10tp2 | afd_dp8tp2 | afd_dp6tp2 | afd_dp6tp4 | afd_dp3tp8 | afd_dp4tp2 | ep16 | ep16_dp4tp4 | ep16_dp8tp2 | ep16_dp2tp8 | ep32 | dual_ep16_router) ;;
  *)
    echo "Invalid topology: ${TOPOLOGY}" >&2
    exit 2
    ;;
esac
case "${CHUNK_SIZE}" in
  4096 | 8192 | 16384 | 32768 | 49152 | 65536) ;;
  *)
    echo "Invalid chunk size: ${CHUNK_SIZE}" >&2
    exit 2
    ;;
esac
if [[ ! "${REQUEST_RATE}" =~ ^([0-9]+([.][0-9]+)?|inf)$ ]]; then
  echo "Invalid request rate: ${REQUEST_RATE}; use a number or inf" >&2
  exit 2
fi
if [[ ! "${REPEAT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repeat must be a positive integer: ${REPEAT}" >&2
  exit 2
fi
if [[ ! "${NUM_WARMUPS}" =~ ^[0-9]+$ ]]; then
  echo "Warmup count must be a non-negative integer: ${NUM_WARMUPS}" >&2
  exit 2
fi

TOPOLOGY_METADATA=()
case "${TOPOLOGY}" in
  afd_dp3tp4_ep8)
    TOPOLOGY_METADATA=(
      "attention_dp=3" "attention_tp=4" "ffn_ep=8"
      "active_npu_dies=20" "reserved_npu_dies=32"
      "ffn_graph_mode=${PREFILL_FFN_GRAPH_MODE:-EAGER}" "async_moe_ubatching=true"
      "compressor_workspace_requested=true" "layered_gmm=true"
      "ffn_hccl_buffsize=${FFN_HCCL_BUFFSIZE}"
    )
    ;;
  afd_dp12tp2 | afd_dp10tp2 | afd_dp8tp2 | afd_dp6tp2)
    BENCH_DP=${TOPOLOGY#afd_dp}
    BENCH_DP=${BENCH_DP%tp2}
    TOPOLOGY_METADATA=(
      "attention_dp=${BENCH_DP}" "attention_tp=2" "ffn_ep=8"
      "active_npu_dies=$((BENCH_DP * 2 + 8))" "reserved_npu_dies=32"
      "ffn_hccl_buffsize=${FFN_HCCL_BUFFSIZE}"
    )
    ;;
esac

CONCURRENCY_ARGS=()
case "${MAX_CONCURRENCY}" in
  "" | 0 | none) ;;
  *)
    if [[ ! "${MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Max concurrency must be a positive integer, 0, none, or omitted" >&2
      exit 2
    fi
    CONCURRENCY_ARGS=(--max-concurrency "${MAX_CONCURRENCY}")
    ;;
esac

prepare_bench_dataset
DATASET_PATH=${DSV4_BENCH_DATASET_PATH}
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

verify_bench_dataset "${DATASET_PATH}" "${DATASET_SHA256}"

RESULT_DIR=${DSV4_BENCH_RESULT_DIR:-${RESULT_ROOT}/${TOPOLOGY}/chunk_${CHUNK_SIZE}/rps_${REQUEST_RATE}/repeat_${REPEAT}}
RESULT_PATH=${RESULT_DIR}/${RESULT_FILENAME}
if [[ -e "${RESULT_PATH}" && "${OVERWRITE}" != 1 ]]; then
  echo "Refusing to overwrite existing result: ${RESULT_PATH}" >&2
  echo "Set DSV4_BENCH_OVERWRITE=1 to replace it." >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"

echo "Benchmark: topology=${TOPOLOGY} chunk=${CHUNK_SIZE} offered_rps=${REQUEST_RATE} repeat=${REPEAT} endpoint=${BENCH_HOST}:${BENCH_PORT} result=${RESULT_PATH}"
"${VLLM_CLI}" bench serve \
  --backend vllm \
  --label "${TOPOLOGY}" \
  --model "${SERVED_MODEL_NAME}" \
  --tokenizer "${DSV4_MODEL}" \
  --tokenizer-mode deepseek_v4 \
  --trust-remote-code \
  --endpoint /v1/completions \
  --host "${BENCH_HOST}" \
  --port "${BENCH_PORT}" \
  --dataset-name custom \
  --dataset-path "${DATASET_PATH}" \
  --skip-chat-template \
  --custom-output-len 1 \
  --num-prompts -1 \
  --request-rate "${REQUEST_RATE}" \
  ${CONCURRENCY_ARGS[@]+"${CONCURRENCY_ARGS[@]}"} \
  --no-oversample \
  --disable-shuffle \
  --num-warmups "${NUM_WARMUPS}" \
  --temperature 0 \
  --extra-body '{"add_special_tokens":false}' \
  --request-id-prefix "${TOPOLOGY}-c${CHUNK_SIZE}-rps${REQUEST_RATE}-rep${REPEAT}-" \
  --percentile-metrics ttft,e2el \
  --metric-percentiles 25,50,90,95,99 \
  --metadata \
    "topology=${TOPOLOGY}" \
    ${TOPOLOGY_METADATA[@]+"${TOPOLOGY_METADATA[@]}"} \
    "chunk_size=${CHUNK_SIZE}" \
    "offered_rps=${REQUEST_RATE}" \
    "repeat=${REPEAT}" \
    "dataset_sha256=${DATASET_SHA256}" \
    "prefix_cache=false" \
    "kv_connector=false" \
  --save-result \
  --save-detailed \
  --result-dir "${RESULT_DIR}" \
  --result-filename "${RESULT_FILENAME}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
