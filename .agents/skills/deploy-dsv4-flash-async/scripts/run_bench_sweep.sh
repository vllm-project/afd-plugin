#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Runtime path is relative to this script; syntax/behavior are checked together.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

TOPOLOGY=${DSV4_BENCH_TOPOLOGY:-${PREFILL_TOPOLOGY}}
CHUNK_SIZE=${DSV4_BENCH_CHUNK_SIZE:-${PREFILL_MAX_NUM_BATCHED_TOKENS}}
REPEATS=${DSV4_BENCH_REPEATS:-3}
RATES=${DSV4_BENCH_RATES:-2,3,4,5}
RESULT_ROOT=${DSV4_BENCH_RESULT_ROOT:-${REPO_ROOT}/bench_results/dsv4-flash}
BENCH_HOST=${DSV4_BENCH_HOST:-127.0.0.1}
BENCH_PORT=${DSV4_BENCH_PORT:-${PREFILL_PORT}}
HEALTH_TIMEOUT=${DSV4_BENCH_HEALTH_TIMEOUT:-1800}
OOM_REASON=

usage() {
  cat <<'EOF'
Usage: bash scripts/dsv4-flash/run_bench_sweep.sh [options]

Options:
  --topology NAME          afd_dp3tp4_ep8, afd_dp6tp4, afd_dp3tp8, afd_dp4tp2,
                           ep16[_dp4tp4|_dp8tp2|
                           _dp2tp8], ep32, or dual_ep16_router.
  --chunk-size TOKENS      4096, 8192, 16384, 32768, 49152, or 65536.
  --repeats N              Repetitions per request rate (default: 3).
  --rates CSV              Offered request rates (default: 2,3,4,5).
  --record-oom REASON      Record this topology/chunk as undeployable and exit.
  -h, --help               Show this help message.

The service must already have been launched with the same topology and chunk.
EOF
}

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
    --repeats)
      REPEATS=${2:?--repeats requires a value}
      shift 2
      ;;
    --repeats=*)
      REPEATS=${1#*=}
      shift
      ;;
    --rates)
      RATES=${2:?--rates requires a value}
      shift 2
      ;;
    --rates=*)
      RATES=${1#*=}
      shift
      ;;
    --record-oom)
      OOM_REASON=${2:?--record-oom requires a reason}
      shift 2
      ;;
    --record-oom=*)
      OOM_REASON=${1#*=}
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${TOPOLOGY}" in
  afd_dp3tp4_ep8 | afd_dp6tp4 | afd_dp3tp8 | afd_dp4tp2 | ep16 | ep16_dp4tp4 | ep16_dp8tp2 | ep16_dp2tp8 | ep32 | dual_ep16_router) ;;
  *) echo "Invalid topology: ${TOPOLOGY}" >&2; exit 2 ;;
esac
case "${CHUNK_SIZE}" in
  4096 | 8192 | 16384 | 32768 | 49152 | 65536) ;;
  *) echo "Invalid chunk size: ${CHUNK_SIZE}" >&2; exit 2 ;;
esac
if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repeats must be a positive integer: ${REPEATS}" >&2
  exit 2
fi

STATUS_DIR=${RESULT_ROOT}/${TOPOLOGY}/chunk_${CHUNK_SIZE}
STATUS_PATH=${STATUS_DIR}/deployment_status.json

record_status() {
  local status=$1
  local reason=$2
  mkdir -p "${STATUS_DIR}"
  "${PYTHON}" - "${STATUS_PATH}" "${status}" "${TOPOLOGY}" "${CHUNK_SIZE}" "${reason}" <<'PY'
import datetime
import json
import pathlib
import sys

path, status, topology, chunk_size, reason = sys.argv[1:]
payload = {
    "status": status,
    "topology": topology,
    "chunk_size": int(chunk_size),
    "reason": reason,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY
  echo "Recorded deployment status: ${STATUS_PATH}"
}

if [[ -n "${OOM_REASON}" ]]; then
  record_status oom_undeployable "${OOM_REASON}"
  exit 0
fi

DATASET_PATH=${DSV4_BENCH_DATASET_PATH:-}
EXPECTED_DATASET_SHA256=${DSV4_BENCH_DATASET_SHA256:-1ebccbd149bc8f28568d3d5eced3911d2a9473fb015bdb829b898a26abf63d08}
prepare_bench_dataset
DATASET_PATH=${DSV4_BENCH_DATASET_PATH}
if [[ ! -f "${DATASET_PATH}" ]]; then
  record_status dataset_missing "Dataset not found: ${DATASET_PATH}"
  exit 1
fi
if ! verify_bench_dataset "${DATASET_PATH}" "${EXPECTED_DATASET_SHA256}"; then
  record_status dataset_invalid "Dataset SHA-256 validation failed; inspect the benchmark log."
  exit 1
fi

IFS=',' read -r -a RATE_LIST <<< "${RATES}"
for rate in "${RATE_LIST[@]}"; do
  if [[ ! "${rate}" =~ ^([0-9]+([.][0-9]+)?|inf)$ ]]; then
    echo "Invalid request rate in --rates: ${rate}" >&2
    exit 2
  fi
done

HEALTH_URL="http://${BENCH_HOST}:${BENCH_PORT}/health"
echo "Waiting up to ${HEALTH_TIMEOUT}s for ${HEALTH_URL}"
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    record_status health_timeout "Service did not become healthy at ${HEALTH_URL} within ${HEALTH_TIMEOUT}s; inspect server logs and record OOM explicitly if applicable."
    exit 1
  fi
  sleep 10
done
record_status ready "Service passed health check; benchmark sweep started."

for repeat in $(seq 1 "${REPEATS}"); do
  for rate in "${RATE_LIST[@]}"; do
    result_path=${RESULT_ROOT}/${TOPOLOGY}/chunk_${CHUNK_SIZE}/rps_${rate}/repeat_${repeat}/result.json
    if [[ -e "${result_path}" && "${DSV4_BENCH_OVERWRITE:-0}" != 1 ]]; then
      echo "Validating existing result: ${result_path}"
    elif ! DSV4_BENCH_HOST="${BENCH_HOST}" \
      DSV4_BENCH_PORT="${BENCH_PORT}" \
      DSV4_BENCH_RESULT_ROOT="${RESULT_ROOT}" \
      bash "${SCRIPT_DIR}/run_bench.sh" \
        --topology "${TOPOLOGY}" \
        --chunk-size "${CHUNK_SIZE}" \
        --request-rate "${rate}" \
        --repeat "${repeat}"; then
      record_status benchmark_failed "Benchmark failed at repeat=${repeat}, offered_rps=${rate}; inspect the service and benchmark logs."
      exit 1
    fi
    if ! "${PYTHON}" - "${result_path}" "${TOPOLOGY}" "${CHUNK_SIZE}" "${rate}" "${repeat}" "${EXPECTED_DATASET_SHA256}" <<'PY'
import json
import math
import pathlib
import sys

path, topology, chunk_size, offered_rps, repeat, dataset_sha256 = sys.argv[1:]
result = json.loads(pathlib.Path(path).read_text())
expected = {
    "completed": 1536,
    "failed": 0,
    "total_input_tokens": 15803063,
    "total_output_tokens": 1536,
}
for key, value in expected.items():
    if result.get(key) != value:
        raise SystemExit(f"{path}: expected {key}={value}, got {result.get(key)!r}")
for key in ("input_lens", "ttfts"):
    if len(result.get(key, [])) != 1536:
        raise SystemExit(
            f"{path}: expected 1536 detailed {key}, got {len(result.get(key, []))}"
        )
if sum(result["input_lens"]) != expected["total_input_tokens"]:
    raise SystemExit(f"{path}: detailed input lengths do not match total input tokens")
if max(result["input_lens"]) != 63778:
    raise SystemExit(f"{path}: detailed input maximum does not match the frozen workload")
if result.get("output_lens") != [1] * expected["completed"]:
    raise SystemExit(f"{path}: every measured request must produce exactly one token")
errors = result.get("errors", [])
if len(errors) != expected["completed"] or any(errors):
    raise SystemExit(f"{path}: expected 1536 detailed requests without errors")
if not all(math.isfinite(value) and value > 0 for value in result["ttfts"]):
    raise SystemExit(f"{path}: measured TTFT values must be finite and positive")
metadata = {
    "topology": topology,
    "chunk_size": chunk_size,
    "offered_rps": offered_rps,
    "repeat": repeat,
    "dataset_sha256": dataset_sha256,
}
for key, value in metadata.items():
    if str(result.get(key)) != value:
        raise SystemExit(
            f"{path}: expected metadata {key}={value}, got {result.get(key)!r}"
        )
PY
    then
      record_status benchmark_invalid "Result validation failed at repeat=${repeat}, offered_rps=${rate}."
      exit 1
    fi
  done
done

record_status completed "All configured request rates and repetitions completed."
