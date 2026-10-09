#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

require_common_network

LOG_FILE=${PROXY_LOG_FILE:-${LOG_DIR}/proxy.log}
PID_FILE=${PROXY_PID_FILE:-${PID_DIR}/proxy.pid}
PROXY_SCRIPT=${PROXY_SCRIPT:-${PYTHON_VLLM_ASCEND_PATH}/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py}

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
  echo "Proxy script not found: ${PROXY_SCRIPT}" >&2
  exit 1
fi

nohup "${PYTHON}" "${PROXY_SCRIPT}" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --prefiller-hosts "${P_NODE_IP}" \
  --prefiller-ports "${PREFILL_PORT}" \
  --decoder-hosts "${D_NODE_IP}" \
  --decoder-ports "${DECODE_PORT}" > "${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "Started PD proxy pid=$(<"${PID_FILE}") log=${LOG_FILE} port=${PROXY_PORT}"
