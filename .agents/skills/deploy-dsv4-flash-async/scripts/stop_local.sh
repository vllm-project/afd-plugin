#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PID_DIR=${PID_DIR:-${SCRIPT_DIR}/logs/pids}

if [[ ! -d "${PID_DIR}" ]]; then
  echo "No pid directory: ${PID_DIR}"
  exit 0
fi

for pid_file in "${PID_DIR}"/*.pid; do
  [[ -e "${pid_file}" ]] || continue
  pid=$(<"${pid_file}")
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "Stopped pid=${pid} (${pid_file})"
  fi
  rm -f "${pid_file}"
done
