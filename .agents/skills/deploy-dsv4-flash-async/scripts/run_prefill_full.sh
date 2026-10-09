#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

case "${PREFILL_ENABLE_KV_CONNECTOR}" in
  0 | 1) ;;
  *) echo "PREFILL_ENABLE_KV_CONNECTOR must be 0 or 1" >&2; exit 2 ;;
esac
require_prefill_network

case "${PREFILL_MAX_NUM_BATCHED_TOKENS}" in
  4096 | 8192 | 16384 | 32768 | 65536) ;;
  *)
    echo "PREFILL_MAX_NUM_BATCHED_TOKENS must be one of 4096, 8192, 16384, 32768, 65536" >&2
    exit 1
    ;;
esac

FULL_VISIBLE_DEVICES=${FULL_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

case "${PREFILL_TOPOLOGY}:${PREFILL_NODE_ID}" in
  ep16:0 | ep16_dp4tp4:0)
    FULL_DP_SIZE=4
    FULL_DP_SIZE_LOCAL=4
    FULL_DP_START_RANK=0
    FULL_HEADLESS=0
    FULL_TP_SIZE=4
    ;;
  ep16_dp8tp2:0)
    FULL_DP_SIZE=8
    FULL_DP_SIZE_LOCAL=8
    FULL_DP_START_RANK=0
    FULL_HEADLESS=0
    FULL_TP_SIZE=2
    ;;
  ep16_dp2tp8:0)
    FULL_DP_SIZE=2
    FULL_DP_SIZE_LOCAL=2
    FULL_DP_START_RANK=0
    FULL_HEADLESS=0
    FULL_TP_SIZE=8
    ;;
  ep16:1 | ep16_dp4tp4:1 | ep16_dp8tp2:1 | ep16_dp2tp8:1)
    echo "EP16 is a single-node topology; launch only PREFILL_NODE_ID=0" >&2
    exit 2
    ;;
  ep32:0)
    FULL_DP_SIZE=4
    FULL_DP_SIZE_LOCAL=2
    FULL_DP_START_RANK=0
    FULL_HEADLESS=0
    FULL_TP_SIZE=8
    ;;
  ep32:1)
    FULL_DP_SIZE=4
    FULL_DP_SIZE_LOCAL=2
    FULL_DP_START_RANK=2
    FULL_HEADLESS=1
    FULL_TP_SIZE=8
    ;;
  *)
    echo "run_prefill_full.sh requires PREFILL_TOPOLOGY=ep16, ep16_dp4tp4, ep16_dp8tp2, ep16_dp2tp8, or ep32 and PREFILL_NODE_ID=0 or 1" >&2
    exit 2
    ;;
esac

if (( PREFILL_DP_SIZE != FULL_DP_SIZE || PREFILL_TP_SIZE != FULL_TP_SIZE )); then
  echo "PREFILL_DP_SIZE/TP_SIZE must match the selected full-model topology" >&2
  exit 2
fi

LOG_FILE=${PREFILL_FULL_LOG_FILE:-${LOG_DIR}/prefill_${PREFILL_TOPOLOGY}_node${PREFILL_NODE_ID}.log}
PID_FILE=${PREFILL_FULL_PID_FILE:-${PID_DIR}/prefill_${PREFILL_TOPOLOGY}_node${PREFILL_NODE_ID}.pid}

export FULL_DP_SIZE FULL_DP_SIZE_LOCAL FULL_DP_START_RANK FULL_HEADLESS
export FULL_TP_SIZE FULL_VISIBLE_DEVICES

echo "Starting DSV4 full prefill: topology=${PREFILL_TOPOLOGY} node=${PREFILL_NODE_ID} global=DP${FULL_DP_SIZE}TP${FULL_TP_SIZE}/EP$((FULL_DP_SIZE * FULL_TP_SIZE)) local_dp=${FULL_DP_SIZE_LOCAL} start_rank=${FULL_DP_START_RANK} headless=${FULL_HEADLESS} devices=${FULL_VISIBLE_DEVICES} chunk=${PREFILL_MAX_NUM_BATCHED_TOKENS}"
nohup bash "${SCRIPT_DIR}/run_prefill_full_rank.sh" > "${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "Started ${PREFILL_TOPOLOGY} node=${PREFILL_NODE_ID} pid=$(<"${PID_FILE}") log=${LOG_FILE}"
