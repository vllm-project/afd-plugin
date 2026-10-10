#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Runtime path is relative to this script; syntax/behavior are checked together.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

# On the two-node AFD layouts node 0 owns only Attention, while node 1 owns
# the remaining Attention DP ranks and all FFN ranks. FFN can wait at the
# CamAsync rendezvous while the cross-node Attention group initializes.
if [[ "${PREFILL_START_FFN}" == 1 ]]; then
  bash "${SCRIPT_DIR}/run_prefill_ffn.sh"
fi
bash "${SCRIPT_DIR}/run_prefill_attention.sh"

echo "P node launch submitted: topology=${PREFILL_TOPOLOGY} node=${PREFILL_NODE_ID}. Watch ${LOG_DIR}/*node${PREFILL_NODE_ID}.log"
