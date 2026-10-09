#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${PREFILL_HOST_IP:?Set PREFILL_HOST_IP to the Prefill API address}"
: "${ATTENTION_HOST_IP:?Set ATTENTION_HOST_IP to the Attention API address}"
exec "$PYTHON_BIN" "$ROOT_DIR/tools/proxy_server.py" \
  --host "${PROXY_HOST:-0.0.0.0}" \
  --port "${PROXY_PORT:-9000}" \
  --workers 1 \
  --prefiller-hosts "$PREFILL_HOST_IP" \
  --prefiller-ports "${PREFILL_API_PORT:-8100}" \
  --decoder-hosts "$ATTENTION_HOST_IP" \
  --decoder-ports "${ATTENTION_API_PORT:-8910}"
