#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

PROXY_IP=${PROXY_IP:-127.0.0.1}
PROXY_PORT=${PROXY_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv4}

curl -fsS --connect-timeout 10 --max-time "${REQUEST_TIMEOUT:-120}" "http://${PROXY_IP}:${PROXY_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Who are you?\"}],
    \"max_tokens\": 32,
    \"temperature\": 0
  }"
