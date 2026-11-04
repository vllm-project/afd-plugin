#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# P2pNcclAFDConnector entry point of the shared Qwen3-30B-A3B graph recipe,
# recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh,
# which documents the layout and settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFD_CONNECTOR=P2pNcclAFDConnector exec bash \
    "${SCRIPT_DIR}/../../../DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh" "$@"
