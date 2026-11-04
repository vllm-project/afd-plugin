#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# P2pNcclAFDConnector entry point of the shared Qwen3-235B-A22B-FP8 graph
# recipe in the 4A2F layout (one 6-GPU node),
# recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/graph.sh,
# which documents the layout and settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT=${LAYOUT:-4A2F} AFD_CONNECTOR=P2pNcclAFDConnector exec bash \
    "${SCRIPT_DIR}/../../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/graph.sh" "$@"
