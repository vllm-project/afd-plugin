#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# P2pNcclAFDConnector entry point of the shared Qwen3-235B-A22B-FP8 4A2F recipe.
# The topology and every environment knob are documented in
# recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFD_CONNECTOR=P2pNcclAFDConnector exec bash \
    "${SCRIPT_DIR}/../../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh"
