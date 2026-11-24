#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-235B-A22B-FP8 4A4F EPLB on DeepEPM2NAFDConnector: two 4-GPU pods, each
# holding two Attention and two FFN ranks (LAYOUT=2A2F,2A2F). Attention is
# DP4 x TP1 and FFN is DP4 x TP1 (expert parallel); each role is led by pod 0
# and pod 1 runs headless from its start rank. FULL_DECODE_ONLY CUDA graphs.
# The FFN runs EPLB with 16 redundant experts, like the single-node
# 4a2f_graph_dp4tp1_eplb.sh recipe.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-2A2F,2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
CONNECTOR_CONFIG='"payload_format": "fp8_block128_fp32"'   # FP8 payload
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,48,64,96,112,128,192,256}
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 256 --max-num-batched-tokens 3072 --max-model-len 4096)

# EPLB is FFN-only: Attention maps its logical top-k ids through the FFN's
# placement, so it needs no EPLB flags. Sync EPLB is required by the connector.
# With 4 FFN ranks the 16 redundant experts make 144 physical experts, 36 per
# rank, so every rank still owns a whole number of experts.
EPLB_NUM_REDUNDANT_EXPERTS=${EPLB_NUM_REDUNDANT_EXPERTS:-16}
EPLB_STEP_INTERVAL=${EPLB_STEP_INTERVAL:-3000}
EPLB_LOG_INTERVAL=${EPLB_LOG_INTERVAL:-100}
FFN_ARGS="--enable-eplb --eplb-config {\"num_redundant_experts\":${EPLB_NUM_REDUNDANT_EXPERTS},\"step_interval\":${EPLB_STEP_INTERVAL},\"use_async\":false,\"log_balancedness\":true,\"log_balancedness_interval\":${EPLB_LOG_INTERVAL}}${FFN_ARGS:+ ${FFN_ARGS}}"

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
