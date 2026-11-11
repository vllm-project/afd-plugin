#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-235B-A22B-FP8 on DeepEPM2NAFDConnector, two 4A1F pods with
# FULL_DECODE_ONLY CUDA graphs and DBO: Attention DP8 x TP1, FFN DP2 x TP1.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-4A1F,4A1F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
CONNECTOR_CONFIG='"payload_format": "fp8_block128_fp32"'   # FP8 payload
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64,96,128,192,256,384,512}   # ascending
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 512 --max-num-batched-tokens 3072 --max-model-len 4096
            --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
