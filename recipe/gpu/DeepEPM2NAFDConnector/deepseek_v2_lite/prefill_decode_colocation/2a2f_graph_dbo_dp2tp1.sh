#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# DeepSeek-V2-Lite (BF16) on DeepEPM2NAFDConnector, 2A2F FULL_DECODE_ONLY
# CUDA graphs and DBO: FFN DP2 x TP1 and Attention DP2 x TP1.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
ATTN_TP=${ATTN_TP:-1} FFN_TP=${FFN_TP:-1}
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64}   # ascending
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096
            --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
