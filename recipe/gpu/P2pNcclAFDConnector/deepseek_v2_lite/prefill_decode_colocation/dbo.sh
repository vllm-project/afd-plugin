#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# DeepSeek-V2-Lite (BF16) on P2pNcclAFDConnector, FULL_DECODE_ONLY graphs + DBO.
# Placement: one pod per comma-separated LAYOUT entry (default 2A2F), pod k
# running POD=AFD_<k> with the same LAYOUT; ATTN_TP/FFN_TP set the TP size
# of each role, e.g. ATTN_TP=2 FFN_TP=2 for DP1 x TP2, LAYOUT=4A4F ATTN_TP=2
# FFN_TP=2 on 8 GPUs, or LAYOUT=2A0F,0A2F ATTN_TP=2 FFN_TP=2 on two pods.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-P2pNcclAFDConnector}
LAYOUT=${LAYOUT:-2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-64}   # ascending
SERVE_ARGS=(--max-num-seqs 64 --max-num-batched-tokens 64
            --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
