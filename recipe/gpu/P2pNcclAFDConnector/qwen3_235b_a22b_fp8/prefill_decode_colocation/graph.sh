#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-235B-A22B-FP8 on P2pNcclAFDConnector, FULL_DECODE_ONLY CUDA graphs.
# Placement: one pod per comma-separated LAYOUT entry (default 4A2F), pod k
# running POD=AFD_<k> with the same LAYOUT; the default is one
# 6-GPU node. Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the
# P2P FFN hangs on its first step.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-P2pNcclAFDConnector}
LAYOUT=${LAYOUT:-4A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,12,16,24,32,48,64,96,128,192,256}   # ascending
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 256 --max-num-batched-tokens 3072 --max-model-len 4096)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
