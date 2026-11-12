#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3.5-122B-A10B-FP8 on P2pNcclAFDConnector, FULL_DECODE_ONLY CUDA graphs.
# Placement: one pod per comma-separated LAYOUT entry (default 2A0F,2A0F,0A2F), pod k
# running POD=AFD_<k> with the same LAYOUT; the default runs
# 4A2F on three 2-GPU pods (Attention DP4 over pods 0-1, FFN DP2 on pod 2);
# LAYOUT=4A2F runs it on one 6-GPU node.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-P2pNcclAFDConnector}
LAYOUT=${LAYOUT:-2A0F,2A0F,0A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-32}   # ascending
SERVE_ARGS=(--dtype bfloat16 --language-model-only --max-model-len 114688 --mamba-cache-mode align
            --all2all-backend allgather_reducescatter --seed 0 --max-num-seqs 32 --max-num-batched-tokens 8192)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
