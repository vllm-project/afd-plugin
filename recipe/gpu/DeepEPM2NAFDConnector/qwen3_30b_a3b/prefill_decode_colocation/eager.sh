#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B (BF16) on DeepEPM2NAFDConnector, eager mode.
# Placement: one pod per comma-separated LAYOUT entry (default 2A2F), pod k
# running POD=AFD_<k> with the same LAYOUT; FFN on each pod's first GPUs.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
MODE=eager
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
