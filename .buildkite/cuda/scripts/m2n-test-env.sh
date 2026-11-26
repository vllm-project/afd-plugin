#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Environment for the DeepEPM2NAFDConnector steps in test-weekly.yml. The step
# commands source it with `. .buildkite/cuda/scripts/m2n-test-env.sh`, so it is
# POSIX sh (the step shell may be dash).

# afdEP-comm builds its transport against the NCCL wheel shipped in the image.
AFDEP_NCCL_ROOT=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])")
export AFDEP_NCCL_ROOT
export LD_LIBRARY_PATH="${AFDEP_NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# All ranks share one node (one NVLink domain), so NCCL GIN is not used.
export EP_DISABLE_GIN=1

# JIT caches on the node's hostPath (the afd-jit-cache volume of the h100_4
# preset in ci_mirror_hardwares.yml), so later runs on the same node skip the
# kernel compilation that dominates a cold run. Every cache keys its entries by
# a source/toolchain fingerprint, so a new pin compiles into new entries.
AFD_JIT_CACHE_ROOT="${AFD_JIT_CACHE_ROOT:-/root/.cache/afd-jit}"
export AFDEP_CACHE_DIR="${AFD_JIT_CACHE_ROOT}/afdep"
export EP_JIT_CACHE_DIR="${AFD_JIT_CACHE_ROOT}/deep_ep"
export DG_JIT_CACHE_DIR="${AFD_JIT_CACHE_ROOT}/deep_gemm"
mkdir -p "${AFDEP_CACHE_DIR}" "${EP_JIT_CACHE_DIR}" "${DG_JIT_CACHE_DIR}"
echo "M2N env: AFDEP_NCCL_ROOT=${AFDEP_NCCL_ROOT} JIT cache=${AFD_JIT_CACHE_ROOT}"
