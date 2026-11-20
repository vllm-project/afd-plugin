#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

ROOT_DIR=$1
SOC_VERSION=$2

case "$SOC_VERSION" in
  910c|ascend910_9392|ascend910_93*)
    SOC_ARG="ascend910_93"
    ;;
  950|ascend950*|Ascend950*)
    SOC_ARG="ascend950"
    ;;
  *)
    echo "AFD A2E/E2A custom ACLNN ops are currently built only for Ascend 910C/950; got ${SOC_VERSION}."
    exit 0
    ;;
esac

NPU_CSRC_DIR="${ROOT_DIR}/csrc/npu"
rm -rf "${NPU_CSRC_DIR}/build" "${NPU_CSRC_DIR}/output"
echo "building AFD ACLNN custom ops selected from the registry for ${SOC_ARG}"
bash "${NPU_CSRC_DIR}/scripts/compile_ascend_proj.sh" \
  "${NPU_CSRC_DIR}" \
  "${SOC_ARG}" \
  "${CMAKE_BUILD_TYPE:-Release}" \
  "${NPU_CSRC_DIR}/output"

INSTALL_PATH="${ROOT_DIR}/afd_plugin/_cann_ops_custom"
rm -rf "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}"
run_file="${NPU_CSRC_DIR}/output/AFD_${SOC_ARG}.run"
if [ ! -f "${run_file}" ]; then
  echo "expected run package at ${run_file} but it is missing"
  exit 1
fi
"${run_file}" --install-path="${INSTALL_PATH}"
