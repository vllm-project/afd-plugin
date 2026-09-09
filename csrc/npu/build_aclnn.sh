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

cd "${ROOT_DIR}/csrc/npu"
rm -rf build output
echo "building AFD ACLNN custom ops a2e;e2a for ${SOC_ARG}"
bash build.sh -n "a2e;e2a" -c "${SOC_ARG}"

INSTALL_PATH="${ROOT_DIR}/afd_plugin/_cann_ops_custom"
rm -rf "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}"
shopt -s nullglob
run_files=(./output/CANN-custom_ops*.run)
if [ "${#run_files[@]}" -ne 1 ]; then
  echo "expected one CANN-custom_ops installer under output/, found ${#run_files[@]}"
  exit 1
fi
"${run_files[0]}" --install-path="${INSTALL_PATH}"
