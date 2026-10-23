# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
cmake_minimum_required(VERSION 3.16)
project(ffn_faas_mixed_probe LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(SOC_VERSION "Ascend910_9392" CACHE STRING "Offline compiler target")
set(RUN_MODE "npu" CACHE STRING "Compile device binary; do not execute it")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE PATH "Matched CANN installation")
include("${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
ascendc_library(ffn_faas_mixed_probe SHARED probe_kernel.cpp)
ascendc_compile_definitions(ffn_faas_mixed_probe PRIVATE ASCENDC_CUBE_ONLY)
ascendc_compile_options(ffn_faas_mixed_probe PRIVATE
    -O2 --cce-auto-sync=off -Werror -Wno-deprecated-declarations)
ascendc_include_directories(ffn_faas_mixed_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(probe_tiling EXCLUDE_FROM_ALL probe_tiling.cpp)
target_compile_definitions(probe_tiling PRIVATE _GLIBCXX_USE_CXX11_ABI=0
    PROBE_SOC_VERSION="${SOC_VERSION}")
target_include_directories(probe_tiling PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api")
target_link_directories(probe_tiling PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(probe_tiling PRIVATE tiling_api platform register unified_dlog c_sec)
