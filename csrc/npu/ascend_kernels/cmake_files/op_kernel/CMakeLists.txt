# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# op_kernel build for the AFD Ascend custom-op run package (npu_op_* build
# system): collects kernel compile options, then builds the AscendC kernel
# library linked against the tiling library.

# Collect all kernel compile options into a single list to avoid overwriting
# custom_compile_options.ini. Options carried over from the legacy
# per-operator CMakeLists (a2e/e2a): --cce-auto-sync=off,
# -Wno-deprecated-declarations, -Werror.
set(KERNEL_COMPILE_OPTS)

# Common options: add current source dir as global header search path
# (resolves the shared utils/op_kernel headers copied next to the kernels).
list(APPEND KERNEL_COMPILE_OPTS -I${CMAKE_CURRENT_SOURCE_DIR} -DASCENDC_DUMP=0 --cce-auto-sync=off -Wno-deprecated-declarations -Werror)

# Mega uses CANN's W4A8 BlockMmad directly; ordinary operator builds do not need it.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/afd_mega_moe_pull_native.cpp")
    find_path(AFD_MC2_INCLUDE_DIR
        NAMES template_linear_algebra_v2/catlass.hpp
        HINTS "${ASCEND_CANN_PACKAGE_PATH}" "$ENV{ASCEND_HOME_PATH}"
        PATH_SUFFIXES opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/mega_moe/arch22
        NO_DEFAULT_PATH)
    if(NOT AFD_MC2_INCLUDE_DIR OR NOT EXISTS "${AFD_MC2_INCLUDE_DIR}/utils/block_mmad_w4a4.hpp")
        message(FATAL_ERROR "AfdMegaMoePullNative requires the CANN 9.1 MC2 W4A8 headers in ASCEND_CANN_PACKAGE_PATH")
    endif()
    list(APPEND KERNEL_COMPILE_OPTS -I${AFD_MC2_INCLUDE_DIR})
endif()

# set custom compile options (single call to avoid overwriting)
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options("" ALL OPTIONS -g -O0 ${KERNEL_COMPILE_OPTS})
else()
    npu_op_kernel_options("" ALL OPTIONS ${KERNEL_COMPILE_OPTS})
endif()

# Collect AscendC kernel source files
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

# Compile AscendC kernels and link with tiling library
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

# Register kernel library into the package
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
