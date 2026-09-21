# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# op_host build for the AFD Ascend custom-op run package (npu_op_* build
# system): generates ACLNN stubs / op_proto / fallback tiling sources, then
# builds the cust_opapi / cust_op_proto / cust_optiling libraries.

# =====================================================
# Detect CANN version include path globally
# =====================================================
if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH} AND NOT "$ENV{ASCEND_CANN_PACKAGE_PATH}" STREQUAL "")
    set(CANN_ROOT "$ENV{ASCEND_CANN_PACKAGE_PATH}")
elseif(DEFINED ASCEND_CANN_PACKAGE_PATH AND NOT "${ASCEND_CANN_PACKAGE_PATH}" STREQUAL "")
    set(CANN_ROOT "${ASCEND_CANN_PACKAGE_PATH}")
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_ROOT "$ENV{ASCEND_TOOLKIT_HOME}")
else()
    set(CANN_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CANN83_PATH "${CANN_ROOT}/include/platform/platform_infos_def.h")
set(CANN82_PATH "${CANN_ROOT}/include/experiment/platform/platform/platform_infos_def.h")

if(EXISTS "${CANN83_PATH}")
    add_definitions(-DUSE_CANN83_PATH)
    set(ACTIVE_CANN_MACRO "-DUSE_CANN83_PATH")
    message(STATUS ">>> [op_host] Detected CANN 8.3+, defined USE_CANN83_PATH")
elseif(EXISTS "${CANN82_PATH}")
    add_definitions(-DUSE_CANN82_PATH)
    set(ACTIVE_CANN_MACRO "-DUSE_CANN82_PATH")
    message(STATUS ">>> [op_host] Detected CANN 8.2, defined USE_CANN82_PATH")
else()
    message(FATAL_ERROR ">>> [op_host] platform_infos_def.h not found!\n  Checked:\n   - ${CANN83_PATH}\n   - ${CANN82_PATH}")
endif()

# =====================================================
# AFD Ascend 950 (A5) tiling gate
# =====================================================
# Mc2CcTilingConfig::SetCommEngine only exists in CANN releases that support
# Ascend 950; gate its use so 910_93 builds against older CANN still compile.
# The define must reach add_definitions() (for the cust_optiling compile) and
# CODE_GEN_COMPILE_OPTS (for npu_op_code_gen's separate compile process).
# Initialise CODE_GEN_COMPILE_OPTS first, then append the A5 gate so the
# set() below does not clobber the list(APPEND).

# Custom compile options for npu_op_code_gen (passed to
# compile_ascendc_all_ops_so.py via --compile-options). The current source dir
# is added so op_host sources resolve the ../op_kernel/ tiling headers.
set(CODE_GEN_COMPILE_OPTS -I${CMAKE_CURRENT_SOURCE_DIR})
if(DEFINED ASCEND_COMPUTE_UNIT AND "${ASCEND_COMPUTE_UNIT}" MATCHES "ascend950")
    add_definitions(-DAFD_TILING_HAS_COMM_ENGINE=1)
    list(APPEND CODE_GEN_COMPILE_OPTS -DAFD_TILING_HAS_COMM_ENGINE=1)
    message(STATUS ">>> [op_host] ascend950 detected, defined AFD_TILING_HAS_COMM_ENGINE")
endif()

# =====================================================
# Host-side include paths
# =====================================================
# The grouped_matmul_swiglu_quant_v2_layered op_host/op_api sources use the
# l0op composite pattern and CANN op_common headers that resolve outside the
# canonical ASCEND_HOME_PATH/include tree. Add the pkg_inc and aclnn include
# directories to both the in-tree compile (include_directories) and the
# separate npu_op_code_gen process (CODE_GEN_COMPILE_OPTS), mirroring the
# upstream cam comm_operator op_host build.
set(CANN_ARCH_DIR "")
foreach(cand "aarch64-linux" "x86_64-linux" "linux")
    if(EXISTS "${CANN_ROOT}/${cand}/pkg_inc/op_common")
        set(CANN_ARCH_DIR "${CANN_ROOT}/${cand}")
        break()
    endif()
endforeach()
if(NOT "${CANN_ARCH_DIR}" STREQUAL "")
    include_directories("${CANN_ARCH_DIR}/pkg_inc/op_common")
    list(APPEND CODE_GEN_COMPILE_OPTS -I${CANN_ARCH_DIR}/pkg_inc/op_common)
    # migrated op_host sources include "util/math_util.h"
    # (pkg_inc/op_common/op_host/util)
    if(EXISTS "${CANN_ARCH_DIR}/pkg_inc/op_common/op_host")
        include_directories("${CANN_ARCH_DIR}/pkg_inc/op_common/op_host")
        list(APPEND CODE_GEN_COMPILE_OPTS -I${CANN_ARCH_DIR}/pkg_inc/op_common/op_host)
    endif()
    # log/log.h includes "op_common/op_host/util/opbase_export.h" and "dlog_pub.h"
    if(EXISTS "${CANN_ARCH_DIR}/pkg_inc")
        include_directories("${CANN_ARCH_DIR}/pkg_inc")
        list(APPEND CODE_GEN_COMPILE_OPTS -I${CANN_ARCH_DIR}/pkg_inc)
    endif()
    if(EXISTS "${CANN_ARCH_DIR}/pkg_inc/base")
        include_directories("${CANN_ARCH_DIR}/pkg_inc/base")
        list(APPEND CODE_GEN_COMPILE_OPTS -I${CANN_ARCH_DIR}/pkg_inc/base)
    endif()
endif()
if(EXISTS "${CANN_ROOT}/include/aclnn")
    include_directories("${CANN_ROOT}/include/aclnn")
    list(APPEND CODE_GEN_COMPILE_OPTS -I${CANN_ROOT}/include/aclnn)
endif()

# Collect all host-side source files recursively.
file(GLOB_RECURSE ops_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# =====================================================
# Step 1: Generate ACLNN stubs, op_proto.cc, fallback files
# =====================================================
npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${package_name}
    COMPILE_OPTIONS -g ${ACTIVE_CANN_MACRO} ${CODE_GEN_COMPILE_OPTS}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

# =====================================================
# Step 2: ACLNN single-operator invocation library
# =====================================================
# Copy pre-generated autogen files from pregen/build_out/autogen/
file(GLOB pregen_files "${CMAKE_CURRENT_SOURCE_DIR}/../pregen/build_out/autogen/*")
file(COPY ${pregen_files} DESTINATION ${ASCEND_AUTOGEN_PATH})

# Copy any op_api/ files that exist locally
file(GLOB_RECURSE op_api_files "${CMAKE_CURRENT_SOURCE_DIR}/op_api/*")
if(op_api_files)
    file(COPY ${op_api_files} DESTINATION ${ASCEND_AUTOGEN_PATH})
endif()

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn*.cpp)

npu_op_library(cust_opapi ACLNN
    ${aclnn_src}
)
# The grouped_matmul_swiglu_quant_v2_layered aclnn sources use the l0op
# composite pattern (l0op::Contiguous / ReFormat / TransData / ViewCopy live in
# the CANN built-in libopapi.so). The ACLNN target template only links
# nnopbase/ascendcl, so add libopapi as a direct dependency: it becomes a
# DT_NEEDED entry of libcust_opapi.so and the loader resolves the l0op builtins
# from it no matter whether torch_npu has libopapi in the global dlopen scope.
target_link_libraries(cust_opapi PRIVATE opapi)
target_compile_options(cust_opapi PRIVATE
    -fvisibility=hidden
)

# =====================================================
# Step 3: Operator prototype library (for graph compilation)
# =====================================================
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${ops_srcs}
    ${proto_src}
)
target_include_directories(cust_op_proto PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(cust_op_proto PRIVATE
    -fvisibility=hidden
)

# =====================================================
# Step 4: Tiling library
# =====================================================
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${fallback_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_optiling TILING
    ${ops_srcs}
    ${fallback_src}
)
target_include_directories(cust_optiling PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(cust_optiling PRIVATE
    -fvisibility=hidden
)

# =====================================================
# Step 5: Register all host-side libraries into the package
# =====================================================
npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
