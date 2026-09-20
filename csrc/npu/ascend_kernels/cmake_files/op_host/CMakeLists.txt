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

# Collect all host-side source files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

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
