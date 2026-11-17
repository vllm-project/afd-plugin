# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# The CANN ops-transformer installer places Blaze, Tensor API and MC2/OpBase
# kernel headers under ascendc/common. Resolve these once for the generated
# msopgen project so host codegen and kernel OPC use the same SDK installation.
set(MEGAMOE_CANN_HINTS
    "${ASCEND_CANN_PACKAGE_PATH}"
    "$ENV{ASCEND_CANN_PACKAGE_PATH}"
    "$ENV{ASCEND_HOME_PATH}"
    "$ENV{ASCEND_TOOLKIT_HOME}")
list(FILTER MEGAMOE_CANN_HINTS EXCLUDE REGEX "^$")
if(NOT MEGAMOE_CANN_HINTS)
    list(APPEND MEGAMOE_CANN_HINTS "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(MEGAMOE_COMMON_SUFFIX
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common")
find_path(MEGAMOE_TRANSFORMER_COMMON_DIR
    NAMES blaze/gemm/block/block_mmad.h
    HINTS ${MEGAMOE_CANN_HINTS}
    PATH_SUFFIXES "${MEGAMOE_COMMON_SUFFIX}"
        "aarch64-linux/${MEGAMOE_COMMON_SUFFIX}"
        "x86_64-linux/${MEGAMOE_COMMON_SUFFIX}"
    NO_DEFAULT_PATH)
if(NOT MEGAMOE_TRANSFORMER_COMMON_DIR)
    message(FATAL_ERROR
        "Split MegaMoE requires CANN 9.2 ops-transformer source headers. "
        "Set ASCEND_CANN_PACKAGE_PATH or MEGAMOE_TRANSFORMER_COMMON_DIR "
        "to the installed ascendc/common directory.")
endif()

# Check the shared SDK components before invoking the separate OPC compiler.
# Do not silently mix the removed vendored headers with a different SDK copy.
foreach(header
    op_kernel/mc2_moe_context.h op_kernel/mc2_kernel_utils.h
    op_kernel/quantize_functions.h op_kernel/mc2_exception_dump.h
    op_kernel/math_util.h
    blaze/gemm/block/block_mmad_qbmm_mx.h blaze/gemm/utils/layout_struct.h
    blaze/gemm/tile/copy_weight_ub_to_l1.h blaze/gemm/tile/shift_w4_to_w8.h
    blaze/gemm/tile/tile_trait.h blaze/gemm/tile/copy_gm_to_ub.h
    tensor_api/include/tensor_api/tensor.h)
    if(NOT EXISTS "${MEGAMOE_TRANSFORMER_COMMON_DIR}/${header}")
        message(FATAL_ERROR "Split MegaMoE SDK header missing: ${MEGAMOE_TRANSFORMER_COMMON_DIR}/${header}")
    endif()
endforeach()
if(NOT IS_DIRECTORY "${MEGAMOE_TRANSFORMER_COMMON_DIR}/tensor_api/impl")
    message(FATAL_ERROR "Split MegaMoE requires the SDK Tensor API impl directory")
endif()

find_path(MEGAMOE_OPBASE_INCLUDE_DIR
    NAMES op_common/op_host/util/math_util.h
    HINTS ${MEGAMOE_CANN_HINTS}
    PATH_SUFFIXES include pkg_inc aarch64-linux/include aarch64-linux/pkg_inc
        x86_64-linux/include x86_64-linux/pkg_inc
    NO_DEFAULT_PATH)
if(NOT MEGAMOE_OPBASE_INCLUDE_DIR)
    message(FATAL_ERROR
        "Split MegaMoE SDK header op_common/op_host/util/math_util.h missing. "
        "Set MEGAMOE_OPBASE_INCLUDE_DIR to the SDK include or pkg_inc directory.")
endif()

set(MEGAMOE_KERNEL_SDK_INCLUDES
    "${MEGAMOE_TRANSFORMER_COMMON_DIR}"
    "${MEGAMOE_TRANSFORMER_COMMON_DIR}/op_kernel"
    "${MEGAMOE_TRANSFORMER_COMMON_DIR}/tensor_api/include"
    "${MEGAMOE_TRANSFORMER_COMMON_DIR}/tensor_api"
    "${MEGAMOE_OPBASE_INCLUDE_DIR}")
message(STATUS "Split MegaMoE SDK common headers: ${MEGAMOE_TRANSFORMER_COMMON_DIR}")
message(STATUS "Split MegaMoE SDK OpBase headers: ${MEGAMOE_OPBASE_INCLUDE_DIR}")
