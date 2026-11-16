/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#pragma message( \
    "impl/tensor_api/arch/vector/gm_to_ub/copy_impl/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H

#include "impl/tensor_api/arch/utils/arch_utils.h"

namespace asc {
namespace te {

class copy_gm_to_ub_instr {
public:
    template <typename DataType>
    __aicore__ inline static void data_copy(
        __ubuf__ DataType* dst, __gm__ DataType* src, const uint16_t block_count, const uint32_t block_len,
        const uint8_t left_padding_count, const uint8_t right_padding_count, const bool enable_constant_pad,
        const asc_load_l2_cache_mode cache_mode, const int64_t src_stride, const int64_t dst_stride)
    {
        using PaddingElementType = Std::conditional_t<(sizeof(DataType) > sizeof(uint32_t)), uint32_t, DataType>;
        TENSOR_API_DEBUG_CHECK(
            debug_check_block_limit, block_count, debug_block_count_max, "block_count", "copy_gm_to_ub instruction");
        TENSOR_API_DEBUG_CHECK(
            debug_check_block_limit, block_len, debug_gm_ub_block_len_max, "block_len", "copy_gm_to_ub instruction");
        TENSOR_API_DEBUG_CHECK(
            debug_check_gm2ub_stride, dst_stride, block_len, block_count, "copy_gm_to_ub instruction");
        TENSOR_API_DEBUG_CHECK(
            debug_check_gm2ub_padding<PaddingElementType>, left_padding_count, right_padding_count, dst_stride,
            block_len, block_count, "copy_gm_to_ub instruction");

        if constexpr (sizeof(DataType) == 1) {
            asc_copy_gm2ub_align(
                reinterpret_cast<__ubuf__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src), block_count,
                block_len, left_padding_count, right_padding_count, enable_constant_pad, cache_mode, src_stride,
                dst_stride);
        } else if constexpr (sizeof(DataType) == 2) {
            asc_copy_gm2ub_align(
                reinterpret_cast<__ubuf__ uint16_t*>(dst), reinterpret_cast<__gm__ uint16_t*>(src), block_count,
                block_len, left_padding_count, right_padding_count, enable_constant_pad, cache_mode, src_stride,
                dst_stride);
        } else if constexpr (sizeof(DataType) == 4) {
            asc_copy_gm2ub_align(
                reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src), block_count,
                block_len, left_padding_count, right_padding_count, enable_constant_pad, cache_mode, src_stride,
                dst_stride);
        } else if constexpr (sizeof(DataType) == 8) {
            asc_copy_gm2ub_align(
                reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src), block_count,
                block_len, left_padding_count, right_padding_count, enable_constant_pad, cache_mode, src_stride,
                dst_stride);
        } else {
            static_assert(
                sizeof(DataType) == 1 || sizeof(DataType) == 2 || sizeof(DataType) == 4 || sizeof(DataType) == 8,
                "Unsupported data type size for CopyGmToUbufAlignV2");
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H

#if defined( \
    UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_INSTRUCTION_H
#endif
