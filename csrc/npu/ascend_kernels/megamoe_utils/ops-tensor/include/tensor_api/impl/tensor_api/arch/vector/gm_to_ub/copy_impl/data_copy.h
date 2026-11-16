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
    "impl/tensor_api/arch/vector/gm_to_ub/copy_impl/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/vector/utils/copy_utils.h"
#include "impl/tensor_api/arch/vector/gm_to_ub/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_gm_to_ub_common {
protected:
    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static void emit_copy(
        const DstTensor& dst, const SrcTensor& src, uint16_t block_count, uint32_t block_len, int64_t src_stride,
        int64_t dst_stride, const gm_to_ub_params& params)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        adjust_b4_copy_params<src_type, dst_type>(block_len, src_stride, dst_stride);

        auto cache_mode = static_cast<asc_load_l2_cache_mode>(src.engine().get_cache_mode());
        copy_gm_to_ub_instr::data_copy(
            dst.data().get(), src.data().get(), block_count, block_len, params.left_padding_count,
            params.right_padding_count, params.enable_constant_pad, cache_mode, src_stride, dst_stride);
    }

    template <typename T, typename U, typename DstOffset, typename SrcOffset>
    __aicore__ inline static void emit_copy(
        const T& dst, const U& src, const DstOffset& dst_offset, const SrcOffset& src_offset, uint16_t block_count,
        uint32_t block_len, int64_t src_stride, int64_t dst_stride, const gm_to_ub_params& params)
    {
        using src_type = typename U::element_type;
        using dst_type = typename T::element_type;
        adjust_b4_copy_params<src_type, dst_type>(block_len, src_stride, dst_stride);
        auto cache_mode = static_cast<asc_load_l2_cache_mode>(src.engine().get_cache_mode());
        copy_gm_to_ub_instr::data_copy(
            (dst.data() + dst_offset).get(), (src.data() + src_offset).get(), block_count, block_len,
            params.left_padding_count, params.right_padding_count, params.enable_constant_pad, cache_mode, src_stride,
            dst_stride);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H

#if defined( \
    UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_DATA_COPY_H
#endif
