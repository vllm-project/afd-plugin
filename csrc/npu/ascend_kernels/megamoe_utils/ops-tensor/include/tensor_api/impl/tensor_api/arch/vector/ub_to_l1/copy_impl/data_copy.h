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
    "impl/tensor_api/arch/vector/ub_to_l1/copy_impl/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/vector/ub_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_ub_to_l1_common {
protected:
    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static void emit_copy(
        const DstTensor& dst, const SrcTensor& src, uint16_t block_count, uint32_t block_len, int64_t src_gap,
        int64_t dst_gap)
    {
        copy_ub_to_l1_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, src_gap, dst_gap);
    }

    template <typename T, typename U, typename DstOffset, typename SrcOffset>
    __aicore__ inline static void emit_copy(
        const T& dst, const U& src, const DstOffset& dst_offset, const SrcOffset& src_offset, uint16_t block_count,
        uint32_t block_len, int64_t src_gap, int64_t dst_gap)
    {
        copy_ub_to_l1_instr::data_copy(
            (dst.data() + dst_offset).get(), (src.data() + src_offset).get(), block_count, block_len, src_gap, dst_gap);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H

#if defined( \
    UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_L1_COPY_IMPL_DATA_COPY_H
#endif
