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
    "impl/tensor_api/arch/vector/utils/copy_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H
#endif

/*!
 * \file copy_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H
#define IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename SrcType, typename DstType>
__aicore__ inline void adjust_b4_copy_params(uint32_t& block_len, int64_t& src_stride, int64_t& dst_stride)
{
    if constexpr (is_b4_type<SrcType>) {
        block_len >>= 1;
        src_stride >>= 1;
    }

    if constexpr (is_b4_type<DstType>) {
        dst_stride >>= 1;
    }
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UTILS_COPY_UTILS_H
#endif
