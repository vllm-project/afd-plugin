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
    "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/dn2dn.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H
#endif

/*!
 * \file dn2dn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H
#define IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H

#include "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/data_copy.h"

namespace asc {
namespace te {

class copy_ub_to_ub_dn : private copy_ub_to_ub_common {
public:
    template <const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
    }

    template <
        const ub_to_ub_trait& trait, typename T, typename U, typename DstCoord, typename SrcCoord, typename ShapeType>
    __aicore__ inline static void run(
        const T& dst, const U& src, const DstCoord& dst_coord, const SrcCoord& src_coord, const ShapeType& copy_shape)
    {
        check_template<trait, T, U>();
        using src_type = typename U::element_type;
        using dst_type = typename T::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto block_count = get_shape_columns(src_shape);
        auto block_len = get_shape_rows(src_shape) * sizeof(src_type);
        auto src_gap = get_column_stride(src.layout()) * sizeof(src_type) - block_len;
        auto dst_gap = get_column_stride(dst.layout()) * sizeof(dst_type) - block_len;
        emit_copy(dst, src, dst.layout()(dst_coord), src.layout()(src_coord), block_count, block_len, src_gap, dst_gap);
    }

private:
    template <const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<SrcTensor, DstTensor>();
        check_data_type::check_ub_to_ub_data_type<DstTensor, SrcTensor>();
    }

    template <const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        check_template<trait, DstTensor, SrcTensor>();

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t block_count = get_total_column_shape(src_layout);
        TENSOR_API_DEBUG_CHECK(debug_check_block_count, block_count, "src column shape size", "copy_ub_to_ub DN path");
        uint32_t block_len = get_total_row_shape(src_layout) * sizeof(src_type);
        int64_t src_gap = get_column_stride(src_layout) * sizeof(src_type) - block_len;
        int64_t dst_gap = get_column_stride(dst_layout) * sizeof(dst_type) - block_len;

        emit_copy(dst, src, block_count, block_len, src_gap, dst_gap);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H

#if defined( \
    UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_IMPL_DN2DN_H
#endif
