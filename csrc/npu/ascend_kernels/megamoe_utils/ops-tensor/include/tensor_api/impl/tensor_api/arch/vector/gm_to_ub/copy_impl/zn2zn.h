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
    "impl/tensor_api/arch/vector/gm_to_ub/copy_impl/zn2zn.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H
#endif

/*!
 * \file zn2zn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H
#define IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H

#include "impl/tensor_api/arch/vector/gm_to_ub/copy_impl/data_copy.h"

namespace asc {
namespace te {

class copy_gm_to_ub_zn : private copy_gm_to_ub_common {
public:
    template <const gm_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const gm_to_ub_params& params)
    {
        data_copy_impl<trait, DstTensor, SrcTensor>(dst, src, params);
    }

    template <
        const gm_to_ub_trait& trait, typename T, typename U, typename DstCoord, typename SrcCoord, typename ShapeType>
    __aicore__ inline static void run(
        const T& dst, const U& src, const DstCoord& dst_coord, const SrcCoord& src_coord, const ShapeType& copy_shape,
        const gm_to_ub_params& params)
    {
        check_template<trait, T, U>();
        using src_type = typename U::element_type;
        using dst_type = typename T::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto block_count =
            get_shape_batch_size(src_shape) * Std::ceil_division(get_shape_rows(src_shape), c0_element<src_type>);
        auto block_len = get_shape_columns(src_shape) * c0_size<>;
        auto src_stride = get_row_stride(src.layout()) * sizeof(src_type);
        auto dst_stride = get_row_stride(dst.layout()) * sizeof(dst_type);
        emit_copy(
            dst, src, dst.layout()(dst_coord), src.layout()(src_coord), block_count, block_len, src_stride, dst_stride,
            params);
    }

private:
    template <const gm_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<SrcTensor, DstTensor>();
        check_data_type::check_gm_to_ub_data_type<DstTensor, SrcTensor>();
    }

    template <const gm_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(
        const DstTensor& dst, const SrcTensor& src, const gm_to_ub_params& params)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        check_template<trait, DstTensor, SrcTensor>();

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t block_count = get_element<attr_info::shape, attr_info::row, 1>(src_layout);
        TENSOR_API_DEBUG_CHECK(debug_check_block_count, block_count, "src row shape size", "copy_gm_to_ub ZN path");
        uint32_t block_len = get_total_column_shape(src_layout) *
                             get_element<attr_info::shape, attr_info::row, 0>(src_layout) * sizeof(src_type);
        int64_t src_stride = get_element<attr_info::stride, attr_info::row, 1>(src_layout) * sizeof(src_type);
        int64_t dst_stride = get_element<attr_info::stride, attr_info::row, 1>(dst_layout) * sizeof(dst_type);

        emit_copy(dst, src, block_count, block_len, src_stride, dst_stride, params);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H

#if defined( \
    UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_GM_TO_UB_COPY_IMPL_ZN2ZN_H
#endif
