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
#warning                                                                                                               \
    "impl/tensor_api/arch/cube/l1_to_bt/copy_impl/nd2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_BT_COPY_IMPL_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_BT_COPY_IMPL_ND2ND_H

#include "impl/tensor_api/arch/cube/l1_to_bt/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_l1_to_biastable_nd {
public:
    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        // Batch layouts carry a leading B axis: (B, (row, col)) -> depth 3,
        // (B, ((1, row), (1, col))) -> depth 5. Non-batch layouts are depth 2/4.
        constexpr auto l1_depth = nesting_depth_v<decltype(src.layout().shape())>;
        if constexpr (l1_depth == three_dim_data || l1_depth == five_dim_data) {
            batch_data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
        } else {
            data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
        }
    }

    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        check_template<trait, DstTensor, SrcTensor>();
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint32_t src_col;
        uint16_t block_count;
        if constexpr (SrcTensor::layout_type::depth == two_dim_data) {
            block_count = get<0>(src_shape);
            src_col = get<1>(src_shape);
        } else {
            block_count = get<0, 1>(src_shape);
            src_col = get<1, 1>(src_shape);
        }
        uint32_t src_row;
        uint32_t dst_row;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_row = get_element<attr_info::stride, attr_info::row>(src.layout());
            dst_row = get_element<attr_info::stride, attr_info::row>(dst.layout());
        } else {
            src_row = get_element<attr_info::stride, attr_info::row, 1>(src.layout());
            dst_row = get_element<attr_info::stride, attr_info::row, 1>(dst.layout());
        }
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
        bool conv_control = is_one_of_attr_v<src_type, half> && is_one_of_attr_v<dst_type, float>;
        uint16_t block_len = Std::ceil_division(src_col, c0_element<src_type>);
        if constexpr (is_one_of_attr_v<src_type, float, int32_t>) {
            block_len = Std::ceil_align(block_len, 2);
        }
        uint16_t src_stride = (src_row - src_col) / c0_element<src_type>;
        uint16_t dst_stride = Std::ceil_align((dst_row - src_col) / c0_element<dst_type>, 2);
        copy_l1_to_biastable_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, conv_control, block_count,
                                                          block_len, src_stride, dst_stride);
    }

private:
    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<DstTensor, SrcTensor>();
        check_data_type::check_l1_to_biastable_data_type<DstTensor, SrcTensor>();
    }

    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t src_col;
        uint16_t src_row;
        uint16_t dst_col;
        uint16_t dst_row;
        uint16_t block_count;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_col = get_element<attr_info::shape, attr_info::column>(src_layout);
            src_row = get_element<attr_info::stride, attr_info::row>(src_layout);
            block_count = get_element<attr_info::shape, attr_info::row>(src_layout);
        } else {
            src_col = get_element<attr_info::shape, attr_info::column, 1>(src_layout);
            src_row = get_element<attr_info::stride, attr_info::row, 1>(src_layout);
            block_count = get_element<attr_info::shape, attr_info::row, 1>(src_layout);
        }
        TENSOR_API_DEBUG_CHECK(debug_check_block_count, block_count, "src row shape size", "copy_l1_to_biastable");
        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            dst_col = get_element<attr_info::shape, attr_info::column>(dst_layout);
            dst_row = get_element<attr_info::stride, attr_info::row>(dst_layout);
        } else {
            dst_col = get_element<attr_info::shape, attr_info::column, 1>(dst_layout);
            dst_row = get_element<attr_info::stride, attr_info::row, 1>(dst_layout);
        }

        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        bool conv_control = false;
        if (is_one_of_attr_v<src_type, half> && is_one_of_attr_v<dst_type, float>) {
            conv_control = true;
        }

        uint16_t block_len = Std::ceil_division(src_col, c0_element<src_type>);
        if constexpr (is_one_of_attr_v<src_type, float, int32_t>) {
            block_len = Std::ceil_align(block_len, 2);
        }

        uint16_t src_stride = (src_row - src_col) / c0_element<src_type>;
        uint16_t dst_stride = Std::ceil_align((dst_row - src_col) / c0_element<dst_type>, 2);
        copy_l1_to_biastable_instr::data_copy(reinterpret_cast<uint64_t>(dst.data().get()), src.data().get(),
                                              conv_control, block_count, block_len, src_stride, dst_stride);
    }

    // Batch case: layout is (B, (M, N)) with strides (sB, (sM, sN)). Per-matrix internal
    // compactness is assumed (sM == N), so a single matrix can be moved as one block_len-sized block.
    // Four DataCopy params are derived directly from the batched layout:
    //   - get_element<Shape, Row/Column> on the batched layout returns sub-matrix M/N (the
    //     select_row_col_tuples helper in is_format.h handles batch-axis stripping).
    //   - get<0>(Shape/Stride) returns the batch size and per-matrix start-to-start stride.
    //   - When batches are also contiguous (src_batch_stride == M*N && dst_batch_stride == M*N),
    //     fold the B blocks into a single B*M*N block; otherwise emit B blocks, one per matrix.
    //   - Both src_stride and dst_stride encode the end-to-next-start gap in c0_element units, in
    //     keeping with the L1->BT instruction's stride convention.
    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void batch_data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();

        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        uint16_t batch_size = get<0>(src_layout.shape());
        TENSOR_API_DEBUG_CHECK(debug_check_batch_match, batch_size, get<0>(dst_layout.shape()),
                               "copy_l1_to_biastable batch path");
        TENSOR_API_DEBUG_CHECK(debug_check_batch_count, batch_size, "copy_l1_to_biastable batch path");
        uint32_t src_batch_stride = get<0>(src_layout.stride());
        uint32_t dst_batch_stride = get<0>(dst_layout.stride());

        // Strip the leading B axis before calling GetElement, otherwise the dim=0/1 split below
        // would read the batch axis as row/column.
        auto src_inner = get<1>(src_layout);
        TENSOR_API_DEBUG_CHECK(debug_check_l12bt_column_bytes,
                               static_cast<uint64_t>(get_total_column_shape(src_inner)) * sizeof(src_type),
                               static_cast<uint64_t>(get_total_column_shape(get<1>(dst_layout))) * sizeof(dst_type),
                               "copy_l1_to_biastable batch path");

        uint32_t src_shape_rows;
        uint32_t src_shape_columns;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_shape_rows = get_element<attr_info::shape, attr_info::row>(src_inner);
            src_shape_columns = get_element<attr_info::shape, attr_info::column>(src_inner);
        } else {
            src_shape_rows = get_element<attr_info::shape, attr_info::row, 1>(src_inner);
            src_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(src_inner);
        }

        bool conv_control = false;
        if (is_one_of_attr_v<src_type, half> && is_one_of_attr_v<dst_type, float>) {
            conv_control = true;
        }

        uint32_t matrix_elems = src_shape_rows * src_shape_columns;

        // batch-strided: B blocks, stride encodes per-matrix end-to-next-start gap.
        uint16_t block_count = batch_size;
        uint16_t block_len = Std::ceil_division(matrix_elems, c0_element<src_type>);
        if constexpr (is_one_of_attr_v<src_type, float, int32_t>) {
            block_len = Std::ceil_align(block_len, 2);
        }
        uint16_t src_stride = (src_batch_stride - matrix_elems) / c0_element<src_type>;
        uint16_t dst_stride = Std::ceil_align((dst_batch_stride - matrix_elems) / c0_element<dst_type>, 2);

        copy_l1_to_biastable_instr::data_copy(reinterpret_cast<uint64_t>(dst.data().get()), src.data().get(),
                                              conv_control, block_count, block_len, src_stride, dst_stride);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_BT_COPY_IMPL_ND2ND_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
