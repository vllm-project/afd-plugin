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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nd2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_ND2ND_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_gm_to_l1_nd2nd {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        // Batch layouts carry a leading B axis: (B, (row, col)) -> depth 3,
        // (B, ((1, row), (1, col))) -> depth 5. Non-batch layouts are depth 2/4.
        constexpr auto gm_depth = nesting_depth_v<decltype(src.layout().shape())>;
        if constexpr (gm_depth == three_dim_data || gm_depth == five_dim_data) {
            TENSOR_API_DEBUG_CHECK(debug_check_batch_count, get<0>(src.layout().shape()), "copy_gm_to_l1 nd2nd path");
            batch_data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
        } else {
            data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
        }
    }

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        check_template<trait, DstTensor, SrcTensor>();

        using type = typename SrcTensor::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint32_t src_shape_rows = get_shape_rows(src_shape);
        uint32_t src_shape_columns = get_shape_columns(src_shape);
        uint32_t src_stride_rows = get_matrix_element<attr_info::stride, attr_info::row, 1>(src.layout());
        uint32_t dst_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(dst.layout());
        uint32_t dst_stride_rows = get_matrix_element<attr_info::stride, attr_info::row, 1>(dst.layout());

        uint32_t block_count = src_shape_rows;
        uint32_t block_len = src_shape_columns * sizeof(type);
        uint64_t src_stride = src_stride_rows * sizeof(type);
        uint32_t dst_stride = dst_stride_rows * sizeof(type);
        if ((src_shape_rows == 1) || (src_shape_columns == 1)
            || (src_stride_rows == src_shape_columns && dst_stride_rows == dst_shape_columns
                && src_stride_rows == dst_stride_rows)) {
            block_count = 1;
            block_len = src_shape_rows * src_shape_columns * sizeof(type);
            src_stride = 0;
            dst_stride = block_len;
        }
        if constexpr (is_b4_type<type>) {
            block_len >>= 1;
            src_stride >>= 1;
            dst_stride >>= 1;
        }
        uint8_t cache_mode = src.engine().get_cache_mode();
        copy_gm_to_l1_align_v2_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len, 0,
                                                            0, cache_mode, src_stride, dst_stride);
    }

private:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<SrcTensor, DstTensor>();
        check_data_type::check_gm_to_l1_align_v2_nd_data_type<DstTensor, SrcTensor>();
    }

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();

        using type = typename SrcTensor::element_type;
        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint32_t src_shape_rows;
        uint32_t src_shape_columns;
        uint32_t src_stride_rows;
        uint32_t dst_shape_columns;
        uint32_t dst_stride_rows;

        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            src_shape_rows = get_element<attr_info::shape, attr_info::row>(src_layout);
            src_shape_columns = get_element<attr_info::shape, attr_info::column>(src_layout);
            src_stride_rows = get_element<attr_info::stride, attr_info::row>(src_layout);
        } else {
            src_shape_rows = get_element<attr_info::shape, attr_info::row, 1>(src_layout);
            src_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(src_layout);
            src_stride_rows = get_element<attr_info::stride, attr_info::row, 1>(src_layout);
        }

        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            dst_shape_columns = get_element<attr_info::shape, attr_info::column>(dst_layout);
            dst_stride_rows = get_element<attr_info::stride, attr_info::row>(dst_layout);
        } else {
            dst_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(dst_layout);
            dst_stride_rows = get_element<attr_info::stride, attr_info::row, 1>(dst_layout);
        }

        uint8_t cache_mode = src.engine().get_cache_mode();

        // normal mode, dst_stride % c0_size should be 0
        // compact mode, block_len equals dst_stride
        // multi rows copy, dst non-contiguous case
        uint32_t block_count = src_shape_rows;
        uint32_t block_len = src_shape_columns * sizeof(type);
        uint64_t src_stride = src_stride_rows * sizeof(type);
        uint32_t dst_stride = dst_stride_rows * sizeof(type);

        if ((src_shape_rows == 1) || (src_shape_columns == 1)
            || (src_stride_rows == src_shape_columns && dst_stride_rows == dst_shape_columns
                && src_stride_rows == dst_stride_rows)) {
            // compact mode, one line
            block_count = 1;
            // must use srcShape, there is a scenario of small to large, using dstShape will cause src out of bound
            block_len = src_shape_rows * src_shape_columns * sizeof(type);
            src_stride = 0;
            dst_stride = block_len;
        }
        if constexpr (is_b4_type<type>) {
            // move fp4 as b8, need to be divided by 2
            block_len = block_len >> 1;
            src_stride = src_stride >> 1;
            dst_stride = dst_stride >> 1;
        }
        copy_gm_to_l1_align_v2_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, 0, 0,
                                                cache_mode, src_stride, dst_stride);
    }

    // Batch case: layout is (B, (M, N)) with strides (sB, (sM, sN)). Per-matrix internal
    // compactness is assumed (sM == N), so a single matrix can be moved as one block_len-sized block.
    // Four DataCopy params are derived directly from the batched layout:
    //   - get_element<Shape, Row/Column> on the batched layout returns sub-matrix M/N (the
    //     select_row_col_tuples helper in is_format.h handles batch-axis stripping).
    //   - get<0>(Shape/Stride) returns the batch size and per-matrix start-to-start stride.
    //   - When batches are also contiguous (src_batch_stride == M*N && dst_batch_stride == M*N),
    //     fold the B blocks into a single B*M*N block; otherwise emit B blocks, one per matrix.
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void batch_data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();

        using type = typename SrcTensor::element_type;
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        uint32_t batch_size = get<0>(src_layout.shape());
        uint64_t src_batch_stride = get<0>(src_layout.stride());
        uint32_t dst_batch_stride = get<0>(dst_layout.stride());

        // Strip the leading B axis before calling GetElement, otherwise the dim=0/1 split below
        // would read the batch axis as row/column.
        auto src_inner = get<1>(src_layout);

        uint32_t src_shape_rows;
        uint32_t src_shape_columns;
        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            src_shape_rows = get_element<attr_info::shape, attr_info::row>(src_inner);
            src_shape_columns = get_element<attr_info::shape, attr_info::column>(src_inner);
        } else {
            src_shape_rows = get_element<attr_info::shape, attr_info::row, 1>(src_inner);
            src_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(src_inner);
        }

        uint32_t matrix_elems = src_shape_rows * src_shape_columns;
        uint8_t cache_mode = src.engine().get_cache_mode();

        // batch-strided: B blocks, stride = per-matrix start-to-start in bytes.
        uint32_t block_count = batch_size;
        uint32_t block_len = matrix_elems * sizeof(type);
        uint64_t src_stride = src_batch_stride * sizeof(type);
        uint32_t dst_stride = dst_batch_stride * sizeof(type);

        if constexpr (is_b4_type<type>) {
            // move fp4 as b8, need to be divided by 2
            block_len = block_len >> 1;
            src_stride = src_stride >> 1;
            dst_stride = dst_stride >> 1;
        }
        copy_gm_to_l1_align_v2_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, 0, 0,
                                                cache_mode, src_stride, dst_stride);
    }
};
} // namespace te
} // namespace asc

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
