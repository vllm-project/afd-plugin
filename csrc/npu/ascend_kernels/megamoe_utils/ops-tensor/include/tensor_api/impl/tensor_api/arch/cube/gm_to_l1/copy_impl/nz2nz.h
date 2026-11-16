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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nz2nz.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nz2nz.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NZ2NZ_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NZ2NZ_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/copy_common.h"

namespace asc {
namespace te {

class copy_gm_to_l1_nz2nz {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        run_gm_to_l1_batched<trait, copy_gm_to_l1_nz2nz, DstTensor, SrcTensor>(dst, src);
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
        auto block_count =
            get_shape_batch_size(src_shape) * Std::ceil_division(get_shape_columns(src_shape), c0_element<type>);
        auto block_len = get_shape_rows(src_shape) * c0_size<>;
        auto src_stride = get_matrix_element<attr_info::stride, attr_info::column, 1>(src.layout()) * sizeof(type);
        auto dst_stride = get_matrix_element<attr_info::stride, attr_info::column, 1>(dst.layout()) * sizeof(type);
        if constexpr (is_b4_type<type>) {
            src_stride >>= 1;
            dst_stride >>= 1;
        }
        copy_gm_to_l1_align_v2_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len, 0,
                                                            0, src.engine().get_cache_mode(), src_stride, dst_stride);
    }

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<SrcTensor, DstTensor>();
        check_data_type::check_gm_to_l1_align_v2_nd_data_type<DstTensor, SrcTensor>();
    }

    // Extracts single-matrix parameters from the (batch-stripped) src/dst layouts and emits the
    // copy. matrix_num carries the batch dimension (1 when there is no batch). The align_v2
    // instruction has no hardware batch loop, so the batch axis is folded into block_count: NZ
    // matrices are laid out along the column (big-fractal) direction, so matrix_num matrices of
    // big_fractal_size big-fractals each become matrix_num * big_fractal_size contiguous blocks that
    // share the same per-big-fractal stride. src_nd_matrix_stride/dst_nz_matrix_stride are unused here
    // because that contiguity makes the single src_stride/dst_stride describe the whole sequence.
    template <typename DstTensor, typename SrcTensor, typename SrcLayout, typename DstLayout>
    __aicore__ inline static void emit_copy(const DstTensor& dst, const SrcTensor& src, const SrcLayout& src_layout,
                                            const DstLayout& dst_layout, uint16_t matrix_num,
                                            uint64_t src_nd_matrix_stride, uint32_t dst_nz_matrix_stride)
    {
        using type = typename SrcTensor::element_type;

        auto small_fractal_size = get_element<attr_info::shape, attr_info::row, 0>(src_layout)
                                  * get_element<attr_info::shape, attr_info::row, 1>(src_layout);
        auto big_fractal_size = get_element<attr_info::shape, attr_info::column, 1>(src_layout);
        auto src_stride_size = get_element<attr_info::stride, attr_info::column, 1>(src_layout);
        auto dst_stride_size = get_element<attr_info::stride, attr_info::column, 1>(dst_layout);

        uint8_t left_padding_cnt = 0;
        uint8_t right_padding_cnt = 0;
        uint8_t cache_mode = src.engine().get_cache_mode();

        auto block_count = big_fractal_size * matrix_num;
        auto block_len = small_fractal_size * c0_size<>;
        auto src_stride = src_stride_size * sizeof(type);
        auto dst_stride = dst_stride_size * sizeof(type);
        if constexpr (is_b4_type<type>) {
            // move fp4 as b8, need to be divided by 2
            src_stride = src_stride >> 1;
            dst_stride = dst_stride >> 1;
        }
        copy_gm_to_l1_align_v2_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len,
                                                left_padding_cnt, right_padding_cnt, cache_mode, src_stride,
                                                dst_stride);
    }
};
} // namespace te
} // namespace asc

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
