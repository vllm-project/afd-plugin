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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scalea_zz2zz.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file scalea_zz2zz.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_SCALEA_ZZ2ZZ_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_SCALEA_ZZ2ZZ_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_gm_to_l1_scalea_zz2zz {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        if constexpr (SrcTensor::layout_type::depth == five_dim_data) {
            TENSOR_API_DEBUG_CHECK(debug_check_batch_count, get<0>(src.layout().shape()),
                                   "copy_gm_to_l1 scalea_zz2zz path");
        }
        data_copy_impl<trait>(dst, src);
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
        constexpr auto depth = SrcTensor::layout_type::depth;
        uint32_t src_shape_rows_b;
        uint32_t src_shape_rows_s;
        uint32_t src_shape_columns;
        uint32_t src_stride_rows_b;
        uint32_t src_stride_rows_s;
        uint32_t dst_stride_rows;
        if constexpr (SrcTensor::layout_type::depth == five_dim_data) {
            src_shape_rows_b = get<1, 0, 1>(src_shape);
            src_shape_rows_s = get<1, 0, 0>(src_shape);
            src_shape_columns = get<1, 1, 1>(src_shape);
            src_stride_rows_b = get<1, 0, 1>(src.layout().stride());
            src_stride_rows_s = get<1, 0, 0>(src.layout().stride());
            dst_stride_rows = get<1, 0, 1>(dst.layout().stride());
        } else {
            src_shape_rows_b = get<0, 1>(src_shape);
            src_shape_rows_s = get<0, 0>(src_shape);
            src_shape_columns = get<1, 1>(src_shape);
            src_stride_rows_b = get<0, 1>(src.layout().stride());
            src_stride_rows_s = get<0, 0>(src.layout().stride());
            dst_stride_rows = get<0, 1>(dst.layout().stride());
        }
        uint32_t block_len = src_shape_columns * sizeof(type) * src_shape_rows_s * src_stride_rows_s;
        uint64_t src_stride = src_stride_rows_b * sizeof(type);
        uint32_t dst_stride = dst_stride_rows * sizeof(type);
        uint32_t batch_num = get_shape_batch_size(src_shape);
        uint64_t src_batch_stride = 0;
        if constexpr (depth == five_dim_data) {
            src_batch_stride = get<0>(src.layout().stride());
        }
        for (uint32_t i = 0; i < batch_num; ++i) {
            if constexpr (SrcTensor::layout_type::depth == five_dim_data) {
                auto dst_batch = make_single_batch_sub_tensor(dst, i);
                copy_gm_to_l1_align_v2_instr::data_copy_with_offset(
                    dst_batch, src, dst_offset, src_offset + i * src_batch_stride, src_shape_rows_b, block_len, 0, 0,
                    src.engine().get_cache_mode(), src_stride, dst_stride);
            } else {
                copy_gm_to_l1_align_v2_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, src_shape_rows_b,
                                                                    block_len, 0, 0, src.engine().get_cache_mode(),
                                                                    src_stride, dst_stride);
            }
        }
    }

private:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<SrcTensor, DstTensor>();
        check_data_type::check_gm_to_l1_scale_data_type<DstTensor, SrcTensor>();
    }

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();
        if constexpr (SrcTensor::layout_type::depth == five_dim_data) {
            auto src_layout = src.layout();
            auto dst_layout = dst.layout();
            emit_copy(dst, src, remove_batch_dim(src_layout), remove_batch_dim(dst_layout), get<0>(src_layout.shape()),
                      get<0>(src_layout.stride()), get<0>(dst_layout.stride()));
        } else {
            emit_copy(dst, src, src.layout(), dst.layout(), 1, 0, 0);
        }
    }

    template <typename DstTensor, typename SrcTensor, typename SrcLayout, typename DstLayout>
    __aicore__ inline static void emit_copy(const DstTensor& dst, const SrcTensor& src, const SrcLayout& src_layout,
                                            const DstLayout& dst_layout, uint16_t batch_num, uint64_t src_batch_stride,
                                            uint64_t dst_batch_stride)
    {
        using type = typename SrcTensor::element_type;

        auto src_shape_rows_b = get_element<attr_info::shape, attr_info::row, 1>(src_layout);
        auto src_shape_rows_s = get_element<attr_info::shape, attr_info::row, 0>(src_layout);
        auto src_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(src_layout);
        auto src_stride_rows_b = get_element<attr_info::stride, attr_info::row, 1>(src_layout);
        auto src_stride_rows_s = get_element<attr_info::stride, attr_info::row, 0>(src_layout);

        auto dst_shape_columns = get_element<attr_info::shape, attr_info::column, 1>(dst_layout);
        auto dst_stride_rows = get_element<attr_info::stride, attr_info::row, 1>(dst_layout);

        uint8_t cache_mode = src.engine().get_cache_mode();

        // lprp mode, dst_stride % c0_size should be 0
        // multi rows copy, dst non-contiguous case

        uint32_t block_count = src_shape_rows_b;
        uint32_t block_len = src_shape_columns * sizeof(type) * src_shape_rows_s * src_stride_rows_s;
        uint64_t src_stride = src_stride_rows_b * sizeof(type);
        uint32_t dst_stride = dst_stride_rows * sizeof(type);

        uint8_t left_padding_cnt = 0;
        uint8_t right_padding_cnt = 0;
        for (uint16_t batch_index = 0; batch_index < batch_num; ++batch_index) {
            copy_gm_to_l1_align_v2_instr::data_copy(
                reinterpret_cast<__cbuf__ half*>((dst.data() + batch_index * dst_batch_stride).get()),
                reinterpret_cast<__gm__ half*>((src.data() + batch_index * src_batch_stride).get()), block_count,
                block_len, left_padding_cnt, right_padding_cnt, cache_mode, src_stride, dst_stride);
        }
    }
};

} // namespace te
} // namespace asc

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
