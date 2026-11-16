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
    "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/zn2nzb8b4.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file zn2nzb8b4.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_ZN2NZB8B4_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_ZN2NZB8B4_H

#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/instruction.h"

namespace asc {
namespace te {

class load_data_l1_to_l0a_zn2nz_b8b4 {
public:
    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();
        if constexpr (DstTensor::layout_type::depth == five_dim_data) {
            batch_load_data_impl<trait, DstTensor, SrcTensor>(dst, src);
        } else if constexpr (DstTensor::layout_type::depth == four_dim_data) {
            load_data_impl<trait, DstTensor, SrcTensor>(dst, src);
        } else {
            static_assert(DstTensor::layout_type::depth == four_dim_data
                              || DstTensor::layout_type::depth == five_dim_data,
                          "load_data_l1_to_l0a_zn2nz_b8b4 only supports the plain fractal layout "
                          "((row0,row1),(col0,col1)) or the batch layout (B,((row0,row1),(col0,col1))).");
        }
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        check_template<trait, DstTensor, SrcTensor>();
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        using dst_type = typename DstTensor::element_type;
        constexpr uint32_t stride_unit = c0_element<dst_type> * fractal_fixed;
        auto src_row_size = get_shape_rows(src_shape);
        auto dst_row_size = get_shape_rows(dst.layout().shape());
        auto m1 = src_row_size - dst_row_size;
        auto m_step = get_shape_columns(dst.layout().shape()) / fractal_fixed;
        auto k_step = Std::ceil_division(src_row_size, c0_element<dst_type>);
        auto src_stride = get_element<attr_info::stride, attr_info::row, 1>(src.layout()) / stride_unit;
        auto dst_stride = get_element<attr_info::stride, attr_info::column, 1>(dst.layout()) / stride_unit;
        if (m1 < fractal_fixed) {
            load_l1_to_l0a_instr::load_data_with_offset<true>(dst, src, dst_offset, src_offset, 0, 0, m_step, k_step,
                                                              src_stride, dst_stride);
        } else if constexpr (is_b4_type<dst_type>) {
            load_split_with_offset<m_step_min_val_b4>(dst, src, dst_offset, src_offset, m_step, k_step, src_stride,
                                                      dst_stride);
        } else {
            load_split_with_offset<m_step_min_val_b8>(dst, src, dst_offset, src_offset, m_step, k_step, src_stride,
                                                      dst_stride);
        }
    }

private:
    static constexpr uint8_t m_step_min_val_b4 = 4;
    static constexpr uint8_t m_step_min_val_b8 = 2;

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_layout_pattern<DstTensor, SrcTensor>();
        check_data_type::check_l1_to_l0a_data_type<DstTensor, SrcTensor>();
    }

    template <uint8_t m_step_min_val, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data_impl_split(const DstTensor& dst, const SrcTensor& src,
                                                       uint16_t m_start_position, uint16_t k_start_position,
                                                       uint8_t m_step, uint8_t k_step, int16_t src_stride,
                                                       uint16_t dst_stride)
    {
        uint16_t m_loop = m_step / m_step_min_val;
        m_step = m_step_min_val;
        for (uint16_t idx = 0; idx < m_loop; ++idx) {
            auto slice_dst = dst(make_coord(make_coord(0, 0), make_coord(0, idx)));
            load_l1_to_l0a_instr::load_data<true>(slice_dst.data().get(), src.data().get(), m_start_position,
                                                  k_start_position, m_step, k_step, src_stride, dst_stride);
            m_start_position += m_step_min_val;
        }
    }

    template <uint8_t m_step_min_val, typename DstTensor, typename SrcTensor, typename DstOffset, typename SrcOffset>
    __aicore__ inline static void load_split_with_offset(const DstTensor& dst, const SrcTensor& src,
                                                         const DstOffset& dst_offset, const SrcOffset& src_offset,
                                                         uint8_t m_step, uint8_t k_step, int16_t src_stride,
                                                         uint16_t dst_stride)
    {
        uint16_t m_loop = m_step / m_step_min_val;
        for (uint16_t idx = 0; idx < m_loop; ++idx) {
            auto sub_dst = dst(make_coord(make_coord(0, 0), make_coord(0, idx)));
            load_l1_to_l0a_instr::load_data_with_offset<true>(sub_dst, src, dst_offset, src_offset,
                                                              idx * m_step_min_val, 0, m_step_min_val, k_step,
                                                              src_stride, dst_stride);
        }
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename DstLayout,
              typename SrcLayout>
    __aicore__ inline static void load_data_fractal(const DstTensor& dst, const SrcTensor& src,
                                                    const DstLayout& dst_layout, const SrcLayout& src_layout)
    {
        using dst_type = typename DstTensor::element_type;
        uint16_t m_start_position = 0;
        uint16_t k_start_position = 0;
        auto m1 = get_element<attr_info::shape, attr_info::row, 1>(src_layout)
                      * get_element<attr_info::shape, attr_info::row, 0>(src_layout)
                  - get_element<attr_info::shape, attr_info::row, 1>(dst_layout)
                        * get_element<attr_info::shape, attr_info::row, 0>(dst_layout);
        auto m_step = get_element<attr_info::shape, attr_info::column, 1>(dst_layout)
                      * get_element<attr_info::shape, attr_info::column, 0>(dst_layout) / fractal_fixed;
        auto k_step = get_element<attr_info::shape, attr_info::row, 1>(src_layout);
        TENSOR_API_DEBUG_CHECK(debug_check_l0_step, k_step, "k_step", "copy_l1_to_l0a zn2nz b8/b4 path");
        TENSOR_API_DEBUG_CHECK(debug_check_l0_m_step, m_step, is_b4_type<dst_type>, "copy_l1_to_l0a zn2nz b8/b4 path");
        // Zn -> Nz
        constexpr uint32_t stride_unit = c0_element<dst_type> * fractal_fixed;
        auto src_stride = get_element<attr_info::stride, attr_info::row, 1>(src_layout) / stride_unit;
        auto dst_stride = get_element<attr_info::stride, attr_info::column, 1>(dst_layout) / stride_unit;
        if (m1 < fractal_fixed) {
            TENSOR_API_DEBUG_CHECK(debug_check_l0_step, m_step, "m_step", "copy_l1_to_l0a zn2nz b8/b4 path");
            load_l1_to_l0a_instr::load_data<true>(dst.data().get(), src.data().get(), m_start_position,
                                                  k_start_position, m_step, k_step, src_stride, dst_stride);
        } else if constexpr (is_b4_type<dst_type>) {
            load_data_impl_split<m_step_min_val_b4>(dst, src, m_start_position, k_start_position, m_step, k_step,
                                                    src_stride, dst_stride);
        } else {
            load_data_impl_split<m_step_min_val_b8>(dst, src, m_start_position, k_start_position, m_step, k_step,
                                                    src_stride, dst_stride);
        }
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data_impl(const DstTensor& dst, const SrcTensor& src)
    {
        load_data_fractal<trait>(dst, src, dst.layout(), src.layout());
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void batch_load_data_impl(const DstTensor& dst, const SrcTensor& src)
    {
        auto dst_layout = dst.layout();
        auto src_layout = src.layout();
        auto dst_no_batch_layout = remove_batch_dim(dst_layout);
        auto src_no_batch_layout = remove_batch_dim(src_layout);
        auto batch_num = get<0>(dst_layout.shape());
        for (uint32_t i = 0; i < batch_num; ++i) {
            load_data_fractal<trait>(make_single_batch_sub_tensor(dst, i), make_single_batch_sub_tensor(src, i),
                                     dst_no_batch_layout, src_no_batch_layout);
        }
    }
};
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_ZN2NZB8B4_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
