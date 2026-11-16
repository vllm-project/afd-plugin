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
    "impl/tensor_api/arch/cube/l1_to_l0scaleb/copy_impl/scaleb.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file scaleb.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0SCALEB_COPY_IMPL_SCALE_B_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0SCALEB_COPY_IMPL_SCALE_B_H

#include "impl/tensor_api/arch/cube/l1_to_l0scaleb/copy_impl/instruction.h"

namespace asc {
namespace te {

class load_data_l1_to_l0b_microscale {
public:
    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor>
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
                          "load_data_l1_to_l0b_microscale only supports the plain fractal layout "
                          "((row0,row1),(col0,col1)) or the batch layout (B,((row0,row1),(col0,col1))).");
        }
    }

    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        check_template<trait, DstTensor, SrcTensor>();
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        auto m_step = get<1, 1>(src_shape);
        auto k_step = get<0, 1>(src_shape);
        auto src_stride = get_element<attr_info::stride, attr_info::column, 1>(src.layout()) >> 5;
        auto dst_stride = get_element<attr_info::stride, attr_info::column, 1>(dst.layout()) >> 5;
        uint64_t mx_dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>((dst.data() + dst_offset).get()));
        load_l1_to_l0b_scale_instr::load_data_with_offset(mx_dst_addr, src, _0{}, src_offset, 0, 0, m_step, k_step,
                                                          src_stride, dst_stride);
    }

private:
    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_data_type::check_l1_to_l0scaleb_data_type<DstTensor, SrcTensor>();
        check_layout_pattern<DstTensor, SrcTensor>();
    }

    template <typename DstTensor, typename SrcTensor, typename DstLayout, typename SrcLayout>
    __aicore__ inline static void load_data_fractal(const DstTensor& dst, const SrcTensor& src,
                                                    const DstLayout& dst_layout, const SrcLayout& src_layout)
    {
        uint16_t m_start_position = 0;
        uint16_t k_start_position = 0;
        auto m_step = get_element<attr_info::shape, attr_info::column, 1>(dst_layout);
        auto k_step = get_element<attr_info::shape, attr_info::row, 1>(dst_layout);
        TENSOR_API_DEBUG_CHECK(debug_check_l0_step, m_step, "m_step", "copy_l1_to_l0scaleb path");
        TENSOR_API_DEBUG_CHECK(debug_check_l0_step, k_step, "k_step", "copy_l1_to_l0scaleb path");
        auto src_stride = get_element<attr_info::stride, attr_info::column, 1>(src_layout) >> 5;
        auto dst_stride = k_step;
        uint64_t mx_dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.data().get()));
        load_l1_to_l0b_scale_instr::load_data(mx_dst_addr, src.data().get(), m_start_position, k_start_position, m_step,
                                              k_step, src_stride, dst_stride);
    }

    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data_impl(const DstTensor& dst, const SrcTensor& src)
    {
        load_data_fractal(dst, src, dst.layout(), src.layout());
    }

    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void batch_load_data_impl(const DstTensor& dst, const SrcTensor& src)
    {
        auto dst_no_batch_layout = remove_batch_dim(dst.layout());
        auto src_no_batch_layout = remove_batch_dim(src.layout());
        auto batch_num = get<0>(dst.layout().shape());
        for (uint32_t i = 0; i < batch_num; ++i) {
            load_data_fractal(dst(make_coord(i, make_coord(make_coord(0, 0), make_coord(0, 0)))),
                              src(make_coord(i, make_coord(make_coord(0, 0), make_coord(0, 0)))), dst_no_batch_layout,
                              src_no_batch_layout);
        }
    }
};
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0SCALEB_COPY_IMPL_SCALE_B_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
