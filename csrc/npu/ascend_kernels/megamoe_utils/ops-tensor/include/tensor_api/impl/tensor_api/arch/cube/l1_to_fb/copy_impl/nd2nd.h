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
    "impl/tensor_api/arch/cube/l1_to_fb/copy_impl/nd2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H

#include "impl/tensor_api/arch/cube/l1_to_fb/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_l1_to_fixbuf_nd {
public:
    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        data_copy_impl<trait, DstTensor, SrcTensor>(dst, src);
    }

    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        check_template<trait, DstTensor, SrcTensor>();
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
        constexpr uint32_t addr_align = 128;
        constexpr uint16_t burst_unit = 64;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint16_t src_col = get_shape_columns(src_shape);
        uint16_t block_count = get_shape_rows(src_shape);
        uint16_t src_row;
        uint16_t dst_row;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_row = get_element<attr_info::stride, attr_info::row>(src.layout());
            dst_row = get_element<attr_info::stride, attr_info::row>(dst.layout());
        } else {
            src_row = get_element<attr_info::stride, attr_info::row, 1>(src.layout());
            dst_row = get_element<attr_info::stride, attr_info::row, 1>(dst.layout());
        }
        uint16_t block_len = Std::ceil_align(src_col * sizeof(src_type), addr_align) / burst_unit;
        uint16_t src_stride = Std::ceil_division(src_row * sizeof(src_type), c0_size<>);
        uint16_t dst_stride = Std::ceil_align(dst_row * sizeof(dst_type), addr_align) / burst_unit;
        copy_l1_to_fixbuf_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len,
                                                       src_stride, dst_stride);
    }

private:
    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_template()
    {
        check_data_type::check_l1_to_fixbuf_data_type<DstTensor, SrcTensor>();
        check_layout_pattern<SrcTensor, DstTensor>();
    }

    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src)
    {
        check_template<trait, DstTensor, SrcTensor>();

        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        constexpr uint32_t deq_tensor_addr_align_value = 128;
        constexpr uint16_t fbuf_burst_len_unit = 64;

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
        TENSOR_API_DEBUG_CHECK(debug_check_block_count, block_count, "src row shape size", "copy_l1_to_fixbuf");
        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            dst_col = get_element<attr_info::shape, attr_info::column>(dst_layout);
            dst_row = get_element<attr_info::stride, attr_info::row>(dst_layout);
        } else {
            dst_col = get_element<attr_info::shape, attr_info::column, 1>(dst_layout);
            dst_row = get_element<attr_info::stride, attr_info::row, 1>(dst_layout);
        }

        uint16_t block_len =
            Std::ceil_align(src_col * sizeof(src_type), deq_tensor_addr_align_value) / fbuf_burst_len_unit;
        uint16_t src_stride = Std::ceil_division(src_row * sizeof(src_type), c0_size<>);
        uint16_t dst_stride =
            Std::ceil_align(dst_row * sizeof(dst_type), deq_tensor_addr_align_value) / fbuf_burst_len_unit;

        copy_l1_to_fixbuf_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, src_stride,
                                           dst_stride);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
