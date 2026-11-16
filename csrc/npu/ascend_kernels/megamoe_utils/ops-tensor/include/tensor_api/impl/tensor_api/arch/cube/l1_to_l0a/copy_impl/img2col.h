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
    "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/img2col.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file img2col.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_IMG2COL_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_IMG2COL_H

#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/instruction.h"

namespace asc {
namespace te {
class load_data_l1_to_l0a_img2col {
public:
    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename PaddingValue>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src,
                                      const img2col_params<PaddingValue>& params)
    {
        // PaddingValue should match the L0A dst element type so the padding register bit pattern is correct;
        // the caller picks it when constructing img2col_params<PaddingValue>.
        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        // Window start offsets come from params (default 0). Non-zero values let an M tile start
        // mid-row (m_start_pos = wout_start) or a K chunk start inside a C0 block (k_start_pos).
        uint16_t k_start_pos = params.k_start_pos;
        uint16_t m_start_pos = params.m_start_pos;

        // src L1 NC1HWC0 (N, C1, H, W, C0): l1_h/l1_w are the feature-map dims, channel_size = C1*C0.
        uint16_t l1_h = get<2>(src_layout.shape());
        uint16_t l1_w = get<3>(src_layout.shape());
        uint16_t channel_size = get<1>(src_layout.shape()) * get<4>(src_layout.shape());

        load_l1_to_l0a_img2col_instr::set_f_matrix(l1_h, l1_w, params.pad_list);
        load_l1_to_l0a_img2col_instr::set_padding(params.pad_value);
        load_l1_to_l0a_img2col_instr::set_repeat(
            static_cast<uint16_t>(Std::ceil_division(params.m_extension, fractal_fixed)));
        load_l1_to_l0a_img2col_instr::load_data(
            dst.data().get(), src.data().get(), params.k_extension, params.m_extension, k_start_pos, m_start_pos,
            params.stride_w, params.stride_h, params.filter_w, params.filter_h, params.dilation_filter_w,
            params.dilation_filter_h, params.enable_filter_w_extend, params.enable_filter_h_extend,
            params.enable_transpose, params.enable_f_matrix_ctrl, channel_size);
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape, typename PaddingValue>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape,
                                      const img2col_params<PaddingValue>& params)
    {
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint16_t l1_h = get<2>(src_shape);
        uint16_t l1_w = get<3>(src_shape);
        uint16_t channel_size = get<1>(src_shape) * get<4>(src_shape);
        load_l1_to_l0a_img2col_instr::set_f_matrix(l1_h, l1_w, params.pad_list);
        load_l1_to_l0a_img2col_instr::set_padding(params.pad_value);
        load_l1_to_l0a_img2col_instr::set_repeat(
            static_cast<uint16_t>(Std::ceil_division(params.m_extension, fractal_fixed)));
        load_l1_to_l0a_img2col_instr::load_data_with_offset(
            dst, src, dst_offset, src_offset, params.k_extension, params.m_extension, 0, 0, params.stride_w,
            params.stride_h, params.filter_w, params.filter_h, params.dilation_filter_w, params.dilation_filter_h,
            params.enable_filter_w_extend, params.enable_filter_h_extend, params.enable_transpose,
            params.enable_f_matrix_ctrl, channel_size);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_IMG2COL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
