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
    "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/nz2nhwc.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nz2nhwc.h
 * \brief L0C(NZ) -> GM(NHWC) copy via the nz2nd fixpipe path (no quant).
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_NZ2NHWC_H
#define IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_NZ2NHWC_H

#include "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/instruction.h"

namespace asc {
namespace te {

class data_copy_l0c_to_gm_nz2nhwc {
public:
    template <const l0c_to_gm_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const l0c_to_gm_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor>();
        check_data_type::check_l0c_to_gm_data_type<quant_pre, DstTensor, SrcTensor>();
        set_register_impl<DstTensor, SrcTensor>(dst, src);

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        // dst_stride = C = Stride[2] (row-major NHWC stride is (H*W*C, W*C, C, 1)).
        uint32_t n_size = get<3>(dst_layout.shape());
        uint32_t m_size = get<1>(dst_layout.shape()) * get<2>(dst_layout.shape());
        uint32_t src_stride = get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed;
        uint32_t dst_stride = get<2>(dst_layout.stride());

        uint8_t cache_mode = dst.engine().get_cache_mode();
        bool relu_en = trait.enable_relu;
        uint8_t unit_flag = static_cast<uint8_t>(params.unit_flag);
        bool is_channel_split = trait.enable_channel_split;

        copy_l0c_to_gm_instr::data_copy<quant_pre>(dst.data().get(), src.data().get(), n_size, m_size, src_stride,
                                                   dst_stride, cache_mode, relu_en, unit_flag, is_channel_split, true,
                                                   false);
    }

    template <const l0c_to_gm_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape,
                                      const l0c_to_gm_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor>();
        check_data_type::check_l0c_to_gm_data_type<quant_pre, DstTensor, SrcTensor>();
        set_register_impl<DstTensor, SrcTensor>(dst, src);
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint32_t n_size = get_shape_columns(src_shape);
        uint32_t m_size = get_shape_rows(src_shape);
        uint32_t src_stride = get_element<attr_info::stride, attr_info::column, 1>(src.layout()) / fractal_fixed;
        uint32_t dst_stride = get<2>(dst.layout().stride());
        copy_l0c_to_gm_instr::data_copy_with_offset<quant_pre>(
            dst, src, dst_offset, src_offset, n_size, m_size, src_stride, dst_stride, dst.engine().get_cache_mode(),
            trait.enable_relu, static_cast<uint8_t>(params.unit_flag), trait.enable_channel_split, true, false);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_NZ2NHWC_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
