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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nhwc2nc1hwc0.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nhwc2nc1hwc0.h
 * \brief GM(NHWC) -> L1(NC1HWC0) copy via the nd2nz DMA path.
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NHWC2NC1HWC0_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NHWC2NC1HWC0_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_gm_to_l1_nhwc2nc1hwc0 {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        using type = typename SrcTensor::element_type;
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        // src NHWC (N, H, W, C): take the H/W/C axes. N is the leading batch axis (kept as N=1 here).
        uint16_t nd_num = get<1>(src_layout.shape());  // src_h
        uint16_t n_value = get<2>(src_layout.shape()); // src_w
        uint32_t d_value = get<3>(src_layout.shape()); // src_c
        // NHWC row-major stride: (H*W*C, W*C, C, 1) -> stride[1]=W*C, stride[2]=C.
        uint64_t loop4_src_stride = static_cast<uint64_t>(get<1>(src_layout.stride())) * sizeof(type); // src_w*src_c
        uint64_t loop1_src_stride = static_cast<uint64_t>(get<2>(src_layout.stride())) * sizeof(type); // src_c

        // dst NC1HWC0 (N, C1, H, W, C0): loop4_dst_stride = dst_w = Shape[3]; loop3_dst_stride = dst_h*dst_w
        // = Shape[2]*Shape[3] (no stride carries the bare H*W product).
        uint16_t loop2_dst_stride = 1;
        uint16_t loop4_dst_stride = get<3>(dst_layout.shape());                              // dst_w
        uint16_t loop3_dst_stride = get<2>(dst_layout.shape()) * get<3>(dst_layout.shape()); // dst_h*dst_w

        if constexpr (is_b4_type<type>) {
            d_value = d_value >> 1;
            loop4_src_stride = loop4_src_stride >> 1;
            loop1_src_stride = loop1_src_stride >> 1;
        }
        uint8_t cache_mode = src.engine().get_cache_mode();

        copy_gm_to_l1_multi_nd2nz_instr::data_copy(dst.data().get(), src.data().get(), nd_num, loop2_dst_stride,
                                                   loop3_dst_stride, loop4_dst_stride, loop1_src_stride, cache_mode,
                                                   n_value, d_value, loop4_src_stride, false);
    }

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        using type = typename SrcTensor::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint16_t nd_num = get<1>(src_shape);
        uint16_t n_value = get<2>(src_shape);
        uint32_t d_value = get<3>(src_shape);
        uint64_t loop4_src_stride = static_cast<uint64_t>(get<1>(src.layout().stride())) * sizeof(type);
        uint64_t loop1_src_stride = static_cast<uint64_t>(get<2>(src.layout().stride())) * sizeof(type);
        uint16_t loop4_dst_stride = get<3>(dst.layout().shape());
        uint16_t loop3_dst_stride = get<2>(dst.layout().shape()) * get<3>(dst.layout().shape());
        if constexpr (is_b4_type<type>) {
            d_value >>= 1;
            loop4_src_stride >>= 1;
            loop1_src_stride >>= 1;
        }
        copy_gm_to_l1_multi_nd2nz_instr::data_copy_with_offset(
            dst, src, dst_offset, src_offset, nd_num, 1, loop3_dst_stride, loop4_dst_stride, loop1_src_stride,
            src.engine().get_cache_mode(), n_value, d_value, loop4_src_stride, false);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NHWC2NC1HWC0_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
