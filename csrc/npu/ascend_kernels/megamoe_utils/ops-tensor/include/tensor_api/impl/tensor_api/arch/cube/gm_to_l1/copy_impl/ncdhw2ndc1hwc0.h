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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/ncdhw2ndc1hwc0.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file ncdhw2ndc1hwc0.h
 * \brief GM(NCDHW) -> L1(NDC1HWC0) copy: one NCHW->NC1HWC0 (dn2nz) per depth slice.
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NCDHW2NDC1HWC0_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NCDHW2NDC1HWC0_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

// GM(NCDHW) -> L1(NDC1HWC0). conv3D adds a depth axis D on top of conv2D's NCHW->NC1HWC0. Each depth
// slice d is an independent NCHW->NC1HWC0 (dn2nz path), so this loops over D and reuses the exact
// same per-slice parameters as copy_gm_to_l1_nchw2nc1hwc0, only with indices shifted for the extra D
// dimension and a per-slice pointer offset on both src and dst.
//
//   src NCDHW  (N, C, D, H, W),        row-major stride (C*D*H*W, D*H*W, H*W, W, 1)
//   dst NDC1HWC0 (N, D, C1, H, W, C0), row-major stride (D*C1*H*W*C0, C1*H*W*C0, H*W*C0, W*C0, C0, 1)
//
// Per slice (fixed d): dn_num=H, n_value=W, d_value=C. Src C-axis step is Stride[1] (=D*H*W here, larger
// than conv2D's H*W because D sits between C and H) and src H-axis step is Stride[3] (=W). Dst nz
// strides: loop2=1, loop3=H*W, loop4=W. The depth step is Stride[2]=H*W on src and Stride[1]=C1*H*W*C0
// on dst.
class copy_gm_to_l1_ncdhw2ndc1hwc0 {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        using type = typename SrcTensor::element_type;
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        // src NCDHW (N, C, D, H, W): dn_num=H, n_value=W, d_value=C.
        uint16_t depth = get<2>(src_layout.shape());   // src_d (loop count)
        uint16_t dn_num = get<3>(src_layout.shape());  // src_h
        uint16_t n_value = get<4>(src_layout.shape()); // src_w
        uint32_t d_value = get<1>(src_layout.shape()); // src_c
        // src stride (C*D*H*W, D*H*W, H*W, W, 1): Stride[1]=D*H*W (C step), Stride[3]=W (H step).
        uint64_t loop1_src_stride = static_cast<uint64_t>(get<1>(src_layout.stride())) * sizeof(type); // C step
        uint64_t loop4_src_stride = static_cast<uint64_t>(get<3>(src_layout.stride())) * sizeof(type); // H step

        // dst NDC1HWC0 (N, D, C1, H, W, C0): loop4_dst_stride=dst_w=Shape[4]; loop3_dst_stride=dst_h*dst_w.
        uint16_t loop2_dst_stride = 1;
        uint16_t loop4_dst_stride = get<4>(dst_layout.shape());                              // dst_w
        uint16_t loop3_dst_stride = get<3>(dst_layout.shape()) * get<4>(dst_layout.shape()); // dst_h*dst_w

        if constexpr (is_b4_type<type>) {
            d_value = d_value >> 1;
            loop4_src_stride = loop4_src_stride >> 1;
            loop1_src_stride = loop1_src_stride >> 1;
        }
        uint8_t cache_mode = src.engine().get_cache_mode();

        // Per depth slice: index the slice with tensor(coord) to offset the address (engine moves by
        // Layout()(coord); pattern/trait preserved), same idiom as the batch-loop scenarios. The DMA
        // params are computed once above from the original layout and do not depend on d; only the
        // src/dst pointers move. src coord picks D at index 2 (NCDHW), dst at index 1 (NDC1HWC0).
        for (uint16_t d = 0; d < depth; ++d) {
            auto sub_src = src(make_coord(0, 0, d, 0, 0));    // NCDHW: engine += d*(H*W)
            auto sub_dst = dst(make_coord(0, d, 0, 0, 0, 0)); // NDC1HWC0: engine += d*(C1*H*W*C0)
            copy_gm_to_l1_multi_dn2nz_instr::data_copy(
                sub_dst.data().get(), sub_src.data().get(), dn_num, loop2_dst_stride, loop3_dst_stride,
                loop4_dst_stride, loop1_src_stride, cache_mode, n_value, d_value, loop4_src_stride, false);
        }
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
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        uint16_t depth = get<2>(src_shape);
        uint16_t dn_num = get<3>(src_shape);
        uint16_t n_value = get<4>(src_shape);
        uint32_t d_value = get<1>(src_shape);
        uint64_t loop1_src_stride = static_cast<uint64_t>(get<1>(src_layout.stride())) * sizeof(type);
        uint64_t loop4_src_stride = static_cast<uint64_t>(get<3>(src_layout.stride())) * sizeof(type);
        uint16_t loop2_dst_stride = 1;
        uint16_t loop4_dst_stride = get<4>(dst_layout.shape());
        uint16_t loop3_dst_stride = get<3>(dst_layout.shape()) * get<4>(dst_layout.shape());
        if constexpr (is_b4_type<type>) {
            d_value >>= 1;
            loop4_src_stride >>= 1;
            loop1_src_stride >>= 1;
        }

        for (uint16_t d = 0; d < depth; ++d) {
            auto depth_src_offset = src_offset + d * get<2>(src_layout.stride());
            auto depth_dst_offset = dst_offset + d * get<1>(dst_layout.stride());
            copy_gm_to_l1_multi_dn2nz_instr::data_copy_with_offset(dst, src, depth_dst_offset, depth_src_offset, dn_num,
                                                                   loop2_dst_stride, loop3_dst_stride, loop4_dst_stride,
                                                                   loop1_src_stride, src.engine().get_cache_mode(),
                                                                   n_value, d_value, loop4_src_stride, false);
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NCDHW2NDC1HWC0_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
