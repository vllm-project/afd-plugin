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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nc1hwc02nc1hwc0.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nc1hwc02nc1hwc0.h
 * \brief GM(NC1HWC0) -> L1(NC1HWC0) copy for conv feature maps.
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NC1HWC02NC1HWC0_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NC1HWC02NC1HWC0_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

// GM(NC1HWC0) -> L1(NC1HWC0). Dispatched by copy_gm_to_l1_routing when both src/dst carry the
// nc1hwc0_layout_ptn tag. W is loaded fully when contiguous (one burst per C1 of H*W*C0);
// otherwise a per-H loop emits C1 bursts of W*C0 each.
class copy_gm_to_l1_nc1hwc02nc1hwc0 {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        using type = typename SrcTensor::element_type;
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();

        // (N, C1, H, W, C0): C1 axis stride = H*W*C0, H axis stride = W*C0.
        uint8_t cache_mode = src.engine().get_cache_mode();
        uint32_t c1 = get<1>(dst_layout.shape());
        uint32_t row_elems = get<2>(dst_layout.stride()); // W*C0 (dst packed H stride)

        uint64_t src_stride = get<1>(src_layout.stride()) * sizeof(type); // burst_src_stride
        uint32_t dst_stride = get<1>(dst_layout.stride()) * sizeof(type); // burst_dst_stride
        if constexpr (is_b4_type<type>) {
            src_stride = src_stride >> 1;
            dst_stride = dst_stride >> 1;
        }

        if (get<2>(src_layout.stride()) == row_elems) {
            // W fully loaded: src H rows contiguous, one burst per C1 of H*W*C0.
            uint32_t block_len = get<1>(dst_layout.stride()) * sizeof(type);
            if constexpr (is_b4_type<type>) {
                block_len = block_len >> 1;
            }
            copy_gm_to_l1_align_v2_instr::data_copy(dst.data().get(), src.data().get(), c1, block_len, 0, 0, cache_mode,
                                                    src_stride, dst_stride);
        } else {
            // W not fully loaded (sliced src): per-H loop, C1 bursts of W*C0 each.
            uint32_t h = get<2>(dst_layout.shape());
            uint32_t block_len = row_elems * sizeof(type);
            if constexpr (is_b4_type<type>) {
                block_len = block_len >> 1;
            }
            for (uint32_t i = 0; i < h; ++i) {
                auto src_h = src(make_coord(0, 0, i, 0, 0));
                auto dst_h = dst(make_coord(0, 0, i, 0, 0));
                copy_gm_to_l1_align_v2_instr::data_copy(dst_h.data().get(), src_h.data().get(), c1, block_len, 0, 0,
                                                        cache_mode, src_stride, dst_stride);
            }
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
        const auto& dst_layout = dst.layout();
        uint32_t c1 = get<1>(dst_layout.shape());
        uint32_t row_elems = get<2>(dst_layout.stride());
        uint64_t src_stride = get<1>(src.layout().stride()) * sizeof(type);
        uint32_t dst_stride = get<1>(dst_layout.stride()) * sizeof(type);
        if constexpr (is_b4_type<type>) {
            src_stride >>= 1;
            dst_stride >>= 1;
        }
        if (get<2>(src.layout().stride()) == row_elems) {
            uint32_t block_len = get<1>(dst_layout.stride()) * sizeof(type);
            if constexpr (is_b4_type<type>) {
                block_len >>= 1;
            }
            copy_gm_to_l1_align_v2_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, c1, block_len, 0, 0,
                                                                src.engine().get_cache_mode(), src_stride, dst_stride);
        } else {
            uint32_t block_len = row_elems * sizeof(type);
            if constexpr (is_b4_type<type>) {
                block_len >>= 1;
            }
            for (uint32_t i = 0; i < get<2>(src_shape); ++i) {
                auto dst_h = dst(make_coord(0, 0, i, 0, 0));
                auto row_offset = src_offset + i * get<2>(src.layout().stride());
                copy_gm_to_l1_align_v2_instr::data_copy_with_offset(dst_h, src, dst_offset, row_offset, c1, block_len,
                                                                    0, 0, src.engine().get_cache_mode(), src_stride,
                                                                    dst_stride);
            }
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_NC1HWC02NC1HWC0_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
