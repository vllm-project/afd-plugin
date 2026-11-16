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
    "impl/tensor_api/arch/cube/l1_to_l0a/copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_H

#include "impl/tensor_api/arch/cube/l1_to_l0a/routing.h"

namespace asc {
namespace te {

struct copy_l1_to_l0a_base {
public:
    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename PaddingValue = int16_t>
    __aicore__ inline static void load_data_impl(const DstTensor& dst, const SrcTensor& src,
                                                 const img2col_params<PaddingValue>& params = default_img2col_params)
    {
        using dst_pos = get_mem_location<DstTensor>;
        using src_pos = get_mem_location<SrcTensor>;
        static_assert(Std::is_same_v<dst_pos, location::l0a>,
                      "For copy_l1_to_l0a, the destination tensor must be located in L0A.");
        static_assert(Std::is_same_v<src_pos, location::l1>,
                      "For copy_l1_to_l0a, the source tensor must be located in L1.");
        using dst_layout = typename DstTensor::layout_type;
        using src_layout = typename SrcTensor::layout_type;
        using dst_pattern = get_layout_pattern<dst_layout>;
        using src_pattern = get_layout_pattern<src_layout>;
        TENSOR_API_DEBUG_CHECK(debug_check_layout, dst.layout(), "dst", "copy_l1_to_l0a");
        TENSOR_API_DEBUG_CHECK(debug_check_layout, src.layout(), "src", "copy_l1_to_l0a");
        TENSOR_API_DEBUG_CHECK(debug_check_copy_size, src, dst, "copy_l1_to_l0a");
        if constexpr (Std::is_same_v<dst_pattern, nz_layout_ptn> && Std::is_same_v<src_pattern, nc1hwc0_layout_ptn>
                      && !is_b4_type<typename DstTensor::element_type>) {
            TENSOR_API_DEBUG_CHECK(debug_check_img2col_coord<typename DstTensor::element_type>, params,
                                   "copy_l1_to_l0a");
        }
        if constexpr (Std::is_same_v<src_pattern, nc1hwc0_layout_ptn>
                      || Std::is_same_v<src_pattern, ndc1hwc0_layout_ptn>) {
            // conv2D (NC1HWC0) and conv3D (NDC1HWC0) both take the img2col path; routing picks the
            // 2D or 3D implementation selected by the source pattern.
            using copy_l1_to_l0a_impl =
                typename copy_l1_to_l0a_routing<current_arch_version, dst_pattern, src_pattern>::type;
            copy_l1_to_l0a_impl::template run<trait, DstTensor, SrcTensor, PaddingValue>(dst, src, params);
        } else {
            if constexpr (dst_layout::depth == five_dim_data && src_layout::depth == five_dim_data) {
                TENSOR_API_DEBUG_CHECK(debug_check_batch_match, get<0>(src.layout().shape()),
                                       get<0>(dst.layout().shape()), "copy_l1_to_l0a");
                TENSOR_API_DEBUG_CHECK(debug_check_l0_batch_stride, get<0>(src.layout().stride()),
                                       remove_batch_dim(src.layout()).capacity(), get<0>(dst.layout().stride()),
                                       remove_batch_dim(dst.layout()).capacity(), "copy_l1_to_l0a");
            }
            using copy_l1_to_l0a_impl =
                typename copy_l1_to_l0a_routing<current_arch_version, dst_pattern, src_pattern>::type;
            copy_l1_to_l0a_impl::template run<trait, DstTensor, SrcTensor>(dst, src);
        }
    }

    template <const l1_to_l0a_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape, typename PaddingValue = int16_t>
    __aicore__ inline static void load_data_impl(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                                 const SrcCoord& src_coord, const CopyShape& copy_shape,
                                                 const img2col_params<PaddingValue>& params = default_img2col_params)
    {
        using dst_pos = get_mem_location<DstTensor>;
        using src_pos = get_mem_location<SrcTensor>;
        static_assert(Std::is_same_v<dst_pos, location::l0a>,
                      "For copy_l1_to_l0a, the destination tensor must be located in L0A.");
        static_assert(Std::is_same_v<src_pos, location::l1>,
                      "For copy_l1_to_l0a, the source tensor must be located in L1.");

        using dst_pattern = get_layout_pattern<typename DstTensor::layout_type>;
        using src_pattern = get_layout_pattern<typename SrcTensor::layout_type>;
        if constexpr (Std::is_same_v<src_pattern, nc1hwc0_layout_ptn>) {
            using copy_l1_to_l0a_impl =
                typename copy_l1_to_l0a_routing<current_arch_version, dst_pattern, src_pattern>::type;
            auto resolved_dst_coord = resolve_copy_coord(dst.layout(), copy_shape, dst_coord);
            auto resolved_src_coord = resolve_copy_coord(src.layout(), copy_shape, src_coord);
            copy_l1_to_l0a_impl::template run<trait, DstTensor, SrcTensor, decltype(resolved_dst_coord),
                                              decltype(resolved_src_coord), CopyShape, PaddingValue>(
                dst, src, resolved_dst_coord, resolved_src_coord, copy_shape, params);
        } else {
            using copy_l1_to_l0a_impl =
                typename copy_l1_to_l0a_routing<current_arch_version, dst_pattern, src_pattern>::type;
            auto resolved_dst_coord = resolve_copy_coord(dst.layout(), copy_shape, dst_coord);
            auto resolved_src_coord = resolve_copy_coord(src.layout(), copy_shape, src_coord);
            copy_l1_to_l0a_impl::template run<trait, DstTensor, SrcTensor>(dst, src, resolved_dst_coord,
                                                                           resolved_src_coord, copy_shape);
        }
    }
};

template <typename Trait, const Trait& trait, typename... Args>
__aicore__ inline void copy_l1_to_l0a::copy(const Args&... args)
{
    copy_l1_to_l0a_base::load_data_impl<trait>(args...);
}

struct copy_l1_to_l0a_with : public copy_l1_to_l0a_base {
public:
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args)
    {
        load_data_impl<trait>(args...);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
