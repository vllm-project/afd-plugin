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
    "impl/tensor_api/arch/cube/l1_to_fb/copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_H

#include "impl/tensor_api/arch/cube/l1_to_fb/routing.h"

namespace asc {
namespace te {

template <typename Trait, const Trait& trait, typename... Args>
__aicore__ inline void copy_l1_to_fixbuf::copy(const Args&... args)
{
    data_copy_impl<trait, Args...>(args...);
}

template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor>
__aicore__ inline void copy_l1_to_fixbuf::data_copy_impl(const DstTensor& dst, const SrcTensor& src)
{
    using dst_pos = get_mem_location<DstTensor>;
    using src_pos = get_mem_location<SrcTensor>;
    static_assert(Std::is_same_v<dst_pos, location::fixbuf>,
                  "For copy_l1_to_fixbuf, the destination tensor must be located in FIXBUF.");
    static_assert(Std::is_same_v<src_pos, location::l1>,
                  "For copy_l1_to_fixbuf, the source tensor must be located in L1.");
    using dst_layout = typename DstTensor::layout_type;
    using src_layout = typename SrcTensor::layout_type;
    using dst_layout_ptn = get_layout_pattern<dst_layout>;
    using src_layout_ptn = get_layout_pattern<src_layout>;
    TENSOR_API_DEBUG_CHECK(debug_check_layout, dst.layout(), "dst", "copy_l1_to_fixbuf");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, src.layout(), "src", "copy_l1_to_fixbuf");
    TENSOR_API_DEBUG_CHECK(debug_check_copy_size, src, dst, "copy_l1_to_fixbuf");
    using copy_l1_to_fb_impl =
        typename copy_l1_to_fixbuf_routing<current_arch_version, dst_layout_ptn, src_layout_ptn>::type;
    copy_l1_to_fb_impl::template run<trait, DstTensor, SrcTensor>(dst, src);
}

template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
    typename SrcCoord, typename CopyShape>
__aicore__ inline void copy_l1_to_fixbuf::data_copy_impl(const DstTensor& dst, const SrcTensor& src,
    const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape)
{
    using dst_pos = get_mem_location<DstTensor>;
    using src_pos = get_mem_location<SrcTensor>;
    static_assert(Std::is_same_v<dst_pos, location::fixbuf>,
                  "For copy_l1_to_fixbuf, the destination tensor must be located in FIXBUF.");
    static_assert(Std::is_same_v<src_pos, location::l1>,
                  "For copy_l1_to_fixbuf, the source tensor must be located in L1.");
    using dst_pattern = get_layout_pattern<typename DstTensor::layout_type>;
    using src_pattern = get_layout_pattern<typename SrcTensor::layout_type>;
    using copy_l1_to_fb_impl =
        typename copy_l1_to_fixbuf_routing<current_arch_version, dst_pattern, src_pattern>::type;
    auto resolved_dst_coord = resolve_copy_coord(dst.layout(), copy_shape, dst_coord);
    auto resolved_src_coord = resolve_copy_coord(src.layout(), copy_shape, src_coord);
    copy_l1_to_fb_impl::template run<trait, DstTensor, SrcTensor>(
        dst, src, resolved_dst_coord, resolved_src_coord, copy_shape);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
