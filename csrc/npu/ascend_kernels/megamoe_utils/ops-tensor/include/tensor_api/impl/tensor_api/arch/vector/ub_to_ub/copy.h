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
#pragma message( \
    "impl/tensor_api/arch/vector/ub_to_ub/copy.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H
#define IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/vector/ub_to_ub/routing.h"

namespace asc {
namespace te {

template <typename Trait, const Trait& trait, typename... Args>
__aicore__ inline void copy_ub_to_ub::copy(const Args&... args)
{
    data_copy_impl<trait, Args...>(args...);
}

template <const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
__aicore__ inline void copy_ub_to_ub::data_copy_impl(const DstTensor& dst, const SrcTensor& src)
{
    using dst_pos = get_mem_location<DstTensor>;
    using src_pos = get_mem_location<SrcTensor>;
    static_assert(
        Std::is_same_v<dst_pos, location::ub>, "For copy_ub_to_ub, the destination tensor must be located in UB.");
    static_assert(Std::is_same_v<src_pos, location::ub>, "For copy_ub_to_ub, the source tensor must be located in UB.");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, dst.layout(), "dst", "copy_ub_to_ub");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, src.layout(), "src", "copy_ub_to_ub");
    TENSOR_API_DEBUG_CHECK(debug_check_copy_size, src, dst, "copy_ub_to_ub");
    using dst_layout_ptn = get_layout_pattern<typename DstTensor::layout_type>;
    using src_layout_ptn = get_layout_pattern<typename SrcTensor::layout_type>;
    using copy_ub_to_ub_impl =
        typename copy_ub_to_ub_routing<current_arch_version, dst_layout_ptn, src_layout_ptn>::type;
    copy_ub_to_ub_impl::template run<trait, DstTensor, SrcTensor>(dst, src);
}

template <
    const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
    typename ShapeType>
__aicore__ inline void copy_ub_to_ub::data_copy_impl(
    const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
    const ShapeType& copy_shape)
{
    using dst_pos = get_mem_location<DstTensor>;
    using src_pos = get_mem_location<SrcTensor>;
    static_assert(Std::is_same_v<dst_pos, location::ub>, "When Copy tensor from UB to UB, dst tensor must on UB");
    static_assert(Std::is_same_v<src_pos, location::ub>, "When Copy tensor from UB to UB, src tensor must on UB");
    using dst_layout_ptn = get_layout_pattern<typename DstTensor::layout_type>;
    using src_layout_ptn = get_layout_pattern<typename SrcTensor::layout_type>;
    using copy_ub_to_ub_impl =
        typename copy_ub_to_ub_routing<current_arch_version, dst_layout_ptn, src_layout_ptn>::type;
    auto resolved_dst = resolve_copy_coord(dst.layout(), copy_shape, dst_coord);
    auto resolved_src = resolve_copy_coord(src.layout(), copy_shape, src_coord);
    copy_ub_to_ub_impl::template run<trait, DstTensor, SrcTensor>(dst, src, resolved_dst, resolved_src, copy_shape);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_COPY_H
#endif
