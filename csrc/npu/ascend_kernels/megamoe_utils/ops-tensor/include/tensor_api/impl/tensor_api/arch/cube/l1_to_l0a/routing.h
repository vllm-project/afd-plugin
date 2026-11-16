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
    "impl/tensor_api/arch/cube/l1_to_l0a/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */

#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_ROUTING_H

#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/nz2nz.h"
#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/zn2nz.h"
#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/img2col.h"
#include "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/img2col3d.h"

namespace asc {
namespace te {

class copy_l1_to_l0a_ignore {
public:
    template <const l1_to_l0a_trait& trait, typename... Args>
    __aicore__ inline void static run(const Args&... args)
    {
        static_assert(Std::is_same_v<Args..., void>, "copy_l1_to_l0a_ignore should not be called");
    }
};

template <uint32_t version, typename DstLayoutPattern, typename SrcLayoutPattern>
struct copy_l1_to_l0a_routing {
    using type = copy_l1_to_l0a_ignore;
};

template <uint32_t version>
struct copy_l1_to_l0a_routing<version, nz_layout_ptn, nz_layout_ptn> {
    using type = load_data_l1_to_l0a_nz2nz;
};

template <uint32_t version>
struct copy_l1_to_l0a_routing<version, nz_layout_ptn, zn_layout_ptn> {
    using type = load_data_l1_to_l0a_zn2nz;
};

// Img2Col: L1(NC1HWC0) -> L0A(NZ) for conv feature maps. Dispatched when src carries the
// nc1hwc0_layout_ptn tag and params is bound via copy_l1_to_l0a.with(Img2ColParams).
template <uint32_t version>
struct copy_l1_to_l0a_routing<version, nz_layout_ptn, nc1hwc0_layout_ptn> {
    using type = load_data_l1_to_l0a_img2col;
};

// conv3D img2col: L1(NDC1HWC0) -> L0A(NZ). Same img2col hardware path; the extra depth axis is
// selected by the caller (tensor(coord)) and looped outside, so this reads one (C1,H,W,C0) plane
// with indices shifted for the leading D axis.
template <uint32_t version>
struct copy_l1_to_l0a_routing<version, nz_layout_ptn, ndc1hwc0_layout_ptn> {
    using type = load_data_l1_to_l0a_img2col3d;
};

} // namespace te
} // namespace asc
#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
