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
    "impl/tensor_api/arch/vector/ub_to_ub/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"tensor_api/tensor.h\"\" and use public functions or variables defined in interface headers files.")
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H
#define IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H

#include "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/dn2dn.h"
#include "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/nd2nd.h"
#include "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/nz2nz.h"
#include "impl/tensor_api/arch/vector/ub_to_ub/copy_impl/zn2zn.h"

namespace asc {
namespace te {

class copy_ub_to_ub_ignore {
public:
    template <const ub_to_ub_trait& trait, typename... Args>
    __aicore__ inline static void run(const Args&... args)
    {
        static_assert(Std::is_same_v<Args..., void>, "copy_ub_to_ub_ignore should not be called");
    }
};

template <uint32_t version, typename DstLayoutPtn, typename SrcLayoutPtn>
struct copy_ub_to_ub_routing {
    using type = copy_ub_to_ub_ignore;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, nd_ext_layout_ptn, nd_ext_layout_ptn> {
    using type = copy_ub_to_ub_nd;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, nd_layout_ptn, nd_layout_ptn> {
    using type = copy_ub_to_ub_nd;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, one_dim_layout_ptn, one_dim_layout_ptn> {
    using type = copy_ub_to_ub_one_dim;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, dn_ext_layout_ptn, dn_ext_layout_ptn> {
    using type = copy_ub_to_ub_dn;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, dn_layout_ptn, dn_layout_ptn> {
    using type = copy_ub_to_ub_dn;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, nz_layout_ptn, nz_layout_ptn> {
    using type = copy_ub_to_ub_nz;
};

template <uint32_t version>
struct copy_ub_to_ub_routing<version, zn_layout_ptn, zn_layout_ptn> {
    using type = copy_ub_to_ub_zn;
};
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ARCH_VECTOR_UB_TO_UB_ROUTING_H
#endif
