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
    "impl/tensor_api/arch/cube/mmad/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_MMAD_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_MMAD_ROUTING_H

#include "impl/tensor_api/arch/cube/mmad/mmad_impl/mmad.h"
#include "impl/tensor_api/arch/cube/mmad/mmad_impl/mmad_with_bias.h"

namespace asc {
namespace te {

class mmad_ignore {
public:
    template <const mmad_trait& trait, typename... Args>
    __aicore__ inline static void run(const Args&... args)
    {
        static_assert(Std::is_same_v<Args..., void>, "mmad_ignore should not be called");
    }
};

template <uint32_t version, typename DstLayoutPtn, typename FmLayoutPtn, typename FilterLayoutPtn, typename BiasPos>
struct mmad_routing {
    using type = mmad_ignore;
};

template <uint32_t version>
struct mmad_routing<version, nz_layout_ptn, nz_layout_ptn, zn_layout_ptn, location::invalid> {
    using type = mmad_executor;
};

template <uint32_t version>
struct mmad_routing<version, nz_layout_ptn, nz_layout_ptn, zn_layout_ptn, location::l0c> {
    using type = mmad_with_bias;
};

template <uint32_t version>
struct mmad_routing<version, nz_layout_ptn, nz_layout_ptn, zn_layout_ptn, location::bias> {
    using type = mmad_with_bias;
};
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_MMAD_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
