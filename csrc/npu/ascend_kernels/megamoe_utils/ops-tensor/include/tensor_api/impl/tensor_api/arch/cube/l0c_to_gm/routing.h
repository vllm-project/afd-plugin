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
    "impl/tensor_api/arch/cube/l0c_to_gm/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_ROUTING_H

#include "impl/tensor_api/arch/cube/utils/l0c2out_utils.h"
#include "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/data_copy.h"
#include "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/nz2nchw.h"
#include "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/nz2ncdhw.h"
#include "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/nz2nhwc.h"

namespace asc {
namespace te {

class copy_l0c_to_gm_ignore {
public:
    template <const l0c_to_gm_trait& trait, typename... Args>
    __aicore__ inline static void run(const Args&... args)
    {
        static_assert(Std::is_same_v<Args..., void>, "copy_l0c_to_gm_ignore should not be called");
    }
};

template <uint32_t version, typename DstLayoutPtn, typename SrcLayoutPtn>
struct copy_l0c_to_gm_routing {
    using type = copy_l0c_to_gm_ignore;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nd_ext_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};
template <uint32_t version>
struct copy_l0c_to_gm_routing<version, dn_ext_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};
template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nd_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};
template <uint32_t version>
struct copy_l0c_to_gm_routing<version, dn_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nz_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nc1hwc0_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nhwc_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm_nz2nhwc;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, nchw_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm_nz2nchw;
};

template <uint32_t version>
struct copy_l0c_to_gm_routing<version, ncdhw_layout_ptn, nz_layout_ptn> {
    using type = data_copy_l0c_to_gm_nz2ncdhw;
};
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
