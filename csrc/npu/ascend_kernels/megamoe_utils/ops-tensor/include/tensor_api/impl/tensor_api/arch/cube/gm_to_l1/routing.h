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
    "impl/tensor_api/arch/cube/gm_to_l1/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_ROUTING_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/dn2nz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/dn2zn.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nd2nd.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nd2nz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nd2zn.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nc1hwc02nc1hwc0.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nchw2nc1hwc0.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/ncdhw2ndc1hwc0.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nhwc2nc1hwc0.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nz2nz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/zn2zn.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scalea_nd2zz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scalea_dn2zz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scalea_zz2zz.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scaleb_nd2nn.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scaleb_dn2nn.h"
#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/scaleb_nn2nn.h"

namespace asc {
namespace te {

class copy_gm_to_l1_ignore {
public:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        static_assert(!Std::is_same_v<DstTensor, DstTensor>, "copy_gm_to_l1: unsupported layout pattern combination.");
    }
};

template <uint32_t version, typename DstLayoutPtn, typename SrcLayoutPtn>
struct copy_gm_to_l1_routing {
    using type = copy_gm_to_l1_ignore;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nd_ext_layout_ptn, nd_ext_layout_ptn> {
    using type = copy_gm_to_l1_nd2nd;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nd_layout_ptn, nd_layout_ptn> {
    using type = copy_gm_to_l1_nd2nd;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nz_layout_ptn, nd_ext_layout_ptn> {
    using type = copy_gm_to_l1_nd2nz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nz_layout_ptn, nd_layout_ptn> {
    using type = copy_gm_to_l1_nd2nz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zn_layout_ptn, nd_ext_layout_ptn> {
    using type = copy_gm_to_l1_nd2zn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zn_layout_ptn, nd_layout_ptn> {
    using type = copy_gm_to_l1_nd2zn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nz_layout_ptn, dn_ext_layout_ptn> {
    using type = copy_gm_to_l1_dn2nz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nz_layout_ptn, dn_layout_ptn> {
    using type = copy_gm_to_l1_dn2nz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zn_layout_ptn, dn_ext_layout_ptn> {
    using type = copy_gm_to_l1_dn2zn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zn_layout_ptn, dn_layout_ptn> {
    using type = copy_gm_to_l1_dn2zn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nz_layout_ptn, nz_layout_ptn> {
    using type = copy_gm_to_l1_nz2nz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zn_layout_ptn, zn_layout_ptn> {
    using type = copy_gm_to_l1_zn2zn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zz_layout_ptn, scalea_nd_layout_ptn> {
    using type = copy_gm_to_l1_scalea_nd2zz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zz_layout_ptn, scalea_dn_layout_ptn> {
    using type = copy_gm_to_l1_scalea_dn2zz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, zz_layout_ptn, zz_layout_ptn> {
    using type = copy_gm_to_l1_scalea_zz2zz;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nn_layout_ptn, scaleb_nd_layout_ptn> {
    using type = copy_gm_to_l1_scaleb_nd2nn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nn_layout_ptn, scaleb_dn_layout_ptn> {
    using type = copy_gm_to_l1_scaleb_dn2nn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nn_layout_ptn, nn_layout_ptn> {
    using type = copy_gm_to_l1_scaleb_nn2nn;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nc1hwc0_layout_ptn, nc1hwc0_layout_ptn> {
    using type = copy_gm_to_l1_nc1hwc02nc1hwc0;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nc1hwc0_layout_ptn, nhwc_layout_ptn> {
    using type = copy_gm_to_l1_nhwc2nc1hwc0;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, nc1hwc0_layout_ptn, nchw_layout_ptn> {
    using type = copy_gm_to_l1_nchw2nc1hwc0;
};

template <uint32_t version>
struct copy_gm_to_l1_routing<version, ndc1hwc0_layout_ptn, ncdhw_layout_ptn> {
    using type = copy_gm_to_l1_ncdhw2ndc1hwc0;
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
