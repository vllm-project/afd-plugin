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
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H
#endif

/*!
 * \file copy_trait.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H
#define INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H

namespace asc {
namespace te {

struct gm_to_ub_trait {};
struct ub_to_l1_trait {};
struct ub_to_gm_trait {};
struct ub_to_ub_trait {};

struct gm_to_ub_trait_default {
    using trait_type = gm_to_ub_trait;
    static constexpr const trait_type value = {};
};

struct ub_to_l1_trait_default {
    using trait_type = ub_to_l1_trait;
    static constexpr const trait_type value = {};
};

struct ub_to_gm_trait_default {
    using trait_type = ub_to_gm_trait;
    static constexpr const trait_type value = {};
};

struct ub_to_ub_trait_default {
    using trait_type = ub_to_ub_trait;
    static constexpr const trait_type value = {};
};

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_TRAIT_H
#endif
