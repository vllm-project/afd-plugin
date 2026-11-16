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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H
#endif

/*!
 * \file copy_op.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H
#define INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H

#include "tensor_api/arch/vector/copy_trait.h"

namespace asc {
namespace te {

struct gm_to_ub_params {
    __aicore__ constexpr gm_to_ub_params() {}

    __aicore__ constexpr gm_to_ub_params(
        uint8_t left_padding_count, uint8_t right_padding_count, bool enable_constant_pad = true)
        : left_padding_count(left_padding_count),
          right_padding_count(right_padding_count),
          enable_constant_pad(enable_constant_pad)
    {}

    uint8_t left_padding_count = 0;
    uint8_t right_padding_count = 0;
    bool enable_constant_pad = true;
};

struct copy_gm_to_ub {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};

struct copy_ub_to_l1 {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const ub_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <
        const ub_to_l1_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
        typename ShapeType>
    __aicore__ inline static void data_copy_impl(
        const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
        const ShapeType& copy_shape);
};

struct copy_ub_to_gm {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const ub_to_gm_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <
        const ub_to_gm_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
        typename ShapeType>
    __aicore__ inline static void data_copy_impl(
        const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
        const ShapeType& copy_shape);
};

struct copy_ub_to_ub {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <
        const ub_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
        typename ShapeType>
    __aicore__ inline static void data_copy_impl(
        const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
        const ShapeType& copy_shape);
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/arch/vector/gm_to_ub/copy.h"
#include "impl/tensor_api/arch/vector/ub_to_l1/copy.h"
#include "impl/tensor_api/arch/vector/ub_to_gm/copy.h"
#include "impl/tensor_api/arch/vector/ub_to_ub/copy.h"

#endif // INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_INCLUDE_TENSOR_API_ARCH_VECTOR_COPY_OP_H
#endif
