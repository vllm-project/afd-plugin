/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_transto5hd_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_TRANSTO5HD_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_TRANSTO5HD_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl/utils_impl.h"

// __aicore__ inline void asc_transto5hd_b8_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
// uint16_t src_stride, bool dst_high_half, bool src_high_half)
#define asc_transto5hd_b8_impl(dst, src, repeat, dst_stride, src_stride, dst_high_half, src_high_half) \
    do {                                                                                               \
        if ASC_IS_AIV {                                                                                \
            scatter_vnchwconv_b8(                                                                      \
                (dst), (src), (repeat), (dst_stride), (src_stride), (dst_high_half), (src_high_half)); \
        }                                                                                              \
    } while (0)

// __aicore__ inline void asc_transto5hd_b8_sync_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t
// dst_stride, uint16_t src_stride, bool dst_high_half, bool src_high_half)
#define asc_transto5hd_b8_sync_impl(dst, src, repeat, dst_stride, src_stride, dst_high_half, src_high_half)           \
    do {                                                                                                              \
        asc_transto5hd_b8_impl((dst), (src), (repeat), (dst_stride), (src_stride), (dst_high_half), (src_high_half)); \
        asc_sync_post_process();                                                                                      \
    } while (0)

// __aicore__ inline void asc_transto5hd_b16_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
// uint16_t src_stride)
#define asc_transto5hd_b16_impl(dst, src, repeat, dst_stride, src_stride)              \
    do {                                                                               \
        if ASC_IS_AIV {                                                                \
            scatter_vnchwconv_b16((dst), (src), (repeat), (dst_stride), (src_stride)); \
        }                                                                              \
    } while (0)

// __aicore__ inline void asc_transto5hd_b16_sync_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t
// dst_stride, uint16_t src_stride)
#define asc_transto5hd_b16_sync_impl(dst, src, repeat, dst_stride, src_stride)       \
    do {                                                                             \
        asc_transto5hd_b16_impl((dst), (src), (repeat), (dst_stride), (src_stride)); \
        asc_sync_post_process();                                                     \
    } while (0)

// __aicore__ inline void asc_transto5hd_b32_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
// uint16_t src_stride)
#define asc_transto5hd_b32_impl(dst, src, repeat, dst_stride, src_stride)              \
    do {                                                                               \
        if ASC_IS_AIV {                                                                \
            scatter_vnchwconv_b32((dst), (src), (repeat), (dst_stride), (src_stride)); \
        }                                                                              \
    } while (0)

// __aicore__ inline void asc_transto5hd_b32_sync_impl(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t
// dst_stride, uint16_t src_stride)
#define asc_transto5hd_b32_sync_impl(dst, src, repeat, dst_stride, src_stride)       \
    do {                                                                             \
        asc_transto5hd_b32_impl((dst), (src), (repeat), (dst_stride), (src_stride)); \
        asc_sync_post_process();                                                     \
    } while (0)

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
