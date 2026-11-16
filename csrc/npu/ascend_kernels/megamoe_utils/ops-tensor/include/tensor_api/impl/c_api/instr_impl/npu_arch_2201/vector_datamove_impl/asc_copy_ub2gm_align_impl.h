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
 * \file asc_copy_ub2gm_align_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_DATAMOVE_IMPL_ASC_COPY_UB2GM_ALIGN_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_DATAMOVE_IMPL_ASC_COPY_UB2GM_ALIGN_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

#ifndef ASC_C_API_DEFAULT_ZERO_VALUE
#define ASC_C_API_DEFAULT_ZERO_VALUE 0
#endif

#ifndef ASC_C_API_DEFAULT_NBURST
#define ASC_C_API_DEFAULT_NBURST 1
#endif

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b8(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ int8_t* dst, __ubuf__ int8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b8(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ half* dst, __ubuf__ half* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b8(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b16(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ int16_t* dst, __ubuf__ int16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b16(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b16(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ float* dst, __ubuf__ float* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ int32_t* dst, __ubuf__ int32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ double* dst, __ubuf__ double* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ double* dst, __ubuf__ double* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ double* dst, __ubuf__ double* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ int64_t* dst, __ubuf__ int64_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_ub2gm_align_impl(
    __gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_b32(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, n_burst, len_burst, left_padding_num, right_padding_num, src_gap,
            dst_gap);
    }
}

__aicore__ inline void asc_copy_ub2gm_align_impl(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(
        dst, src, ASC_C_API_DEFAULT_NBURST, size, ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE,
        ASC_C_API_DEFAULT_ZERO_VALUE, ASC_C_API_DEFAULT_ZERO_VALUE);
}

__aicore__ inline void asc_copy_ub2gm_align_sync_impl(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
    asc_sync_post_process();
}

#undef ASC_C_API_DEFAULT_ZERO_VALUE
#undef ASC_C_API_DEFAULT_NBURST

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
