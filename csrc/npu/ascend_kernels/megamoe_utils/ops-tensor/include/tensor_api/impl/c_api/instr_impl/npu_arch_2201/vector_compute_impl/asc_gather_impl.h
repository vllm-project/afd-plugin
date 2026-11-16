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
 * \file asc_gather_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_GATHER_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_GATHER_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

// asc_gather int16_t
__aicore__ inline void asc_gather_impl(
    __ubuf__ int16_t* dst, __ubuf__ int16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride,
    uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather((__ubuf__ uint16_t*)dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ int16_t* dst, __ubuf__ int16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ int16_t* dst, __ubuf__ int16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather uint16_t
__aicore__ inline void asc_gather_impl(
    __ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride,
    uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather(dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather half
__aicore__ inline void asc_gather_impl(
    __ubuf__ half* dst, __ubuf__ half* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather((__ubuf__ uint16_t*)dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ half* dst, __ubuf__ half* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ half* dst, __ubuf__ half* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather bfloat16_t
__aicore__ inline void asc_gather_impl(
    __ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride,
    uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather((__ubuf__ uint16_t*)dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather int32_t
__aicore__ inline void asc_gather_impl(
    __ubuf__ int32_t* dst, __ubuf__ int32_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride,
    uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather((__ubuf__ uint32_t*)dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ int32_t* dst, __ubuf__ int32_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ int32_t* dst, __ubuf__ int32_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather uint32_t
__aicore__ inline void asc_gather_impl(
    __ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride,
    uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather(dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

// asc_gather float
__aicore__ inline void asc_gather_impl(
    __ubuf__ float* dst, __ubuf__ float* src, __ubuf__ uint32_t* src_offset, uint16_t dst_repeat_stride, uint8_t repeat)
{
    if ASC_IS_AIV {
        vgather((__ubuf__ uint32_t*)dst, src_offset, (uint64_t)src, dst_repeat_stride, repeat);
    }
}

__aicore__ inline void asc_gather_impl(
    __ubuf__ float* dst, __ubuf__ float* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_set_mask_count_begin(count);
    asc_gather_impl(dst, src, src_offset, ASC_C_API_DEFAULT_REPEAT_STRIDE.U16, ASC_C_API_DEFAULT_REPEAT.U8);
    asc_set_mask_count_end();
}

__aicore__ inline void asc_gather_sync_impl(
    __ubuf__ float* dst, __ubuf__ float* src, __ubuf__ uint32_t* src_offset, uint32_t count)
{
    asc_gather_impl(dst, src, src_offset, count);
    asc_sync_post_process();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
