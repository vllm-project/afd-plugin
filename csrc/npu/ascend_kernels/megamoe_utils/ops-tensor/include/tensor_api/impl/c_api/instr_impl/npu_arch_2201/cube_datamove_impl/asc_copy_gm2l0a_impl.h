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
 * \file asc_copy_gm2l0a_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_GM2L0A_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_GM2L0A_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca(dst, src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l0a_impl(
    __ca__ int4b_t* dst, __gm__ int4b_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_ca_s4((__ca__ void*)dst, (__gm__ void*)src, base_idx, repeat, src_stride, dst_gap, 0, inc);
    }
}

__aicore__ inline void asc_copy_gm2l0a_sync_impl(
    __ca__ int4b_t* dst, __gm__ int4b_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l0a_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
