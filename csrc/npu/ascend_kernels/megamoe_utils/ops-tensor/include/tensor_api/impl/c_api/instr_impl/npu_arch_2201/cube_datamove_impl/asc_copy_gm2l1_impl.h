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
 * \file asc_copy_gm2l1_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_GM2L1_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_GM2L1_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

// PAD_NONE
__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_NONE);
    }
}

__aicore__ inline void asc_copy_gm2l1_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE1
__aicore__ inline void asc_copy_gm2l1_pad1_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE1);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad1_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad1_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad1_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad1_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE2
__aicore__ inline void asc_copy_gm2l1_pad2_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE2);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad2_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad2_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad2_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad2_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE3
__aicore__ inline void asc_copy_gm2l1_pad3_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE3);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad3_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad3_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad3_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad3_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE4
__aicore__ inline void asc_copy_gm2l1_pad4_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE4);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad4_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad4_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad4_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad4_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE5
__aicore__ inline void asc_copy_gm2l1_pad5_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE5);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad5_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad5_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad5_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad5_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE6
__aicore__ inline void asc_copy_gm2l1_pad6_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE6);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad6_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad6_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad6_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad6_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE7
__aicore__ inline void asc_copy_gm2l1_pad7_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE7);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad7_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad7_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad7_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad7_impl(dst, src, size);
    asc_sync_post_process();
}

// PAD_MODE8
__aicore__ inline void asc_copy_gm2l1_pad8_impl(
    __cbuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_gm_to_cbuf(dst, src, 0, n_burst, len_burst, src_gap, dst_gap, PAD_MODE8);
    }
}

__aicore__ inline void asc_copy_gm2l1_pad8_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad8_impl(dst, src, 1, size / ASC_C_API_ONE_DATABLOCK_SIZE, 0, 0);
}

__aicore__ inline void asc_copy_gm2l1_pad8_sync_impl(__cbuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2l1_pad8_impl(dst, src, size);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ half* dst, __gm__ half* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ float* dst, __gm__ float* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ int32_t* dst, __gm__ int32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ int8_t* dst, __gm__ int8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_gm2l1_impl(
    __cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        load_gm_to_cbuf(dst, src, base_idx, repeat, src_stride, dst_gap, static_cast<uint8_t>(0), addr_cal_mode_t::inc);
    }
}

__aicore__ inline void asc_copy_gm2l1_sync_impl(
    __cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
    uint16_t dst_gap)
{
    asc_copy_gm2l1_impl(dst, src, base_idx, repeat, src_stride, dst_gap);
    asc_sync_post_process();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
