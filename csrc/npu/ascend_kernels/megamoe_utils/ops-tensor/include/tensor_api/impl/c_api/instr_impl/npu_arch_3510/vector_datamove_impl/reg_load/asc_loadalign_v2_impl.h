/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_V2_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_V2_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// =========asc_loadalign(fp4x2_e2m1/fp4x2_e1m2/u8/s8/f8_e4m3/fp8_e5m2/fp8_e8m0/u16/s16/bf16/half/u32/s32/float/u64/s64)=========
__simd_callee__ inline void asc_loadalign_impl(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_int64_t& dst, __ubuf__ int64_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_uint64_t& dst, __ubuf__ uint64_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, NORM);
    }
}

__simd_callee__ inline void asc_loadalign_impl(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        pld(dst, src, offset, NORM);
    }
}

// =========BRC_B8(fp4x2_e2m1/fp4x2_e1m2/u8/s8/f8_e4m3/fp8_e5m2/fp8_e8m0)=========
__simd_callee__ inline void asc_loadalign_brc_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, BRC_B8);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, BRC_B8);
    }
}

// =========BRC_B16(u16/s16/bf16/half)=========
__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B16);
    }
}

// =========BRC_B32(u32/s32/float)=========
__simd_callee__ inline void asc_loadalign_brc_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BRC_B32);
    }
}

// =========US(B8/B16)=========
__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, US_B8);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, US_B16);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_impl(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        pld(dst, src, offset, US);
    }
}

// =========DS(B8/B16)=========
__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, DS_B8);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, DS_B16);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_impl(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        pld(dst, src, offset, DS);
    }
}

// DINTLV(B8/B16/B32)
__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst0, dst1, src_tmp, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst0, dst1, src_tmp, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst0), reinterpret_cast<vector_fp4x2_e1m2_t&>(dst1), src_tmp, offset,
            DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst0, dst1, src_tmp, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst0), reinterpret_cast<vector_uint8_t&>(dst1),
            reinterpret_cast<__ubuf__ uint8_t*>(src), offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst0, dst1, src_tmp, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst0, dst1, src_tmp, offset, DINTLV_B8);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_half& dst0, vector_half& dst1, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B16);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B32);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B32);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_impl(
    vector_float& dst0, vector_float& dst1, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst0, dst1, src, offset, DINTLV_B32);
    }
}

// unpack(B8/B16/B32)
__simd_callee__ inline void asc_loadalign_unpack_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, UNPK_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B16);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B32);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK_B32);
    }
}

// unpack v2
__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, UNPK4_B8);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_v2_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, UNPK4_B8);
    }
}

// brc_v2
__simd_callee__ inline void asc_loadalign_brc_v2_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(dst, src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vld(dst, src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vld(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vld(dst, src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vld(dst, src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vld(dst, src_tmp, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v2_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, BLK);
    }
}

/// brc_v3(B16/B32)
__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B16);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B32);
    }
}

__simd_callee__ inline void asc_loadalign_brc_v3_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vld(dst, src, offset, E2B_B32);
    }
}

// asc_loadalign_brc_elem_impl forwards to asc_loadalign_brc_impl
__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

// asc_loadalign_unpack4_impl forwards to asc_loadalign_unpack_v2_impl
__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

// asc_loadalign_brc_datablock_impl forwards to asc_loadalign_brc_v2_impl
__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(
    vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_impl(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

// asc_loadalign_brc_elem2datablock_impl forwards to asc_loadalign_brc_v3_impl
__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_impl(
    vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

#endif
