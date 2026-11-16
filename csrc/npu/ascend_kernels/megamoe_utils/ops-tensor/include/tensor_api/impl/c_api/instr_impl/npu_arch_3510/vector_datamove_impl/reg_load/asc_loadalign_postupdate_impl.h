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
 * \file asc_loadalign_postupdate_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_POSTUPDATE_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADALIGN_POSTUPDATE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// postupdate, norm
__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_int64_t& dst, __ubuf__ int64_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_uint64_t& dst, __ubuf__ uint64_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        plds(dst, src, offset, NORM, POST_UPDATE);
    }
}

// postupdate, brc
__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, BRC_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_impl(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BRC_B32, POST_UPDATE);
    }
}

// postupdate, upsample
__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, US_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, US_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate_impl(
    vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        plds(dst, src, offset, US, POST_UPDATE);
    }
}

// postupdate, downsample
__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, DS_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, DS_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate_impl(
    vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        plds(dst, src, offset, DS, POST_UPDATE);
    }
}

// postupdate, unpack
__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, UNPK_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK_B32, POST_UPDATE);
    }
}

// postupdate, unpack_v2
__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate_v2_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, UNPK4_B8, POST_UPDATE);
    }
}

// postupdate, brc_v2
__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst, src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst, src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst, src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst, src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst, src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(reinterpret_cast<vector_uint8_t&>(dst), src_tmp, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v2_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, BLK, POST_UPDATE);
    }
}

// postupdate, brc_v3
__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_brc_postupdate_v3_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst, src, offset, E2B_B32, POST_UPDATE);
    }
}

// postupdate, deintlv
__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vlds(dst0, dst1, src_tmp, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(dst0, dst1, src_tmp, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vlds(
            reinterpret_cast<vector_fp4x2_e1m2_t&>(dst0), reinterpret_cast<vector_fp4x2_e1m2_t&>(dst1), src_tmp, offset,
            DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vlds(dst0, dst1, src_tmp, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vlds(dst0, dst1, src_tmp, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vlds(dst0, dst1, src_tmp, offset, DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t*& src_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(src);
        vlds(
            reinterpret_cast<vector_uint8_t&>(dst0), reinterpret_cast<vector_uint8_t&>(dst1), src_tmp, offset,
            DINTLV_B8, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_half& dst0, vector_half& dst1, __ubuf__ half*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B16, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate_impl(
    vector_float& dst0, vector_float& dst1, __ubuf__ float*& src, int32_t offset)
{
    if ASC_IS_AIV {
        vlds(dst0, dst1, src, offset, DINTLV_B32, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_int64_t& dst, __ubuf__ int64_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(src);
        vsldb(dst, src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(
            reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*&>(src),
            (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(src);
        vsldb(dst, src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t*& src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(src);
        vsldb(dst, src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(src);
        vsldb(dst, src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vsldb(
            reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask,
            POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t*& src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*&>(src);
        vsldb(dst, src_tmp, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_loadalign_postupdate_impl(
    vector_float& dst, __ubuf__ float*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        vsldb(dst, src, (block_stride << 16u | repeat_stride & 0xFFFFU), mask, POST_UPDATE);
    }
}

// asc_loadalign_brc_elem_postupdate_impl forwards to asc_loadalign_brc_postupdate_impl
__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

// asc_loadalign_unpack4_postupdate_impl forwards to asc_loadalign_unpack_postupdate_v2_impl
__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

// asc_loadalign_brc_datablock_postupdate_impl forwards to asc_loadalign_brc_postupdate_v2_impl
__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

// asc_loadalign_brc_elem2datablock_postupdate_impl forwards to asc_loadalign_brc_postupdate_v3_impl
__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate_impl(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
