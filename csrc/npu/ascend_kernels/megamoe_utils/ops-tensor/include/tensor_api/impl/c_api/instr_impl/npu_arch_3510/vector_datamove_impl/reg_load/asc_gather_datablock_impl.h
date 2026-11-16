/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_gather_datablock_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_VECTOR_DATAMOVE_IMPL_NPU_ARCH_3510_ASC_GATHER_DATABLOCK_IMPL_H
#define IMPL_C_API_INSTR_IMPL_VECTOR_DATAMOVE_IMPL_NPU_ARCH_3510_ASC_GATHER_DATABLOCK_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// ==========有mask参数=========
__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint64_t& dst, __ubuf__ uint64_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vgatherb(dst, src_tmp, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t* src_tmp = reinterpret_cast<__ubuf__ uint8_t*>(src);
        vgatherb(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vgatherb(dst, src_tmp, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vgatherb(dst, src_tmp, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vgatherb(dst, src_tmp, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vgatherb(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, index, mask);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vgatherb(dst, src_tmp, index, mask);
    }
}
// ==========无mask参数=========
__simd_callee__ inline void asc_gather_datablock_impl(vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_half& dst, __ubuf__ half* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(vector_float& dst, __ubuf__ float* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(dst, src, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vgatherb(dst, src_tmp, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        vgatherb(reinterpret_cast<vector_uint8_t&>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vgatherb(dst, src_tmp, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vgatherb(dst, src_tmp, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vgatherb(dst, src_tmp, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vgatherb(reinterpret_cast<vector_fp4x2_e1m2_t&>(dst), src_tmp, index);
    }
}

__simd_callee__ inline void asc_gather_datablock_impl(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vgatherb(dst, src_tmp, index);
    }
}

#endif
