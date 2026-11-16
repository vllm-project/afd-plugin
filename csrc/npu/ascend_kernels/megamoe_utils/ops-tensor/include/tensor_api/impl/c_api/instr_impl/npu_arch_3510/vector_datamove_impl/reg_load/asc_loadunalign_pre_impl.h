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
 * \file asc_loadunalign_pre_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADUNALIGN_PRE_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOADUNALIGN_PRE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int8_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint8_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int4b_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ fp8_e8m0_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ fp8_e5m2_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vldas(dst, src_tmp);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ hifloat8_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, reinterpret_cast<__ubuf__ uint8_t*>(src));
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int16_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint16_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ half* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ bfloat16_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int32_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint32_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ float* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int64_t* src)
{
    if ASC_IS_AIV {
        vldas(dst, src);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* src_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* src_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* src_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* src_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(src);
        vlda(dst, src_tmp, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, reinterpret_cast<__ubuf__ uint8_t*>(src), offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ half* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(
    vector_load_unalign& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ float* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

__simd_callee__ inline void asc_loadunalign_pre_impl(vector_load_unalign& dst, __ubuf__ int64_t* src, addr_reg offset)
{
    if ASC_IS_AIV {
        vlda(dst, src, offset);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
