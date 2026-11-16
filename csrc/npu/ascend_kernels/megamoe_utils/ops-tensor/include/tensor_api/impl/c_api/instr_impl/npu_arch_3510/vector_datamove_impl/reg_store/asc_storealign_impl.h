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
 * \file asc_storealign_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_STORE_ASC_STOREALIGN_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_STORE_ASC_STOREALIGN_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// ==========偏移固定传入0，由用户自行更新目的操作数的地址=========
// storealign
__simd_callee__ inline void asc_storealign_impl(__ubuf__ int8_t* dst_align32b, vector_int8_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, 0, NORM_B8, mask);
    }
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int8_t* dst_align32b, vector_int8_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int16_t* dst_align32b, vector_int16_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int32_t* dst_align32b, vector_int32_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ half* dst_align32b, vector_half src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ float* dst_align32b, vector_float src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, 0, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, 0, ONEPT_B8, mask);
    }
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK_B16, mask);
    }
}

// storealign_pack_v2
__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, 0, PK4_B32, mask);
    }
}

// storealign_intlv
__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(__ubuf__ half* dst_align32b, vector_half src0, vector_half src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, 0, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(
            *reinterpret_cast<vector_uint8_t*>(&src0), *reinterpret_cast<vector_uint8_t*>(&src1), dst_align32b_tmp, 0,
            INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(
            *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src0), *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src1),
            dst_align32b_tmp, 0, INTLV_B8, mask);
    }
}

//===========使用vsts通过int32_t传入偏移，用户可以选择更新偏移或者更新src地址
__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ half* dst_align32b, vector_half src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ float* dst_align32b, vector_float src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask);
    }
}

// storealign_pack_v2
__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask);
    }
}

// storealign_intlv
__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsts(
            *reinterpret_cast<vector_uint8_t*>(&src0), *reinterpret_cast<vector_uint8_t*>(&src1), dst_align32b_tmp,
            offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vsts(
            *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src0), *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src1),
            dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

//===========使用vsts通过int32_t传入偏移，增加PostMode可以设置成硬件自动Post Update
__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B8, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B8, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, NORM_B16, mask, POST_UPDATE);
    }
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B8, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B8, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_1st_postupdate_impl(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vsts(src, dst_align32b, offset, ONEPT_B16, mask, POST_UPDATE);
    }
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ int64_t*& dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B64, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ uint64_t*& dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B64, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK_B16, mask, POST_UPDATE);
    }
}

// storealign_pack_v2
__simd_callee__ inline void asc_storealign_pack_postupdate_v2_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_v2_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_v2_impl(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vsts(src, dst_align32b, offset, PK4_B32, mask, POST_UPDATE);
    }
}

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新目的操作数的地址=========
// storealign
__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, NORM_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, offset, NORM_B8, mask);
    }
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ half* dst_align32b, vector_half src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(__ubuf__ float* dst_align32b, vector_float src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src, dst_align32b, offset, ONEPT_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(src, dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_1st_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src), dst_align32b_tmp, offset, ONEPT_B8, mask);
    }
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B64, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK_B16, mask);
    }
}

// storealign_pack_v2
__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK4_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_pack_v2_impl(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    if ASC_IS_AIV {
        vst(src, dst_align32b, offset, PK4_B32, mask);
    }
}

// storealign_intlv
__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B32, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        vst(src0, src1, dst_align32b, offset, INTLV_B16, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_uint8_t*>(&src0), *reinterpret_cast<vector_uint8_t*>(&src1), dst_align32b_tmp,
            offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vst(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vst(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vst(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e2m1x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst_align32b);
        vst(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(src0, src1, dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

__simd_callee__ inline void asc_storealign_intlv_impl(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, addr_reg offset)
{
    if ASC_IS_AIV {
        vector_bool mask;
        __ubuf__ float4_e1m2x2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst_align32b);
        vst(*reinterpret_cast<vector_fp4x2_e1m2_t*>(&src0), *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src1),
            dst_align32b_tmp, offset, INTLV_B8, mask);
    }
}

union asc_store_align_config {
    int32_t config;
    struct {
        uint32_t repeat_stride : 16;
        uint32_t block_stride : 16;
    };
};

// ==========asc_storealign_impl(int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/fp8_e4m3fn_t/fp8_e5m2_t/fp8_e8m0_t/bfloat16_t/half/float)==========
__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ uint8_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*>(dst_align32b);
        vsstb(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e4m3_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e5m2_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e8m0_t* dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ half* dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

__simd_callee__ inline void asc_storealign_impl(
    __ubuf__ float* dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask);
    }
}

union asc_store_align_config_post {
    int32_t config;
    struct {
        uint32_t repeat_stride : 16;
        uint32_t block_stride : 16;
    };
};

// ==========asc_storealign_postupdate_impl(int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/fp8_e4m3fn_t/fp8_e5m2_t/fp8_e8m0_t/bfloat16_t/half/float)==========
__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ hifloat8_t*& dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ uint8_t*& dst_align32b_tmp = reinterpret_cast<__ubuf__ uint8_t*&>(dst_align32b);
        vsstb(*reinterpret_cast<vector_uint8_t*>(&src), dst_align32b_tmp, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ fp8_e4m3fn_t*& dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e4m3_t*& dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*&>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ fp8_e5m2_t*& dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e5m2_t*& dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*&>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ fp8_e8m0_t*& dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        __ubuf__ float8_e8m0_t*& dst_align32b_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*&>(dst_align32b);
        vsstb(src, dst_align32b_tmp, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ half*& dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(
    __ubuf__ float*& dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    if ASC_IS_AIV {
        asc_store_align_config_post config;
        config.block_stride = block_stride;
        config.repeat_stride = repeat_stride;
        vsstb(src, dst_align32b, config.config, mask, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, dst, offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, dst, offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset)
{
    if ASC_IS_AIV {
        pst(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint32_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, dst, 0, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint32_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, dst, 0, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint16_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), 0, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint16_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), 0, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint8_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), 0, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint8_t* dst, vector_bool src)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), 0, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, dst, offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, dst, offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_impl(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, NORM);
    }
}

__simd_callee__ inline void asc_storealign_pack_impl(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, reinterpret_cast<__ubuf__ uint32_t*>(dst), offset, PK);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(__ubuf__ uint32_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, dst, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ uint32_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        psts(src, dst, offset, PK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(__ubuf__ uint16_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint32_t*& temp_dst = reinterpret_cast<__ubuf__ uint32_t*&>(dst);
        psts(src, temp_dst, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(
    __ubuf__ uint16_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint32_t*& temp_dst = reinterpret_cast<__ubuf__ uint32_t*&>(dst);
        psts(src, temp_dst, offset, PK, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_postupdate_impl(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint32_t*& temp_dst = reinterpret_cast<__ubuf__ uint32_t*&>(dst);
        psts(src, temp_dst, offset, NORM, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storealign_pack_postupdate_impl(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ uint32_t*& temp_dst = reinterpret_cast<__ubuf__ uint32_t*&>(dst);
        psts(src, temp_dst, offset, PK, POST_UPDATE);
    }
}

// asc_storealign_pack_quarter_impl forwards to asc_storealign_pack_v2_impl
__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_impl(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

// asc_storealign_pack_quarter_postupdate_impl forwards to asc_storealign_pack_postupdate_v2_impl
__simd_callee__ inline void asc_storealign_pack_quarter_postupdate_impl(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate_impl(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate_impl(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

#endif
