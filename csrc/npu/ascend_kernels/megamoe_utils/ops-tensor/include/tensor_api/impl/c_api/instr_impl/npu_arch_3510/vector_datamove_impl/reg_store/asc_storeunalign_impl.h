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
 * \file asc_storeunalign_impl.h
 * \brief
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_STORE_ASC_STOREUNALIGN_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_STORE_ASC_STOREUNALIGN_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, src1, dst);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst);
        vstus(src0, count, src1, dst_tmp);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        vstus(src0, count, *reinterpret_cast<vector_uint8_t*>(&src1), reinterpret_cast<__ubuf__ uint8_t*&>(dst));
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst);
        vstus(src0, count, src1, dst_tmp);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst);
        vstus(src0, count, src1, dst_tmp);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* dst_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst);
        vstus(src0, count, src1, dst_tmp);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst);
        vstus(src0, count, src1, dst_tmp);
    }
}

__simd_callee__ inline void asc_storeunalign_impl(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst);
        vstus(src0, count, *reinterpret_cast<vector_fp4x2_e1m2_t*>(&src1), dst_tmp);
    }
}

#endif
