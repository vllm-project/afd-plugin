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
 * \file asc_deintlv_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_DEINTLV_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_DEINTLV_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_deintlv_impl(
    vector_int32_t& dst0, vector_int32_t& dst1, vector_int32_t src0, vector_int32_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_uint32_t& dst0, vector_uint32_t& dst1, vector_uint32_t src0, vector_uint32_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_int16_t& dst0, vector_int16_t& dst1, vector_int16_t src0, vector_int16_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_uint16_t& dst0, vector_uint16_t& dst1, vector_uint16_t src0, vector_uint16_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_int8_t& dst0, vector_int8_t& dst1, vector_int8_t src0, vector_int8_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_uint8_t& dst0, vector_uint8_t& dst1, vector_uint8_t src0, vector_uint8_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, vector_hifloat8_t src0, vector_hifloat8_t src1)
{
    if ASC_IS_AIV {
        vdintlv(
            reinterpret_cast<vector_uint8_t&>(dst0), reinterpret_cast<vector_uint8_t&>(dst1),
            *reinterpret_cast<vector_uint8_t*>(&src0), *reinterpret_cast<vector_uint8_t*>(&src1));
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, vector_bfloat16_t src0, vector_bfloat16_t src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(
    vector_float& dst0, vector_float& dst1, vector_float src0, vector_float src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_impl(vector_half& dst0, vector_half& dst1, vector_half src0, vector_half src1)
{
    if ASC_IS_AIV {
        vdintlv(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_b8_impl(
    vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
{
    if ASC_IS_AIV {
        pdintlv_b8(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_b16_impl(
    vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
{
    if ASC_IS_AIV {
        pdintlv_b16(dst0, dst1, src0, src1);
    }
}

__simd_callee__ inline void asc_deintlv_b32_impl(
    vector_bool& dst0, vector_bool& dst1, vector_bool src0, vector_bool src1)
{
    if ASC_IS_AIV {
        pdintlv_b32(dst0, dst1, src0, src1);
    }
}

#endif
