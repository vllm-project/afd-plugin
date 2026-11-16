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
 * \file asc_max_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_VECTOR_COMPUTE_IMPL_NPU_ARCH_3510_ASC_MAX_IMPL_H
#define IMPL_C_API_INSTR_VECTOR_COMPUTE_IMPL_NPU_ARCH_3510_ASC_MAX_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// asc_max uint8_t
__simd_callee__ inline void asc_max_impl(
    vector_uint8_t& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max int8_t
__simd_callee__ inline void asc_max_impl(vector_int8_t& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max uint16_t
__simd_callee__ inline void asc_max_impl(
    vector_uint16_t& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max int16_t
__simd_callee__ inline void asc_max_impl(
    vector_int16_t& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max int32_t
__simd_callee__ inline void asc_max_impl(
    vector_int32_t& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max uint32_t
__simd_callee__ inline void asc_max_impl(
    vector_uint32_t& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max float
__simd_callee__ inline void asc_max_impl(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max half
__simd_callee__ inline void asc_max_impl(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

// asc_max bfloat16
__simd_callee__ inline void asc_max_impl(
    vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vmax(dst, src0, src1, mask, MODE_ZEROING);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
