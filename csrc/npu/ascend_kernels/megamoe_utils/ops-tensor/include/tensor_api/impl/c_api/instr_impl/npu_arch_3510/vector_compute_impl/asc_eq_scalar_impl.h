/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_SCALAR_EQ_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_SCALAR_EQ_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_uint8_t src0, uint8_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_int8_t src0, int8_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(
    vector_bool& dst, vector_bfloat16_t src0, bfloat16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_uint16_t src0, uint16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_int16_t src0, int16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_half src0, half value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_uint32_t src0, uint32_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_int32_t src0, int32_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

__simd_callee__ inline void asc_eq_scalar_impl(vector_bool& dst, vector_float src0, float value, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmps_eq(dst, src0, value, mask);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
