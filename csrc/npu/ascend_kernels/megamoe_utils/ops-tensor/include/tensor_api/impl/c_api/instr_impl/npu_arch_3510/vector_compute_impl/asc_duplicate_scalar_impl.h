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
 * \file asc_duplicate_scalar_impl.h
 * \brief
 */

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_DUPLICATE_SCALAR_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_DUPLICATE_SCALAR_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// vdup
__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint8_t& dst, uint8_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int8_t& dst, int8_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_fp8_e4m3fn_t& dst, fp8_e4m3fn_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        float8_e4m3_t* value_tmp = reinterpret_cast<float8_e4m3_t*>(&value);
        vdup(dst, *value_tmp, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_fp8_e5m2_t& dst, fp8_e5m2_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        float8_e5m2_t* value_tmp = reinterpret_cast<float8_e5m2_t*>(&value);
        vdup(dst, *value_tmp, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint16_t& dst, uint16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int16_t& dst, int16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_half& dst, half value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_bfloat16_t& dst, bfloat16_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint32_t& dst, uint32_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int32_t& dst, int32_t value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_float& dst, float value, vector_bool mask)
{
    if ASC_IS_AIV {
        vdup(dst, value, mask, MODE_ZEROING);
    }
}

// vbr
__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint8_t& dst, uint8_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int8_t& dst, int8_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint16_t& dst, uint16_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int16_t& dst, int16_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_uint32_t& dst, uint32_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_int32_t& dst, int32_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_half& dst, half value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_float& dst, float value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_bfloat16_t& dst, bfloat16_t value)
{
    if ASC_IS_AIV {
        vbr(dst, value);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_fp8_e4m3fn_t& dst, fp8_e4m3fn_t value)
{
    if ASC_IS_AIV {
        float8_e4m3_t* value_tmp = reinterpret_cast<float8_e4m3_t*>(&value);
        vbr(dst, *value_tmp);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_fp8_e5m2_t& dst, fp8_e5m2_t value)
{
    if ASC_IS_AIV {
        float8_e5m2_t* value_tmp = reinterpret_cast<float8_e5m2_t*>(&value);
        vbr(dst, *value_tmp);
    }
}

__simd_callee__ inline void asc_duplicate_scalar_impl(vector_fp8_e8m0_t& dst, fp8_e8m0_t value)
{
    if ASC_IS_AIV {
        float8_e8m0_t* value_tmp = reinterpret_cast<float8_e8m0_t*>(&value);
        vbr(dst, *value_tmp);
    }
}

#endif
