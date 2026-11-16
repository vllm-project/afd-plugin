/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_LT_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_LT_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_uint8_t src0, vector_uint8_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_int8_t src0, vector_int8_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_half src0, vector_half src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_uint16_t src0, vector_uint16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_int16_t src0, vector_int16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_float src0, vector_float src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_uint32_t src0, vector_uint32_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(vector_bool& dst, vector_int32_t src0, vector_int32_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

__simd_callee__ inline void asc_lt_impl(
    vector_bool& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
{
    if ASC_IS_AIV {
        vcmp_lt(dst, src0, src1, mask);
    }
}

#endif
