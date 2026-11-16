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
 * \file asc_load_impl.h
 * \brief
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOAD_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_REG_LOAD_ASC_LOAD_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_load_impl(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_half& dst, __ubuf__ half* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_float& dst, __ubuf__ float* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_int64_t& dst, __ubuf__ int64_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

__simd_callee__ inline void asc_load_impl(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    if ASC_IS_AIV {
        vector_load_unalign ureg;
        asc_loadunalign_pre_impl(ureg, src);
        asc_loadunalign_impl(dst, ureg, src);
    }
}

#endif
