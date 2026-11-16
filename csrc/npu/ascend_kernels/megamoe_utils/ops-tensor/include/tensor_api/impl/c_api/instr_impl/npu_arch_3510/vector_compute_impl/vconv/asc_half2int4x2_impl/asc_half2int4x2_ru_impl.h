/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCONV_ASC_HALF2INT4X2_IMPL_ASC_HALF2INT4X2_RU_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCONV_ASC_HALF2INT4X2_IMPL_ASC_HALF2INT4X2_RU_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// ru - sat:dis/en - v1 v2 v3 v4
__simd_callee__ inline void asc_half2int4x2_ru_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_DISABLE, PART_P0, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_sat_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_ENABLE, PART_P0, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_v2_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_DISABLE, PART_P1, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_sat_v2_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_ENABLE, PART_P1, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_v3_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_DISABLE, PART_P2, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_sat_v3_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_ENABLE, PART_P2, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_v4_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_DISABLE, PART_P3, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_half2int4x2_ru_sat_v4_impl(vector_int4x2_t& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_f162s4(dst, src, mask, ROUND_C, RS_ENABLE, PART_P3, MODE_ZEROING);
    }
}

#endif
