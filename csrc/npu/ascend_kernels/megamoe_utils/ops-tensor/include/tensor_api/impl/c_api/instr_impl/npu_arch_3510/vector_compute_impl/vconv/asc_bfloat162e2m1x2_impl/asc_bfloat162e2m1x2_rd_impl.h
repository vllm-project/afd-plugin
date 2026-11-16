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

#ifndef IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCON_ASC_BFLOAT162E2M1X2_IMPL_ASC_BFLOAT162E2M1X2_RD_IMPL_H
#define IMPL_CAPI_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCON_ASC_BFLOAT162E2M1X2_IMPL_ASC_BFLOAT162E2M1X2_RD_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P0, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v2_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P1, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v3_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P2, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_v4_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P3, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask,
    std::integral_constant<asc_position_quarter_mode, asc_position_quarter_mode::DISPERSE_FIRST_QUARTER>
        dst_quarter_pos)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P0, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask,
    std::integral_constant<asc_position_quarter_mode, asc_position_quarter_mode::DISPERSE_SECOND_QUARTER>
        dst_quarter_pos)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P1, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask,
    std::integral_constant<asc_position_quarter_mode, asc_position_quarter_mode::DISPERSE_THIRD_QUARTER>
        dst_quarter_pos)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P2, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_bfloat162e2m1x2_rd_impl(
    vector_fp4x2_e2m1_t& dst, vector_bfloat16_t src, vector_bool mask,
    std::integral_constant<asc_position_quarter_mode, asc_position_quarter_mode::DISPERSE_FOURTH_QUARTER>
        dst_quarter_pos)
{
    if ASC_IS_AIV {
        vcvt(dst, src, mask, ROUND_F, PART_P3, MODE_ZEROING);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
