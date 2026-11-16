/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#error "reg_store.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H
#endif

#ifndef INCLUDE_C_API_REG_COMPUTE_REG_STORE_H
#define INCLUDE_C_API_REG_COMPUTE_REG_STORE_H
#include "impl/utils/common_types.h"
#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl/utils_type.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl.h"
#endif

//==============asc_scatter=============
__simd_callee__ inline void asc_scatter(
    __ubuf__ int8_t* dst, vector_int8_t src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint8_t* dst, vector_uint8_t src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ int16_t* dst, vector_int16_t src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint16_t* dst, vector_uint16_t src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ int32_t* dst, vector_int32_t src, vector_uint32_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint32_t* dst, vector_uint32_t src, vector_uint32_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(
    __ubuf__ bfloat16_t* dst, vector_bfloat16_t src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(__ubuf__ half* dst, vector_half src, vector_uint16_t index, vector_bool mask);

__simd_callee__ inline void asc_scatter(__ubuf__ float* dst, vector_float src, vector_uint32_t index, vector_bool mask);

// ==========asc_store=========

// ==========偏移固定传入0，由用户自行更新src地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(__ubuf__ int8_t* dst_align32b, vector_int8_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, vector_bool mask);

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src);

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(__ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ int64_t* dst_align32b, vector_int64_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint64_t* dst_align32b, vector_uint64_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask);

// ==========asc_storealign_pack_v2(float/u32/s32)=========
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask);

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_storealign_intlv(__ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1);

__simd_callee__ inline void asc_storealign_intlv(__ubuf__ half* dst_align32b, vector_half src0, vector_half src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1);

//===========使用vsts通过int32_t传入偏移，用户可以选择更新偏移或者更新src地址
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset, vector_bool mask);

// storealign_1st
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset);

// storealign_pack
__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask);

// storealign_pack_v2
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask);

// storealign_intlv
__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, int32_t offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, int32_t offset);

//===========使用vsts通过int32_t传入偏移，增加PostMode可以设置成硬件自动Post Update
__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask);

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset);

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset);

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int64_t*& dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint64_t*& dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask);

// storealign_pack_v2
[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. Please use "
             "asc_storealign_pack_quarter_postupdate instead.")]]
__simd_callee__ inline void asc_storealign_pack_postupdate_v2(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. Please use "
             "asc_storealign_pack_quarter_postupdate instead.")]]
__simd_callee__ inline void asc_storealign_pack_postupdate_v2(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. Please use "
             "asc_storealign_pack_quarter_postupdate instead.")]]
__simd_callee__ inline void asc_storealign_pack_postupdate_v2(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask);

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新目的操作数的地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset, vector_bool mask);

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset);

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset);

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask);

// ==========asc_storealign_pack_v2(float/u32/s32)=========
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask);

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. Please use asc_storealign_pack_quarter instead.")]]
__simd_callee__ inline void asc_storealign_pack_v2(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask);

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, addr_reg offset);

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, addr_reg offset);

// ==========asc_storeunalign(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count);

// ==========使用uint32_t作为存储偏移量，每调用一次接口更新目的操作数在UB上的地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t*& dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t*& dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t*& dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t*& dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half*& dst, vector_store_unalign& src0, vector_half src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float*& dst, vector_store_unalign& src0, vector_float src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t*& dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ hifloat8_t*& dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t*& dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t*& dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t*& dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign& src0, vector_bool src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign& src0, vector_bool src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign& src0, vector_bool src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1);

// ==========使用addr_reg作为存储偏移量，每调用一次接口更新目的操作数在UB上的地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, addr_reg& count);

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, addr_reg& count);

// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/f8e4m3/f8e5m2/f8e8m0)=========
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

// ==========asc_storealign_postupdate(u8/s8/half/u16/s16/float/u32/s32/bf16/f8e4m3/f8e5m2/f8e8m0)=========
__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ hifloat8_t*& dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e5m2_t*& dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e8m0_t*& dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask);

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ hifloat8_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src, int32_t offset);

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src);

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ hifloat8_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src, addr_reg offset);

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src, addr_reg offset);

//===========asc_storeunalign_post_postupdate(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int8_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int4b_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e8m0_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e5m2_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ hifloat8_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int16_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ half*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ bfloat16_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int32_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ float*& dst, vector_store_unalign src, int32_t offset);

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int64_t*& dst, vector_store_unalign src, int32_t offset);

// ===========asc_store(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_store(__ubuf__ int8_t* dst, vector_int8_t src);

__simd_callee__ inline void asc_store(__ubuf__ uint8_t* dst, vector_uint8_t src);

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src);

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src);

__simd_callee__ inline void asc_store(__ubuf__ int4b_t* dst, vector_int4x2_t src);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e8m0_t* dst, vector_fp8_e8m0_t src);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e5m2_t* dst, vector_fp8_e5m2_t src);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src);

__simd_callee__ inline void asc_store(__ubuf__ hifloat8_t* dst, vector_hifloat8_t src);

__simd_callee__ inline void asc_store(__ubuf__ int16_t* dst, vector_int16_t src);

__simd_callee__ inline void asc_store(__ubuf__ uint16_t* dst, vector_uint16_t src);

__simd_callee__ inline void asc_store(__ubuf__ half* dst, vector_half src);

__simd_callee__ inline void asc_store(__ubuf__ bfloat16_t* dst, vector_bfloat16_t src);

__simd_callee__ inline void asc_store(__ubuf__ int32_t* dst, vector_int32_t src);

__simd_callee__ inline void asc_store(__ubuf__ uint32_t* dst, vector_uint32_t src);

__simd_callee__ inline void asc_store(__ubuf__ float* dst, vector_float src);

__simd_callee__ inline void asc_store(__ubuf__ int64_t* dst, vector_int64_t src);

// ==========asc_store(with
// count)(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)============
__simd_callee__ inline void asc_store(__ubuf__ int8_t* dst, vector_int8_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ uint8_t* dst, vector_uint8_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ int16_t* dst, vector_int16_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ uint16_t* dst, vector_uint16_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ int32_t* dst, vector_int32_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ uint32_t* dst, vector_uint32_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ float* dst, vector_float src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ int64_t* dst, vector_int64_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ bfloat16_t* dst, vector_bfloat16_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e8m0_t* dst, vector_fp8_e8m0_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e5m2_t* dst, vector_fp8_e5m2_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ hifloat8_t* dst, vector_hifloat8_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src, uint32_t count);

__simd_callee__ inline void asc_store(__ubuf__ int4b_t* dst, vector_int4x2_t src, uint32_t count);

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset);
__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset);
__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset);

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src);
__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src);
__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src);

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset);

__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint32_t*& dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint32_t*& dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint16_t*& dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint16_t*& dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset);
__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset);

// ==========asc_storealign_pack_quarter==========
__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask);

// ==========asc_storealign_pack_quarter_postupdate==========
__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask);

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask);

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H

#endif
