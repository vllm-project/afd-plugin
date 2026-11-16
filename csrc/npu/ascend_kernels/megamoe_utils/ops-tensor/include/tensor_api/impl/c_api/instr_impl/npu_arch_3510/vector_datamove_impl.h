/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_get_mask_spr_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadalign_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadalign_v2_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadalign_postupdate_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadunalign_pre_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_storealign_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_storeunalign_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_storeunalign_postupdate_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_gather_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_gather_datablock_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ub2gm_loop_size_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ub2gm_loop1_stride_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ub2gm_loop2_stride_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadunalign_postupdate_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_loadunalign_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_load/asc_load_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_loop_size_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_loop1_stride_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_loop2_stride_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ndim_loop_stride_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_storeunalign_post_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_storeunalign_post_postupdate_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_ndim_copy_gm2ub_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ndim_pad_count_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_ndim_pad_value_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_copy_pad_val_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_store_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/reg_store/asc_scatter_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_scatter_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_gm2ub_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2gm_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_gm2ub_align_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2gm_align_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2ub_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_ndim_copy_dci_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_set_gm2ub_pad_impl.h"

// ==========偏移固定传入0，由用户自行更新目的操作数的地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(__ubuf__ int8_t* dst_align32b, vector_int8_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, mask);
}

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src)
{
    asc_storealign_1st_impl(dst_align32b, src);
}

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(__ubuf__ int16_t* dst_align32b, vector_int16_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ int64_t* dst_align32b, vector_int64_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint64_t* dst_align32b, vector_uint64_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ half* dst_align32b, vector_half src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, mask);
}

// ==========asc_storealign_pack_v2(float/u32/s32)=========
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, mask);
}

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_storealign_intlv(__ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(__ubuf__ half* dst_align32b, vector_half src0, vector_half src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1);
}

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新目的操作数的地址=========
// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

// ==========asc_storealign_1st(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, addr_reg offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

// ==========asc_storealign_pack(half/u16/s16/float/u32/s32/u64/s64/bf16)=========
__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ half* dst_align32b, vector_half src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

// ==========asc_storealign_pack_v2(float/u32/s32)=========
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

// ==========asc_storealign_intlv(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, addr_reg offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

//===========使用vsts通过int32_t传入偏移，用户可以选择更新偏移或者更新src地址
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, offset, mask);
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st(__ubuf__ int8_t* dst_align32b, vector_int8_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint8_t* dst_align32b, vector_uint8_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ half* dst_align32b, vector_half src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ float* dst_align32b, vector_float src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st(__ubuf__ int4b_t* dst_align32b, vector_int4x2_t src, int32_t offset)
{
    asc_storealign_1st_impl(dst_align32b, src, offset);
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ int64_t* dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ uint64_t* dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ half* dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_impl(dst_align32b, src, offset, mask);
}

// storealign_pack_v2
[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter instead.")]] __simd_callee__ inline void
asc_storealign_pack_v2(__ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_v2_impl(dst_align32b, src, offset, mask);
}

// storealign_intlv
__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src0, vector_int8_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src0, vector_uint8_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src0, vector_int16_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src0, vector_uint16_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src0, vector_int32_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src0, vector_uint32_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ half* dst_align32b, vector_half src0, vector_half src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src0, vector_bfloat16_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src0, vector_fp8_e4m3fn_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src0, vector_hifloat8_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src0, vector_fp8_e5m2_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src0, vector_fp8_e8m0_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e2m1_t* dst_align32b, vector_fp4x2_e2m1_t src0, vector_fp4x2_e2m1_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ fp4x2_e1m2_t* dst_align32b, vector_fp4x2_e1m2_t src0, vector_fp4x2_e1m2_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

__simd_callee__ inline void asc_storealign_intlv(
    __ubuf__ int4b_t* dst_align32b, vector_int4x2_t src0, vector_int4x2_t src1, int32_t offset)
{
    asc_storealign_intlv_impl(dst_align32b, src0, src1, offset);
}

//===========使用vsts通过int32_t传入偏移，增加PostMode可以设置成硬件自动Post Update
__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, offset, mask);
}

// storealign_1st
__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(__ubuf__ half*& dst_align32b, vector_half src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

__simd_callee__ inline void asc_storealign_1st_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset)
{
    asc_storealign_1st_postupdate_impl(dst_align32b, src, offset);
}

// storealign_pack
__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ int64_t*& dst_align32b, vector_int64_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ uint64_t*& dst_align32b, vector_uint64_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_impl(dst_align32b, src, offset, mask);
}

// storealign_pack_v2
[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter_postupdate instead.")]] __simd_callee__ inline void
asc_storealign_pack_postupdate_v2(__ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter_postupdate instead.")]] __simd_callee__ inline void
asc_storealign_pack_postupdate_v2(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

[[deprecated("NOTICE: asc_storealign_pack_postupdate_v2 is deprecated. "
             "Please use asc_storealign_pack_quarter_postupdate instead.")]] __simd_callee__ inline void
asc_storealign_pack_postupdate_v2(__ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_postupdate_v2_impl(dst_align32b, src, offset, mask);
}

// ==========asc_storeunalign(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count)
{
    asc_storeunalign_impl(dst, src0, src1, count);
}

// ==========偏移固定传入0，由用户自行更新目的操作数的地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t*& dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t*& dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t*& dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t*& dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half*& dst, vector_store_unalign& src0, vector_half src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float*& dst, vector_store_unalign& src0, vector_float src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t*& dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ hifloat8_t*& dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t*& dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t*& dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t*& dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign& src0, vector_bool src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign& src0, vector_bool src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign& src0, vector_bool src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1);
}

// ==========通过地址寄存器传入偏移，用户可以选择更新偏移或者更新目的操作数的地址=========
// ==========asc_storeunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e1m2/e2m1)=========
__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_storeunalign_postupdate(
    __ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, addr_reg& count)
{
    asc_storeunalign_postupdate_impl(dst, src0, src1, count);
}

// ==========源操作数为起初地址，目的数为寄存器=========
// ==========asc_gather(u8/s8/half/u16/s16/float/u32/s32/bf16/e4m3/e5m2/e8m0)=========
__simd_callee__ inline void asc_gather(
    vector_int16_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_uint16_t& dst, __ubuf__ uint8_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint16_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_impl(dst, src, index, mask);
}

// ==========源操作数和目的数都为寄存器=========
// ==========asc_gather(u8/s8/half/u16/s16/u32/s32/bf16/e4m3/e5m2/e8m0)=========
__simd_callee__ inline void asc_gather(vector_int8_t& dst, vector_int8_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_uint8_t& dst, vector_uint8_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_int16_t& dst, vector_int16_t src, vector_uint16_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_uint16_t& dst, vector_uint16_t src, vector_uint16_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_int32_t& dst, vector_int32_t src, vector_uint32_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_uint32_t& dst, vector_uint32_t src, vector_uint32_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_half& dst, vector_half src, vector_uint16_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_uint16_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_hifloat8_t& dst, vector_hifloat8_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_uint8_t index)
{
    asc_gather_impl(dst, src, index);
}

// ==========有mask参数=========
// ==========asc_gather_datablock(u8/s8/half/u16/s16/float/u32/s32/u64/s64/bf16/e4m3/e5m2/e2m1/e1m2/e8m0)=========
__simd_callee__ inline void asc_gather_datablock(
    vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_uint64_t& dst, __ubuf__ uint64_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index, vector_bool mask)
{
    asc_gather_datablock_impl(dst, src, index, mask);
}

// ==========无mask参数=========
// ==========asc_gather_datablock(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e2m1/e1m2/e8m0)=========
__simd_callee__ inline void asc_gather_datablock(vector_int8_t& dst, __ubuf__ int8_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_int64_t& dst, __ubuf__ int64_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_half& dst, __ubuf__ half* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_float& dst, __ubuf__ float* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

__simd_callee__ inline void asc_gather_datablock(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint32_t index)
{
    asc_gather_datablock_impl(dst, src, index);
}

// ==========asc_get_mask_spr==========
__simd_callee__ inline vector_bool asc_get_mask_spr_b16() { return asc_get_mask_spr_b16_impl(); }

__simd_callee__ inline vector_bool asc_get_mask_spr_b32() { return asc_get_mask_spr_b32_impl(); }

// ==========asc_loadalign(int8_t/uint8_t/fp4x2_e2m1_t/fp4x2_e1m2_t/fp8_e8m0_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t)==========
__simd_callee__ inline void asc_loadalign(vector_int8_t& dst, __ubuf__ int8_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_uint8_t& dst, __ubuf__ uint8_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_int4x2_t& dst, __ubuf__ int4b_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_int16_t& dst, __ubuf__ int16_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_half& dst, __ubuf__ half* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_int32_t& dst, __ubuf__ int32_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_float& dst, __ubuf__ float* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_int64_t& dst, __ubuf__ int64_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign(vector_uint64_t& dst, __ubuf__ uint64_t* src)
{
    asc_loadalign_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src) { asc_loadalign_impl(dst, src); }

__simd_callee__ inline void asc_loadalign_brc(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_upsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_downsample_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_unpack_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_unpack_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_v2_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_v3_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_half& dst0, vector_half& dst1, __ubuf__ half* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign_deintlv(vector_float& dst0, vector_float& dst1, __ubuf__ float* src)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src);
}

__simd_callee__ inline void asc_loadalign(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int64_t& dst, __ubuf__ int64_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint64_t& dst, __ubuf__ uint64_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_impl(dst, src, offset);
}

// =========asc_loadalign(addr_reg)(begin)============
__simd_callee__ inline void asc_loadalign(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_int64_t& dst, __ubuf__ int64_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_uint64_t& dst, __ubuf__ uint64_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

// =========BRC_B16(u16/s16/bf16/half)=========
__simd_callee__ inline void asc_loadalign_brc(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

// =========BRC_B32(u32/s32/float)=========
__simd_callee__ inline void asc_loadalign_brc(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

// =========US(B8/B16)=========
__simd_callee__ inline void asc_loadalign_upsample(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

// =========DS(B8/B16)=========
__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bool& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

// DINTLV(B8/B16/B32)
__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_half& dst0, vector_half& dst1, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_float& dst0, vector_float& dst1, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

// unpack(B8/B16/B32)
__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

// unpack v2
[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

// brc_v2
[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

/// brc_v3(B16/B32)
[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

// =========asc_loadalign(addr_reg)(end)============

__simd_callee__ inline void asc_loadalign_brc(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_upsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_downsample_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_unpack_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_v2 is deprecated. "
             "Please use asc_loadalign_unpack4 instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_unpack_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v2(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock instead.")]] __simd_callee__ inline void
asc_loadalign_brc_v3(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_half& dst0, vector_half& dst1, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv(
    vector_float& dst0, vector_float& dst1, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_deintlv_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_int64_t& dst, __ubuf__ int64_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_postupdate(vector_uint64_t& dst, __ubuf__ uint64_t*& src, int32_t offset)
{
    asc_loadalign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_postupdate(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_upsample_postupdate(vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_upsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_downsample_postupdate(
    vector_bool& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_downsample_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack_postupdate(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_unpack_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_unpack4_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_unpack_postupdate_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_unpack_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v2 is deprecated. "
             "Please use asc_loadalign_brc_datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v2(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v2_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

[[deprecated("NOTICE: asc_loadalign_brc_postupdate_v3 is deprecated. "
             "Please use asc_loadalign_brc_elem2datablock_postupdate instead.")]] __simd_callee__ inline void
asc_loadalign_brc_postupdate_v3(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_postupdate_v3_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_int8_t& dst0, vector_int8_t& dst1, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_uint8_t& dst0, vector_uint8_t& dst1, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_fp4x2_e2m1_t& dst0, vector_fp4x2_e2m1_t& dst1, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_fp4x2_e1m2_t& dst0, vector_fp4x2_e1m2_t& dst1, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_int4x2_t& dst0, vector_int4x2_t& dst1, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_fp8_e8m0_t& dst0, vector_fp8_e8m0_t& dst1, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_fp8_e5m2_t& dst0, vector_fp8_e5m2_t& dst1, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_fp8_e4m3fn_t& dst0, vector_fp8_e4m3fn_t& dst1, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_hifloat8_t& dst0, vector_hifloat8_t& dst1, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_int16_t& dst0, vector_int16_t& dst1, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_uint16_t& dst0, vector_uint16_t& dst1, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_half& dst0, vector_half& dst1, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_bfloat16_t& dst0, vector_bfloat16_t& dst1, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_int32_t& dst0, vector_int32_t& dst1, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_uint32_t& dst0, vector_uint32_t& dst1, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

__simd_callee__ inline void asc_loadalign_deintlv_postupdate(
    vector_float& dst0, vector_float& dst1, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_deintlv_postupdate_impl(dst0, dst1, src, offset);
}

// repeat stride模式读入
__simd_callee__ inline void asc_loadalign(
    vector_int8_t& dst, __ubuf__ int8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_uint8_t& dst, __ubuf__ uint8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_int16_t& dst, __ubuf__ int16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_int32_t& dst, __ubuf__ int32_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_int64_t& dst, __ubuf__ int64_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_int4x2_t& dst, __ubuf__ int4b_t* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_half& dst, __ubuf__ half* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign(
    vector_float& dst, __ubuf__ float* src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_int8_t& dst, __ubuf__ int8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_int32_t& dst, __ubuf__ int32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_int64_t& dst, __ubuf__ int64_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_half& dst, __ubuf__ half*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_loadalign_postupdate(
    vector_float& dst, __ubuf__ float*& src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_loadalign_postupdate_impl(dst, src, block_stride, repeat_stride, mask);
}

// ==========asc_loadunalign_pre(int8_t/uint8_t/fp4x2_e2m1_t/fp4x2_e1m2_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t)==========
__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int8_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint8_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int4b_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int16_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint16_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ half* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int32_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint32_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ float* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int64_t* src)
{
    asc_loadunalign_pre_impl(dst, src);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadunalign_pre(vector_load_unalign& dst, __ubuf__ int64_t* src, addr_reg offset)
{
    asc_loadunalign_pre_impl(dst, src, offset);
}

// ==========asc_loadunalign(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_loadunalign(vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_half& dst, vector_load_unalign& src0, __ubuf__ half* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_float& dst, vector_load_unalign& src0, __ubuf__ float* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(
    vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

__simd_callee__ inline void asc_loadunalign(vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t* src1)
{
    asc_loadunalign_impl(dst, src0, src1);
}

// ==========asc_loadunalign_postupdate(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e2m1/e1m2/e8m0)=========
__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_half& dst, vector_load_unalign& src0, __ubuf__ half*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_float& dst, vector_load_unalign& src0, __ubuf__ float*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t*& src1, uint32_t count)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, count);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int8_t& dst, vector_load_unalign& src0, __ubuf__ int8_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint8_t& dst, vector_load_unalign& src0, __ubuf__ uint8_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int16_t& dst, vector_load_unalign& src0, __ubuf__ int16_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint16_t& dst, vector_load_unalign& src0, __ubuf__ uint16_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int32_t& dst, vector_load_unalign& src0, __ubuf__ int32_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_uint32_t& dst, vector_load_unalign& src0, __ubuf__ uint32_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_half& dst, vector_load_unalign& src0, __ubuf__ half* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_float& dst, vector_load_unalign& src0, __ubuf__ float* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int64_t& dst, vector_load_unalign& src0, __ubuf__ int64_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_bfloat16_t& dst, vector_load_unalign& src0, __ubuf__ bfloat16_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e4m3fn_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e4m3fn_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_hifloat8_t& dst, vector_load_unalign& src0, __ubuf__ hifloat8_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e5m2_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e5m2_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp4x2_e2m1_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e2m1_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp4x2_e1m2_t& dst, vector_load_unalign& src0, __ubuf__ fp4x2_e1m2_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_int4x2_t& dst, vector_load_unalign& src0, __ubuf__ int4b_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

__simd_callee__ inline void asc_loadunalign_postupdate(
    vector_fp8_e8m0_t& dst, vector_load_unalign& src0, __ubuf__ fp8_e8m0_t* src1, addr_reg& offset, uint32_t inc)
{
    asc_loadunalign_postupdate_impl(dst, src0, src1, offset, inc);
}

// ==========asc_load(u8/s8/half/u16/s16/float/u32/s32/s64/bf16/e4m3/e5m2/e8m0/e2m1/e1m2)=========
__simd_callee__ inline void asc_load(vector_int8_t& dst, __ubuf__ int8_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_uint8_t& dst, __ubuf__ uint8_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_int16_t& dst, __ubuf__ int16_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_uint16_t& dst, __ubuf__ uint16_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_int32_t& dst, __ubuf__ int32_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_uint32_t& dst, __ubuf__ uint32_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_half& dst, __ubuf__ half* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_float& dst, __ubuf__ float* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_int64_t& dst, __ubuf__ int64_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src) { asc_load_impl(dst, src); }

__simd_callee__ inline void asc_load(vector_int4x2_t& dst, __ubuf__ int4b_t* src) { asc_load_impl(dst, src); }

// ==========asc_storealign(u8/s8/half/u16/s16/float/u32/s32/bf16/f8e4m3/f8e5m2/f8e8m0)=========
__simd_callee__ inline void asc_storealign(
    __ubuf__ int8_t* dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint8_t* dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int16_t* dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint16_t* dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ bfloat16_t* dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e4m3fn_t* dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ hifloat8_t* dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e5m2_t* dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ fp8_e8m0_t* dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ half* dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign(
    __ubuf__ float* dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

// ==========asc_storealign_postupdate(u8/s8/half/u16/s16/float/u32/s32/bf16/f8e4m3/f8e5m2/f8e8m0)=========
__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int8_t*& dst_align32b, vector_int8_t src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint8_t*& dst_align32b, vector_uint8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int16_t*& dst_align32b, vector_int16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint16_t*& dst_align32b, vector_uint16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ bfloat16_t*& dst_align32b, vector_bfloat16_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst_align32b, vector_fp8_e4m3fn_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ hifloat8_t*& dst_align32b, vector_hifloat8_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e5m2_t*& dst_align32b, vector_fp8_e5m2_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ fp8_e8m0_t*& dst_align32b, vector_fp8_e8m0_t src, uint16_t block_stride, uint16_t repeat_stride,
    vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ half*& dst_align32b, vector_half src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

__simd_callee__ inline void asc_storealign_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, uint16_t block_stride, uint16_t repeat_stride, vector_bool mask)
{
    asc_storealign_postupdate_impl(dst_align32b, src, block_stride, repeat_stride, mask);
}

//===========asc_set_ub2gm_loop_size=============
__aicore__ inline void asc_set_ub2gm_loop_size(uint32_t loop1_size, uint32_t loop2_size)
{
    asc_set_ub2gm_loop_size_impl(loop1_size, loop2_size);
}

//===========asc_set_ub2gm_loop1_stride===========
__aicore__ inline void asc_set_ub2gm_loop1_stride(uint64_t loop1_src_stride, uint64_t loop1_dst_stride)
{
    asc_set_ub2gm_loop1_stride_impl(loop1_src_stride, loop1_dst_stride);
}

//===========asc_set_ub2gm_loop2_stride===========
__aicore__ inline void asc_set_ub2gm_loop2_stride(uint64_t loop2_src_stride, uint64_t loop2_dst_stride)
{
    asc_set_ub2gm_loop2_stride_impl(loop2_src_stride, loop2_dst_stride);
}

__aicore__ inline void asc_set_gm2ub_loop_size(uint64_t loop1_size, uint64_t loop2_size)
{
    asc_set_gm2ub_loop_size_impl(loop1_size, loop2_size);
}

__aicore__ inline void asc_set_gm2ub_loop1_stride(uint64_t loop1_src_stride, uint64_t loop1_dst_stride)
{
    asc_set_gm2ub_loop1_stride_impl(loop1_src_stride, loop1_dst_stride);
}

__aicore__ inline void asc_set_gm2ub_loop2_stride(uint64_t loop2_src_stride, uint64_t loop2_dst_stride)
{
    asc_set_gm2ub_loop2_stride_impl(loop2_src_stride, loop2_dst_stride);
}

__aicore__ inline void asc_set_ndim_loop0_stride(uint64_t dst_stride, uint64_t src_stride)
{
    asc_set_ndim_loop0_stride_impl(dst_stride, src_stride);
}

__aicore__ inline void asc_set_ndim_loop1_stride(uint64_t dst_stride, uint64_t src_stride)
{
    asc_set_ndim_loop1_stride_impl(dst_stride, src_stride);
}

__aicore__ inline void asc_set_ndim_loop2_stride(uint64_t dst_stride, uint64_t src_stride)
{
    asc_set_ndim_loop2_stride_impl(dst_stride, src_stride);
}

__aicore__ inline void asc_set_ndim_loop3_stride(uint64_t dst_stride, uint64_t src_stride)
{
    asc_set_ndim_loop3_stride_impl(dst_stride, src_stride);
}

__aicore__ inline void asc_set_ndim_loop4_stride(uint64_t dst_stride, uint64_t src_stride)
{
    asc_set_ndim_loop4_stride_impl(dst_stride, src_stride);
}

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ hifloat8_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src)
{
    asc_storeunalign_post_impl(dst, src);
}

//===========asc_storeunalign_post(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int8_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint8_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int4b_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ hifloat8_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int16_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint16_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ half* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ bfloat16_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int32_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ uint32_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ float* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post(__ubuf__ int64_t* dst, vector_store_unalign src, addr_reg offset)
{
    asc_storeunalign_post_impl(dst, src, offset);
}

//===========asc_storeunalign_post_postupdate(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float/int64)===========
__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int8_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint8_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int4b_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e8m0_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e5m2_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ hifloat8_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int16_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint16_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ half*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ bfloat16_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int32_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ uint32_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ float*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storeunalign_post_postupdate(
    __ubuf__ int64_t*& dst, vector_store_unalign src, int32_t offset)
{
    asc_storeunalign_post_postupdate_impl(dst, src, offset);
}

//===========asc_store==================
__simd_callee__ inline void asc_store(__ubuf__ int8_t* dst, vector_int8_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ uint8_t* dst, vector_uint8_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ int16_t* dst, vector_int16_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ uint16_t* dst, vector_uint16_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ int32_t* dst, vector_int32_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ uint32_t* dst, vector_uint32_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ half* dst, vector_half src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ float* dst, vector_float src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ int64_t* dst, vector_int64_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ bfloat16_t* dst, vector_bfloat16_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ hifloat8_t* dst, vector_hifloat8_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ fp8_e5m2_t* dst, vector_fp8_e5m2_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ fp8_e8m0_t* dst, vector_fp8_e8m0_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src) { asc_store_impl(dst, src); }

__simd_callee__ inline void asc_store(__ubuf__ int4b_t* dst, vector_int4x2_t src) { asc_store_impl(dst, src); }

// ==========asc_store(with count)============
__simd_callee__ inline void asc_store(__ubuf__ int8_t* dst, vector_int8_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ uint8_t* dst, vector_uint8_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ int16_t* dst, vector_int16_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ uint16_t* dst, vector_uint16_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ int32_t* dst, vector_int32_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ uint32_t* dst, vector_uint32_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ half* dst, vector_half src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ float* dst, vector_float src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ int64_t* dst, vector_int64_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ bfloat16_t* dst, vector_bfloat16_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ fp8_e8m0_t* dst, vector_fp8_e8m0_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ hifloat8_t* dst, vector_hifloat8_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ fp8_e5m2_t* dst, vector_fp8_e5m2_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

__simd_callee__ inline void asc_store(__ubuf__ int4b_t* dst, vector_int4x2_t src, uint32_t count)
{
    asc_store_impl(dst, src, count);
}

//===========asc_set_ndim_pad_count===========
__aicore__ inline void asc_set_ndim_pad_count(asc_ndim_pad_count_config& config)
{
    asc_set_ndim_pad_count_impl(config);
}

//===========asc_set_ndim_pad_value(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float)===========
__aicore__ inline void asc_set_ndim_pad_value(int8_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(uint8_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(int16_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(uint16_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(half pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(bfloat16_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(int32_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(uint32_t pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

__aicore__ inline void asc_set_ndim_pad_value(float pad_value) { return asc_set_ndim_pad_value_impl(pad_value); }

//===========asc_ndim_copy_gm2ub(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float)===========
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e2m1_t* dst, __gm__ fp4x2_e2m1_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e1m2_t* dst, __gm__ fp4x2_e1m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e8m0_t* dst, __gm__ fp8_e8m0_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ half* dst, __gm__ half* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ float* dst, __gm__ float* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    asc_load_l2_cache_mode l2_cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, l2_cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e2m1_t* dst, __gm__ fp4x2_e2m1_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e1m2_t* dst, __gm__ fp4x2_e1m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e8m0_t* dst, __gm__ fp8_e8m0_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ half* dst, __gm__ half* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

[[deprecated("NOTICE: asc_ndim_copy_gm2ub the uint8_t cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_ndim_copy_gm2ub(
    __ubuf__ float* dst, __gm__ float* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    asc_ndim_copy_gm2ub_impl(
        dst, src, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size, loop0_lp_count, loop0_rp_count,
        padding_mode, cache_mode);
}

//==============asc_scatter=============
__simd_callee__ inline void asc_scatter(
    __ubuf__ int8_t* dst, vector_int8_t src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint8_t* dst, vector_uint8_t src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ int16_t* dst, vector_int16_t src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint16_t* dst, vector_uint16_t src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ int32_t* dst, vector_int32_t src, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ uint32_t* dst, vector_uint32_t src, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(
    __ubuf__ bfloat16_t* dst, vector_bfloat16_t src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(__ubuf__ half* dst, vector_half src, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

__simd_callee__ inline void asc_scatter(__ubuf__ float* dst, vector_float src, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(dst, src, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_int8_t& src, __ubuf__ int8_t* dst, vector_uint16_t index, vector_bool mask) "
             "is deprecated. "
             "Please use asc_scatter(__ubuf__ int8_t* dst, vector_int8_t src, vector_uint16_t index, vector_bool mask) "
             "instead.")]] __simd_callee__ inline void
asc_scatter(vector_int8_t& src, __ubuf__ int8_t* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_uint8_t& src, __ubuf__ uint8_t* dst, vector_uint16_t index, vector_bool mask) "
             "is deprecated. "
             "Please use asc_scatter(__ubuf__ uint8_t* dst, vector_uint8_t src, vector_uint16_t index, vector_bool "
             "mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_uint8_t& src, __ubuf__ uint8_t* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_int16_t& src, __ubuf__ int16_t* dst, vector_uint16_t index, vector_bool mask) "
             "is deprecated. "
             "Please use asc_scatter(__ubuf__ int16_t* dst, vector_int16_t src, vector_uint16_t index, vector_bool "
             "mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_int16_t& src, __ubuf__ int16_t* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_uint16_t& src, __ubuf__ uint16_t* dst, vector_uint16_t index, vector_bool "
             "mask) is deprecated. "
             "Please use asc_scatter(__ubuf__ uint16_t* dst, vector_uint16_t src, vector_uint16_t index, vector_bool "
             "mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_uint16_t& src, __ubuf__ uint16_t* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_int32_t& src, __ubuf__ int32_t* dst, vector_uint32_t index, vector_bool mask) "
             "is deprecated. "
             "Please use asc_scatter(__ubuf__ int32_t* dst, vector_int32_t src, vector_uint32_t index, vector_bool "
             "mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_int32_t& src, __ubuf__ int32_t* dst, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_uint32_t& src, __ubuf__ uint32_t* dst, vector_uint32_t index, vector_bool "
             "mask) is deprecated. "
             "Please use asc_scatter(__ubuf__ uint32_t* dst, vector_uint32_t src, vector_uint32_t index, vector_bool "
             "mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_uint32_t& src, __ubuf__ uint32_t* dst, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_bfloat16_t& src, __ubuf__ bfloat16_t* dst, vector_uint16_t index, vector_bool "
             "mask) is deprecated. "
             "Please use asc_scatter(__ubuf__ bfloat16_t* dst, vector_bfloat16_t src, vector_uint16_t index, "
             "vector_bool mask) instead.")]] __simd_callee__ inline void
asc_scatter(vector_bfloat16_t& src, __ubuf__ bfloat16_t* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated(
    "NOTICE: asc_scatter(vector_half& src, __ubuf__ half* dst, vector_uint16_t index, vector_bool mask) is deprecated. "
    "Please use asc_scatter(__ubuf__ half* dst, vector_half src, vector_uint16_t index, vector_bool mask) "
    "instead.")]] __simd_callee__ inline void
asc_scatter(vector_half& src, __ubuf__ half* dst, vector_uint16_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

[[deprecated("NOTICE: asc_scatter(vector_float& src, __ubuf__ float* dst, vector_uint32_t index, vector_bool mask) is "
             "deprecated. "
             "Please use asc_scatter(__ubuf__ float* dst, vector_float src, vector_uint32_t index, vector_bool mask) "
             "instead.")]] __simd_callee__ inline void
asc_scatter(vector_float& src, __ubuf__ float* dst, vector_uint32_t index, vector_bool mask)
{
    asc_scatter_impl(src, dst, index, mask);
}

// ==========asc_copy_gm2ub_align(int8_t/uint8_t/fp8_e5m2_t/fp8_e4m3fn_t/hifloat8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ half* dst, __gm__ half* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ float* dst, __gm__ float* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint16_t n_burst, uint32_t len_burst,
    uint8_t left_padding_num, uint8_t right_padding_num, bool enable_constant_pad, asc_load_l2_cache_mode l2_cache_mode,
    uint64_t src_stride, uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ half* dst, __gm__ half* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ float* dst, __gm__ float* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t left_padding_num,
    uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode, uint64_t src_stride,
    uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

[[deprecated("NOTICE: asc_copy_gm2ub_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_load_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint16_t n_burst, uint32_t len_burst,
    uint8_t left_padding_num, uint8_t right_padding_num, bool enable_constant_pad, uint8_t l2_cache_mode,
    uint64_t src_stride, uint32_t dst_stride)
{
    asc_copy_gm2ub_align_impl(
        dst, src, n_burst, len_burst, left_padding_num, right_padding_num, enable_constant_pad, l2_cache_mode,
        src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ half* dst, __gm__ half* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ half* dst, __gm__ half* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ float* dst, __gm__ float* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ float* dst, __gm__ float* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_align_sync(__ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t size)
{
    asc_copy_gm2ub_align_sync_impl(dst, src, size);
}

// ==========asc_copy_ub2gm_align(int8_t/uint8_t/fp8_e5m2_t/fp8_e4m3fn_t/hifloat8_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float)==========
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int8_t* dst, __ubuf__ int8_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int16_t* dst, __ubuf__ int16_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int32_t* dst, __ubuf__ int32_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ half* dst, __ubuf__ half* src, uint16_t n_burst, uint32_t len_burst, asc_store_l2_cache_mode l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ float* dst, __ubuf__ float* src, uint16_t n_burst, uint32_t len_burst, asc_store_l2_cache_mode l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ hifloat8_t* dst, __ubuf__ hifloat8_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ fp8_e5m2_t* dst, __ubuf__ fp8_e5m2_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ fp8_e4m3fn_t* dst, __ubuf__ fp8_e4m3fn_t* src, uint16_t n_burst, uint32_t len_burst,
    asc_store_l2_cache_mode l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int8_t* dst, __ubuf__ int8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int16_t* dst, __ubuf__ int16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ int32_t* dst, __ubuf__ int32_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ half* dst, __ubuf__ half* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ float* dst, __ubuf__ float* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ hifloat8_t* dst, __ubuf__ hifloat8_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ fp8_e5m2_t* dst, __ubuf__ fp8_e5m2_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

[[deprecated("NOTICE: asc_copy_ub2gm_align the uint8_t l2_cache_mode parameter is deprecated. "
             "Please use enum class asc_store_l2_cache_mode instead")]]
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ fp8_e4m3fn_t* dst, __ubuf__ fp8_e4m3fn_t* src, uint16_t n_burst, uint32_t len_burst, uint8_t l2_cache_mode,
    uint64_t dst_stride, uint32_t src_stride)
{
    asc_copy_ub2gm_align_impl(dst, src, n_burst, len_burst, l2_cache_mode, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ hifloat8_t* dst, __ubuf__ hifloat8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ hifloat8_t* dst, __ubuf__ hifloat8_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ fp8_e5m2_t* dst, __ubuf__ fp8_e5m2_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ fp8_e5m2_t* dst, __ubuf__ fp8_e5m2_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ fp8_e4m3fn_t* dst, __ubuf__ fp8_e4m3fn_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm_align_sync(__gm__ fp8_e4m3fn_t* dst, __ubuf__ fp8_e4m3fn_t* src, uint32_t size)
{
    asc_copy_ub2gm_align_sync_impl(dst, src, size);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src, addr_reg offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src) { asc_storealign_impl(dst, src); }

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src)
{
    asc_storealign_pack_impl(dst, src);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src) { asc_storealign_impl(dst, src); }

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src)
{
    asc_storealign_pack_impl(dst, src);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src) { asc_storealign_impl(dst, src); }

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src)
{
    asc_storealign_pack_impl(dst, src);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint32_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint16_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack(__ubuf__ uint8_t* dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint32_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint32_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint16_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint16_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_postupdate(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_storealign_pack_postupdate(__ubuf__ uint8_t*& dst, vector_bool src, int32_t offset)
{
    asc_storealign_pack_postupdate_impl(dst, src, offset);
}

//=============asc_ndim_copy_dci===============
__aicore__ inline void asc_ndim_copy_dci() { asc_ndim_copy_dci_impl(); }

//=============asc_copy_ub2l1===============
__aicore__ inline void asc_copy_ub2l1(__cbuf__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2l1_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2l1(
    __cbuf__ void* dst, __ubuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    asc_copy_ub2l1_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
}

__aicore__ inline void asc_copy_ub2l1_sync(__cbuf__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2l1_sync_impl(dst, src, size);
}
//=============asc_copy_gm2ub===============
__aicore__ inline void asc_copy_gm2ub(
    __ubuf__ void* dst, __gm__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_stride,
    uint16_t dst_stride)
{
    asc_copy_gm2ub_impl(dst, src, n_burst, len_burst, src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2ub_impl(dst, src, size);
}

__aicore__ inline void asc_copy_gm2ub_sync(__ubuf__ void* dst, __gm__ void* src, uint32_t size)
{
    asc_copy_gm2ub_sync_impl(dst, src, size);
}

//=============asc_copy_ub2gm===============
__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2gm_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2gm(
    __gm__ void* dst, __ubuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t dst_stride,
    uint16_t src_stride)
{
    asc_copy_ub2gm_impl(dst, src, n_burst, len_burst, dst_stride, src_stride);
}

__aicore__ inline void asc_copy_ub2gm_sync(__gm__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2gm_sync_impl(dst, src, size);
}

//=============asc_copy_ub2ub===============
__aicore__ inline void asc_copy_ub2ub(__ubuf__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2ub_impl(dst, src, size);
}

__aicore__ inline void asc_copy_ub2ub(
    __ubuf__ void* dst, __ubuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    asc_copy_ub2ub_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
}

__aicore__ inline void asc_copy_ub2ub_sync(__ubuf__ void* dst, __ubuf__ void* src, uint32_t size)
{
    asc_copy_ub2ub_sync_impl(dst, src, size);
}

//===========asc_set_copy_pad_val(int8/uint8/int16/uint16/half/bfloat16/int32/uint32/float)===========
__aicore__ inline void asc_set_copy_pad_val(int8_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(uint8_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(int16_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(uint16_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(half pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(bfloat16_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(int32_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(uint32_t pad_value) { asc_set_copy_pad_val_impl(pad_value); }

__aicore__ inline void asc_set_copy_pad_val(float pad_value) { asc_set_copy_pad_val_impl(pad_value); }

// ========== asc_loadalign_brc_elem ==========
__simd_callee__ inline void asc_loadalign_brc_elem(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_elem_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_elem_impl(dst, src, offset);
}

// ========== asc_loadalign_unpack4 ==========
__simd_callee__ inline void asc_loadalign_unpack4(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_unpack4_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_unpack4_impl(dst, src, offset);
}

// ========== asc_loadalign_brc_datablock ==========
__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int8_t& dst, __ubuf__ int8_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int8_t& dst, __ubuf__ int8_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int4x2_t& dst, __ubuf__ int4b_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint8_t& dst, __ubuf__ uint8_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int8_t& dst, __ubuf__ int8_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_datablock_impl(dst, src, offset);
}

// ========== asc_loadalign_brc_elem2datablock ==========
__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_int16_t& dst, __ubuf__ int16_t* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_uint16_t& dst, __ubuf__ uint16_t* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_half& dst, __ubuf__ half* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_int32_t& dst, __ubuf__ int32_t* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_uint32_t& dst, __ubuf__ uint32_t* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_float& dst, __ubuf__ float* src)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_int16_t& dst, __ubuf__ int16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_half& dst, __ubuf__ half* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_int32_t& dst, __ubuf__ int32_t* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_float& dst, __ubuf__ float* src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_uint16_t& dst, __ubuf__ uint16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_int16_t& dst, __ubuf__ int16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_half& dst, __ubuf__ half* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_uint32_t& dst, __ubuf__ uint32_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(
    vector_int32_t& dst, __ubuf__ int32_t* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock(vector_float& dst, __ubuf__ float* src, addr_reg offset)
{
    asc_loadalign_brc_elem2datablock_impl(dst, src, offset);
}

// ========== asc_loadalign_brc_elem_postupdate ==========
__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem_postupdate(vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_elem_postupdate_impl(dst, src, offset);
}

// ========== asc_loadalign_unpack4_postupdate ==========
__simd_callee__ inline void asc_loadalign_unpack4_postupdate(vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_unpack4_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_unpack4_postupdate_impl(dst, src, offset);
}

// ========== asc_loadalign_brc_datablock_postupdate ==========
__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_int8_t& dst, __ubuf__ int8_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_uint8_t& dst, __ubuf__ uint8_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_fp4x2_e2m1_t& dst, __ubuf__ fp4x2_e2m1_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_fp4x2_e1m2_t& dst, __ubuf__ fp4x2_e1m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_int4x2_t& dst, __ubuf__ int4b_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_hifloat8_t& dst, __ubuf__ hifloat8_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_datablock_postupdate(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_datablock_postupdate_impl(dst, src, offset);
}

// ========== asc_loadalign_brc_elem2datablock_postupdate ==========
__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_int16_t& dst, __ubuf__ int16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_uint16_t& dst, __ubuf__ uint16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_half& dst, __ubuf__ half*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_bfloat16_t& dst, __ubuf__ bfloat16_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_int32_t& dst, __ubuf__ int32_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_uint32_t& dst, __ubuf__ uint32_t*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

__simd_callee__ inline void asc_loadalign_brc_elem2datablock_postupdate(
    vector_float& dst, __ubuf__ float*& src, int32_t offset)
{
    asc_loadalign_brc_elem2datablock_postupdate_impl(dst, src, offset);
}

// ========== asc_storealign_pack_quarter ==========
__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ int32_t* dst_align32b, vector_int32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst_align32b, vector_uint32_t src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ float* dst_align32b, vector_float src, addr_reg offset, vector_bool mask)
{
    asc_storealign_pack_quarter_impl(dst_align32b, src, offset, mask);
}

// ========== asc_storealign_pack_quarter_postupdate ==========
__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ int32_t*& dst_align32b, vector_int32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ uint32_t*& dst_align32b, vector_uint32_t src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_postupdate_impl(dst_align32b, src, offset, mask);
}

__simd_callee__ inline void asc_storealign_pack_quarter_postupdate(
    __ubuf__ float*& dst_align32b, vector_float src, int32_t offset, vector_bool mask)
{
    asc_storealign_pack_quarter_postupdate_impl(dst_align32b, src, offset, mask);
}

// ========== asc_set_gm2ub_pad ==========
[[deprecated("NOTICE: asc_set_gm2ub_pad is deprecated. "
             "Please use asc_set_copy_pad_val instead for pad value setting")]] __aicore__ inline void
asc_set_gm2ub_pad(uint32_t pad_val)
{
    asc_set_gm2ub_pad_impl(pad_val);
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
