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

#ifndef IMPL_C_API_INSTR_IMPL_CUBE_COMPUTE_IMPL_CUBE_COMPUTE_C_API_IMPL_H
#define IMPL_C_API_INSTR_IMPL_CUBE_COMPUTE_IMPL_CUBE_COMPUTE_C_API_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_mmad_sparse_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_set_fp32_mode_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_mmad_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_set_l0c2gm_config_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_set_mmad_direction_m_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_set_mmad_direction_n_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_enable_hf32_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_set_hf32_round_mode_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_enable_hf32_trans_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_get_l0c2gm_config_impl.h"

__aicore__ inline void asc_enable_hf32() { asc_enable_hf32_impl(); }

__aicore__ inline void asc_set_hf32_round_mode(asc_hf32_round_mode hf32_round_mode)
{
    asc_set_hf32_round_mode_impl(hf32_round_mode);
}

__aicore__ inline void asc_enable_hf32_trans(uint32_t mode) { asc_enable_hf32_trans_impl(mode); }

__aicore__ inline void asc_mmad_sparse(
    __cc__ int32_t* c, __ca__ int8_t* a, __cb__ int8_t* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unit_flag,
    bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sparse_impl(c, a, b, m, k, n, unit_flag, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sparse_sync(
    __cc__ int32_t* c, __ca__ int8_t* a, __cb__ int8_t* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unit_flag,
    bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sparse_sync_impl(c, a, b, m, k, n, unit_flag, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_set_fp32_mode() { asc_set_fp32_mode_impl(); }

__aicore__ inline void asc_set_mmad_direction_m() { asc_set_mmad_direction_m_impl(); }

__aicore__ inline void asc_set_mmad_direction_n() { asc_set_mmad_direction_n_impl(); }

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ bfloat16_t* a_matrix, __cb__ bfloat16_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ bfloat16_t* a_matrix, __cb__ bfloat16_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ bfloat16_t* a_matrix, __cb__ bfloat16_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag,
    bool k_direction_align, bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ bfloat16_t* a_matrix, __cb__ bfloat16_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag,
    bool k_direction_align, bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ half* a_matrix, __cb__ half* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ half* a_matrix, __cb__ half* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ half* a_matrix, __cb__ half* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ half* a_matrix, __cb__ half* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ float* c_matrix, __ca__ float* a_matrix, __cb__ float* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ int32_t* c_matrix, __ca__ int8_t* a_matrix, __cb__ int8_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ int32_t* c_matrix, __ca__ int8_t* a_matrix, __cb__ int8_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad(
    __cc__ int32_t* c_matrix, __ca__ int8_t* a_matrix, __cb__ int8_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_sync(
    __cc__ int32_t* c_matrix, __ca__ int8_t* a_matrix, __cb__ int8_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_s4(
    __cc__ int32_t* c_matrix, __ca__ int4b_t* a_matrix, __cb__ int4b_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_s4_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_s4_sync(
    __cc__ int32_t* c_matrix, __ca__ int4b_t* a_matrix, __cb__ int4b_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t unit_flag, bool k_direction_align, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_s4_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, k_direction_align, c_matrix_source,
        c_matrix_init_val);
}

__aicore__ inline void asc_mmad_s4(
    __cc__ int32_t* c_matrix, __ca__ int4b_t* a_matrix, __cb__ int4b_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_s4_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_mmad_s4_sync(
    __cc__ int32_t* c_matrix, __ca__ int4b_t* a_matrix, __cb__ int4b_t* b_matrix, uint16_t left_height, uint16_t n_dim,
    uint16_t right_width, uint8_t feat_offset, uint8_t smask_offset, uint8_t unit_flag, bool k_direction_align,
    bool is_weight_offset, bool c_matrix_source, bool c_matrix_init_val)
{
    asc_mmad_s4_sync_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, feat_offset, smask_offset, unit_flag,
        k_direction_align, is_weight_offset, c_matrix_source, c_matrix_init_val);
}

__aicore__ inline void asc_set_l0c2gm_config(uint64_t relu_pre, uint64_t quant_pre, bool enable_unit_flag)
{
    asc_set_l0c2gm_config_impl(relu_pre, quant_pre, enable_unit_flag);
}

__aicore__ inline uint64_t asc_get_l0c2gm_relu() { return asc_get_l0c2gm_relu_impl(); }

__aicore__ inline uint64_t asc_get_l0c2gm_unitflag() { return asc_get_l0c2gm_unitflag_impl(); }

__aicore__ inline uint64_t asc_get_l0c2gm_prequant() { return asc_get_l0c2gm_prequant_impl(); }

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
