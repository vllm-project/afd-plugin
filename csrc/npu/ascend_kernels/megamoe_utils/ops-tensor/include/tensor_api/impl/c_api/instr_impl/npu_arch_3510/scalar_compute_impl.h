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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_popc_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_clz_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_zero_bits_cnt_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_ffs_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_store_dev_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_float2int32_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_sflbits_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_ffz_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_clear_nthbit_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_set_nthbit_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_add_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_sub_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_exch_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_max_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_min_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_cas_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_and_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_or_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_xor_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_inc_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl/asc_atomic_dec_impl.h"

// ==========asc_clz==========
__aicore__ inline int64_t asc_clz(uint64_t value_in) { return asc_clz_impl(value_in); }

__aicore__ inline int64_t asc_zero_bits_cnt(uint64_t value) { return asc_zero_bits_cnt_impl(value); }

__aicore__ inline int64_t asc_ffs(uint64_t value) { return asc_ffs_impl(value); }

__aicore__ inline int64_t asc_popc(uint64_t value) { return asc_popc_impl(value); }

__aicore__ inline void asc_store_dev(__gm__ int8_t* addr, int8_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ uint8_t* addr, uint8_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ int16_t* addr, int16_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ uint16_t* addr, uint16_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ int32_t* addr, int32_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ uint32_t* addr, uint32_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ int64_t* addr, int64_t value) { asc_store_dev_impl(addr, value); }

__aicore__ inline void asc_store_dev(__gm__ uint64_t* addr, uint64_t value) { asc_store_dev_impl(addr, value); }

// ==========asc_float2int32 (rd/ru/rn/rna)==========
__aicore__ inline int32_t asc_float2int32_rd(float value) { return asc_float2int32_rd_impl(value); }

__aicore__ inline int32_t asc_float2int32_ru(float value) { return asc_float2int32_ru_impl(value); }

__aicore__ inline int32_t asc_float2int32_rn(float value) { return asc_float2int32_rn_impl(value); }

__aicore__ inline int32_t asc_float2int32_rna(float value) { return asc_float2int32_rna_impl(value); }

//=============asc_sflbits============
__aicore__ inline int64_t asc_sflbits(int64_t value) { return asc_sflbits_impl(value); }

//=============asc_ffz================
__aicore__ inline int64_t asc_ffz(uint64_t value) { return asc_ffz_impl(value); }

//=============asc_clear_nthbit================
__aicore__ inline uint64_t asc_clear_nthbit(uint64_t bits, int64_t idx) { return asc_clear_nthbit_impl(bits, idx); }

//=============asc_set_nthbit================
__aicore__ inline uint64_t asc_set_nthbit(uint64_t bits, int64_t idx) { return asc_set_nthbit_impl(bits, idx); }

namespace __asc_aicore {
// ==========asc_atomic_add==========
__aicore__ inline int32_t asc_atomic_add(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_add_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_add(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_add_impl(address, val);
}

__aicore__ inline float asc_atomic_add(__gm__ float* address, float val) { return asc_atomic_add_impl(address, val); }

__aicore__ inline int64_t asc_atomic_add(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_add_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_add(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_add_impl(address, val);
}

// ==========asc_atomic_sub==========
__aicore__ inline int32_t asc_atomic_sub(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_sub_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_sub(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_sub_impl(address, val);
}

__aicore__ inline float asc_atomic_sub(__gm__ float* address, float val) { return asc_atomic_sub_impl(address, val); }

__aicore__ inline int64_t asc_atomic_sub(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_sub_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_sub(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_sub_impl(address, val);
}

// ==========asc_atomic_exch==========
__aicore__ inline int32_t asc_atomic_exch(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_exch_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_exch(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_exch_impl(address, val);
}

__aicore__ inline float asc_atomic_exch(__gm__ float* address, float val) { return asc_atomic_exch_impl(address, val); }

__aicore__ inline int64_t asc_atomic_exch(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_exch_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_exch(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_exch_impl(address, val);
}

// ==========asc_atomic_max==========
__aicore__ inline int32_t asc_atomic_max(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_max_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_max(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_max_impl(address, val);
}

__aicore__ inline float asc_atomic_max(__gm__ float* address, float val) { return asc_atomic_max_impl(address, val); }

__aicore__ inline int64_t asc_atomic_max(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_max_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_max(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_max_impl(address, val);
}

// ==========asc_atomic_min==========
__aicore__ inline int32_t asc_atomic_min(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_min_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_min(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_min_impl(address, val);
}

__aicore__ inline float asc_atomic_min(__gm__ float* address, float val) { return asc_atomic_min_impl(address, val); }

__aicore__ inline int64_t asc_atomic_min(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_min_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_min(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_min_impl(address, val);
}

// ==========asc_atomic_cas==========
__aicore__ inline int32_t asc_atomic_cas(__gm__ int32_t* address, int32_t compare, int32_t val)
{
    return asc_atomic_cas_impl(address, compare, val);
}

__aicore__ inline uint32_t asc_atomic_cas(__gm__ uint32_t* address, uint32_t compare, uint32_t val)
{
    return asc_atomic_cas_impl(address, compare, val);
}

__aicore__ inline float asc_atomic_cas(__gm__ float* address, float compare, float val)
{
    return asc_atomic_cas_impl(address, compare, val);
}

__aicore__ inline int64_t asc_atomic_cas(__gm__ int64_t* address, int64_t compare, int64_t val)
{
    return asc_atomic_cas_impl(address, compare, val);
}

__aicore__ inline uint64_t asc_atomic_cas(__gm__ uint64_t* address, uint64_t compare, uint64_t val)
{
    return asc_atomic_cas_impl(address, compare, val);
}

// ==========asc_atomic_and==========
__aicore__ inline int32_t asc_atomic_and(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_and_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_and(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_and_impl(address, val);
}

__aicore__ inline int64_t asc_atomic_and(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_and_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_and(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_and_impl(address, val);
}

// ==========asc_atomic_or==========
__aicore__ inline int32_t asc_atomic_or(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_or_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_or(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_or_impl(address, val);
}

__aicore__ inline int64_t asc_atomic_or(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_or_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_or(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_or_impl(address, val);
}

// ==========asc_atomic_xor==========
__aicore__ inline int32_t asc_atomic_xor(__gm__ int32_t* address, int32_t val)
{
    return asc_atomic_xor_impl(address, val);
}

__aicore__ inline uint32_t asc_atomic_xor(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_xor_impl(address, val);
}

__aicore__ inline int64_t asc_atomic_xor(__gm__ int64_t* address, int64_t val)
{
    return asc_atomic_xor_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_xor(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_xor_impl(address, val);
}

// ==========asc_atomic_inc==========
__aicore__ inline uint32_t asc_atomic_inc(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_inc_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_inc(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_inc_impl(address, val);
}

// ==========asc_atomic_dec==========
__aicore__ inline uint32_t asc_atomic_dec(__gm__ uint32_t* address, uint32_t val)
{
    return asc_atomic_dec_impl(address, val);
}

__aicore__ inline uint64_t asc_atomic_dec(__gm__ uint64_t* address, uint64_t val)
{
    return asc_atomic_dec_impl(address, val);
}
} // namespace __asc_aicore

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
