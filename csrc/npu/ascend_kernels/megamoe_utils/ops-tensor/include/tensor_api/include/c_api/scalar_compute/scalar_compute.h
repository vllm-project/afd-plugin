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
#error "scalar_compute.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H
#endif

#ifndef INCLUDE_C_API_SCALAR_COMPUTE_H
#define INCLUDE_C_API_SCALAR_COMPUTE_H
#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include "impl/c_api/instr_impl/npu_arch_2201/scalar_compute_impl.h"
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "impl/c_api/instr_impl/npu_arch_3510/scalar_compute_impl.h"
#endif

__aicore__ inline int64_t asc_sflbits(int64_t value);

__aicore__ inline int64_t asc_ffz(uint64_t value);

__aicore__ inline int64_t asc_ffs(uint64_t value);

__aicore__ inline int64_t asc_popc(uint64_t value);

__aicore__ inline int64_t asc_clz(uint64_t value_in);
__aicore__ inline int64_t asc_zero_bits_cnt(uint64_t value);
__aicore__ inline uint64_t asc_set_nthbit(uint64_t bits, int64_t idx);
__aicore__ inline uint64_t asc_clear_nthbit(uint64_t bits, int64_t idx);
__aicore__ inline int32_t asc_float2int32_rd(float value);
__aicore__ inline int32_t asc_float2int32_ru(float value);
__aicore__ inline int32_t asc_float2int32_rn(float value);
__aicore__ inline int32_t asc_float2int32_rna(float value);
__aicore__ inline void asc_store_dev(__gm__ int8_t* addr, int8_t value);
__aicore__ inline void asc_store_dev(__gm__ uint8_t* addr, uint8_t value);
__aicore__ inline void asc_store_dev(__gm__ int16_t* addr, int16_t value);
__aicore__ inline void asc_store_dev(__gm__ uint16_t* addr, uint16_t value);
__aicore__ inline void asc_store_dev(__gm__ int32_t* addr, int32_t value);
__aicore__ inline void asc_store_dev(__gm__ uint32_t* addr, uint32_t value);
__aicore__ inline void asc_store_dev(__gm__ int64_t* addr, int64_t value);
__aicore__ inline void asc_store_dev(__gm__ uint64_t* addr, uint64_t value);
namespace __asc_aicore {
__aicore__ inline int32_t asc_atomic_add(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_add(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline float asc_atomic_add(__gm__ float* address, float val);
__aicore__ inline int64_t asc_atomic_add(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_add(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_sub(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_sub(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline float asc_atomic_sub(__gm__ float* address, float val);
__aicore__ inline int64_t asc_atomic_sub(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_sub(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_exch(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_exch(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline float asc_atomic_exch(__gm__ float* address, float val);
__aicore__ inline int64_t asc_atomic_exch(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_exch(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_max(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_max(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline float asc_atomic_max(__gm__ float* address, float val);
__aicore__ inline int64_t asc_atomic_max(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_max(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_min(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_min(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline float asc_atomic_min(__gm__ float* address, float val);
__aicore__ inline int64_t asc_atomic_min(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_min(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_cas(__gm__ int32_t* address, int32_t compare, int32_t val);
__aicore__ inline uint32_t asc_atomic_cas(__gm__ uint32_t* address, uint32_t compare, uint32_t val);
__aicore__ inline float asc_atomic_cas(__gm__ float* address, float compare, float val);
__aicore__ inline int64_t asc_atomic_cas(__gm__ int64_t* address, int64_t compare, int64_t val);
__aicore__ inline uint64_t asc_atomic_cas(__gm__ uint64_t* address, uint64_t compare, uint64_t val);
__aicore__ inline int32_t asc_atomic_and(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_and(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline int64_t asc_atomic_and(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_and(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_or(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_or(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline int64_t asc_atomic_or(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_or(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline int32_t asc_atomic_xor(__gm__ int32_t* address, int32_t val);
__aicore__ inline uint32_t asc_atomic_xor(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline int64_t asc_atomic_xor(__gm__ int64_t* address, int64_t val);
__aicore__ inline uint64_t asc_atomic_xor(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline uint32_t asc_atomic_inc(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline uint64_t asc_atomic_inc(__gm__ uint64_t* address, uint64_t val);
__aicore__ inline uint32_t asc_atomic_dec(__gm__ uint32_t* address, uint32_t val);
__aicore__ inline uint64_t asc_atomic_dec(__gm__ uint64_t* address, uint64_t val);
} // namespace __asc_aicore

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H
#endif
