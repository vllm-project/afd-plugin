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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_ATOMIC_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_ATOMIC_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_none_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_get_store_atomic_config_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_store_atomic_config_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_float_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_bfloat_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_float16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int8_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int16_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_add_int32_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_max_int32_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/atomic_impl/asc_set_atomic_min_int32_impl.h"

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with half precision (half).
__aicore__ inline void asc_set_atomic_add_float16() { asc_set_atomic_add_float16_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with single precision (float).
__aicore__ inline void asc_set_atomic_max_float() { asc_set_atomic_max_float_impl(); }

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with bfloat16 precision (bfloat16_t).
__aicore__ inline void asc_set_atomic_add_bfloat() { asc_set_atomic_add_bfloat_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with single precision (float).
__aicore__ inline void asc_set_atomic_min_float() { asc_set_atomic_min_float_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with bfloat16 precision (bfloat16_t).
__aicore__ inline void asc_set_atomic_min_bfloat() { asc_set_atomic_min_bfloat_impl(); }

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with single precision (float).
__aicore__ inline void asc_set_atomic_add_float() { asc_set_atomic_add_float_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with half precision (half).
__aicore__ inline void asc_set_atomic_max_float16() { asc_set_atomic_max_float16_impl(); }

/// Clears the atomic operation state, disabling any active atomic add/max/min.
__aicore__ inline void asc_set_atomic_none() { asc_set_atomic_none_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with bfloat16 precision (bfloat16_t).
__aicore__ inline void asc_set_atomic_max_bfloat() { asc_set_atomic_max_bfloat_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with half precision (half).
__aicore__ inline void asc_set_atomic_min_float16() { asc_set_atomic_min_float16_impl(); }

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 8-bit signed integer (int8_t).
__aicore__ inline void asc_set_atomic_add_int8() { asc_set_atomic_add_int8_impl(); }

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 16-bit signed integer (int16_t).
__aicore__ inline void asc_set_atomic_add_int16() { asc_set_atomic_add_int16_impl(); }

/// Enables atomic add for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 32-bit signed integer (int32_t).
__aicore__ inline void asc_set_atomic_add_int32() { asc_set_atomic_add_int32_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 8-bit signed integer (int8_t).
__aicore__ inline void asc_set_atomic_max_int8() { asc_set_atomic_max_int8_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 16-bit signed integer (int16_t).
__aicore__ inline void asc_set_atomic_max_int16() { asc_set_atomic_max_int16_impl(); }

/// Enables atomic max for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 32-bit signed integer (int32_t).
__aicore__ inline void asc_set_atomic_max_int32() { asc_set_atomic_max_int32_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 8-bit signed integer (int8_t).
__aicore__ inline void asc_set_atomic_min_int8() { asc_set_atomic_min_int8_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 16-bit signed integer (int16_t).
__aicore__ inline void asc_set_atomic_min_int16() { asc_set_atomic_min_int16_impl(); }

/// Enables atomic min for data transfer from Unified Buffer/L0C Buffer/L1 Buffer to
/// Global Memory with 32-bit signed integer (int32_t).
__aicore__ inline void asc_set_atomic_min_int32() { asc_set_atomic_min_int32_impl(); }

// ==================== Deprecated interfaces ====================

/// \deprecated Use asc_atomic_add for atomic add operation instead.
[[deprecated("NOTICE: asc_set_store_atomic_config_v2 is deprecated."
             "Please use asc_atomic_add instead for atomic add operation.")]] __aicore__ inline void
asc_set_store_atomic_config_v2(uint16_t type, uint16_t op)
{
    asc_set_store_atomic_config_v2_impl(type, op);
}

/// \deprecated Use asc_atomic_add for atomic add operation instead.
[[deprecated("NOTICE: asc_get_store_atomic_config is deprecated on Ascend 950PR/Ascend 950DT."
             "Please use asc_atomic_add instead for atomic add operation.")]] __aicore__ inline void
asc_get_store_atomic_config(asc_store_atomic_config& config)
{
    asc_get_store_atomic_config_impl(config);
}

/// \deprecated Use asc_set_atomic_add_int32 instead.
[[deprecated("NOTICE: asc_set_atomic_add_int is deprecated. Please use asc_set_atomic_add_int32 instead.")]]
__aicore__ inline void asc_set_atomic_add_int()
{
    asc_set_atomic_add_int_impl();
}

/// \deprecated Use asc_set_atomic_max_int32 instead.
[[deprecated("NOTICE: asc_set_atomic_max_int is deprecated. Please use asc_set_atomic_max_int32 instead.")]]
__aicore__ inline void asc_set_atomic_max_int()
{
    asc_set_atomic_max_int_impl();
}

/// \deprecated Use asc_set_atomic_min_int32 instead.
[[deprecated("NOTICE: asc_set_atomic_min_int is deprecated. Please use asc_set_atomic_min_int32 instead.")]]
__aicore__ inline void asc_set_atomic_min_int()
{
    asc_set_atomic_min_int_impl();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
