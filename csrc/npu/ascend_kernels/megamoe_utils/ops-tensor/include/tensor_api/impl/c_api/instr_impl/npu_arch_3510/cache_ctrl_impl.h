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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CACHE_CTRL_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CACHE_CTRL_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_icache_preload_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_data_cache_preload_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_dcci_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_get_icache_preload_status_impl.h"
#include "impl/c_api/instr_impl/npu_arch_3510/cache_ctrl_impl/asc_dci_impl.h"

// ==========asc_icache_preload==========
__aicore__ inline void asc_icache_preload(const void* addr, int64_t prefetch_len)
{
    asc_icache_preload_impl(addr, prefetch_len);
}

__aicore__ inline void asc_datacache_preload(__gm__ uint64_t* address, int64_t offset)
{
    asc_datacache_preload_impl(address, offset);
}

namespace __asc_aicore {
__aicore__ inline void asc_dcci_single(__gm__ void* dst) { asc_dcci_single_impl(dst); }
} // namespace __asc_aicore

__aicore__ inline void asc_ub_dcci_single(__ubuf__ void* dst) { asc_ub_dcci_single_impl(dst); }

__aicore__ inline void asc_dcci_entire_all() { asc_dcci_entire_all_impl(); }

__aicore__ inline void asc_dcci_entire_out() { asc_dcci_entire_out_impl(); }

__aicore__ inline void asc_dcci_entire_atomic() { asc_dcci_entire_atomic_impl(); }

//=========asc_get_icache_preload_status========
__aicore__ inline int64_t asc_get_icache_preload_status() { return asc_get_icache_preload_status_impl(); }

//==============asc_dci===============
__aicore__ inline void asc_dci() { asc_dci_impl(); }

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
