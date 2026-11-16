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
#error "asc_simd.h cannot be used with compile flag --enable-simt enabled."
#endif

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H
#endif

#ifndef INCLUDE_C_API_ASC_SIMD_H
#define INCLUDE_C_API_ASC_SIMD_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

#include "c_api/atomic/atomic.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "c_api/cube_compute/cube_compute.h"
#include "c_api/cube_datamove/cube_datamove.h"
#include "c_api/misc/misc.h"
#include "c_api/scalar_compute/scalar_compute.h"
#include "c_api/sync/sync.h"
#include "c_api/sys_var/sys_var.h"
#include "c_api/vector_datamove/vector_datamove.h"
#include "c_api/vector_compute/vector_compute.h"

#if (ASC_DEVKIT_VERSION_NUM >= 902000000)
#if !defined(__NPU_ARCH__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
#include "c_api/reg_compute/reg_convert.h"
#include "c_api/reg_compute/reg_load.h"
#include "c_api/reg_compute/reg_store.h"
#include "c_api/reg_compute/reg_vector.h"
#endif
#else
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
#include "c_api/reg_compute/reg_convert.h"
#include "c_api/reg_compute/reg_load.h"
#include "c_api/reg_compute/reg_store.h"
#include "c_api/reg_compute/reg_vector.h"
#endif
#endif
#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_C_API_H
#endif
