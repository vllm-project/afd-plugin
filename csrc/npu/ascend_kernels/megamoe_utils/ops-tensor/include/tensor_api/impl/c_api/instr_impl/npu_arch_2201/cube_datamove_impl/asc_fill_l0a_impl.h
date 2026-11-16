/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_fill_l0a_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_FILL_L0A_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_FILL_L0A_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

// asc_fill_l0a half
__aicore__ inline void asc_fill_l0a_impl(__ca__ half* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ half* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ half* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ half* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a float
__aicore__ inline void asc_fill_l0a_impl(__ca__ float* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ float* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ float* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ float* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a int16_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ int16_t* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ int16_t* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ int16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ int16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a int32_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ int32_t* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ int32_t* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ int32_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ int32_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a uint16_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ uint16_t* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ uint16_t* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ uint16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ uint16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a uint32_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ uint32_t* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ uint32_t* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

__aicore__ inline void asc_fill_l0a_impl(__ca__ uint32_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ uint32_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a bfloat16_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ bfloat16_t* dst, bfloat16_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix_bf16(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(
    __ca__ bfloat16_t* dst, bfloat16_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a bfloat16_t-half
__aicore__ inline void asc_fill_l0a_impl(__ca__ bfloat16_t* dst, half value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix_h(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(__ca__ bfloat16_t* dst, half value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

// asc_fill_l0a bfloat16_t-uint32_t
__aicore__ inline void asc_fill_l0a_impl(__ca__ bfloat16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    if ASC_IS_AIC {
        create_ca_matrix_ui(dst, config.config, value);
    }
}

__aicore__ inline void asc_fill_l0a_sync_impl(
    __ca__ bfloat16_t* dst, uint32_t value, const asc_fill_value_config& config)
{
    asc_fill_l0a_impl(dst, value, config);
    asc_sync_post_process();
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
