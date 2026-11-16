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
 * \file asc_set_ndim_pad_value_impl.h
 * \brief
 */
#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_ASC_SET_NDIM_PAD_VALUE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_DATAMOVE_IMPL_ASC_SET_NDIM_PAD_VALUE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

//===========asc_set_ndim_pad_value(int8/uint8/e2m1/e1m2/e8m0/e5m2/e4m3fn/int16/uint16/half/bfloat16/int32/uint32/float)===========
__aicore__ inline void asc_set_ndim_pad_value_impl(int8_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint8_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(uint8_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint8_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(int16_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint16_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(uint16_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint16_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(half pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint16_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(bfloat16_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint16_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(int32_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint32_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(uint32_t pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint32_t*>(&pad_value));
    }
}

__aicore__ inline void asc_set_ndim_pad_value_impl(float pad_value)
{
    if ASC_IS_AIV {
        return set_pad_val_nddma(*reinterpret_cast<uint32_t*>(&pad_value));
    }
}

#endif
