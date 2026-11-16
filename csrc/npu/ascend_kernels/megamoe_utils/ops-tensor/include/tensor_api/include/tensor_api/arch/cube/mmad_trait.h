/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mmad_trait.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_TRAIT_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_TRAIT_H

#include "tensor_api/utils/utils.h"

namespace asc {
namespace te {
enum class mmad_type : uint8_t { normal = 0, mx = 1 };

struct mmad_trait {
    __aicore__ constexpr mmad_trait(){};

    __aicore__ constexpr mmad_trait(int32_t fm_offset, bool k_direction_align, bool init_with_btbuf,
                                    bool disable_gemv, asc::te::mmad_type mmad_type) :
        fm_offset(fm_offset), k_direction_align(k_direction_align), init_with_btbuf(init_with_btbuf),
        disable_gemv(disable_gemv), mmad_type(mmad_type){};

    int32_t fm_offset = 0;
    bool k_direction_align = false;
    bool init_with_btbuf = false;
    bool disable_gemv = true;
    asc::te::mmad_type mmad_type = asc::te::mmad_type::normal;
};

struct mmad_trait_default {
    using trait_type = mmad_trait;
    static constexpr const trait_type value = {};
};

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_TRAIT_H
