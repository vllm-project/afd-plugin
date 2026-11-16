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
 * \file copy_trait.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_COPY_TRAIT_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_COPY_TRAIT_H

#include "tensor_api/utils/utils.h"

namespace asc {
namespace te {

enum class round_mode : uint8_t { default_round = 0, hybrid = 1 };

enum class dual_dst_mode : uint8_t { disable = 0, split_m = 1, split_n = 2 };

struct gm_to_l1_trait {};
struct l1_to_ub_trait {};
struct l1_to_l0a_trait {};
struct l1_to_l0b_trait {};
struct l1_to_l0scalea_trait {};
struct l1_to_l0scaleb_trait {};
struct l1_to_fixbuf_trait {};
struct l1_to_biastable_trait {};

struct l0c_to_gm_trait {
    __aicore__ constexpr l0c_to_gm_trait() = default;
    __aicore__ constexpr l0c_to_gm_trait(asc::te::round_mode round_mode, bool enable_relu,
                                         bool enable_channel_split);

    asc::te::round_mode round_mode = asc::te::round_mode::default_round;
    bool enable_relu = false;
    bool enable_channel_split = false;
};

struct l0c_to_ub_trait {
    __aicore__ constexpr l0c_to_ub_trait() = default;
    __aicore__ constexpr l0c_to_ub_trait(asc::te::round_mode round_mode, bool enable_relu,
                                         bool enable_channel_split, dual_dst_mode dual_dst_ctl);

    asc::te::round_mode round_mode = asc::te::round_mode::default_round;
    bool enable_relu = false;
    bool enable_channel_split = false;
    dual_dst_mode dual_dst_ctl = dual_dst_mode::disable;
};

struct l0c_to_l1_trait {
    __aicore__ constexpr l0c_to_l1_trait() = default;
    __aicore__ constexpr l0c_to_l1_trait(asc::te::round_mode round_mode, bool enable_relu,
                                         bool enable_channel_split);

    asc::te::round_mode round_mode = asc::te::round_mode::default_round;
    bool enable_relu = false;
    bool enable_channel_split = false;
};

struct gm_to_l1_trait_default {
    using trait_type = gm_to_l1_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_ub_trait_default {
    using trait_type = l1_to_ub_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_l0a_trait_default {
    using trait_type = l1_to_l0a_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_l0b_trait_default {
    using trait_type = l1_to_l0b_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_l0scalea_trait_default {
    using trait_type = l1_to_l0scalea_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_l0scaleb_trait_default {
    using trait_type = l1_to_l0scaleb_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_fixbuf_trait_default {
    using trait_type = l1_to_fixbuf_trait;
    static constexpr const trait_type value = {};
};

struct l1_to_biastable_trait_default {
    using trait_type = l1_to_biastable_trait;
    static constexpr const trait_type value = {};
};

struct l0c_to_gm_trait_default {
    using trait_type = l0c_to_gm_trait;
    static constexpr const trait_type value = {};
};

struct l0c_to_ub_trait_default {
    using trait_type = l0c_to_ub_trait;
    static constexpr const trait_type value = {};
};

struct l0c_to_l1_trait_default {
    using trait_type = l0c_to_l1_trait;
    static constexpr const trait_type value = {};
};

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_ARCH_CUBE_COPY_TRAIT_H
