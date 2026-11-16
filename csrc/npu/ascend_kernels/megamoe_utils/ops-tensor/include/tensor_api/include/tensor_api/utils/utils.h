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
 * \file utils.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_UTILS_UTILS_H
#define INCLUDE_TENSOR_API_UTILS_UTILS_H

#include "utils/base/sys_macros.h"
#include "utils/base/sys_constants.h"
#include "utils/common_types.h"
#include "utils/std/type_traits.h"
#include "tensor_api/experimental/utils/reg_enum.h"

namespace AscendC {
namespace Std {}
} // namespace AscendC

namespace asc {
namespace te {

namespace Std = AscendC::Std;

enum class unit_flag_mode : uint8_t { disable = 0, enable_keep = 2, enable_update = 3 };

enum class cache_mode : uint8_t { normal = 0, disable = 4, last = 5, persistent = 6 };

using _0 = Std::Int<0>;
using _1 = Std::Int<1>;
using _2 = Std::Int<2>;
using _3 = Std::Int<3>;
using _4 = Std::Int<4>;
using _5 = Std::Int<5>;
using _6 = Std::Int<6>;
using _7 = Std::Int<7>;
using _8 = Std::Int<8>;
using _9 = Std::Int<9>;
using _10 = Std::Int<10>;
using _16 = Std::Int<16>;
using _24 = Std::Int<24>;
using _32 = Std::Int<32>;
using _64 = Std::Int<64>;
using _128 = Std::Int<128>;
using _256 = Std::Int<256>;
using _512 = Std::Int<512>;
using _1024 = Std::Int<1024>;
using _2048 = Std::Int<2048>;
using _4096 = Std::Int<4096>;

struct keep_mark {
    __aicore__ inline constexpr keep_mark() = default;
};
inline constexpr keep_mark _{};

namespace location {
struct invalid {};
struct gm {};
struct ub {};
struct l1 {};
struct l0a {};
struct l0b {};
struct l0scalea {};
struct l0scaleb {};
struct l0c {};
struct bias {};
struct fixbuf {};
struct ssbuf {};
} // namespace location

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_UTILS_UTILS_H
