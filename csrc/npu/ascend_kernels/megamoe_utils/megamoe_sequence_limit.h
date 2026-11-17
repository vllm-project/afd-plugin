// Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
#pragma once
#include <stdint.h>

namespace CamMegamoe {
constexpr int64_t SPLIT_SEQUENCE_MAX = 32;
// The count payload carries seven generation bits. The half-ring must exceed
// the two-generation skew allowed by the single input and output slots.
static_assert(SPLIT_SEQUENCE_MAX >= 6 && SPLIT_SEQUENCE_MAX <= 128 && SPLIT_SEQUENCE_MAX % 2 == 0,
              "Invalid split sequence ring");
}
