// Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
#pragma once
#include "mega_sequence_limit.h"

namespace AfdAsync {
__aicore__ inline bool SplitSequenceReached(uint32_t observed, uint32_t current, bool previous)
{
    constexpr uint32_t limit = static_cast<uint32_t>(SPLIT_SEQUENCE_MAX);
    // Zero exists only before the first publication. A wait for generation 2's
    // predecessor forces this word off zero before generation 1 can recur.
    if (observed == 0U) { return previous && current == 1U; }
    uint32_t target = current;
    if (previous) { target = current == 1U ? limit : current - 1U; }
    return (observed + limit - target) % limit < limit / 2U;
}
}
