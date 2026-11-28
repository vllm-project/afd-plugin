/* Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2. */
#pragma once
#include <cstdint>

namespace AfdAsync {
// Offsets from the end of the diagnostic area in the destination rank's window.
struct MteWindowOffsets {
    int64_t routes, taskMeta, counts, ffnEnd;
    int64_t inputs, results, attentionEnd;
};

template <typename Sizes>
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
__aicore__
#endif
inline MteWindowOffsets MteOffsets(const Sizes &s)
{
    const int64_t routes = s.dataOffset;
    const int64_t taskMeta = routes + s.maskRecvSize;
    const int64_t counts = taskMeta + s.routeTaskMetaSize;
    const int64_t inputs = s.dataOffset;
    const int64_t results = inputs + s.dispatchRecordAreaSize;
    return {routes, taskMeta, counts, counts + s.expertCountRecvSize,
            inputs, results, results + s.combineSendSize};
}
} // namespace AfdAsync
