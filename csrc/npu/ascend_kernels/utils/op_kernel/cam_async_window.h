// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_WINDOW_H
#define AFD_CAM_ASYNC_WINDOW_H

#include "kernel_operator.h"

#if !((defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__))
#include "cam_async_comm_group.h"
#endif

namespace Cam {

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
// CANN 9.0.0 exports this resource type through kernel_operator.h. Require the
// 64-rank MTE window layout so an SDK ABI change fails visibly at compile time.
constexpr uint32_t ASYNC_A5_WINDOW_RANK_CAPACITY = 64;
static_assert(sizeof(AscendC::HcclCombineOpParam::windowsIn) / sizeof(uint64_t)
    >= ASYNC_A5_WINDOW_RANK_CAPACITY, "A5 async MTE requires at least 64 data windows");
#endif

// MC2 must supply the matching MTE server resource context. A5 windowsIn maps
// data windows; windowsOut is the server state window. Compact-v2 keeps its own
// metadata and ready flags in the data window on both platforms. Runtime window
// initialization and remote MTE completion still require device qualification.
class AsyncDataWindow {
public:
    __aicore__ inline void Init(GM_ADDR context, uint32_t localRank)
    {
        context_ = context;
        localRank_ = localRank;
    }

    __aicore__ inline GM_ADDR GetDataWindow(const int32_t rankId) const
    {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
        auto *context = (__gm__ AscendC::HcclCombineOpParam *)context_;
        return (GM_ADDR)context->windowsIn[rankId];
#else
        auto *context = (__gm__ HcclOpResParam *)context_;
        if (rankId == localRank_) {
            return (GM_ADDR)context->localWindowsIn;
        }
        return (GM_ADDR)((HcclRankRelationResV2 *)
            context->remoteRes[rankId].nextDevicePtr)->windowsIn;
#endif
    }

private:
    GM_ADDR context_{nullptr};
    uint32_t localRank_{0};
};

} // namespace Cam
#endif // AFD_CAM_ASYNC_WINDOW_H
