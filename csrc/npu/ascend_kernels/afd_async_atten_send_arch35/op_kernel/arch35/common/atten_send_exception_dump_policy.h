/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Provide shared exception dump policy definitions for split Mega execution.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#ifndef MEGA_DISPATCH_SEND_ARCH35_EXCEPTION_DUMP_POLICY_H
#define MEGA_DISPATCH_SEND_ARCH35_EXCEPTION_DUMP_POLICY_H
// ### PATCH START: AFD merged source layout

#include "mc2_exception_dump.h"
// ### PATCH END: AFD merged source layout
#include "atten_send_workspace.h"
// ### PATCH START: AFD merged source layout
#include "../../afd_async_atten_send_tiling.h"
// ### PATCH END: AFD merged source layout

namespace MegaDispatchSendImpl {

enum class Stage : uint32_t {
    INIT = 0,
    INPUT_PREPARE = 1,
    SHARED_EXPERT_GMM1 = 2,
    CROSS_RANK_SYNC_INPUT = 3,
    MOE_GMM1_ACTIVATION = 4,
    MOE_GMM2_COMBINE = 5,
    SHARED_EXPERT_GMM2 = 6,
    CROSS_RANK_SYNC_OUTPUT = 7,
    UNPERMUTE = 8,
    COMPLETE = 9,
    END = 10 // 公共文件会用END判断，小于END的stage才会dump
};

enum class LoopCountIndex : uint32_t {
    GMM1 = 0,
    GMM2 = 1
};

constexpr size_t GMM_LOOP_COUNT_ALIGNMENT = 64U;

struct alignas(GMM_LOOP_COUNT_ALIGNMENT) GmmLoopCount {
    uint64_t gmm1Count;
    uint64_t gmm2Count;
};

constexpr size_t GMM_LOOP_COUNT_DATA_SIZE = 2U * sizeof(uint64_t);
constexpr size_t GMM_LOOP_COUNT_REGION_OFFSET =
    MC2ExceptionDump::DUMP_HEADER_REGION_SIZE + MC2ExceptionDump::DUMP_TILING_REGION_SIZE +
    MC2ExceptionDump::DUMP_BLOCK_STAGE_REGION_SIZE +
    MC2ExceptionDump::MAX_DUMP_ENTRIES * sizeof(MC2ExceptionDump::DumpParams);
constexpr size_t MAX_AIV_CORE_NUM = 72U;

struct MegaDispatchSendExceptionDumpPolicy {
    using TilingDataT = MegaDispatchSendTilingData;
    using StageEnumT = Stage;
    // Public exception-dump wire tag; it is independent of this operator's registered name.
    static constexpr MC2ExceptionDump::OpType OP_TYPE = MC2ExceptionDump::OpType::OP_TYPE_MEGA_MOE;
};

using ExceptionDumpEngine = MC2ExceptionDump::ExceptionDump<MegaDispatchSendExceptionDumpPolicy>;

__aicore__ inline __gm__ GmmLoopCount *RegisterAttenSendExceptionDump(ExceptionDumpEngine &engine, GM_ADDR dumpBase,
                                                                    GM_ADDR tilingGM,
                                                                    const MegaDispatchSendTilingData *tilingData,
                                                                    const MegaDispatchSendImpl::PeermemInfo &peermemInfo,
                                                                    GM_ADDR epRankIdAddr)
{
    engine.Init(dumpBase, tilingGM);
    engine.UpdateStage(Stage::INIT);
    engine.Dump(epRankIdAddr, sizeof(uint32_t));
    size_t routeRecvSize = static_cast<size_t>(CalcRouteIndexRecvSize(
        CalcDispatchRouteIndexAlignSize(tilingData), static_cast<int64_t>(tilingData->moeExpertPerRank),
        static_cast<int64_t>(tilingData->epWorldSize)));
    engine.Dump(peermemInfo.rankSyncInWorldPtr, static_cast<size_t>(MegaDispatchSendImpl::PEERMEM_DATA_OFFSET));
    engine.Dump(peermemInfo.maskRecvPtr, routeRecvSize);
    GM_ADDR loopCountBase = dumpBase + GMM_LOOP_COUNT_REGION_OFFSET;
    engine.Dump(loopCountBase, tilingData->blockAivNum, GMM_LOOP_COUNT_DATA_SIZE, sizeof(GmmLoopCount));

    if ASCEND_IS_NOT_AIV {
        return nullptr;
    }
    __gm__ GmmLoopCount *loopCount = reinterpret_cast<__gm__ GmmLoopCount *>(loopCountBase) + GetBlockIdx();
    loopCount->gmm1Count = 0U;
    loopCount->gmm2Count = 0U;
    GlobalTensor<uint8_t> loopCountGlobal;
    loopCountGlobal.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(loopCount));
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(loopCountGlobal);
    return loopCount;
}

__aicore__ inline void UpdateGmmLoopCount(__gm__ GmmLoopCount *loopCount, LoopCountIndex index, uint64_t count)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (unlikely(loopCount == nullptr)) {
        return;
    }
    __gm__ uint64_t *countAddr = &loopCount->gmm2Count;
    if (index == LoopCountIndex::GMM1) {
        countAddr = &loopCount->gmm1Count;
    }
    *countAddr = count;
    GlobalTensor<uint8_t> countGlobal;
    countGlobal.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(countAddr));
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countGlobal);
}

} // namespace MegaDispatchSendImpl

#endif // MEGA_DISPATCH_SEND_ARCH35_EXCEPTION_DUMP_POLICY_H
