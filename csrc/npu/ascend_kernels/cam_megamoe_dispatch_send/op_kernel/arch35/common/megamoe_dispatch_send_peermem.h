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
 * \file megamoe_dispatch_send_peermem.h
 * \brief peermem 跨卡对称窗口的布局与尺寸计算。
 *        窗口各区尺寸一律按全卡一致的 numMaxTokensPerRank 上界推导（可变 bs 下各卡真实 bs 可不同，
 *        但跨卡读写共用同一套偏移，布局必须逐字节一致）。
 *        本文件是 host 侧 tiling 校验与 device 侧地址装配的唯一事实源，两侧不得各写一份公式。
 */

#ifndef MEGAMOE_DISPATCH_SEND_PEERMEM_H
#define MEGAMOE_DISPATCH_SEND_PEERMEM_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t *
// host 侧命名空间级自由函数需显式 inline 保 ODR；类内成员函数本就隐式 inline，加之无副作用。
#define HOST_DEVICE inline
// host 侧自洽：Ops::Base::CeilAlign/CeilDiv 的提供方，不再依赖包含方的 include 顺序。
#include "op_common/op_host/util/math_util.h"
#endif
// ### PATCH START: AFD merged source layout
#include "../../cam_megamoe_dispatch_send_tiling.h"
// ### PATCH END: AFD merged source layout
#include "megamoe_dispatch_send_constants.h"
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/megamoe_layer_selection.h"
// ### PATCH END: AFD merged source layout

namespace MegamoeDispatchSendImpl {

// 原同步布局：前 48 KiB 是 rank 槽位，随后 12 KiB 是各 AIV 的同步计数区。
// URMA 的 rank 槽位超过预留区时动态扩展；MTE 保持原 60 KiB 固定布局。
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 1024 * 48LL;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = PEERMEM_DATA_OFFSET - PEERMEM_MIN_RANK_SYNC_SIZE;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = static_cast<int64_t>(INT_CACHELINE) * sizeof(int32_t);

// 通信拓扑：决定窗口数据区按 MTE 还是 URMA 布局推进。
constexpr int64_t TOPO_TYPE_MTE = 0U;  // mte
constexpr int64_t TOPO_TYPE_URMA = 1U; // urma

HOST_DEVICE int64_t CalcUrmaSyncCountOffset(int64_t epWorldSize)
{
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    return rankSyncSize > PEERMEM_MIN_RANK_SYNC_SIZE ? rankSyncSize : PEERMEM_MIN_RANK_SYNC_SIZE;
}

HOST_DEVICE int64_t CalcPeermemDataOffset(int64_t topoType, int64_t epWorldSize)
{
    if (topoType == TOPO_TYPE_URMA) {
        return Ops::Base::CeilAlign(CalcUrmaSyncCountOffset(epWorldSize) + PEERMEM_SYNC_COUNT_REGION_SIZE, ALIGN_512);
    }
    return PEERMEM_DATA_OFFSET;
}

// ============================ 各区尺寸（host / device 共用） ============================

// URMA layered 路径仍使用 mask 槽；MTE WAVE 不再经过该布局。
HOST_DEVICE int64_t CalcDispatchMaskAlignSizeBy(int64_t numMaxTokensPerRank, int64_t topK)
{
    int64_t sendTotalNum = numMaxTokensPerRank * topK;
    int64_t alignedRouteCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    return Ops::Base::CeilAlign(alignedRouteCount / static_cast<int64_t>(BITS_PER_BYTE), ALIGN_32);
}

HOST_DEVICE int64_t CalcDispatchMaskAlignSize(const MegamoeDispatchSendTilingData *tilingData)
{
    return CalcDispatchMaskAlignSizeBy(static_cast<int64_t>(tilingData->numMaxTokensPerRank),
                                       static_cast<int64_t>(tilingData->topK));
}

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
HOST_DEVICE int64_t CalcDispatchRouteIndexAlignSize(const MegamoeDispatchSendTilingData *tilingData)
{
    return Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData->numMaxTokensPerRank) * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
}

// compact route 接收区不再携带槽尾 count；count 继续使用下方独立连续表。
HOST_DEVICE int64_t CalcRouteIndexRecvSize(int64_t routeIndexAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * routeIndexAlignSize, ALIGN_512);
}

HOST_DEVICE int64_t CalcMaskRecvSize(int64_t maskAlignSize, int64_t moeExpertPerRank, int64_t epWorldSize)
{
    int64_t maskSlotSize = maskAlignSize + ALIGN_32;
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, ALIGN_512);
}

// 独立 raw count 表按 [localExpert][sourceRank] 排列，供接收端一次搬入并做前缀和。
HOST_DEVICE int64_t CalcExpertCountRecvSize(int64_t moeExpertPerRank, int64_t epWorldSize)
{
    return Ops::Base::CeilAlign(moeExpertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);
}

// 单 token 的量化记录字节数 = 量化数据 + mx scale（prefetch 场景再拼 topk 权重），按 32B 对齐。
HOST_DEVICE int64_t CalcQuantTokenScaleBytes(int64_t h, uint32_t elemsPerByte, int64_t topK, bool topkWeightsPrefetch)
{
    uint32_t mxScaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(h), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(h) / elemsPerByte, static_cast<uint32_t>(ALIGN_256)) *
        static_cast<uint32_t>(sizeof(int8_t));
    uint32_t scaleBytes = mxScaleNum * static_cast<uint32_t>(sizeof(int8_t));
    uint32_t tokenScaleBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
    if (topkWeightsPrefetch) {
        uint32_t weightBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)),
                                                    static_cast<uint32_t>(ALIGN_32));
        tokenScaleBytes = Ops::Base::CeilAlign(tokenScaleBytes + weightBytes, static_cast<uint32_t>(ALIGN_32));
    }
    return static_cast<int64_t>(tokenScaleBytes);
}

// combine 接收区单 token 记录字节数：须与 kernel InitCombineBuffers 的 combine record 布局一致。
HOST_DEVICE int64_t CalcCombineTokenBytes(int64_t h, int64_t yDtypeSize, bool isQuantCombine)
{
    if (!isQuantCombine) {
        return h * yDtypeSize;
    }
    int64_t tokenStorageBytes = Ops::Base::CeilAlign(h, ALIGN_256);
    int64_t scaleCount = Ops::Base::CeilDiv(h, static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
    int64_t storedScaleBytes = Ops::Base::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
    return Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, ALIGN_32);
}

// 计算窗口总大小的入参（供 host 侧 tiling 校验 cclBufferSize 使用）。
struct PeermemSizeParams {
    int64_t numMaxTokensPerRank;
    int64_t topK;
    int64_t h;
    int64_t moeExpertPerRank;
    int64_t epWorldSize;
    int64_t yDtypeSize;
    uint32_t elemsPerByte;
    bool topkWeightsPrefetch;
    bool isQuantCombine;
    int64_t topoType;
    int64_t serverNum;
};

// peermem 各分段尺寸。Host 打印、窗口下限校验共用该结果，避免重复维护布局公式。
struct PeermemLayoutSizes {
    int64_t dataOffset;
    int64_t maskAlignSize;
    int64_t maskRecvSize;
    int64_t routeTaskMetaSize;
    int64_t expertCountRecvSize;
    int64_t tokenScaleBytes;
    int64_t relayDataSize;
    int64_t relayFlagSize;
    int64_t dispatchRecordAreaSize;
    int64_t combineTokenBytes;
    int64_t combineSendSize;
};

HOST_DEVICE PeermemLayoutSizes CalcPeermemLayoutSizes(const PeermemSizeParams &params)
{
    PeermemLayoutSizes sizes{};
    sizes.dataOffset = CalcPeermemDataOffset(params.topoType, params.epWorldSize);
    sizes.maskAlignSize = CalcDispatchMaskAlignSizeBy(params.numMaxTokensPerRank, params.topK);
    if (params.topoType == TOPO_TYPE_MTE) {
        int64_t routeIndexAlignSize =
            Ops::Base::CeilAlign(params.numMaxTokensPerRank * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
        sizes.maskRecvSize = CalcRouteIndexRecvSize(routeIndexAlignSize, params.moeExpertPerRank, params.epWorldSize);
    } else {
        sizes.maskRecvSize = CalcMaskRecvSize(sizes.maskAlignSize, params.moeExpertPerRank, params.epWorldSize);
    }
    sizes.routeTaskMetaSize = params.topoType == TOPO_TYPE_MTE ?
        static_cast<int64_t>(CamMegamoeLayer::TaskAreaBytes(static_cast<uint32_t>(params.epWorldSize))) : 0;
    sizes.expertCountRecvSize = CalcExpertCountRecvSize(params.moeExpertPerRank, params.epWorldSize);
    sizes.tokenScaleBytes =
        CalcQuantTokenScaleBytes(params.h, params.elemsPerByte, params.topK, params.topkWeightsPrefetch);
    sizes.dispatchRecordAreaSize = Ops::Base::CeilAlign(params.numMaxTokensPerRank * sizes.tokenScaleBytes, ALIGN_512);
    if (params.topoType == TOPO_TYPE_URMA) {
        int64_t relayRecordBytes = Ops::Base::CeilAlign(sizes.tokenScaleBytes, ALIGN_512);
        sizes.relayDataSize =
            Ops::Base::CeilAlign(params.numMaxTokensPerRank * relayRecordBytes * params.serverNum, ALIGN_512);
        sizes.relayFlagSize = Ops::Base::CeilAlign(
            params.serverNum * params.numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);
        sizes.dispatchRecordAreaSize = sizes.relayDataSize + sizes.relayFlagSize;
    }
    sizes.combineTokenBytes = CalcCombineTokenBytes(params.h, params.yDtypeSize, params.isQuantCombine);
    sizes.combineSendSize =
        Ops::Base::CeilAlign(params.numMaxTokensPerRank * params.topK * sizes.combineTokenBytes, ALIGN_512);
    return sizes;
}

/*
 * peermem 窗口所需的最小字节数，逐段与下方 PeermemInfo 构造函数的偏移推进同源：
 *   同步区(dataOffset) + route/mask 区 + MTE task metadata 区 + count 区 + dispatch 区 + combine 区。
 * host 侧 tiling 用它校验用户传入的 cclBufferSize，不再各自手写一份布局公式。
 */
HOST_DEVICE int64_t CalcPeermemLeastSize(const PeermemSizeParams &params)
{
    PeermemLayoutSizes sizes = CalcPeermemLayoutSizes(params);
    return sizes.dataOffset + sizes.maskRecvSize + sizes.routeTaskMetaSize + sizes.expertCountRecvSize + sizes.dispatchRecordAreaSize +
           sizes.combineSendSize;
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// device 侧窗口各区基址；偏移推进顺序即上方 CalcPeermemLeastSize 的分段顺序。
struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr{nullptr};
    GM_ADDR maskRecvPtr{nullptr};
    GM_ADDR routeTaskMetaPtr{nullptr};
    GM_ADDR expertCountRecvPtr{nullptr};
    GM_ADDR quantTokenScalePtr{nullptr};
    GM_ADDR dispatchReceivePtr{nullptr};
    GM_ADDR dispatchFlagPtr{nullptr};
    GM_ADDR combineSendPtr{nullptr};

    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegamoeDispatchSendTilingData *tilingData, uint32_t elemsPerByte = 1,
                                  uint32_t serverNum = 1)
    {
        PeermemSizeParams params{};
        params.numMaxTokensPerRank = static_cast<int64_t>(tilingData->numMaxTokensPerRank);
        params.topK = static_cast<int64_t>(tilingData->topK);
        params.h = static_cast<int64_t>(tilingData->h);
        params.moeExpertPerRank = static_cast<int64_t>(tilingData->moeExpertPerRank);
        params.epWorldSize = static_cast<int64_t>(tilingData->epWorldSize);
        params.yDtypeSize = static_cast<int64_t>(sizeof(bfloat16_t));
        params.elemsPerByte = elemsPerByte;
        params.topkWeightsPrefetch = tilingData->topkWeightsPrefetch == 1;
        params.isQuantCombine = tilingData->combineQuantMode != COMBINE_NO_QUANT;
        params.topoType = tilingData->topoType;
        params.serverNum = static_cast<int64_t>(serverNum);
        PeermemLayoutSizes sizes = CalcPeermemLayoutSizes(params);

        rankSyncInWorldPtr = base;
        int64_t offset = sizes.dataOffset;
        maskRecvPtr = base + offset;
        offset += sizes.maskRecvSize;

        if (tilingData->topoType == TOPO_TYPE_MTE) {
            routeTaskMetaPtr = base + offset;
            offset += sizes.routeTaskMetaSize;
        }
        expertCountRecvPtr = base + offset;
        offset += sizes.expertCountRecvSize;

        if (tilingData->topoType == TOPO_TYPE_MTE) {
            quantTokenScalePtr = base + offset;
            offset += sizes.dispatchRecordAreaSize;
        }
        if (tilingData->topoType == TOPO_TYPE_URMA) {
            dispatchReceivePtr = base + offset;
            offset += sizes.relayDataSize;
            dispatchFlagPtr = base + offset;
            offset += sizes.relayFlagSize;
        }
        combineSendPtr = base + offset;
    }
};
#endif

} // namespace MegamoeDispatchSendImpl

#endif // MEGAMOE_DISPATCH_SEND_PEERMEM_H
