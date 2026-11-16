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
 * \file megamoe_combine_recv_workspace.h
 * \brief
 */

#ifndef MEGAMOE_COMBINE_RECV_WORKSPACE_H
#define MEGAMOE_COMBINE_RECV_WORKSPACE_H

// peermem 窗口的布局与尺寸（含 HOST_DEVICE / GM_ADDR 宏与基础 include）。
#include "megamoe_combine_recv_peermem.h"

namespace MegamoeCombineRecvImpl {

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL; // 异常dump区
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;

constexpr int64_t INVALID_WORKSPACE_OFFSET = -1LL;

// 每个共享专家按 256-token group 预留独立的同步槽，所有量化模式使用相同布局。
HOST_DEVICE int64_t CalcSharedActivationFlagElementsPerExpert(int64_t tokenNum)
{
    return Ops::Base::CeilDiv(tokenNum, static_cast<int64_t>(L1_TILE_M_256)) * static_cast<int64_t>(INT_CACHELINE);
}

// 仅描述各 workspace 分区相对基址的字节偏移，不持有或构造任何地址。
struct WorkspaceLayout {
    int64_t dispatchRevDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevWeightsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t expertRevTokenNumsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t metaInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagActivationToGmm2Offset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedActivationToGmm2Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagDispatchToGmm1Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagSendCntCalToUpdParamsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagGmmToEpilogueOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2ReadyOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2CombineSyncCounterOffset{INVALID_WORKSPACE_OFFSET};
    int64_t cumsumInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertResultOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertGmm1OutOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1TileStatusOffset{INVALID_WORKSPACE_OFFSET}; // GMM1 tile 就绪状态位区（仅 prefetch 软同步分配）
    int64_t sharedExpertGmm2TileCounterOffset{INVALID_WORKSPACE_OFFSET};
    int64_t maskSlotOffset{INVALID_WORKSPACE_OFFSET};               // urma发送mask临时GM
    int64_t dispatchRelaySendQueueOffset{INVALID_WORKSPACE_OFFSET}; // 按目标 Server 划分的一级中继发送队列
    int64_t dispatchRemoteReadyFlagSnapshotOffset{
        INVALID_WORKSPACE_OFFSET}; // 各逻辑核的固定 256-token 远端就绪标志窗口

    // 连续 flag 通知区（自 flagActivationToGmm2Offset 起）的 int32 元素总数。
    // 在分配处顺手记账，保证 ResetSyncStatus 的清零范围与分配范围恒同源。
    int64_t flagResetElementCount{0};
    // prefetch 软同步 GMM1 tile 状态区的 int32 元素总数（未分配时为 0）。
    int64_t gmm1TileStatusElementCount{0};

    int64_t workspaceSize{0};
    HOST_DEVICE explicit WorkspaceLayout(const MegamoeCombineRecvTilingData *tilingData)
    {
        uint32_t rankNumPerServer =
            tilingData->rankNumPerServer == 0U ? tilingData->epWorldSize : tilingData->rankNumPerServer;
        uint32_t serverNum = Ops::Base::CeilDiv(tilingData->epWorldSize, rankNumPerServer);
        Initialize(tilingData, serverNum);
    }

    HOST_DEVICE WorkspaceLayout(const MegamoeCombineRecvTilingData *tilingData, uint32_t serverNum)
    {
        Initialize(tilingData, serverNum);
    }

private:
    HOST_DEVICE void Initialize(const MegamoeCombineRecvTilingData *tilingData, uint32_t serverNum)
    {
        workspaceSize = 0;
        // Tiling admits only MTE and URMA. Each path preserves its original allocation order.
        if (tilingData->topoType == TOPO_TYPE_MTE) {
            InitializeMte(tilingData);
        } else {
            InitializeUrma(tilingData, serverNum);
        }
    }

    HOST_DEVICE void InitializeMte(const MegamoeCombineRecvTilingData *tilingData)
    {
        flagActivationToGmm2Offset = workspaceSize;
        sharedActivationToGmm2Offset = workspaceSize;
        workspaceSize += SIZE_INT_32 * CalcSharedActivationFlagElementsPerExpert(tilingData->bs) *
                         tilingData->sharedExpertNum;
        flagGmmToEpilogueOffset = workspaceSize;
        workspaceSize += SIZE_INT_32 * tilingData->aicNum * INT_CACHELINE;
        sharedExpertGmm2TileCounterOffset = workspaceSize;
        workspaceSize += SIZE_INT_32 * Ops::Base::CeilDiv(tilingData->bs, uint32_t(L1_TILE_M_256)) *
                         tilingData->sharedExpertNum * INT_CACHELINE;
        flagResetElementCount = workspaceSize / SIZE_INT_32;
        workspaceSize = Ops::Base::CeilAlign(workspaceSize, int64_t(ALIGN_512));
        InitializeSharedExpertBuffers(tilingData);
        workspaceSize = Ops::Base::CeilAlign(workspaceSize + ALIGN_512, int64_t(ALIGN_512));
    }

    HOST_DEVICE void InitializeUrma(const MegamoeCombineRecvTilingData *tilingData, uint32_t serverNum)
    {
        InitializeDispatchBuffers(tilingData);
        InitializeUrmaPrefetchWeights(tilingData);
        int64_t countBytes = static_cast<int64_t>(tilingData->moeExpertPerRank) * ALIGN_32 * tilingData->aicNum;
        InitializeActivationAndMetadata(tilingData, countBytes);
        InitializeUrmaSyncFlags(tilingData);
        // All URMA GMM modes retain dispatch cumsum across activation.
        InitializeGmmIntermediateBuffers(tilingData, true);
        gmm2MmadResOffset = workspaceSize;
        workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->h;
        InitializeGmmTileStatus(tilingData);
        InitializeUrmaBuffers(tilingData, serverNum);
        InitializeSharedExpertBuffers(tilingData);
    }

    HOST_DEVICE void InitializeDispatchBuffers(const MegamoeCombineRecvTilingData *tilingData)
    {
        dispatchRevDataOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScaleOffset = workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * dispatchScaleElementsPerToken, ALIGN_512);
    }

    HOST_DEVICE void InitializeUrmaPrefetchWeights(const MegamoeCombineRecvTilingData *tilingData)
    {
        if (tilingData->topkWeightsPrefetch == 1) {
            dispatchRevWeightsOffset = workspaceSize;
            uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)),
                                                             static_cast<uint32_t>(ALIGN_32));
            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->maxOutputSize) * static_cast<int64_t>(weightAlignBytes), ALIGN_512);
        }
    }

    HOST_DEVICE void InitializeActivationAndMetadata(const MegamoeCombineRecvTilingData *tilingData,
                                                     int64_t expertCountWorkspaceBytes)
    {
        activationQuantDataOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF, ALIGN_512);

        activationQuantScaleOffset = workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize *
                                     Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->hiddenDim / ACTIVATION_N_HALF),
                                                        static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                                     MXFP_MULTI_BASE_SIZE,
                                 ALIGN_512);

        expertRevTokenNumsOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(expertCountWorkspaceBytes, ALIGN_512);

        metaInfoOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);
    }

    HOST_DEVICE void InitializeMteSyncFlags(const MegamoeCombineRecvTilingData *tilingData)
    {
        // Keep all scalar flags contiguous for ResetFlagList.
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);
        flagActivationToGmm2Offset = workspaceSize;
        int64_t wavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
        workspaceSize += SIZE_INT_32 * moeExpertCount * wavesPerExpert * INT_CACHELINE;
        if (tilingData->sharedExpertNum > 0) {
            sharedActivationToGmm2Offset = workspaceSize;
            workspaceSize += SIZE_INT_32 *
                             CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tilingData->bs)) *
                             static_cast<int64_t>(tilingData->sharedExpertNum);
        }
        InitializeRoutingFlags(tilingData);
        bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
        if (isW4Mode || tilingData->combineQuantMode == COMBINE_NO_QUANT) {
            flagGmmToEpilogueOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }
        if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            gmm2ReadyOffset = workspaceSize;
            workspaceSize += SIZE_INT_32 * moeExpertCount * tilingData->aicNum * INT_CACHELINE;
        }
        if (tilingData->sharedExpertNum > 0) {
            sharedExpertGmm2TileCounterOffset = workspaceSize;
            int64_t tokenGroupCount =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs), static_cast<int64_t>(L1_TILE_M_256));
            workspaceSize +=
                SIZE_INT_32 * tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
        }
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;
    }

    HOST_DEVICE void InitializeUrmaSyncFlags(const MegamoeCombineRecvTilingData *tilingData)
    {
        // Keep all scalar flags contiguous for ResetFlagList.
        int64_t flagRegionBeginOffset = workspaceSize;
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);
        flagActivationToGmm2Offset = workspaceSize;
        // URMA activation uses only the first slot of each expert.
        workspaceSize += SIZE_INT_32 * moeExpertCount * INT_CACHELINE;
        InitializeRoutingFlags(tilingData);
        bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
        if (isW4Mode) {
            flagGmmToEpilogueOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }
        gmm2CombineSyncCounterOffset = workspaceSize;
        workspaceSize += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) * moeExpertCount *
                         INT_CACHELINE * SIZE_INT_32;
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;
    }

    HOST_DEVICE void InitializeRoutingFlags(const MegamoeCombineRecvTilingData *tilingData)
    {
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);
        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        flagDispatchToGmm1Offset = workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsOffset = workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * moeExpertCount * tilingData->aicNum;
    }

    HOST_DEVICE void InitializeGmmIntermediateBuffers(const MegamoeCombineRecvTilingData *tilingData, bool reserveCumsum)
    {
        if (reserveCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            cumsumInfoOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_32 * tilingData->moeExpertPerRank *
                                                                       tilingData->epWorldSize),
                                                  ALIGN_32) *
                             tilingData->aicNum;
        }
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 || tilingData->topkWeightsPrefetch == 1) {
            // gmm1MmadRes：GMM1 matmul 输出，布局为 maxOutputSize × hiddenDim 个 BF16。
            gmm1MmadResOffset = workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
    }

    HOST_DEVICE void InitializeGmmTileStatus(const MegamoeCombineRecvTilingData *tilingData)
    {
        // GMM1 tile 状态位区（仅 prefetch 路径分配，用于 AIC→AIV 软同步）。
        // 每个 tile 状态和末尾 allDone 状态都独占一条 64B cache line。
        gmm1TileStatusOffset = INVALID_WORKSPACE_OFFSET;
        if (tilingData->topkWeightsPrefetch == 1) {
            gmm1TileStatusOffset = workspaceSize;
            int64_t statusSlots =
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->maxTilesPerExpert + 1;
            gmm1TileStatusElementCount = statusSlots * INT_CACHELINE;
            int64_t statusBytes = SIZE_INT_32 * statusSlots * INT_CACHELINE;
            workspaceSize += Ops::Base::CeilAlign(statusBytes, ALIGN_512);
        }
    }

    HOST_DEVICE void InitializeUrmaBuffers(const MegamoeCombineRecvTilingData *tilingData, uint32_t serverNum)
    {
        maskSlotOffset = workspaceSize;
        // 远端 mask 槽的 count 位于全卡共用的容量尾部，本地发送暂存必须使用相同槽宽。
        const int64_t maskAlignSize = CalcDispatchMaskAlignSize(tilingData);
        int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(ALIGN_32); // mask + 32B count

        workspaceSize += Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->epWorldSize * maskSlotSize,
            static_cast<int64_t>(ALIGN_512));

        dispatchRelaySendQueueOffset = workspaceSize;
        int64_t serverWorkspaceBytes = static_cast<int64_t>(ALIGN_32) + static_cast<int64_t>(tilingData->bs) * ALIGN_32;
        workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                              static_cast<int64_t>(ALIGN_512));

        dispatchRemoteReadyFlagSnapshotOffset = workspaceSize;
        // 每个逻辑核只缓存当前 256-token 窗口；稀疏 token 按命中的窗口分块读取。
        int64_t flagSnapshotBytes = static_cast<int64_t>(L1_TILE_M_256) * sizeof(uint64_t);
        int64_t relayFlagSnapshotBytesPerBlock =
            Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(ALIGN_512));
        workspaceSize += static_cast<int64_t>(tilingData->aicNum) * relayFlagSnapshotBytesPerBlock;
    }

    HOST_DEVICE void InitializeSharedExpertBuffers(const MegamoeCombineRecvTilingData *tilingData)
    {
        // 共享专家 workspace buffer。
        if (tilingData->sharedExpertNum > 0) {
            // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
            sharedExpertResultOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->h, ALIGN_512);
            // sharedExpertGmm1Out：共享专家 GMM1 输出 [sharedExpertNum × bs × hiddenDim]。
            sharedExpertGmm1OutOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim, ALIGN_512);
            // sharedExpertActivationData：SwiGLU 量化输出 [sharedExpertNum × bs × hiddenDim / 2] FP8。
            sharedExpertActivationDataOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim / ACTIVATION_N_HALF,
                ALIGN_512);
            // sharedExpertActivationScale：每个 MXFP scale group 保存两个 multi-base 分量。
            sharedExpertActivationScaleOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum *
                    Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->hiddenDim / ACTIVATION_N_HALF),
                                       static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                    MXFP_MULTI_BASE_SIZE,
                ALIGN_512);
        }
    }
};

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr{nullptr};
    GM_ADDR dispatchRevScalePtr{nullptr};
    GM_ADDR dispatchRevWeightsPtr{nullptr};
    GM_ADDR activationQuantDataPtr{nullptr};
    GM_ADDR activationQuantScalePtr{nullptr};
    GM_ADDR expertRevTokenNumsPtr{nullptr};
    GM_ADDR metaInfoPtr{nullptr};
    GM_ADDR flagActivationToGmm2Ptr{nullptr};
    GM_ADDR sharedActivationToGmm2Ptr{nullptr};
    GM_ADDR flagDispatchToGmm1Ptr{nullptr};
    GM_ADDR flagSendCntCalToUpdParamsPtr{nullptr};
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertActivationDataPtr{nullptr};
    GM_ADDR sharedExpertActivationScalePtr{nullptr};
    GM_ADDR gmm1TileStatusPtr{nullptr};
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    GM_ADDR maskSlotPtr{nullptr};
    GM_ADDR dispatchRelaySendQueuePtr{nullptr};
    GM_ADDR dispatchRemoteReadyFlagSnapshotPtr{nullptr};

    int64_t flagResetElementCount{0};
    int64_t gmm1TileStatusElementCount{0};

private:
    // 仅在取得真实 workspace 基址后，将有效偏移绑定为可访问地址。
    HOST_DEVICE static GM_ADDR ResolveWorkspaceAddress(GM_ADDR base, int64_t offset)
    {
        if (base == nullptr || offset < 0) {
            return nullptr;
        }
        return base + offset;
    }

public:
    HOST_DEVICE void Bind(GM_ADDR base, const WorkspaceLayout &layout)
    {
        dispatchRevDataPtr = ResolveWorkspaceAddress(base, layout.dispatchRevDataOffset);
        dispatchRevScalePtr = ResolveWorkspaceAddress(base, layout.dispatchRevScaleOffset);
        dispatchRevWeightsPtr = ResolveWorkspaceAddress(base, layout.dispatchRevWeightsOffset);
        activationQuantDataPtr = ResolveWorkspaceAddress(base, layout.activationQuantDataOffset);
        activationQuantScalePtr = ResolveWorkspaceAddress(base, layout.activationQuantScaleOffset);
        expertRevTokenNumsPtr = ResolveWorkspaceAddress(base, layout.expertRevTokenNumsOffset);
        metaInfoPtr = ResolveWorkspaceAddress(base, layout.metaInfoOffset);
        flagActivationToGmm2Ptr = ResolveWorkspaceAddress(base, layout.flagActivationToGmm2Offset);
        sharedActivationToGmm2Ptr = ResolveWorkspaceAddress(base, layout.sharedActivationToGmm2Offset);
        flagDispatchToGmm1Ptr = ResolveWorkspaceAddress(base, layout.flagDispatchToGmm1Offset);
        flagSendCntCalToUpdParamsPtr = ResolveWorkspaceAddress(base, layout.flagSendCntCalToUpdParamsOffset);
        flagGmmToEpiloguePtr = ResolveWorkspaceAddress(base, layout.flagGmmToEpilogueOffset);
        gmm2ReadyPtr = ResolveWorkspaceAddress(base, layout.gmm2ReadyOffset);
        gmm2CombineSyncCounterPtr = ResolveWorkspaceAddress(base, layout.gmm2CombineSyncCounterOffset);
        cumsumInfoPtr = ResolveWorkspaceAddress(base, layout.cumsumInfoOffset);
        gmm1MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm1MmadResOffset);
        gmm2MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm2MmadResOffset);
        sharedExpertResultPtr = ResolveWorkspaceAddress(base, layout.sharedExpertResultOffset);
        sharedExpertGmm1OutPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm1OutOffset);
        sharedExpertInputDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputDataOffset);
        sharedExpertInputScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputScaleOffset);
        sharedExpertActivationDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationDataOffset);
        sharedExpertActivationScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationScaleOffset);
        gmm1TileStatusPtr = ResolveWorkspaceAddress(base, layout.gmm1TileStatusOffset);
        sharedExpertGmm2TileCounterPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm2TileCounterOffset);
        maskSlotPtr = ResolveWorkspaceAddress(base, layout.maskSlotOffset);
        dispatchRelaySendQueuePtr = ResolveWorkspaceAddress(base, layout.dispatchRelaySendQueueOffset);
        dispatchRemoteReadyFlagSnapshotPtr =
            ResolveWorkspaceAddress(base, layout.dispatchRemoteReadyFlagSnapshotOffset);
        flagResetElementCount = layout.flagResetElementCount;
        gmm1TileStatusElementCount = layout.gmm1TileStatusElementCount;
    }
};

} // namespace MegamoeCombineRecvImpl

#endif // MEGAMOE_COMBINE_RECV_WORKSPACE_H
