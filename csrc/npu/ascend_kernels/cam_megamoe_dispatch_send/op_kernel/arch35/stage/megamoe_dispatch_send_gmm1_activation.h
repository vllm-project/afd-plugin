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
 * Description: Implement the gmm1 activation stage of the AFD MegaMoE pipeline.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#ifndef MEGAMOE_DISPATCH_SEND_GMM1_ACTIVATION_H
#define MEGAMOE_DISPATCH_SEND_GMM1_ACTIVATION_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
// ### PATCH START: AFD merged source layout
#include "../blaze/epilogue/cam_megamoe_dispatch_send_block_epilogue_activation_mx_quant.h"
// ### PATCH END: AFD merged source layout
#include "../common/megamoe_dispatch_send_gmm_common.h"

namespace MegamoeDispatchSendImpl {

using namespace AscendC;

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

// 等待当前 GMM1 tile 对应的 Dispatch 输入就绪。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm1InputReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    uint32_t waveIdx = mLoc / config.tileM;
    uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
    uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
    __gm__ int32_t *flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + flagOffset;
    WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
}

namespace GmmKernel {

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC, typename TensorMetaInfo>
struct Gmm1WorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
    TensorMetaInfo &metaInfoGm;
};

template <bool IsWaveFlagGrained>
__aicore__ inline void NotifyGmm2InputReady(__gm__ int32_t *flagBase, uint32_t flagSlotIdx = 0U)
{
    if (flagBase == nullptr) {
        return;
    }
    if constexpr (IsWaveFlagGrained) {
        flagBase += static_cast<uint64_t>(flagSlotIdx) * INT_CACHELINE;
    }
    AscendC::AtomicAdd(flagBase, 1);
}

__aicore__ inline void NotifyGmm1TileStatus(const GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = gmmAddrInfo.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
    AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
}

// 计算一个 Prefetch GMM1 tile 并直接写入 GM。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToGmGeneric(BlockMmad &blockMmad, const Config &config, TensorB &gmB,
                                                  TensorScaleB &gmScaleB, TensorBias &gmBias, TensorC &gmC,
                                                  const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
                                                  uint32_t kLoc, TensorA &gmBlockA, TensorScaleA &gmBlockScaleA)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nLoc),
                                  asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.slice(asc::te::make_coord(0, nLoc), asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = gmC.slice(asc::te::make_coord(mLoc, nLoc),
                                       asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
    } else {
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            uint32_t nOffset = nLoc + weightBlock * config.outputN;
            auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nOffset),
                                      asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nOffset),
                                                asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = gmC.slice(asc::te::make_coord(mLoc, nOffset),
                                           asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
        }
    }
}

// 计算一个非 Prefetch GMM1 tile 并写入固定 tile layout 的 UB。
template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToUbGeneric(BlockMmad &blockMmad, const Config &config, TensorB &gmB,
                                                  TensorScaleB &gmScaleB, TensorBias &gmBias,
                                                  const ActualShape &actualShape, uint32_t nLoc, uint32_t kLoc,
                                                  TensorA &gmBlockA, TensorScaleA &gmBlockScaleA,
                                                  TensorC &l0cOutUbFirst, TensorC &l0cOutUbSecond, uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nLoc),
                                  asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.slice(asc::te::make_coord(0, nLoc), asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
        auto tensorBlockUb = tensorUb.slice(asc::te::make_coord(0, 0),
                                            asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
    } else {
        auto tensorBlockUbFirst = l0cOutUbFirst.slice(
            asc::te::make_coord(0, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockUbSecond = l0cOutUbSecond.slice(
            asc::te::make_coord(0, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto gmBlockB = gmB.slice(asc::te::make_coord(kLoc, nLoc + weightBlock * config.outputN),
                                      asc::te::make_shape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nLoc + weightBlock * config.outputN),
                                                asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                      weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
        }
    }
}

// AIC 执行一个 A8W4 GMM1 MMAD tile；不管理 Dispatch/GMM1 或 GMM1/SwiGLU 同步，
// AIV0 与 AIC 的 Blaze 内部握手仍由 GMM 实现负责。
template <typename BlockMmad, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1AicMmadTileA8W4(BlockMmad &blockMmad, TensorA &gmA, TensorScaleA &gmScaleA,
                                           TensorScaleB &gmScaleB, TensorC &l0cOutGm, const Config &config,
                                           const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc)
{
    auto gmBlockA = gmA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.k));
    auto gmBlockScaleA =
        gmScaleA.slice(asc::te::make_coord(mLoc, 0), asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));

    for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
        uint32_t nOffset = nLoc + weightBlock * config.outputN;
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0, nOffset),
                                            asc::te::make_shape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.slice(asc::te::make_coord(mLoc, nOffset),
                                            asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
    }
}

// Prefetch 路径的 AIC 将 GMM1 tile 写入 GM，并通知 AIV 后处理。
template <typename MmadContext, bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicPrefetchMmadGeneric(WorkSet &workSet, const Params &params,
                                                  const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                  uint32_t startLoopIdx, uint32_t tileNum, uint32_t expertIdx,
                                                  MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.slice(asc::te::make_coord(mLoc, kLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /*
         * E8M0 scales are stored in 64-K pairs. Keep the padded even scale span in the tensor view;
         * the QBMM GM->L1 copy consumes each pair as one 16-bit element.
         */
        auto gmBlockScaleA = workSet.gmScaleA.slice(asc::te::make_coord(mLoc, 0),
                                                    asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }
        Gmm1AicMmadTileToGmGeneric<BlockMmad, IsGmm1Interleaved>(blockMmad, config, workSet.gmB, workSet.gmScaleB,
                                                                 workSet.gmBias, workSet.gmC, actualShape, mLoc, nLoc,
                                                                 kLoc, gmBlockA, gmBlockScaleA);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        NotifyGmm1TileStatus(gmmAddrInfo, expertIdx, loopIdx);
    }
}

// 非 Prefetch 路径的 AIC 将 GMM1 tile 写入 UB，并维持与 AIV 的 ping-pong 同步。
template <typename MmadContext, bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadGeneric(WorkSet &workSet, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                          uint32_t startLoopIdx, uint32_t tileNum, int32_t &vecSetSyncCom,
                                          uint16_t &pingpongIdx, MmadContext *blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto &blockMmad = blockMmadContext->blockMmad;

    using KernelConfig = typename Config::KernelConfig;
    using ElementC = typename KernelConfig::ElementCType;
    using MakeLayoutC = typename KernelConfig::MakeLayoutC;
    uint32_t ubBufSize = config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto ubLayout = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto l0cOutUbFirst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetFirst), ubLayout);
    auto l0cOutUbSecond =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetSecond), ubLayout);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.slice(asc::te::make_coord(mLoc, kLoc),
                                          asc::te::make_shape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.slice(asc::te::make_coord(mLoc, 0),
                                                    asc::te::make_shape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }
        if constexpr (IsGmm1Interleaved) {
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx);
            }
        } else if (vecSetSyncCom != 0) {
            WaitForVector();
        }

        Gmm1AicMmadTileToUbGeneric<BlockMmad, IsGmm1Interleaved>(
            blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, actualShape, nLoc, kLoc, gmBlockA,
            gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, pingpongIdx);

        if constexpr (IsGmm1Interleaved) {
            NotifyVector(pingpongIdx);
            vecSetSyncCom++;
            pingpongIdx = 1U - pingpongIdx;
        } else {
            NotifyVector();
            vecSetSyncCom = 1;
        }
    }
}

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/SwiGLU 的同步关系。
template <typename BlockMmad, bool IsShared, bool TopkWeightsPrefetch, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadA8W4(WorkSet &workSet, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                       int32_t &gmm1TileReadySequence, uint32_t expertIdx)
{
    BlockMmad blockMmad{};
    typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN, L0_TILE_K,
                                               0};
    typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
    blockMmad.Init(matmulShape, l0TileShape, config.blockMmadTiling.l1Params);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }

        Gmm1AicMmadTileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, config,
                            actualShape, mLoc, nLoc);

        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        if constexpr (TopkWeightsPrefetch) {
            NotifyGmm1TileStatus(gmmAddrInfo, expertIdx, loopIdx);
        } else {
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmm1TileReadySequence);
        }
    }
}

// Prefetch 交错路径从 GM 取回一个子 tile，执行 SwiGLU/量化并通知 GMM2。
template <typename MakeLayoutC, typename ElementC, bool IsWaveFlagGrained, typename Config, typename TensorC,
          typename TensorMetaInfo, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric(
    const Config &config, TensorC &gmC, TensorMetaInfo &metaInfoGm, ActivationQuantOp &activationQuantOp,
    const GMMAddrInfo &gmmAddrInfo, uint32_t expertBeforeCnt, const ActualShape &actualShape, uint32_t mLoc,
    uint32_t nLoc)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    auto tensorBlockUb = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(0), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    auto topkWeightTensor = activationQuantOp.GetTopkWeightTensor();

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
        auto tensorBlockGm =
            gmC.slice(asc::te::make_coord(subMLoc, nLoc), asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        asc::te::copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        uint32_t epilogueN = Get<N_VALUE>(actualShape) / ACTIVATION_N_HALF;
        uint32_t epilogueNLoc = nLoc / ACTIVATION_N_HALF;
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, epilogueN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + epilogueNLoc,
            subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            static_cast<int64_t>(subMLoc / L1_TILE_M_256),
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp(epilogueShape, epilogueOffset, 0U);
        NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, subMLoc / L1_TILE_M_256);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

// 非 Prefetch 交错路径直接消费 AIC 的 UB ping-pong tile。
template <bool IsWaveFlagGrained, typename Config, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0InterleavedEpilogueTileGeneric(const Config &config,
                                                              ActivationQuantOp &activationQuantOp,
                                                              const GMMAddrInfo &gmmAddrInfo,
                                                              const ActualShape &actualShape, uint32_t mLoc,
                                                              uint32_t nLoc, uint16_t pingpongIdx)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    uint32_t epilogueN = Get<N_VALUE>(actualShape) / ACTIVATION_N_HALF;
    uint32_t epilogueNLoc = nLoc / ACTIVATION_N_HALF;
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), epilogueN, 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + epilogueNLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        static_cast<int64_t>(mLoc / L1_TILE_M_256),
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    activationQuantOp(epilogueShape, epilogueOffset, pingpongIdx);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
}

// Prefetch 非交错路径从 GM 取回两路 GMM1 输出，再执行 SwiGLU/量化。
template <typename MakeLayoutC, typename ElementC, bool IsWaveFlagGrained, typename Config, typename TensorC,
          typename TensorMetaInfo, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0PrefetchEpilogueTileGeneric(const Config &config, TensorC &gmC,
                                                           TensorMetaInfo &metaInfoGm,
                                                           ActivationQuantOp &activationQuantOp,
                                                           const GMMAddrInfo &gmmAddrInfo, uint32_t expertBeforeCnt,
                                                           const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    uint64_t firstMetaInfoRow = expertBeforeCnt + mLoc;
    auto topkWeightTensor = activationQuantOp.GetTopkWeightTensor();
    AscendC::DataCopy(topkWeightTensor, metaInfoGm[firstMetaInfoRow * INT32_PER_256B],
                      static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);

    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
    auto tensorBlockUbFirst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        if (subOffset != 0) {
            AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
        }

        auto tensorBlockGmFirst =
            gmC.slice(asc::te::make_coord(subMLoc, nLoc), asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond = gmC.slice(asc::te::make_coord(subMLoc, nLoc + config.outputN),
                                             asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        asc::te::copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        asc::te::copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + nLoc,
            subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp(epilogueShape, epilogueOffset);
        NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, subMLoc / L1_TILE_M_256);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

// 非 Prefetch 非交错路径直接消费 AIC 的 UB tile。
template <bool IsWaveFlagGrained, typename Config, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0EpilogueTileGeneric(const Config &config, ActivationQuantOp &activationQuantOp,
                                                   const GMMAddrInfo &gmmAddrInfo, const ActualShape &actualShape,
                                                   uint32_t mLoc, uint32_t nLoc)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                                 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        0,
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    activationQuantOp(epilogueShape, epilogueOffset);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
}

// Prefetch 路径的 AIV1 从 GM 取回 A8W4 GMM1 子 tile，执行 SwiGLU/量化并发布完成状态。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename TensorC, typename TensorMetaInfo,
          typename ActivationQuantOp, typename Config, typename ActualShape>
__aicore__ inline void Gmm1Aiv1PrefetchEpilogueTileA8W4(ActivationQuantOp &activationQuantOp, TensorC &l0cOutGm,
                                                        TensorMetaInfo &metaInfoGm, const GMMAddrInfo &gmmAddrInfo,
                                                        const Config &config, const ActualShape &actualShape,
                                                        uint32_t mLoc, uint32_t nLoc, uint32_t expertBeforeCnt)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }

    constexpr uint32_t subTileM = L1_TILE_M_128;
    uint64_t firstMetaInfoRow = expertBeforeCnt + mLoc;
    auto topkWeightTensor = activationQuantOp.GetTopkWeightTensor();
    AscendC::DataCopy(topkWeightTensor, metaInfoGm[firstMetaInfoRow * INT32_PER_256B],
                      static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);
    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
    auto tensorBlockUbFirst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        if (subOffset != 0) {
            AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
        }
        auto tensorBlockGmFirst =
            l0cOutGm.slice(asc::te::make_coord(subMLoc, nLoc), asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond = l0cOutGm.slice(asc::te::make_coord(subMLoc, nLoc + config.outputN),
                                                  asc::te::make_shape(subM, Get<N_VALUE>(actualShape)));
        asc::te::copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        asc::te::copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + nLoc,
            subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp(epilogueShape, epilogueOffset);
        NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, subMLoc / L1_TILE_M_256);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

// 非 Prefetch 路径的 AIV1 从 GM 取回完整 A8W4 GMM1 tile 并执行 SwiGLU/量化。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename TensorC, typename ActivationQuantOp,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1Aiv1EpilogueTileA8W4(ActivationQuantOp &activationQuantOp, TensorC &l0cOutGm,
                                                const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    auto tensorBlockGmFirst = l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc),
                                             asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto tensorBlockGmSecond =
        l0cOutGm.slice(asc::te::make_coord(mLoc, nLoc + config.outputN),
                       asc::te::make_shape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto layoutL0cUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    uint32_t ubBufferSize =
        config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufferSize) * sizeof(ElementC);
    auto tensorBlockUbFirst =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, ElementC>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    asc::te::copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
    asc::te::copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                                 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        0,
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    activationQuantOp(epilogueShape, epilogueOffset);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
}

// 为一个 GMM1 逻辑 block 的 AIV0 路径展开两段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm1Aiv0PrologueA8W4(Scheduler &scheduler, TensorB &gmB, const Config &config,
                                            uint32_t startLoopIdx, uint32_t tileNum)
{
    BlockPrologue blockPrologue;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto nOffset = nLoc + weightBlock * config.outputN;
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.blockMmadTiling.l1Params.kL1);
        }
    }
}

__aicore__ inline void WaitGmm1TileStatus(const GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = gmmAddrInfo.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
    int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
    WaitUntilGmFlagEquals(statusAddr, roundTag);
}

// Prefetch 路径的 AIV0 等待 GM tile 就绪，再执行 SwiGLU/量化。
template <typename MakeLayoutC, typename ElementC, bool IsGmm1Interleaved, bool IsWaveFlagGrained, typename WorkSet,
          typename Config, typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv0PrefetchEpilogueGeneric(WorkSet &workSet, const Params &params,
                                                       const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                       uint32_t startLoopIdx, uint32_t tileNum,
                                                       ActivationQuantOp &activationQuantOp, uint32_t expertBeforeCnt,
                                                       uint32_t expertIdx)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        WaitGmm1TileStatus(gmmAddrInfo, expertIdx, loopIdx);
        if constexpr (IsGmm1Interleaved) {
            Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
                config, workSet.gmC, workSet.metaInfoGm, activationQuantOp, gmmAddrInfo, expertBeforeCnt, actualShape,
                mLoc, nLoc);
        } else {
            Gmm1Aiv0PrefetchEpilogueTileGeneric<MakeLayoutC, ElementC, IsWaveFlagGrained>(
                config, workSet.gmC, workSet.metaInfoGm, activationQuantOp, gmmAddrInfo, expertBeforeCnt, actualShape,
                mLoc, nLoc);
        }
    }
}

// 非 Prefetch 路径的 AIV0 直接消费 AIC 的 UB tile，并维持 ping-pong 同步。
template <bool IsGmm1Interleaved, bool IsWaveFlagGrained, typename WorkSet, typename Config, typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv0EpilogueGeneric(WorkSet &workSet, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                               uint32_t startLoopIdx, uint32_t tileNum,
                                               ActivationQuantOp &activationQuantOp, uint16_t &pingpongIdx)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        if constexpr (IsGmm1Interleaved) {
            WaitForCube(pingpongIdx);
            Gmm1Aiv0InterleavedEpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo,
                                                                      actualShape, mLoc, nLoc, pingpongIdx);
            NotifyCube(pingpongIdx);
            pingpongIdx = 1U - pingpongIdx;
        } else {
            WaitForCube();
            Gmm1Aiv0EpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo, actualShape, mLoc,
                                                           nLoc);
            NotifyCube();
        }
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename ElementC, typename MakeLayoutC, bool TopkWeightsPrefetch, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename Scheduler, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1ExecGeneric(Scheduler &scheduler, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                       ActivationQuantOp &activationQuantOp, int32_t &vecSetSyncCom,
                                       uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t &pingpongIdx,
                                       BlockMmadContext<BlockMmad> *blockMmadContext, bool allowWeightL2Bypass)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)),
        layouts.a);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)),
        layouts.b);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
                                         layouts.scaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
                                         layouts.scaleB);
    if constexpr (IsWaveFlagGrained && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    GM_ADDR cGlobal = 0UL;
    if constexpr (TopkWeightsPrefetch) {
        cGlobal = gmmAddrInfo.gmm1OutGlobal;
    }
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC *>(cGlobal)), layouts.c);
    AscendC::GlobalTensor<float> metaInfoGm;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmmAddrInfo.metaInfoGlobal));
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, blockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, gmmAddrInfo, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx, blockMmadContext);
            }
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            if constexpr (TopkWeightsPrefetch) {
                Gmm1AicPrefetchMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, expertIdx, &localBlockMmadContext);
            } else {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, gmmAddrInfo, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx,
                    &localBlockMmadContext);
            }
        }
    } else {
        // AIV1 在入口处提前退出。
        if constexpr (TopkWeightsPrefetch) {
            Gmm1Aiv0PrefetchEpilogueGeneric<MakeLayoutC, ElementC, IsGmm1Interleaved, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            Gmm1Aiv0EpilogueGeneric<IsGmm1Interleaved, IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                          tileNum, activationQuantOp, pingpongIdx);
        }
    }
}

// Prefetch 路径的 AIV1 等待 GM tile 就绪，再执行 A8W4 SwiGLU/量化。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv1PrefetchEpilogueA8W4(WorkSet &workSet, const Params &params,
                                                    const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                    uint32_t startLoopIdx, uint32_t tileNum,
                                                    ActivationQuantOp &activationQuantOp, uint32_t expertBeforeCnt,
                                                    uint32_t expertIdx)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        WaitGmm1TileStatus(gmmAddrInfo, expertIdx, loopIdx);
        Gmm1Aiv1PrefetchEpilogueTileA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
            activationQuantOp, workSet.gmC, workSet.metaInfoGm, gmmAddrInfo, config, actualShape, mLoc, nLoc,
            expertBeforeCnt);
    }
}

// 非 Prefetch 路径的 AIV1 按 AIC 序号消费 A8W4 GMM1 tile。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv1EpilogueA8W4(WorkSet &workSet, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                            uint32_t startLoopIdx, uint32_t tileNum, int32_t &gmm1TileReadySequence,
                                            ActivationQuantOp &activationQuantOp)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        int32_t expectedReadySequence = gmm1TileReadySequence + 1;
        WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        Gmm1Aiv1EpilogueTileA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(activationQuantOp, workSet.gmC, gmmAddrInfo,
                                                                           config, actualShape, mLoc, nLoc);
        gmm1TileReadySequence = expectedReadySequence;
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM1/SwiGLU 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC, bool IsShared,
          typename Scheduler, typename Config, bool TopkWeightsPrefetch, bool IsWaveFlagGrained,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1ExecA8W4(Scheduler &scheduler, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                    const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                    int32_t &gmm1TileReadySequence, ActivationQuantOp &activationQuantOp,
                                    uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm1OutGlobal)),
        layouts.c);
    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)),
        layouts.a);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)),
        layouts.b);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
                                         layouts.scaleA);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
                                         layouts.scaleB);
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    AscendC::GlobalTensor<float> metaInfoGm;
    if constexpr (TopkWeightsPrefetch && g_coreType == AscendC::AIV) {
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmmAddrInfo.metaInfoGlobal));
    }

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<BlockMmad, IsShared, TopkWeightsPrefetch>(workSet, params, gmmAddrInfo, config, startLoopIdx,
                                                                  tileNum, gmm1TileReadySequence, expertIdx);
    } else if (GetSubBlockIdx() == 0) {
        Gmm1Aiv0PrologueA8W4<BlockPrologue>(workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        if constexpr (TopkWeightsPrefetch) {
            Gmm1Aiv1PrefetchEpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
                workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, activationQuantOp, expertBeforeCnt,
                expertIdx);
        } else {
            Gmm1Aiv1EpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
                workSet, gmmAddrInfo, config, startLoopIdx, tileNum, gmm1TileReadySequence, activationQuantOp);
        }
    }
}

} // namespace GmmKernel

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1Generic(
    BlockEpilogueActivationMxQuant<EpilogueElementA, ElementC, EpilogueTileM, L1_TILE_N, TopkWeightsPrefetch,
                                   IsGmm1Interleaved> &epilogueOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, const BlockJobContext &blockJob,
    uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t &pingpongIdx, void *blockMmadContext = nullptr,
    bool allowWeightL2Bypass = false)
{
    using GmmConfig =
        GmmKernel::Config<false, COMBINE_NO_QUANT, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, TopkWeightsPrefetch, IsShared, false, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (GetSubBlockIdx() != 0) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    GmmKernel::Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved,
                               IsWaveFlagGrained>(scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum,
                                                  epilogueOp, vecSetSyncCom, expertBeforeCnt, expertIdx, pingpongIdx,
                                                  typedBlockMmadContext, allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1Generic(
    BlockEpilogueActivationMxQuant<EpilogueElementA, ElementC, EpilogueTileM, L1_TILE_N, TopkWeightsPrefetch,
                                   IsGmm1Interleaved> &epilogueOp,
    const Params &params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, int32_t &vecSetSyncCom, uint32_t expertBeforeCnt,
    uint32_t expertIdx, uint16_t &pingpongIdx, void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1Generic<ElementA, EpilogueElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                   Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
        epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom, blockJob, expertBeforeCnt,
        expertIdx, pingpongIdx, blockMmadContext, allowWeightL2Bypass);
}

// RunGmm1A8W4：执行 A8W4 prologue（W4→W8）、GMM1、SwiGLU 和量化。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1A8W4(BlockEpilogueActivationMxQuant<ElementA, ElementC, EpilogueTileM, L1_TILE_N,
                                                                  TopkWeightsPrefetch> &activationQuantOp,
                                   const Params &params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   int32_t &gmm1TileReadySequence, const BlockJobContext &blockJob,
                                   uint32_t expertBeforeCnt, uint32_t expertIdx = 0)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        TopkWeightsPrefetch, IsShared, false>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    if constexpr (g_coreType == AIV) {
        activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        GmmKernel::BlockScheduler::Params{asc::te::make_coord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    GmmKernel::Gmm1ExecA8W4<BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared, GmmKernel::BlockScheduler,
                            decltype(config), TopkWeightsPrefetch, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmm1TileReadySequence, activationQuantOp,
        expertBeforeCnt, expertIdx);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1A8W4(BlockEpilogueActivationMxQuant<ElementA, ElementC, EpilogueTileM, L1_TILE_N,
                                                                  TopkWeightsPrefetch> &activationQuantOp,
                                   const Params &params,
                                   const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   int32_t &gmm1TileReadySequence, uint32_t expertBeforeCnt, uint32_t expertIdx = 0)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, EpilogueTileM,
                TopkWeightsPrefetch, IsShared, IsWaveFlagGrained>(activationQuantOp, params, problemShape, gmmAddrInfo,
                                                                  startBlockIdx, gmm1TileReadySequence, blockJob,
                                                                  expertBeforeCnt, expertIdx);
}

template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool EnableA8W4, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(
    const GmmExecutionConfig &gmmConfig, const MoeSyncWorkspaceLayout &syncLayout, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state, uint32_t rowOffsetInExpert = 0U, uint32_t gmm1TilesPerMGroup = 0U)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = workspace.gmm1MmadResPtr + globalTokenStartIndex * n * sizeof(bfloat16_t);
    }
    if constexpr (TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1TileStatus = reinterpret_cast<__gm__ int32_t *>(workspace.gmm1TileStatusPtr) +
                                     (static_cast<uint64_t>(state.expertIdx) * syncLayout.gmm1TileStatusCountPerExpert +
                                      static_cast<uint64_t>(expertMGroupOffset) * gmm1TilesPerMGroup) *
                                         INT_CACHELINE;
    }
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr;
    gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
                                       expertOffset * syncLayout.activationFlagSlotCountPerExpert +
                                       static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal =
        workspace.dispatchRevDataPtr + globalTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.dispatchRevScalePtr + globalTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (EnableA8W4) {
            runsActivation = GetSubBlockIdx() == 1;
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                globalTokenStartIndex * n / ACTIVATION_N_HALF / outputElementsPerByte,
                globalTokenStartIndex * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
                                         expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
                                         static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

template <typename QuantOutType, typename ActivationOutType, typename QuantScaleType, uint32_t Gmm1TileM,
          uint32_t EpilogueTileM, bool TopkWeightsPrefetch, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false, bool IsShared = false, typename BlockEpilogue>
__aicore__ inline void RunGmm1GenericByWeightFormat(const GmmExecutionConfig &gmmConfig, const Params &gmmParams,
                                                    BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                                    const ProblemShape &problemShape, uint32_t globalTokenStartIndex,
                                                    GmmRuntimeState &runtimeState, uint32_t expertIdx,
                                                    void *persistentBlockMmadContext = nullptr,
                                                    bool allowWeightL2Bypass = false)
{
    if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        RunGmm1Generic<QuantOutType, ActivationOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, true,
                       Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            gmmConfig.blockJob, globalTokenStartIndex, expertIdx, runtimeState.pingpongIdx, persistentBlockMmadContext,
            allowWeightL2Bypass);
    } else {
        RunGmm1Generic<QuantOutType, ActivationOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, false,
                       Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            gmmConfig.blockJob, globalTokenStartIndex, expertIdx, runtimeState.pingpongIdx, persistentBlockMmadContext,
            allowWeightL2Bypass);
    }
}

// 原型：MegamoeDispatchSend::UpdateSharedGlobalBuffer<GMM1>。构造一个共享专家的 GMM1 和 SwiGLU GM 视图。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const MoeStageCommonConfig &commonConfig,
                                                          const GmmExecutionConfig &gmmConfig,
                                                          const WorkspaceInfo &workspace,
                                                          const ExpertWeightTensorListAddrs &weights,
                                                          BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = commonConfig.tokenNum;
    uint64_t n = commonConfig.gmm1OutputDim;
    uint64_t k = commonConfig.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t activationN = n / ACTIVATION_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(activationN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = workspace.sharedExpertInputDataPtr;
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertInputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = workspace.sharedExpertGmm1OutPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor,
                                                              sharedExpertIdx, expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx, expertIdx * n * scaleK);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    if (workspace.sharedActivationToGmm2Ptr != nullptr) {
        uint64_t sharedActivationFlagElementCount =
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(m)));
        gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.sharedActivationToGmm2Ptr) +
                                           expertIdx * sharedActivationFlagElementCount;
    }
    gmmAddrInfo.metaInfoGlobal = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * activationN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

} // namespace MegamoeDispatchSendImpl

#endif // MEGAMOE_DISPATCH_SEND_GMM1_ACTIVATION_H
