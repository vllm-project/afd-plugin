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
 * Description: Compute host tiling parameters and workspace requirements.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

/*!
 * \file mega_ffn_tiling.cpp
 * \brief
 */

#include "mega_host_compat.h"
#include "mega_sequence_limit.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
// ### PATCH START: AFD merged source layout
#include "../../../op_kernel/afd_async_mega_ffn_tiling.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "../../../op_kernel/afd_async_mega_ffn_tiling_key.h"
// ### PATCH END: AFD merged source layout
#include "../../../op_kernel/arch35/common/mega_ffn_constants.h"
#include "../../../op_kernel/arch35/common/mega_ffn_workspace.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace MegaFusedMoeImpl;
namespace optiling {
namespace {
constexpr uint32_t GMM_TILE_N = 256U;
constexpr uint32_t GMM1_MIN_LOGICAL_TILES_PER_CORE = 4U;
constexpr int64_t GMM_TILE_STATUS_COUNT_ALIGN = 16LL;

static uint32_t WeightTensorCount(const gert::TilingContext *context, uint32_t input)
{
    uint32_t count = 0U;
    while (count < AfdAsyncLayer::MAX_TENSORS && context->GetDynamicInputShape(input, count) != nullptr) {
        ++count;
    }
    return context->GetDynamicInputShape(input, count) == nullptr ? count : 0U;
}

// Descriptor-only validation: the current task layer remains device-resident.
static bool CheckWeightBank(const gert::TilingContext *context, MegaFusedMoeTilingData *td)
{
    uint32_t count = WeightTensorCount(context, 1U);
    td->weightLayerCount = AfdAsyncLayer::LayerCount(count, td->moeExpertPerRank, td->isPerExpertWeightTensor);
    if (td->weightLayerCount == 0U) { return false; }
    std::vector<std::vector<int64_t>> dimensions{
        {td->hiddenDim, td->h}, {td->h, td->hiddenDim / 2},
        {td->hiddenDim, td->h / 64, 2}, {td->h, td->hiddenDim / 2 / 64, 2}};
    for (uint32_t input = 1U; input <= 4U; ++input) {
        if (WeightTensorCount(context, input) != count) { return false; }
        const auto *firstDesc = context->GetDynamicInputDesc(input, 0U);
        if (firstDesc == nullptr) { return false; }
        auto expected = dimensions[input - 1U];
        if (!td->isPerExpertWeightTensor) { expected.insert(expected.begin(), td->moeExpertPerRank); }
        for (uint32_t tensor = 0U; tensor < count; ++tensor) {
            const auto *shape = context->GetDynamicInputShape(input, tensor);
            const auto *desc = context->GetDynamicInputDesc(input, tensor);
            if (shape == nullptr || desc == nullptr || desc->GetDataType() != firstDesc->GetDataType() ||
                desc->GetStorageFormat() != firstDesc->GetStorageFormat() ||
                shape->GetOriginShape().GetDimNum() != expected.size()) { return false; }
            for (size_t dim = 0; dim < expected.size(); ++dim) {
                if (shape->GetOriginShape().GetDim(dim) != expected[dim]) { return false; }
            }
        }
    }
    return true;
}

uint32_t CalcMGroupsPerWave(const MegaFusedMoeTilingData *tilingData, uint32_t aicNum)
{
    if (tilingData->hiddenDim == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    /*
     * hiddenDim 包含 gate/up 两部分。交织模式每核至少调度 4 个独立 N tile；非交织模式
     * 每个物理任务成对处理 gate/up，原先每核 2 个物理任务同样等价于 4 个逻辑 N tile。
     * hiddenDim 已校验为 GMM_TILE_N(256) 的倍数，CeilDiv 对
     * 非交织半宽产生的尾 tile 只影响 wave 粒度估算，不影响正确性，两种编译模式共用本公式。
     */
    uint64_t gmm1LogicalTilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->hiddenDim, GMM_TILE_N);
    uint64_t gmm2TilesPerMGroup = ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t gmm1RequiredMGroups = ops::CeilDiv<uint64_t>(
        static_cast<uint64_t>(aicNum) * GMM1_MIN_LOGICAL_TILES_PER_CORE, gmm1LogicalTilesPerMGroup);
    uint64_t gmm2RequiredMGroups = ops::CeilDiv<uint64_t>(static_cast<uint64_t>(aicNum), gmm2TilesPerMGroup);
    return static_cast<uint32_t>(std::max(gmm1RequiredMGroups, gmm2RequiredMGroups));
}
static void SetPrefetchAndWaveParams(MegaFusedMoeTilingData *tilingData, const uint32_t aicNum)
{
    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）。
    // 非交织调度只遍历 hiddenDim / 2，交织调度会遍历完整 hiddenDim；统一按完整 hiddenDim
    // 预留，使两个 clencode 共用同一 workspace 布局并避免交织模式覆盖下一 expert 的状态槽。
    tilingData->maxTilesPerExpert = 0;
    if (tilingData->topkWeightsPrefetch == 1) {
        int64_t maxSchedulerN = static_cast<int64_t>(tilingData->hiddenDim);
        int64_t maxTilesM = ops::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(256));
        int64_t maxTilesN = ops::CeilDiv(maxSchedulerN, static_cast<int64_t>(256));
        tilingData->maxTilesPerExpert =
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, GMM_TILE_STATUS_COUNT_ALIGN));
    }

    tilingData->mGroupsPerWave = CalcMGroupsPerWave(tilingData, aicNum);
}
static uint32_t CalcDispatchCopyBufferBytes(const MegaFusedMoeTilingData *tilingData, uint32_t activationElementsPerByte)
{
    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    return copyBufferBytes;
}

/*
 * 计算 dispatch 阶段不随 ring 深度与 route batch 变化的固定 UB 占用。
 */
static uint32_t CalcDispatchFixedBufferBytes(const MegaFusedMoeTilingData *tilingData)
{
    // fixedBufferBytes 包含 cumsumInfoTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ops::CeilAlign(
            static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank * sizeof(int32_t),
            static_cast<uint64_t>(ALIGN_32))) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->moeExpertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    return fixedBufferBytes;
}

/*
 * 分两步定下 dispatch 阶段的 UB 分配：
 *   第一步：先按一个保守的 route batch 大小起步，把剩下的 UB 全拿去开 ring slot，slot 越多流水越深；
 *   第二步：ring 深度定死之后，UB 若还有富余，再反过来把 route batch 撑大，这样总批数更少。
 * 预算减法使用饱和计算，避免 UB 不足时无符号回绕；bufferCount 必须先限制上限，再保证最小 ring 深度。
 */
static void SelectDispatchRingAndRouteBatch(MegaFusedMoeDispatchBufferConfig &bufferConfig, uint64_t sendTotalNum,
                                            uint64_t alignedTotalRouteItems, uint32_t fixedBufferBytes,
                                            uint32_t dispatchSlotBytes, uint32_t availableUbBytes)
{
    // 第一步：用基准 batch 把 ring 深度定下来。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));

    // MTE 接收侧只保留一个 int32 topK 有效下标 batch，不再分配 mask 和第二个 index tensor。
    // routeIndexBufferBytes 是该有效下标 tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含 count/prefix 固定区和一份有效下标 tensor。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + routeIndexBufferBytes;
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 dispatch slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // 第二步：ring 深度已定，剩余 UB 用来扩 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的全部 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是有效下标 tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        uint32_t expandedRouteItems = routeItemBudgetBytes / static_cast<uint32_t>(sizeof(int32_t));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
        }
    }
}

static MegaFusedMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaFusedMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaFusedMoeDispatchBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank);
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));
    uint32_t copyBufferBytes = CalcDispatchCopyBufferBytes(tilingData, activationElementsPerByte);
    bufferConfig.copyBufferBytes = copyBufferBytes;
    uint32_t fixedBufferBytes = CalcDispatchFixedBufferBytes(tilingData);
    // 一个 dispatch ring slot 包含 token/scale copy buffer 和一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);

    SelectDispatchRingAndRouteBatch(bufferConfig, sendTotalNum, alignedTotalRouteItems, fixedBufferBytes,
                                    dispatchSlotBytes, availableUbBytes);
    return bufferConfig;
}

static uint64_t CalcTopkValidIndexRingSlotBytes(uint32_t routeItemsPerBatch, uint32_t topK)
{
    // routeItemsPerBatch 按 256 个 item 对齐，但 topK 不一定整除 256，因此后续 batch 可能从某个
    // token 的 topK 段中间开始。同一 token 的 topK 专家不重复，单个专家每个 token 至多匹配一个下标；
    // 一个 batch 最多跨越 CeilDiv(routeItemsPerBatch + topK - 1, topK) 个 token，slot 按此上界预留空间。
    uint64_t maxMatchedRouteItems =
        ops::CeilDiv(static_cast<uint64_t>(routeItemsPerBatch) + topK - 1U, static_cast<uint64_t>(topK));
    uint64_t validIndexBytes = ops::CeilAlign(maxMatchedRouteItems * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32));
    return static_cast<uint64_t>(routeItemsPerBatch) / BITS_PER_BYTE + validIndexBytes;
}

// MTE Wave producer：ring slot 同时保存临时 compare mask 与本专家的 topK 有效下标。
static MegaFusedMoeSendMaskBufferConfig CalcTopkValidIndexBufferConfig(const MegaFusedMoeTilingData *tilingData,
                                                                  uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                                  uint32_t availableUbBytes)
{
    MegaFusedMoeSendMaskBufferConfig bufferConfig{};
    // sendTotalNum 表示所有专家合计最多发送的 topK 有效下标数，不是 token 数。发送批网格按
    // numMaxTokensPerRank * topK 的容量上界划分，确保各 Rank 使用一致批次数；Kernel 再按本 Rank
    // 的实际 bs * topK 对每批有效长度进行裁剪。
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->topK;
    uint64_t alignedTotalRouteItems = ops::CeilAlign(sendTotalNum, static_cast<uint64_t>(ALIGN_256));

    // Stage 1：使用基准 batch 确定 route ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>(ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
    bufferConfig.bufferBytes = static_cast<uint32_t>(
        CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(bufferConfig.routeItemsPerBatch), tilingData->topK));

    // topkIdsTensor 和 gather 输出 tensor 各占一份 int32 route batch。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t bytesWithoutRouteBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // routeBufferBudgetBytes 是扣除固定 tensor 和两份 route tensor 后可用于 ring slot 的 UB。
    uint32_t routeBufferBudgetBytes =
        availableUbBytes > bytesWithoutRouteBuffers ? availableUbBytes - bytesWithoutRouteBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(routeBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // ring 深度超过当前核的实际发送次数不会增加流水重叠。
    uint64_t routePushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (routePushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > routePushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(routePushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。容量公式包含 topK 有效下标的占用。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // 每个 slot 预留有效下标的 32B 对齐余量和 batch 边界余量，其余 UB 用于随 batch 线性增长的数据区。
        uint64_t fixedBytesWithRoutePadding =
            static_cast<uint64_t>(fixedBufferBytes) +
            static_cast<uint64_t>(bufferConfig.bufferCount) * (ALIGN_32 + 2U * sizeof(int32_t));
        uint64_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithRoutePadding ? availableUbBytes - fixedBytesWithRoutePadding : 0U;
        // 两份 route tensor 各占 32bit/item；每个 ring slot 占 1bit mask 和约 32/topK bit 有效下标。
        uint64_t expandedRouteItems =
            routeItemBudgetBytes * BITS_PER_BYTE /
            (2U * sizeof(int32_t) * BITS_PER_BYTE + static_cast<uint64_t>(bufferConfig.bufferCount) +
             ops::CeilDiv(static_cast<uint64_t>(bufferConfig.bufferCount) * sizeof(int32_t) * BITS_PER_BYTE,
                          static_cast<uint64_t>(tilingData->topK)));
        expandedRouteItems = expandedRouteItems / ALIGN_256 * ALIGN_256;
        expandedRouteItems = std::min(expandedRouteItems, alignedTotalRouteItems);
        if (expandedRouteItems > static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>(
                ops::CeilDiv(sendTotalNum, static_cast<uint64_t>(bufferConfig.routeItemsPerBatch)));
            bufferConfig.bufferBytes = static_cast<uint32_t>(
                CalcTopkValidIndexRingSlotBytes(static_cast<uint32_t>(expandedRouteItems), tilingData->topK));
        }
    }
    return bufferConfig;
}

/*
 * 计算 unpermute 的 slot 与 scale 尺寸：单 token 的 BF16 搬入区 + FP32 计算区，以及 combine quant 的 scale 展开区。
 * dataSlotBytes 与 scaleBytes 经出参带出，供后续两个阶段共用。
 */
static void CalcUnpermuteSlotAndScaleBytes(const MegaFusedMoeTilingData *tilingData,
                                           MegaFusedMoeUnpermuteBufferConfig &bufferConfig, uint32_t &dataSlotBytes,
                                           uint32_t &scaleBytes)
{
    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }
}

/*
 * 按 weight 元素个数刷新 unpermute 的 weight 区大小：FP32 主区必算，转换中转区只有需要 dtype
 * 转换时才占 UB。两个阶段定完 tokensPerBatch 后都要重算一次，所以抽出来复用。
 */
static void SetUnpermuteWeightBufferBytes(MegaFusedMoeUnpermuteBufferConfig &bufferConfig, uint32_t weightElementCount,
                                          uint32_t topKWeightsConversionElementBytes)
{
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }
}

static MegaFusedMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaFusedMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaFusedMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t dataSlotBytes = 0U;
    uint32_t scaleBytes = 0U;
    CalcUnpermuteSlotAndScaleBytes(tilingData, bufferConfig, dataSlotBytes, scaleBytes);

    // 第一步：先按基准 weight batch 定住每批 token 数和 weight 区，再拿剩余 UB 开输入 ring。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // 第二步：ring 深度已定，UB 还有富余就把 weight batch 撑大。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            SetUnpermuteWeightBufferBytes(bufferConfig, weightElementCount, topKWeightsConversionElementBytes);
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaFusedMoeTilingData *tilingData)
{
    // MTE 统一使用 per-expert AIC ready 表；group counter 服务 URMA layered Combine，量化与非量化均需要。
    if (tilingData->topoType != TOPO_TYPE_URMA || tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    // 上述 guard 保证这里只剩 URMA；layered Combine 仅由 subBlockIdx=1 的半数 AIV 执行。
    uint64_t combineCoreCount = tilingData->blockAivNum / 2U;
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, combineCoreCount);
}

static uint64_t CalcHostFlagElementCount(const MegaFusedMoeTilingData *tilingData)
{
    uint64_t maxWavesPerExpert = ops::CeilDiv<uint64_t>(tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t waveFlagSlotsPerExpert = maxWavesPerExpert * INT_CACHELINE;
    uint64_t activationFlagSlotsPerExpert =
        tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : INT_CACHELINE;
    uint64_t moeExpertCount = tilingData->moeExpertPerRank;

    uint64_t flagElementCount = moeExpertCount * (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
                                                  static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                    tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
    if (isW4Mode || (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
        flagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (tilingData->topoType == TOPO_TYPE_URMA) {
        flagElementCount += tilingData->combineSyncSlotCountPerExpert * moeExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0 && tilingData->topoType == TOPO_TYPE_MTE) {
        uint64_t tokenGroupCount = ops::CeilDiv<uint64_t>(tilingData->bs, L1_TILE_M_256);
        flagElementCount += tokenGroupCount * tilingData->sharedExpertNum * INT_CACHELINE;
        flagElementCount +=
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tilingData->bs))) *
            tilingData->sharedExpertNum;
    }
    return flagElementCount;
}

/*
 * 设置 topK 有效下标发送的两套 UB 配置：先算固定占用，再按 expert 分核的两类 core 各算一套。
 * 本函数读 tilingData->combineSyncSlotCountPerExpert（经 CalcHostFlagElementCount），
 * 该字段必须在调用前写好。
 */
static void SetTopkValidIndexBufferConfigs(MegaFusedMoeTilingData *tilingData, uint32_t activationElementsPerByte,
                                           uint32_t availableUbBytes)
{
    uint64_t totalFlagElementCount = CalcHostFlagElementCount(tilingData);
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 MoE 专家数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->splitMoeRanks * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel topK 有效下标发送的 expert 连续均衡分核一一对应。totalExpertCount 除以 blockAivNum 后，
     * 前 remainder 个 AIV job 各多处理一个 expert，其余 job 处理 quotient 个 expert，因此这里只需
     * 预计算两套配置。Dispatch/Combine 的 wave 内 token 轮转由各自阶段完成，与这里的一次性发送分核无关。
     *
     * 若修改发送阶段的 expert 分核方式、ownedExpertCount 或 routePushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    uint32_t totalExpertCount = tilingData->splitMoeRanks * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcTopkValidIndexBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);
}

/*
 * 设置 Unpermute 的完整 chunk 与 tail chunk 两套 UB 配置，并记录完整 chunk 对应的 core 数。
 */
static void SetUnpermuteBufferConfigs(MegaFusedMoeTilingData *tilingData, ge::DataType topKWeightsDataType,
                                      uint32_t availableUbBytes)
{
    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    if (tilingData->bs == 0U) { return; }
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaFusedMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
}


static ge::graphStatus Tile(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *attrs = context->GetAttrs();
    auto *values = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    auto *members = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto *activation = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    if (!values || values->GetSize() != 17 || !members || members->GetSize() == 0 ||
        members->GetSize() > 768 || !activation || activation->GetSize() != 3) { return ge::GRAPH_FAILED; }
    auto *config = reinterpret_cast<const int64_t *>(values->GetData());
    auto *dp = reinterpret_cast<const int64_t *>(members->GetData());
    auto *parameters = reinterpret_cast<const float *>(activation->GetData());
    if (config[0] < 1 || config[1] < 1 || config[0] + config[1] > 768 || config[2] < config[1] ||
        config[2] > 2048 || (config[0] + config[1]) * (config[2] / config[1]) > 16384 || config[2] % config[1] != 0 || config[2] / config[1] > 1024 ||
        config[3] < 1 || config[3] >= (1 << 24) || config[4] < 1024 || config[4] > 8192 || config[4] % 64 ||
        config[5] < 512 || config[5] > 8192 || config[5] % 256 || config[6] < 1 || config[6] > 32 ||
        config[6] > config[2] || (config[7] != 3 && config[7] != 4) ||
        (config[8] != 0 && config[8] != 3 && config[8] != 4) || (config[9] != 0 && config[9] != 1) ||
        config[11] < 1 || config[11] > AfdAsync::SPLIT_SEQUENCE_MAX || config[12] < 0 || config[12] > 4 ||
        config[13] < 0 || config[13] > config[3] || config[14] < 1 || config[14] > AfdAsync::SPLIT_SEQUENCE_MAX ||
        config[15] < 0 || config[15] > 3 || config[16] < 0 || config[16] > 1) {
        OP_LOGE(nodeName, "Invalid split configuration; see docs/mega_split/README.md");
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < members->GetSize(); ++index) {
        if (dp[index] < 0 || dp[index] >= config[0]) { return ge::GRAPH_FAILED; }
        for (size_t previous = 0; previous < index; ++previous) {
            if (dp[previous] == dp[index]) { return ge::GRAPH_FAILED; }
        }
    }
    auto *td = context->GetTilingData<MegaFusedMoeTilingData>();
    if (!td) { return ge::GRAPH_FAILED; }
    *td = {};
    td->splitAttentionRanks = config[0];
    td->splitMoeRanks = config[1];
    td->epWorldSize = config[0] + config[1];
    td->moeExpertPerRank = config[2] / config[1];
    td->numMaxTokensPerRank = config[3];
    td->h = config[4];
    td->hiddenDim = config[5];
    td->topK = config[6];
    td->combineQuantMode = config[8];
    td->topkWeightsPrefetch = config[9];
    td->splitSequence = config[11];
    td->sharedExpertNum = config[12];
    td->bs = config[13];
    td->splitLaunch = config[14];
    td->actMode = config[15];
    td->actSubMode = config[16];
    td->clampLimit = parameters[0];
    td->activationAlpha = parameters[1];
    td->activationBeta = parameters[2];
    td->topoType = 0;
    td->rankNumPerServer = td->epWorldSize;
    td->splitDpSize = members->GetSize();
    std::copy(dp, dp + members->GetSize(), td->splitDpRanks);
    uint64_t maxOutput = uint64_t(config[3]) * members->GetSize() * std::min(config[6], config[2] / config[1]);
    if (maxOutput > INT32_MAX) { return ge::GRAPH_FAILED; }
    td->maxOutputSize = maxOutput;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    td->aicNum = platform.GetCoreNumAic();
    td->blockAivNum = platform.GetCoreNumAiv();
    td->blockNumPerEP = td->aicNum;
    if (td->aicNum == 0 || td->aicNum > 192 || td->blockAivNum != td->aicNum * 2) { return ge::GRAPH_FAILED; }
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    context->SetBlockDim(platform.CalcTschBlockDim(td->blockAivNum, td->aicNum, td->blockAivNum));
    context->SetScheduleMode(1);
    td->isPerExpertWeightTensor = false;
    td->groupedMatmulMode = GROUPED_MATMUL_MODE_GENERAL;
    if (true) {
        auto *shape = context->GetDynamicInputShape(1, 0);
        auto *desc = context->GetDynamicInputDesc(1, 0);
        if (!shape || !desc) { return ge::GRAPH_FAILED; }
        if (shape->GetOriginShape().GetDimNum() != 2U && shape->GetOriginShape().GetDimNum() != 3U) {
            return ge::GRAPH_FAILED;
        }
        td->isPerExpertWeightTensor = shape->GetOriginShape().GetDimNum() == 2;
        if (!CheckWeightBank(context, td)) {
            OP_LOGE(nodeName, "Expected matching layer-major weight banks: L*N matrices or L tensors of N experts");
            return ge::GRAPH_FAILED;
        }
        auto dtype = desc->GetDataType();
        auto *second = context->GetDynamicInputDesc(2, 0);
        if (!second || second->GetDataType() != dtype) { return ge::GRAPH_FAILED; }
        auto firstFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
        auto secondFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(second->GetStorageFormat()));
        if (dtype == ge::DT_FLOAT4_E2M1) {
            if (desc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32 ||
                second->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32) { return ge::GRAPH_FAILED; }
        } else if (firstFormat != secondFormat ||
                   (firstFormat != ge::FORMAT_ND && firstFormat != ge::FORMAT_FRACTAL_NZ) ||
                   (dtype == ge::DT_FLOAT8_E5M2 && firstFormat != ge::FORMAT_ND)) { return ge::GRAPH_FAILED; }
        if (dtype == ge::DT_FLOAT4_E2M1) {
            if (config[7] != 4) { return ge::GRAPH_FAILED; }
            td->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
        } else if ((dtype == ge::DT_FLOAT8_E4M3FN && config[7] == 4) ||
                   (dtype == ge::DT_FLOAT8_E5M2 && config[7] == 3)) {
            if (firstFormat == ge::FORMAT_FRACTAL_NZ) { td->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W8_NZ; }
        } else { return ge::GRAPH_FAILED; }
    }
    td->sharedExpertNum = 0; // MoE ranks never allocate dense Attention experts.
    SetPrefetchAndWaveParams(td, td->aicNum);
    td->combineSyncSlotCountPerExpert = 0;
    td->dispatchBufferConfig = CalcDispatchBufferConfig(td, 1U, ub);
    PeermemSizeParams sizes{int64_t(td->numMaxTokensPerRank), int64_t(td->topK), int64_t(td->h),
        int64_t(td->moeExpertPerRank), int64_t(td->epWorldSize), 2, 1, bool(td->topkWeightsPrefetch),
        bool(td->combineQuantMode), 0, 1, td->splitAttentionRanks};
    if (CalcPeermemLeastSize(sizes) + EXCEPTION_DUMP_REGION_SIZE > config[10]) {
        OP_LOGE(nodeName, "Split communication window is too small");
        return ge::GRAPH_FAILED;
    }
    WorkspaceLayout workspace(td);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + workspace.workspaceSize + 50 * 1024 * 1024;
    context->SetTilingKey(GET_TPL_TILING_KEY(4, config[7], config[8], 0, bool(config[9])));
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
}
IMPL_OP_OPTILING(AfdAsyncMegaFfnArch35).Tiling(Tile).TilingParse<CompileInfo>(Parse);
}
