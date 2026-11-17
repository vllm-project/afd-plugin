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
 * Description: Define the host-to-kernel tiling parameter layout.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file megamoe_dispatch_send_tiling.h
 * \brief
 */

#ifndef ASCENDC_MEGAMOE_DISPATCH_SEND_TILING
#define ASCENDC_MEGAMOE_DISPATCH_SEND_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// MegamoeDispatchSend adaptive UB buffer policy shared by host tiling and kernel address binding.
// 实际 UB 预算由 host tiling 从平台获取，并通过 availableUbBytes 参与以下配置计算。

/*
 * Unpermute buffer policy
 *
 * Unpermute 保留 1 个累加/输出 buffer，并支持在 EVENT_ID0~EVENT_ID5 上运行 2~6 个输入 buffer。
 * 实际 UB 仅按 tiling 选中的输入 buffer 数量分配；该上限不表示固定预留 6 个 buffer。
 *
 * 最低双 buffer 是基准 batch 的成立前提。按最坏合法规格 k=8192、BF16 topK weight、combine quant 计算：
 *   1 个累加槽 + 2 个输入槽 = 3 * (8192 * 2B + 8192 * 4B) = 144KB；
 *   8192 个 weight item       = 8192 * (4B + 2B)              = 48KB；
 *   BF16/FP32 scale scratch   = 1KB + 4KB                     = 5KB；
 *   合计约 197KB；当前支持平台动态获取的 UB 预算可容纳该布局。
 * 若修改 MAX_H、weight batch 基准、scale scratch 布局或最低 buffer 数，必须重新验证该前提。
 *
 * weight batch 延续原基准：tokensPerBatch * topK 不超过 1024 * 8 个 weight item。
 * CalcUnpermuteBufferConfig 先按该基准确定 batch，再用剩余 UB 反推输入 buffer 数量；上述最坏规格
 * 已保证基准 batch 至少容纳 2 个输入 buffer，因此计算结果可以安全保底到最低 buffer 数。
 *
 * 固定输入 buffer 数量后扩大 weight batch 时，为 FP32 weight 和可选输入转换 buffer 各预留一个
 * 32B 对齐块。统一预留 64B 可覆盖两个独立 buffer 各最多 31B 的 padding。
 *
 * DeQuantMxFp8 的 E8M0 scale 经 DIST_INTLV_B16 交织复制为 2 个 BF16，随后经 DIST_INTLV_B32
 * 交织复制为 4 个 FP32。因此 BF16 scratch 需要 scaleNum * 2 个元素，FP32 scratch 需要
 * scaleNum * 4 个元素；若修改 scale scratch 布局，需同步更新展开倍数。
 */
constexpr int32_t MIN_UNPERMUTE_INPUT_BUFFER_COUNT = 2;
constexpr int32_t MAX_UNPERMUTE_INPUT_BUFFER_COUNT = 6;
constexpr int32_t UNPERMUTE_BASE_TOPK = 8;
constexpr int32_t UNPERMUTE_BASE_TOKENS_PER_BATCH = 1024;
constexpr int32_t UNPERMUTE_WEIGHT_ITEMS_PER_BATCH = UNPERMUTE_BASE_TOPK * UNPERMUTE_BASE_TOKENS_PER_BATCH;
constexpr uint32_t UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES = 64U;
constexpr uint32_t DEQUANT_BF16_SCALE_EXPANSION = 2U;
constexpr uint32_t DEQUANT_FP32_SCALE_EXPANSION = 4U;

/*
 * Route batch shared policy
 *
 * Dispatch 和 SendMask 的 route batch 均按 ALIGN_256 个 item 对齐，使 bit mask 天然按 32B 对齐。
 * 两个基准值当前数值相同，只是各自 UB 预算计算后的结果，彼此独立，不要求保持相等。
 */

/*
 * Dispatch buffer policy
 *
 * Recv route batch 基准用于先选择 dispatch buffer 数量，再固定 buffer 数反推最终 batch。
 * Brecv 表示基准 batch 的 route item 数，D 表示 dispatch buffer 数：
 *   recvVariableBytes(Brecv, D) = Brecv / 8 + 2 * Brecv * sizeof(int32_t)
 *                                  + D * (tokenScaleBytes + 32)；
 *   recvFixedBytes = 32 + Align32(worldSize * moeExpertPerRank * 4) + worldSize * 32
 *                    + Align32(moeExpertPerRank * 4)；
 *   recvFixedBytes + recvVariableBytes(Brecv, D) <= availableUbBytes。
 *
 * 12288 的成立前提按 k=8192、moeExpertNum<=2048、worldSize<=1024、moeExpertPerRank<=1024 和 D=6
 * 保守验证。分别取各 tensor 的独立上限时：
 *   recvFixedBytes             <= 32 + 8192 + 32768 + 4096 = 45088B；
 *   route batch tensor         = 12288 / 8 + 2 * 12288 * 4 = 99840B；
 *   6 个 dispatch slot         <= 6 * (8192B token + 256B scale + 128B weight + 32B triple)
 *                               = 51648B；
 *   合计 196576B，约 192KB；当前支持平台动态获取的 UB 预算可容纳该布局。
 * 因此该基准在所有合法规格下至少支持双 buffer，且最坏规格也能支持当前上限 6 个 buffer。若修改
 * MAX_H、MAX_MOE_EXPERT_NUM、worldSize/moeExpertPerRank 上限、DispatchBuffInit 固定 tensor 布局或
 * dispatch buffer 数量上限，必须重新验证该前提。
 *
 * CalcDispatchBufferConfig 固定选中的 buffer 数后，用剩余 UB 扩大 batch，并向下对齐到 ALIGN_256。
 * 每个 route item 占 1 bit mask 和两个 int32 index，因此：
 *   maxRouteItems = routeItemBudgetBytes * 8 / 65。
 * CopyGMToGMPerToken 的 copyTmp、triple 和 event ID 三类槽位一一对应，统一使用 2~6 个 buffer。
 */
constexpr int32_t MIN_DISPATCH_BUFFER_COUNT = 2;
constexpr int32_t MAX_DISPATCH_BUFFER_COUNT = 6;
constexpr int32_t BASE_RECV_ROUTE_ITEMS_PER_BATCH = 12288;

/*
 * SendMask buffer policy
 *
 * Send route batch 基准用于先选择 mask buffer 数量，再固定 buffer 数反推最终 batch。
 * Bsend 表示基准 batch 的 route item 数，N 表示 mask buffer 数：
 *   sendVariableBytes(Bsend, N) = 2 * Bsend * sizeof(int32_t) + N * (Bsend / 8 + 32)；
 *   sendFixedBytes = resetTensorBytes + 2048 + 2 * xOutTensorBytes + 2 * xInTensorBytes
 *                    + sendCntAccBytes；
 *   sendFixedBytes + sendVariableBytes(Bsend, N) <= availableUbBytes。
 *
 * 12288 的成立前提按最坏合法规格 k=8192、moeExpertNum=2048、N=6 保守验证：
 *   resetTensor                 <= 8KB；
 *   mxTempTensor                = 2KB；
 *   2 个 xOutTensor             <= 2 * (8192B token + 256B scale + 128B weight) = 17152B；
 *   2 个 xInTensor              = 2 * 8192 * sizeof(bfloat16_t) = 32KB；
 *   sendCntAccTensor            <= Align32(2048 * sizeof(int32_t)) = 8KB；
 *   2 个 route int32 tensor     = 2 * 12288 * sizeof(int32_t) = 96KB；
 *   6 个 mask buffer            = 6 * (12288 / 8 + 32) = 9408B；
 *   合计 176064B，约 172KB；当前支持平台动态获取的 UB 预算可容纳该布局。
 * 因此基准 batch 在所有合法规格下至少支持双 buffer，且最坏规格也能支持当前上限 6 个 buffer。
 * 若修改 MAX_H、MAX_MOE_EXPERT_NUM、DISPATCH_RESET_BATCH、Quant buffer 布局或 mask buffer 数量上限，
 * 必须重新验证该前提。
 *
 * CalcTopkValidIndexBufferConfig 固定选中的 buffer 数后，用剩余 UB 扩大 batch，并向下对齐到 ALIGN_256。
 * SendMaskCal 使用 MTE3_V 的 EVENT_ID0~EVENT_ID5 管理 mask push ring。最低保留双 buffer，最多 6 个。
 */
constexpr int32_t MIN_SEND_MASK_BUFFER_COUNT = 2;
constexpr int32_t MAX_SEND_MASK_BUFFER_COUNT = 6;
constexpr int32_t BASE_SEND_ROUTE_ITEMS_PER_BATCH = 12288;

/*
 * Reset buffer policy
 *
 * ResetFlagList 分批清零的粒度，单位为 int32 元素。2048 * sizeof(int32_t) = 8KB；本核 flag 份额
 * 超过该值时复用同一片零 UB 分批推 GM，使 reset UB 与 BS 解耦。若调整 resetTensor 的 UB 预算或
 * 元素类型，需要按 targetResetUbBytes / sizeof(elementType) 重新计算。
 */
constexpr int32_t DISPATCH_RESET_BATCH = 2048;

// GMM2 M 方向的一个 tile 对应一个 combine token group；同步 slot 的粒度必须与该 tile 保持一致。
constexpr uint32_t COMBINE_TOKEN_GROUP_SIZE = 256U;

// Adaptive buffer configs written by host tiling and consumed by the kernel.
struct MegamoeDispatchSendUnpermuteBufferConfig {
    int32_t tokensPerBatch;
    int32_t inputBufferCount;
    uint32_t bf16SlotElementCount;
    uint32_t fp32SlotElementCount;
    uint32_t topKWeightsBufferBytes;
    uint32_t topKWeightsConversionBufferBytes;
};

struct MegamoeDispatchSendDispatchBufferConfig {
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
    int32_t bufferCount;
    uint32_t copyBufferBytes;
};

struct MegamoeDispatchSendSendMaskBufferConfig {
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
    int32_t bufferCount;
    uint32_t bufferBytes;
};

struct MegamoeDispatchSendTilingData {
    uint32_t moeExpertPerRank; // 本卡参与 topK 路由的 MoE 专家数，与 weight1 表达的专家数一致
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;
    uint32_t maxOutputSize;
    uint32_t topK;
    uint32_t aicNum;
    uint32_t blockAivNum;
    int64_t combineQuantMode;
    float clampLimit;
    uint8_t groupedMatmulMode;
    int64_t topoType;
    uint32_t sharedExpertNum; // 独立 dense 路径的共享专家数，不进入 topK/SendMask expert id 空间

    // 每个 MoE 专家固定预留的 GMM2 -> Combine 同步 slot 数。
    uint64_t combineSyncSlotCountPerExpert;

    // Dispatch 的分核不改变 UB 布局，所有 AIV core 共用一套配置；对应 kernel DispatchBuffInit。
    MegamoeDispatchSendDispatchBufferConfig dispatchBufferConfig;
    // SendMask 将全局 expertId 连续均分，前 remainder 个 core 会多处理一个 expert；
    // 对应 kernel SendAndQuantBuffInit 的选择条件和 SendMaskCal 的连续 expert 区间。
    MegamoeDispatchSendSendMaskBufferConfig sendMaskConfigForCoreWithExtraExpert;
    MegamoeDispatchSendSendMaskBufferConfig sendMaskConfigForCoreWithoutExtraExpert;
    uint32_t sendMaskCoreCountWithExtraExpert;
    // Unpermute 的 TilingByCore 生成若干完整 token chunk 和至多一个尾 chunk，分别预计算配置；
    // 对应 kernel UnpermuteBuffInit 的选择条件。
    MegamoeDispatchSendUnpermuteBufferConfig unpermuteConfigForFullTokenChunk;
    MegamoeDispatchSendUnpermuteBufferConfig unpermuteConfigForTailTokenChunk;
    uint32_t unpermuteFullTokenChunkCoreCount;
    // 调度扩展字段统一追加在末尾，保持已有字段偏移不变。
    int32_t topkWeightsPrefetch;
    uint32_t maxTilesPerExpert; // GMM1 tile 状态位区每 expert 的容量，按交织调度的完整 N 上界预留
    uint8_t actMode;            // MegamoeDispatchSendImpl::MegamoeDispatchSendActMode 的底层协议值
    uint8_t actSubMode;         // MegamoeDispatchSendImpl::MegamoeDispatchSendActSubMode 的底层协议值
    float activationAlpha;      // situ: linear_beta; swigluoai: alpha(默认1.702); swiglu/swiglustep: 占位
    float activationBeta;       // situ/swigluoai: beta(默认1.0); swiglu/swiglustep: 占位
    uint32_t mGroupsPerWave;    // 每个 routed expert Wave 消费的 256-row M 分组数
    // MoE 和共享专家使用相同布局：true 表示每个专家一个二维 tensor，false 表示单个三维堆叠 tensor。
    bool isPerExpertWeightTensor;
    // 全卡一致的单卡 token 数上界（attr num_max_tokens_per_rank）。
    // MTE 保留 attr=0 时回退到 bs 的兼容语义；URMA 必须显式传入非 0 上界。
    // 跨卡窗口和 mask 槽几何使用该上界，bs 仅描述本卡真实 token 数。
    uint32_t numMaxTokensPerRank;
    // URMA 分层拓扑；Host workspace 计算与 Device 通道映射必须使用同一个值。
    uint32_t rankNumPerServer;
    // Split-only fields; original payload geometry above remains unchanged.
    uint32_t splitAttentionRanks;
    uint32_t splitMoeRanks;
    uint32_t splitRank;
    uint32_t splitSequence;
    uint32_t splitLaunch;
    uint32_t splitDpSize;
    uint32_t splitDpRanks[768];
    uint32_t splitLayerIndex;
};
#endif
