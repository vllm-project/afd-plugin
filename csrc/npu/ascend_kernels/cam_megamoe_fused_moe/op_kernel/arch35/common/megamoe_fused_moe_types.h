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
 * Description: Provide shared megamoe fused moe types definitions for split MegaMoE execution.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#ifndef MEGAMOE_FUSED_MOE_TYPES_H
#define MEGAMOE_FUSED_MOE_TYPES_H

#include "lib/std/tuple.h"
#include "tensor_api/tensor.h"
// ### PATCH START: AFD merged source layout
#include "../../cam_megamoe_fused_moe_tiling.h"
// ### PATCH END: AFD merged source layout
#include "megamoe_fused_moe_constants.h"
#include "megamoe_fused_moe_workspace.h"
#if defined(ENABLE_MEGAMOE_FUSED_MOE_LAYERED_KERNEL)
#include "adv_api/hcomm/hcomm.h"
#endif
#if __has_include("../../../common/mc2_moe_context.h")
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/mc2_moe_context.h"
// ### PATCH END: AFD merged source layout
#else
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/mc2_moe_context.h"
// ### PATCH END: AFD merged source layout
#endif

namespace MegamoeFusedMoeImpl {

using namespace AscendC;

enum DispatchQuantOutDtype : int64_t {
    E5M2_QUANT = 3U,
    E4M3_QUANT = 4U,
    E2M1_QUANT = 5U,
};

using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

// GMM1/GMM2 逐专家遍历的公共状态。expertIdx 标识当前专家，globalTokenStartIndex 表示该专家
// 在本卡 MoE 专家紧凑 token 序列中的起始索引。
struct ExpertLoopState {
    ProblemShape problemShape;
    int64_t globalTokenStartIndex = 0;
    uint32_t expertIdx = 0U;
    bool expertCountTableReady = false;
};

// GMM1 执行期间共同维护的流水状态；引用成员将更新直接回写到调用方持有的状态。
struct GmmRuntimeState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    uint16_t &pingpongIdx;
};

// 标识 MoE 专家序列中的二维 token 位置。
struct ExpertTokenPosition {
    uint32_t expertIdx = 0U;
    uint32_t tokenIndexInExpert = 0U;
    // 从全部本卡 MoE 专家起点累计的全局连续 row，避免末波 Combine 重新标量扫描 count 表。
    uint64_t globalTokenIndex = 0U;
};

// 标识 MoE 专家紧凑 token 序列中的左闭右开区间 [begin, end)。
struct ExpertTokenRange {
    ExpertTokenPosition begin{};
    ExpertTokenPosition end{};
};

using Mc2MoeContext = Mc2Aclnn::Mc2MoeContext;

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR gmm1OutGlobal;
    GM_ADDR gmm2OutGlobal;
    GM_ADDR metaInfoGlobal;
    __gm__ int32_t *activationToGmm2Flag;
    __gm__ int32_t *dispatchToGmm1Flag;
    __gm__ int32_t *gmm2CombineSyncCounter;
    __gm__ int32_t *gmmToEpilogueFlag;
    __gm__ int32_t *gmm1TileStatus;
    __gm__ int32_t *sharedExpertGmm2TileCounter;
    uint32_t gmm2CombineLogicalCoreCount = 0U;
};

#if defined(ENABLE_MEGAMOE_FUSED_MOE_LAYERED_KERNEL)
struct CombineCommParams {
    Hcomm<COMM_PROTOCOL_UBC_CTP> *hcomm;
};
#endif

// 保存 TensorList 入口地址，供按 expert 布局解析当前专家权重。
struct ExpertWeightTensorListAddrs {
    GM_ADDR weight1 = nullptr;
    GM_ADDR weightScales1 = nullptr;
    GM_ADDR weight2 = nullptr;
    GM_ADDR weightScales2 = nullptr;
    uint32_t layerIndex = 0U;
    uint32_t localExperts = 0U;
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR sharedBGmAddr;
    GM_ADDR sharedBScaleGmAddr;
    GM_ADDR sharedB2GmAddr;
    GM_ADDR sharedB2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegamoeFusedMoeTilingData *tilingData;
#if defined(ENABLE_MEGAMOE_FUSED_MOE_LAYERED_KERNEL)
    CombineCommParams combineCommParams;
#endif
};

enum class AddrUpdateMode : int32_t {
    GMM1,
    GMM2
};

struct BlockJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

// Count/flag workspace 的物理分区；当前与 BlockJobContext 同值，但其编号由 workspace 生产者和消费者共同约定。
struct BlockWorkspaceContext {
    uint32_t blockIdx;
    uint32_t blockNum;
};

template <typename T>
struct PackedElementTraits {
    static constexpr uint32_t ELEMENTS_PER_BYTE = Std::IsSame<T, fp4x2_e2m1_t>::value ? 2U : 1U;
};

struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

struct MoeStageCommonConfig {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t moeExpertPerRank;
    uint32_t sharedExpertNum;
    uint32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
};

// GMM1/GMM2 共用的执行方式：当前 block 的任务分工、矩阵模板模式和专家权重布局。
struct GmmExecutionConfig {
    BlockJobContext blockJob;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

// 各流水阶段在同步 workspace 中为每个专家预留的 slot 数量。
struct MoeSyncWorkspaceLayout {
    int32_t dispatchFlagSlotCountPerExpert;
    int32_t activationFlagSlotCountPerExpert;
    uint32_t gmm1TileStatusCountPerExpert;
};

struct GroupSyncSlotLayout {
    uint32_t baseSlotCountPerGroup;
    uint32_t extraSlotGroupCount;
};

} // namespace MegamoeFusedMoeImpl

#endif // MEGAMOE_FUSED_MOE_TYPES_H
