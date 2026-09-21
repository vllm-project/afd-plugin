/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: grouped_matmul_swiglu_quant_v2_layered_a4w4_post header file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_a4w4_post.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_A4W4_POST_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_A4W4_POST_H

#include "grouped_matmul_swiglu_quant_v2_layered_utils.h"
#include "kernel_operator.h"

#ifdef GMM_SWIGLU_QUANT_V2_A4W4

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA4W4PostProcess {
  public:
    __aicore__ inline GMMA4W4PostProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
                                const GMMSwigluQuantV2 *__restrict gmmSwigluIN);

    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe *pipe);

  private:
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                           WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                           TPipe *pipe);

    __aicore__ inline void VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                         WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyOut(VecConfig &vecConfig, WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void Quant(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void Swiglu(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void InitCoreParams(const GMMSwigluQuantV2BaseParams *baseParams);
    __aicore__ inline void InitGmTensors(const GMAddrParams &gmAddrParams, const GMMSwigluQuantV2 *gmmSwigluIN);
    __aicore__ inline void InitSmoothScaleGm(const GMAddrParams &gmAddrParams);
    __aicore__ inline void InitMmOutGmTensors(const GMAddrParams &gmAddrParams);

    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<float> smoothScaleGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    TQue<QuePosition::VECIN, 1> mmOutQueue;
    TQue<QuePosition::VECOUT, 1> quantOutQueue;
    TQue<QuePosition::VECOUT, 1> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
};

__aicore__ inline void GMMA4W4PostProcess::InitCoreParams(const GMMSwigluQuantV2BaseParams *baseParams)
{
    aicCoreNum = GetBlockNum();
    aivCoreNum = aicCoreNum * NUM_2;
    blockIdx = GetBlockIdx();
    gmmSwigluQuantV2BaseParams = baseParams;
}

__aicore__ inline void GMMA4W4PostProcess::InitGmTensors(const GMAddrParams &gmAddrParams,
                                                         const GMMSwigluQuantV2 *gmmSwigluIN)
{
    gmmSwigluQuantV2 = gmmSwigluIN;
    InitSmoothScaleGm(gmAddrParams);
    InitMmOutGmTensors(gmAddrParams);
    groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM, gmmSwigluQuantV2->groupListLen);
    perTokenScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.xScaleGM, gmmSwigluQuantV2BaseParams->M);
    quantOutputGM.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.yGM, gmmSwigluQuantV2BaseParams->M *
                                                                         gmmSwigluQuantV2->tokenLen /
                                                                         SWIGLU_REDUCE_FACTOR);
    quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.yScaleGM, gmmSwigluQuantV2BaseParams->M);
}

__aicore__ inline void GMMA4W4PostProcess::InitMmOutGmTensors(const GMAddrParams &gmAddrParams)
{
    mmOutGM1.SetGlobalBuffer((__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM));
    mmOutGM2.SetGlobalBuffer(
        (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
}

__aicore__ inline void GMMA4W4PostProcess::InitSmoothScaleGm(const GMAddrParams &gmAddrParams)
{
    smoothScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.smoothScaleGM);
}

__aicore__ inline void GMMA4W4PostProcess::Init(
    const GMAddrParams gmAddrParams, const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
    const GMMSwigluQuantV2 *__restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        InitCoreParams(gmmSwigluQuantV2BaseParamsIN);
        InitGmTensors(gmAddrParams, gmmSwigluIN);
    }
}

__aicore__ inline void GMMA4W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM,
                                                            VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<half> _inMMLocal_0 = mmOutQueue.DeQue<half>();
    const int64_t processNum = vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen;
    DataCopyExtParams copyParams_0{1, static_cast<uint32_t>(processNum * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams_0{false, 0, 0, 0};
    DataCopyPad(_inMMLocal_0[processNum], mmOutGM[vecConfig.curOffset], copyParams_0, padParams_0);

    mmOutQueue.EnQue(_inMMLocal_0);

    LocalTensor<half> _inMMLocal_1 = mmOutQueue.DeQue<half>();
    // 1. fp16 -> fp32
    Cast(_inMMLocal_1.ReinterpretCast<float>(), _inMMLocal_1[processNum], RoundMode::CAST_NONE, processNum);

    mmOutQueue.EnQue(_inMMLocal_1);
    LocalTensor<float> _inMMLocal_2 = mmOutQueue.DeQue<float>();

    vecConfig.curIdx += vecConfig.innerLoopNum;
    vecConfig.curOffset = vecConfig.curIdx * gmmSwigluQuantV2->tokenLen;
    mmOutQueue.EnQue(_inMMLocal_2);
}

__aicore__ inline void GMMA4W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                                         WorkSpaceSplitConfig &workspaceSplitConfig)
{
    // 1. perToken dequantization
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 2. Swiglu
    Swiglu(loopIdx, vecConfig);
    // 3. ApplySmoothScale (skipped when smoothScaleDimNum is 0, meaning smoothScale is a null pointer)
    if (gmmSwigluQuantV2BaseParams->smoothScaleDimNum != 0) {
        ApplySmoothScale(loopIdx, vecConfig, workspaceSplitConfig);
    }
    // 4. Quant
    Quant(loopIdx, vecConfig);
}

__aicore__ inline void GMMA4W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen], mmLocal[loopIdx * gmmSwigluQuantV2->tokenLen], scale,
         gmmSwigluQuantV2->tokenLen);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void GMMA4W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig &vecConfig)
{
    // High-level API swiglu
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        _inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen + gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = _inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen];

    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_V>();
    DataCopyParams repeatParams{
        1, static_cast<uint16_t>((gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0, 0};
    DataCopy(_inMMLocal[loopIdx * gmmSwigluQuantV2->tokenLen], workspaceLocal, repeatParams);

    mmOutQueue.EnQue(_inMMLocal);
}

__aicore__ inline void GMMA4W4PostProcess::ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<float> mmLocal = mmOutQueue.DeQue<float>();

    int64_t smoothScaleDimNum = gmmSwigluQuantV2BaseParams->smoothScaleDimNum;
    int64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR;
    int64_t currentTokenIdx = workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx + loopIdx;

    // Find the group that the current token belongs to
    uint32_t groupIdx = 0;
    int64_t prevM = 0;
    int64_t totalTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    for (uint32_t i = workspaceSplitConfig.rightMatrixExpertStartIndex;
         i <= workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
        int64_t currM = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            currM = groupListGM.GetValue(i);
        } else {
            totalTmp += groupListGM.GetValue(i);
            currM = totalTmp;
        }
        if (currentTokenIdx < currM) {
            groupIdx = i;
            break;
        }
        prevM = currM;
    }

    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;

    if (smoothScaleDimNum == NUM_2) {
        // smoothScale has shape (E, N/2); only the row of the current group is needed
        for (uint32_t j = 0; j < halfTokenLen; j++) {
            float scale = smoothScaleGM.GetValue(groupIdx * halfTokenLen + j);
            float val = mmLocal.GetValue(preOffset + j);
            mmLocal.SetValue(preOffset + j, val * scale);
        }
    } else if (smoothScaleDimNum == 1) {
        // smoothScale has shape (E,), which needs to be broadcast to (N/2)
        float scale = smoothScaleGM.GetValue(groupIdx);
        Muls(mmLocal[preOffset], mmLocal[preOffset], scale, halfTokenLen);
    }

    mmOutQueue.EnQue(mmLocal);
}

__aicore__ inline void GMMA4W4PostProcess::Quant(uint32_t loopIdx, VecConfig &vecConfig)
{
    LocalTensor<float> _inMMLocal = mmOutQueue.DeQue<float>();
    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;
    uint64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / BISECT;
    Abs(_inMMLocal[preOffset + gmmSwigluQuantV2->tokenLen / BISECT], _inMMLocal[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, _inMMLocal[preOffset + gmmSwigluQuantV2->tokenLen / BISECT],
                      reduceTmpLocal, static_cast<uint32_t>(halfTokenLen));
    int32_t eventIdVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    quantScaleLocal.SetValue(loopIdx, quantScale);
    quantScale = QUANT_SCALE_INT8 / reduceResLocal.GetValue(0);
    int32_t eventIdSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(_inMMLocal[preOffset], _inMMLocal[preOffset], quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(preOffset);
    int32_t tempCount = static_cast<int32_t>(halfTokenLen);
    LocalTensor<int8_t> castSpace = reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
    CastFp32ToInt8Template(quantLocal, _inMMLocal, castSpace, dstTempOffset, srcTempOffset, tempCount);
    mmOutQueue.EnQue(_inMMLocal);
    quantOutQueue.EnQue(quantLocal);
}

__aicore__ inline void GMMA4W4PostProcess::UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                                           WorkSpaceSplitConfig &workspaceSplitConfig,
                                                           int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    // Step 1: read grouplist, reduceSum, and compute the total data count
    vecConfig.M = workspaceSplitLoopIdx < workspaceSplitConfig.loopCount - 1 ? workspaceSplitConfig.notLastTaskSize
                                                                             : workspaceSplitConfig.lastLoopTaskSize;
    // Step 2: compute the core splitting
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx =
        blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t prevM = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
    int64_t totalTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex;
         groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++) {
        int64_t currM = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            currM = groupListGM.GetValue(groupIdx);
        } else {
            totalTmp += groupListGM.GetValue(groupIdx);
            currM = totalTmp;
        }
        int64_t tempM = currM - prevM;
        prevM = currM;
        curStartIdx -= tempM;
    }
    // Step 3: compute the total data volume
    vecConfig.outLoopNum =
        (vecConfig.taskNum + gmmSwigluQuantV2->maxProcessRowNum - 1) / gmmSwigluQuantV2->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum
                                ? vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum
                                : gmmSwigluQuantV2->maxProcessRowNum;

    // Step 4: allocate space
    // 2 * row * n * sizeof(float) + row * n / 2 * sizeof(int8) + alignUp<row, 8> * sizeof(float) + n * sizeof(float) +
    // n / 2 *sizeof(float) + 64 < 191 * 1024
    pipe->InitBuffer(mmOutQueue, 1, gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen * sizeof(float));
    pipe->InitBuffer(quantOutQueue, 1,
                     gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR *
                         sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, 1,
                     AlignUp<int32_t>(gmmSwigluQuantV2->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) +
                                          UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA4W4PostProcess::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                   int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        VecConfig vecConfig;
        UpdateVecConfig(blockIdx, vecConfig, workspaceSplitConfig, workspaceSplitLoopIdx, pipe);

        if (blockIdx < vecConfig.usedCoreNum) {
            mmOutGM = (workspaceSplitLoopIdx % NUM_2 == 0 ? mmOutGM1 : mmOutGM2);
            LocalTensor<half> mmLocal = mmOutQueue.AllocTensor<half>();
            LocalTensor<float> quantScaleLocal = quantScaleOutQueue.AllocTensor<float>();
            LocalTensor<int8_t> quantLocal = quantOutQueue.AllocTensor<int8_t>();

            mmOutQueue.EnQue(mmLocal);
            quantScaleOutQueue.EnQue(quantScaleLocal);
            quantOutQueue.EnQue(quantLocal);
            for (uint32_t outLoopIdx = 0; outLoopIdx < vecConfig.outLoopNum; outLoopIdx++) {
                vecConfig.innerLoopNum = outLoopIdx == (vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum
                                                                                  : gmmSwigluQuantV2->maxProcessRowNum;
                int32_t eventIdMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                // 1. Bring in the matmul intermediate result
                customDataCopyIn(outLoopIdx, mmOutGM, vecConfig, workspaceSplitConfig);

                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    // 2. Four-step vector computation (perToken dequantization, Swiglu, SmoothScale, Quant)
                    VectorCompute(innerLoopIdx, vecConfig, workspaceSplitConfig);
                }
                int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                customDataCopyOut(vecConfig, workspaceSplitConfig);
            }
            mmLocal = mmOutQueue.DeQue<half>();
            quantScaleLocal = quantScaleOutQueue.DeQue<float>();
            quantLocal = quantOutQueue.DeQue<int8_t>();

            mmOutQueue.FreeTensor(mmLocal);
            quantScaleOutQueue.FreeTensor(quantScaleLocal);
            quantOutQueue.FreeTensor(quantLocal);
        }
    }
}

__aicore__ inline void GMMA4W4PostProcess::customDataCopyOut(VecConfig &vecConfig,
                                                             WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParams_0{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    DataCopyPad(quantScaleOutputGM[workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx], quantScaleLocal,
                copyParams_0);
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    DataCopyParams copyParams_1{
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0,
        0};
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) *
                              gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR],
                quantLocal, copyParams_1);

    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
    quantOutQueue.EnQue(quantLocal);
    quantScaleOutQueue.EnQue(quantScaleLocal);
}

} // namespace GroupedMatmulDequantSwigluQuant
#endif // GMM_SWIGLU_QUANT_V2_A4W4
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_A4W4_POST_H
