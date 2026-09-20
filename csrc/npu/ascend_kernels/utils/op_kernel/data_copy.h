/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: create data copy header file
 * Create: 2026-01-05
 * Note: This file might be replaced! We will try to use the file from cann-toolkit in env when compiling.
 * History: 2026-01-05 create data copy header file
 */

#ifndef CAM_DATACOPY_GM2GM_H
#define CAM_DATACOPY_GM2GM_H
#include <type_traits>
#include "comm_args.h"

using namespace AscendC;
using namespace Moe;

template <typename T>
FORCE_INLINE_AICORE void SetAtomicOpType(int op)
{
    switch (op) {
        case ADD:
            AscendC::SetAtomicAdd<T>();
            break;
        case MUL:
            // Ignore setting the atomic register when performing mul
            break;
        case MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case MIN:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
    }
}

template <typename T>
FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template<typename T>
FORCE_INLINE_AICORE void CopyUB2UB(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    TBuffAddr srcAddr, dstAddr;
    srcAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    dstAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    srcTensor.SetAddr(srcAddr);
    dstTensor.SetAddr(dstAddr);
    DataCopy(dstTensor, srcTensor, calCount);
}

template <typename T>
FORCE_INLINE_AICORE void SetAtomic(int op)
{
    PipeBarrier<PIPE_ALL>();
    if (op != -1) {
#ifdef __DAV_C220_VEC__
        SetAtomicOpType<T>(op);
#endif
    }
    PipeBarrier<PIPE_ALL>();
}

FORCE_INLINE_AICORE void UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <HardEvent eventType>
FORCE_INLINE_AICORE void SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

template <typename K, typename U = K>
FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain,
                                        const GlobalTensor<U> &sendDataInputGt,
                                        const GlobalTensor<K> &recvDataOutputGT, int op)
{
    // General case (U = K), input/output are the same, share one UB
    // Only when conversion is needed (U->K), UB will be divided into two parts according to the ratio of
    // sizeof(U):sizeof(K) and aligned to 32 bytes
    constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
    constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(K) + sizeof(U)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    constexpr int32_t inputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(U);
    constexpr int32_t outputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(K);

    __gm__ U *input = const_cast<__gm__ U *>(sendDataInputGt.GetPhyAddr());
    __gm__ K *output = const_cast<__gm__ K *>(recvDataOutputGT.GetPhyAddr());
    __ubuf__ U *inputUB[2] = {(__ubuf__ U *)(UB_HEAD_OFFSET), (__ubuf__ U *)(UB_MID_OFFSET)};
    __ubuf__ K *outputUB[2] = {(__ubuf__ K *)inputUB[0], (__ubuf__ K *)inputUB[1]};
    if constexpr (!std::is_same_v<K, U>) {
        outputUB[0] = (__ubuf__ K *)(inputUB[0] + inputUbBlockSize / sizeof(U));
        outputUB[1] = (__ubuf__ K *)(inputUB[1] + inputUbBlockSize / sizeof(U));
    }
    int inputOffsetNum = 0;
    int outputOffsetNum = 0;
    if (dataSizeRemain <= 0) {
        return;
    }

    SetAtomic<K>(op);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2 waits for MTE3
    for (int64_t i = 0; dataSizeRemain > 0; i++) {
        // size and dataSizeRemain both refer to the output size
        uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
        event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(K) * sizeof(U));
        if constexpr (!std::is_same_v<K, U>) {
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl((i & 1) ? outputUB[0] : outputUB[1], (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE,
                     size / sizeof(K));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
        }
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUB2GM(output + outputOffsetNum, (i & 1) ? outputUB[0] : outputUB[1], size);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

        dataSizeRemain -= size;
        inputOffsetNum += (size / sizeof(K));
        outputOffsetNum += (size / sizeof(K));
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2 waits for MTE3

    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);  // Scalar waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);

    UnsetAtomic(op);
    return;
}

template<typename T>
FORCE_INLINE_AICORE void camCpUB2GM(GlobalTensor<T> outputGt, LocalTensor<T> inputLt, uint16_t count, uint32_t size,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams(count, size, srcStride / UB_ALIGN_SIZE, dstStride, 0);
    DataCopyPad(outputGt, inputLt, dataCopyParams);
}

template<typename T>
FORCE_INLINE_AICORE void camCpGM2UB(LocalTensor<T> outputLt, GlobalTensor <T> inputGt, uint32_t size,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(outputLt, inputGt, dataCopyParams, padParams);
}

FORCE_INLINE_AICORE int ceil(int x, int y)
{
    return (x + y - 1) / y;
}

template<typename T>
FORCE_INLINE_AICORE void copyGmToGmWithBlocks(const GlobalTensor<T> &outputGt, const GlobalTensor<T> &inputGt,
    int elementNum, int usedBlockNum, int blockIdx)
{
    int elementPerBlock = ceil(elementNum, usedBlockNum);
    int copyOffset = elementPerBlock * blockIdx;
    int copyLen = elementNum - copyOffset < elementPerBlock ? elementNum - copyOffset : elementPerBlock;
    if (copyLen <= 0) {
        return;
    }
    CpGM2GMPingPong(copyLen * sizeof(T), inputGt[copyOffset], outputGt[copyOffset], COPYONLY);
}

#endif // CAM_DATACOPY_GM2GM_H
