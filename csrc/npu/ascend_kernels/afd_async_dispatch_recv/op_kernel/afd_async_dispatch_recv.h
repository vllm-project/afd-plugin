/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncDispatchRecv operator kernel function header file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncDispatchRecv operator kernel function header file
 */
#ifndef AFD_ASYNC_DISPATCH_RECV_H
#define AFD_ASYNC_DISPATCH_RECV_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "afd_async_dispatch_recv_tiling.h"
#include "cam_async_comm_args.h"
#include "cam_async_comm_group.h"

namespace MoeDistributeDispatchImpl {
constexpr uint64_t CAM_MAX_RANK_SIZE = 384;  // max NPUs supported by the Cam comm library
constexpr uint32_t UB_ALIGN = 32;            // UB aligned to 32 bytes
constexpr uint32_t BATCH_INFO_VAL_NUM = 5;   // number of batch-related info fields
constexpr uint32_t INFO_NUM = 5;             // number of valid batch-info fields; also start/end expert per chunk
constexpr uint32_t DOUBLE = 2;

constexpr uint32_t TOKEN_NUM_INDEX = 0;
constexpr uint32_t ATTN_RANK_ID_INDEX = 1;
constexpr uint32_t LAYER_INDEX_INDEX = 2;
constexpr uint32_t START_EXPERT_INDEX = 3;
constexpr uint32_t END_EXPERT_INDEX = 4;
constexpr uint32_t MOE_PRE_TOKEN_NUM_INDEX = 3;

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename XType, typename ExpandXOutType, bool DynamicQuant
#define TemplateMC2TypeFunc XType, ExpandXOutType, DynamicQuant

using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeClass>
class AfdAsyncDispatchRecv {
public:
    __aicore__ inline AfdAsyncDispatchRecv() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut,
                                GM_ADDR batchInfoOut, GM_ADDR epRecvCountRoutedOut,
                                GM_ADDR workspaceGM, TPipe *pipe,
                                const AfdAsyncDispatchRecvTilingData *tilingData,
                                GM_ADDR commArgs, int32_t isCamComm);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t MathCeil(uint32_t n, uint32_t align)
    {
        return align * (n / align + (n % align == 0 ? 0 : 1));
    }

    __aicore__ inline GM_ADDR GetPeerAddrByRankId(const int32_t rankId)
    {
        // Match the HCCL windowsIn path in the AFD-bundled CAM package.
        if (rankId == moeRankId_) {
            return (GM_ADDR)epWinContext_->localWindowsIn;
        }
        return (GM_ADDR)((HcclRankRelationResV2 *)
            epWinContext_->remoteRes[rankId].nextDevicePtr)->windowsIn;
    }

    __aicore__ inline void DispatchRecv();
    __aicore__ inline void DispatchRecvRouted(uint32_t tpAttnRankId, uint32_t tpIndex,
                                               uint32_t startExpert, uint32_t endExpert);

    __aicore__ inline uint32_t WaitFromAttn();

    __aicore__ inline void ClearFlags(uint32_t attnRankId);

    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t moeRankId_{0};
    uint32_t worldSize_{0};
    uint32_t attnRankNum_{0};
    uint32_t moeRankNum_{0};
    uint32_t routeExpertNumPerMoe_{0};
    uint32_t expertNumPerMoe_{0};
    uint32_t maxSeqLen_{0};
    uint32_t hiddenSize_{0};
    uint32_t topk_{0};
    uint64_t totalUbSize_{0};
    uint64_t totalWorkspaceSize_{0};
    uint64_t workspaceSizePerAttn_{0};
    uint64_t maxTokenNum_{0};
    uint32_t layerIndex_{0};
    uint16_t tpSize_{0};

    GlobalTensor<ExpandXOutType> xOutGMTensor_;
    GlobalTensor<float> quantScaleGMTensor_;
    GlobalTensor<int64_t> epRecvCountRoutedGMTensor_;
    GlobalTensor<int64_t> batchInfoGMTensor_;
    __gm__ HcclOpResParam *epWinContext_{nullptr};

    uint64_t ubReuseSize_{0};

    TPipe *tpipe_{nullptr};
    TBuf<> ubBuffer_;
    LocalTensor<int64_t> batchInfoTensor_;
    LocalTensor<int64_t> tpBatchInfoTensor_;
    LocalTensor<int64_t> epRecvCountOutTensor_;
    // tokens received by each zero-based routed expert in this MoE rank
    LocalTensor<int32_t> expertRecvTokenNumTensor_;
    LocalTensor<int32_t> expertRecvChunkFlagTensor_;  // whether each expert has already been received
    LocalTensor<ExpandXOutType> xOutTensor_;      // staging token
    LocalTensor<float> xOutFloatTensor_;          // quant scale
    LocalTensor<uint32_t> tpExpertRecvTokenNumTensor_;

    LocalTensor<uint16_t> tokenAddrOffsetTensor_;

    // for quantization
    uint32_t hiddenSizeQuant_{0};

    bool isError_{false};
};

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR batchInfoOut, GM_ADDR epRecvCountRoutedOut, GM_ADDR workspaceGM,
    TPipe *pipe, const AfdAsyncDispatchRecvTilingData *tilingData, GM_ADDR commArgs, int32_t isCamComm)
{
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeDispatchInfo.aivNum;
    moeRankId_ = tilingData->moeDistributeDispatchInfo.moeRankId;
    worldSize_ = tilingData->moeDistributeDispatchInfo.worldSize;
    attnRankNum_ = tilingData->moeDistributeDispatchInfo.attnRankNum;
    moeRankNum_ = tilingData->moeDistributeDispatchInfo.moeRankNum;
    routeExpertNumPerMoe_ = tilingData->moeDistributeDispatchInfo.routeExpertNumPerMoe;
    expertNumPerMoe_ = routeExpertNumPerMoe_;
    maxSeqLen_ = tilingData->moeDistributeDispatchInfo.maxSeqLen;
    hiddenSize_ = tilingData->moeDistributeDispatchInfo.hiddenSize;
    topk_ = tilingData->moeDistributeDispatchInfo.topk;
    totalUbSize_ = tilingData->moeDistributeDispatchInfo.totalUbSize;
    totalWorkspaceSize_ = tilingData->moeDistributeDispatchInfo.totalWorkspaceSize;  // total shared mem
    tpSize_ = tilingData->moeDistributeDispatchInfo.tpSize;
    maxTokenNum_ = tilingData->moeDistributeDispatchInfo.maxTokenNum;

    uint32_t maxSeqLenPerRank = maxSeqLen_ / tpSize_;
    uint64_t batchInfoWorkspaceSize = MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    uint64_t expertRecvTokenNumWorkspaceSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t expertRecvChunkFlagWorkspaceSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t tokenWorkspaceSize = 0;
    if constexpr (DynamicQuant) {
        tokenWorkspaceSize = MathCeil((sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) * maxSeqLenPerRank, UB_ALIGN);
    } else {
        tokenWorkspaceSize = MathCeil(sizeof(XType) * hiddenSize_ * maxSeqLenPerRank, UB_ALIGN);
    }
    uint64_t tokenAddrWorkspaceSize = MathCeil(sizeof(uint16_t) * maxSeqLenPerRank * topk_, UB_ALIGN);
    uint64_t totalWorkspaceSize = batchInfoWorkspaceSize + expertRecvTokenNumWorkspaceSize +
    expertRecvChunkFlagWorkspaceSize + tokenWorkspaceSize + tokenAddrWorkspaceSize;
    workspaceSizePerAttn_ = UB_ALIGN * (totalWorkspaceSize / UB_ALIGN);


    xOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)expandXOut);
    quantScaleGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    epRecvCountRoutedGMTensor_.SetGlobalBuffer((__gm__ int64_t *)epRecvCountRoutedOut);
    batchInfoGMTensor_.SetGlobalBuffer((__gm__ int64_t *)batchInfoOut);
    // commArgs is an ABI placeholder; MC2 initializes the HCCL resource context.
    epWinContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    tpipe_ = pipe;
    tpipe_->Reset();
    tpipe_->InitBuffer(ubBuffer_, totalUbSize_);

    uint64_t bufOffset = 0;
    uint64_t bufSize;

    bufSize = MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN); // 32 B
    batchInfoTensor_ = ubBuffer_.GetWithOffset<int64_t>(bufSize / sizeof(int64_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(int64_t) * (INFO_NUM + tpSize_ + expertNumPerMoe_ *  tpSize_), UB_ALIGN);
    tpBatchInfoTensor_ = ubBuffer_.GetWithOffset<int64_t>(bufSize / sizeof(int64_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(int64_t) * expertNumPerMoe_, UB_ALIGN); // 64 B or 96 B
    epRecvCountOutTensor_ = ubBuffer_.GetWithOffset<int64_t>(bufSize / sizeof(int64_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN); // 64 B or 96 B
    expertRecvTokenNumTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN); // 64 B or 96 B
    expertRecvChunkFlagTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
    bufOffset += bufSize;

    if constexpr (DynamicQuant) {
        hiddenSizeQuant_ = (sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) / sizeof(ExpandXOutType);

        bufSize = MathCeil(sizeof(ExpandXOutType) * hiddenSizeQuant_, UB_ALIGN);
        xOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(bufSize / sizeof(ExpandXOutType), bufOffset);
        xOutFloatTensor_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
        bufOffset += bufSize;
    } else {
        bufSize = MathCeil(sizeof(ExpandXOutType) * hiddenSize_, UB_ALIGN); // 14 KB
        xOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(bufSize / sizeof(ExpandXOutType), bufOffset);
        bufOffset += bufSize;
    }

    bufSize = MathCeil(sizeof(uint32_t) * expertNumPerMoe_ * DOUBLE, UB_ALIGN);
    tpExpertRecvTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    // ----- mandatory buffers allocated above -----

    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // remaining reusable UB space
    ubReuseSize_ = totalUbSize_ - bufOffset;

    bufSize = ubReuseSize_;
    tokenAddrOffsetTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::Process()
{
    if (isError_ == true) {
        return;
    }

    DispatchRecv();
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::DispatchRecv()
{
    uint32_t tpAttnRankId = WaitFromAttn();

    tpBatchInfoTensor_(ATTN_RANK_ID_INDEX) = tpAttnRankId;  // attnRankId
    tpBatchInfoTensor_(LAYER_INDEX_INDEX) = batchInfoTensor_(LAYER_INDEX_INDEX);  // layerIndex

    // count tokens received by each expert within the tp group
    GM_ADDR dstGM = GetPeerAddrByRankId(moeRankId_);
    Duplicate(tpExpertRecvTokenNumTensor_, (uint32_t)0, expertNumPerMoe_ * DOUBLE);
    SyncFunc<HardEvent::V_S>();
    GlobalTensor<int32_t> dstExpertRecvTokenNumGMTensor;
    for (uint32_t i = 0; i < tpSize_; ++i) {
        uint32_t attnRankId = tpAttnRankId + i;
        uint64_t attnWorkspaceOffset = workspaceSizePerAttn_ * attnRankId;
        uint64_t expertRecvTokenNumOffset =
            attnWorkspaceOffset + MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
        dstExpertRecvTokenNumGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertRecvTokenNumOffset));

        DataCopyPad(expertRecvTokenNumTensor_, dstExpertRecvTokenNumGMTensor,
            {1U, (uint32_t)(sizeof(int32_t) * expertNumPerMoe_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        SyncFunc<HardEvent::S_MTE2>();

        for (uint32_t j = 0; j < expertNumPerMoe_; ++j) {
            tpExpertRecvTokenNumTensor_(j) += expertRecvTokenNumTensor_(j);
        }
    }

    uint64_t expertRecvChunkFlagOffset = workspaceSizePerAttn_ * tpAttnRankId +
        MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN) +
        MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    GlobalTensor<int32_t> dstExpertRecvChunkFlagGMTensor;
    dstExpertRecvChunkFlagGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertRecvChunkFlagOffset));
    DataCopyPad(expertRecvChunkFlagTensor_, dstExpertRecvChunkFlagGMTensor,
        {1U, (uint32_t)(sizeof(int32_t) * expertNumPerMoe_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();

    // expertNumPerMoe_ is an invalid ID; expert 0 is a valid routed expert.
    uint32_t startExpert = expertNumPerMoe_;  // inclusive once selected
    uint32_t endExpert = expertNumPerMoe_;    // inclusive once selected
    bool foundExpert = false;
    uint32_t tokenCnt = 0;
    for (uint32_t i = 0; i < expertNumPerMoe_; ++i) {
        if (expertRecvChunkFlagTensor_(i) == 0) {
            if (tokenCnt + tpExpertRecvTokenNumTensor_(i) <= maxTokenNum_) {
                if (!foundExpert) {
                    startExpert = i;
                    foundExpert = true;
                }
                tokenCnt += tpExpertRecvTokenNumTensor_(i);
                endExpert = i;
            } else {
                break;
            }
        }
    }
    // Whole-expert chunking requires each pending expert to fit maxTokenNum_.
    // All AIVs see the same counts/flags. Do not treat an unselected chunk as
    // expert 0, read an invalid expert, or publish a completion for missing data.
    if (!foundExpert) {
        isError_ = true;
        return;
    }
    SyncAll<true>();


    for (uint32_t tpIndex = 0; tpIndex < tpSize_; ++tpIndex) {
        DispatchRecvRouted(tpAttnRankId, tpIndex, startExpert, endExpert);
    }

    if (aivId_ == 0) {
        uint32_t tokenNum = 0;
        for (uint32_t i = 0; i < expertNumPerMoe_; ++i) {
            tokenNum += tpExpertRecvTokenNumTensor_(i);
        }
        tpBatchInfoTensor_(TOKEN_NUM_INDEX) = tokenNum;
        tpBatchInfoTensor_(START_EXPERT_INDEX) = startExpert;
        tpBatchInfoTensor_(END_EXPERT_INDEX) = endExpert;


        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(batchInfoGMTensor_, tpBatchInfoTensor_,
            {1U, (uint32_t)(sizeof(int64_t) * (INFO_NUM + tpSize_ + expertNumPerMoe_ * tpSize_)), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();

        for (uint32_t i = 0; i < expertNumPerMoe_; ++i) {
            // final count each expert receives
            epRecvCountOutTensor_(i) = tpExpertRecvTokenNumTensor_(expertNumPerMoe_ + i);
        }

        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(epRecvCountRoutedGMTensor_, epRecvCountOutTensor_,
            {1U, (uint32_t)(sizeof(int64_t) * routeExpertNumPerMoe_), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();

        // record the last-processed attn at the end of shared mem
        GlobalTensor<int64_t> dstBatchInfoGMTensor;
        dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + (totalWorkspaceSize_ - UB_ALIGN)));
        // all experts received this round; start over next time
        if (endExpert == expertNumPerMoe_ - 1) {
            batchInfoTensor_(0) = (tpAttnRankId + tpSize_) % attnRankNum_;
        } else {
            batchInfoTensor_(0) = tpAttnRankId;
        }
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(dstBatchInfoGMTensor, batchInfoTensor_, UB_ALIGN / sizeof(int64_t));
        SyncFunc<HardEvent::MTE3_S>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::DispatchRecvRouted(
    uint32_t tpAttnRankId, uint32_t tpIndex, uint32_t startExpert, uint32_t endExpert)
{
    uint32_t maxSeqLenPerRank = maxSeqLen_ / tpSize_;
    uint32_t attnRankId = tpAttnRankId + tpIndex;
    uint64_t attnWorkspaceOffset = workspaceSizePerAttn_ * attnRankId;
    uint64_t expertRecvTokenNumOffset =
        attnWorkspaceOffset + MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    uint64_t expertRecvChunkFlagOffset =
        expertRecvTokenNumOffset + MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t tokenStoreOffset = expertRecvChunkFlagOffset + MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t tokenAddrStoreOffset = 0;
    if constexpr (DynamicQuant) {
        tokenAddrStoreOffset = tokenStoreOffset
            + MathCeil((sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) * maxSeqLenPerRank, UB_ALIGN);
    } else {
        tokenAddrStoreOffset = tokenStoreOffset + MathCeil(sizeof(XType) * hiddenSize_ * maxSeqLenPerRank, UB_ALIGN);
    }

    uint32_t moeSelfRankId = moeRankId_;
    GM_ADDR dstGM = GetPeerAddrByRankId(moeSelfRankId);

    // reload batchInfoTensor_
    GlobalTensor<int64_t> dstBatchInfoGMTensor;
    dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + attnWorkspaceOffset));
    DataCopyPad(batchInfoTensor_, dstBatchInfoGMTensor,
        {1U, (uint32_t)(sizeof(int64_t) * BATCH_INFO_VAL_NUM), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();
    tpBatchInfoTensor_(INFO_NUM + tpIndex) = batchInfoTensor_(MOE_PRE_TOKEN_NUM_INDEX);

    // reload expertRecvTokenNumTensor_
    GlobalTensor<int32_t> dstExpertRecvTokenNumGMTensor;
    dstExpertRecvTokenNumGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertRecvTokenNumOffset));
    DataCopyPad(expertRecvTokenNumTensor_, dstExpertRecvTokenNumGMTensor,
        {1U, (uint32_t)(sizeof(int32_t) * expertNumPerMoe_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();
    for (uint32_t i = 0; i < expertNumPerMoe_; ++i) {
        tpBatchInfoTensor_(INFO_NUM + tpSize_ + expertNumPerMoe_ * tpIndex + i) = expertRecvTokenNumTensor_(i);
    }

    uint32_t tokenNumSum = 0;
    for (uint32_t i = startExpert; i <= endExpert; ++i) {
        tokenNumSum += expertRecvTokenNumTensor_(i);
    }
    uint32_t preTokenNumSum = 0;
    for (uint32_t i = 0; i < startExpert; ++i) {
        preTokenNumSum += expertRecvTokenNumTensor_(i);
    }
    uint32_t tokenNumPerAiv = tokenNumSum / aivNum_;
    uint32_t tokenNumPerAivRemain = tokenNumSum % aivNum_;
    uint32_t tokenNumPerAivStart = tokenNumPerAiv * aivId_;
    if (aivId_ < tokenNumPerAivRemain) {
        tokenNumPerAiv += 1;
        tokenNumPerAivStart += aivId_;
    } else {
        tokenNumPerAivStart += tokenNumPerAivRemain;
    }
    uint32_t tokenNumPerAivEnd = tokenNumPerAivStart + tokenNumPerAiv;


    GlobalTensor<ExpandXOutType> dstTokenStoreGMTensor;
    GlobalTensor<uint16_t> dstTokenAddrOffsetGMTensor;
    dstTokenStoreGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)(dstGM + tokenStoreOffset));
    dstTokenAddrOffsetGMTensor.SetGlobalBuffer((__gm__ uint16_t *)(dstGM + tokenAddrStoreOffset));

    // Empty AIV slices must not issue a zero-byte token-address DMA.
    if (tokenNumPerAiv > 0) {
        DataCopyPad(tokenAddrOffsetTensor_, dstTokenAddrOffsetGMTensor[tokenNumPerAivStart + preTokenNumSum],
            {1U, (uint32_t)(sizeof(uint16_t) * tokenNumPerAiv), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        SyncFunc<HardEvent::S_MTE2>();
    }

    // each aiv starts from startExpert to find its tokenNumPerAivStart; meanwhile determine
    // which expert each token goes to and how many prior cores sent to each expert, for offset calc
    uint32_t currExpertId = startExpert;
    uint32_t currExpertTokenNum = 0;
    uint32_t nextExpertTokenNum = expertRecvTokenNumTensor_(currExpertId);
    uint32_t preExpertTokenNum = 0;  // total tokens received by experts before the current one
    while (nextExpertTokenNum < tokenNumPerAivStart) {
        preExpertTokenNum += tpExpertRecvTokenNumTensor_(currExpertId);
        currExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
        ++currExpertId;
        nextExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
    }
    for (uint32_t expertSlotOffset = tokenNumPerAivStart; expertSlotOffset < tokenNumPerAivEnd; ++expertSlotOffset) {
        while (expertSlotOffset >= nextExpertTokenNum) {
            preExpertTokenNum += tpExpertRecvTokenNumTensor_(currExpertId);
            currExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
            ++currExpertId;
            nextExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
        }
        // received by prior experts + sent by prior attns to this expert + nth sent by current attn to current expert
        uint32_t tpExpertSlotOffset = preExpertTokenNum + tpExpertRecvTokenNumTensor_(expertNumPerMoe_ + currExpertId) +
            (expertSlotOffset - currExpertTokenNum);

        uint16_t moeTokenSendIdx = tokenAddrOffsetTensor_(expertSlotOffset - tokenNumPerAivStart);

        if constexpr (DynamicQuant) {
            DataCopy(xOutTensor_, dstTokenStoreGMTensor[hiddenSizeQuant_ * moeTokenSendIdx], hiddenSizeQuant_);
            SyncFunc<HardEvent::MTE2_S>();
            SyncFunc<HardEvent::S_MTE2>();

            SyncFunc<HardEvent::S_MTE3>();
            DataCopy(xOutGMTensor_[hiddenSize_ * tpExpertSlotOffset], xOutTensor_, hiddenSize_);
            SyncFunc<HardEvent::MTE3_S>();
            DataCopyPad(quantScaleGMTensor_[tpExpertSlotOffset], xOutFloatTensor_[hiddenSize_ / sizeof(float)],
                {1U, sizeof(float), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
        } else {
            DataCopy(xOutTensor_, dstTokenStoreGMTensor[hiddenSize_ * moeTokenSendIdx], hiddenSize_);
            SyncFunc<HardEvent::MTE2_S>();
            SyncFunc<HardEvent::S_MTE2>();

            SyncFunc<HardEvent::S_MTE3>();
            DataCopy(xOutGMTensor_[hiddenSize_ * tpExpertSlotOffset], xOutTensor_, hiddenSize_);
            SyncFunc<HardEvent::MTE3_S>();
        }
    }

    if (aivId_ == 0) {
        GlobalTensor<int32_t> dstExpertRecvChunkFlagGMTensor;
        dstExpertRecvChunkFlagGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertRecvChunkFlagOffset));
        for (uint32_t i = 0; i <= endExpert; ++i) {
            expertRecvChunkFlagTensor_(i) = 1;
        }
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(dstExpertRecvChunkFlagGMTensor, expertRecvChunkFlagTensor_,
            {1U, (uint32_t)sizeof(int32_t) * expertNumPerMoe_, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }

    if (endExpert == expertNumPerMoe_ - 1) {  // all experts received this round; clear flag
        ClearFlags(attnRankId);
    }

    for (uint32_t i = startExpert; i <= endExpert; ++i) {
        tpExpertRecvTokenNumTensor_(expertNumPerMoe_ + i) += expertRecvTokenNumTensor_(i);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline uint32_t AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::WaitFromAttn()
{
    GM_ADDR dstGM = GetPeerAddrByRankId(moeRankId_);
    GlobalTensor<int64_t> dstBatchInfoGMTensor;
    // the last slot of shared mem marks which dp group is handled this round
    dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + (totalWorkspaceSize_ - UB_ALIGN)));
    DataCopy(batchInfoTensor_, dstBatchInfoGMTensor, UB_ALIGN / sizeof(int64_t));
    SyncFunc<HardEvent::MTE2_S>();
    uint32_t attnRankId = batchInfoTensor_(0);
    SyncFunc<HardEvent::S_MTE2>();

    bool isAllRecv = false;
    if (aivId_ == 0) {
        while (isAllRecv == false) {
            for (uint32_t i = 0; i < (attnRankNum_ / tpSize_); ++i) {
                isAllRecv = true;
                for (uint32_t j = tpSize_; j > 0; --j) {
                    uint32_t srcRankId = (attnRankId + (tpSize_ * i + j - 1)) % attnRankNum_;
                    uint64_t offset = workspaceSizePerAttn_ * srcRankId;
                    dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + offset));

                    DataCopyPad(batchInfoTensor_, dstBatchInfoGMTensor,
                        {1U, (uint32_t)(sizeof(int64_t) * BATCH_INFO_VAL_NUM), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                    SyncFunc<HardEvent::MTE2_S>();
                    SyncFunc<HardEvent::S_MTE2>();

                    if (batchInfoTensor_(TOKEN_NUM_INDEX) == 0) {
                        isAllRecv = false;
                        break;
                    }
                }
                // aiv0 confirms a dp group is fully received; writes a marker at card 0 of each
                // dp group so other aivs see it ready
                if (isAllRecv == true) {
                    attnRankId = (attnRankId + (tpSize_ * i)) % attnRankNum_;

                    DataCopyPad(dstBatchInfoGMTensor[BATCH_INFO_VAL_NUM - 1],
                        batchInfoTensor_, {1U, sizeof(int64_t), 0U, 0U, 0U});
                    SyncFunc<HardEvent::MTE3_S>();
                    break;
                }
            }
        }
    } else {
        while (isAllRecv == false) {
            for (uint32_t i = 0; i < (attnRankNum_ / tpSize_); ++i) {
                uint32_t srcRankId = (attnRankId + (tpSize_ * i)) % attnRankNum_;
                uint64_t offset = workspaceSizePerAttn_ * srcRankId;
                dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + offset));

                DataCopyPad(batchInfoTensor_, dstBatchInfoGMTensor,
                    {1U, (uint32_t)(sizeof(int64_t) * BATCH_INFO_VAL_NUM), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                SyncFunc<HardEvent::MTE2_S>();
                SyncFunc<HardEvent::S_MTE2>();

                if (batchInfoTensor_(BATCH_INFO_VAL_NUM - 1) != 0) {
                    attnRankId = srcRankId;
                    isAllRecv = true;
                    break;
                }
            }
        }
    }

    return attnRankId;
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::ClearFlags(uint32_t attnRankId)
{
    SyncAll<true>();
    if (aivId_ != 0) {
        return;
    }

    uint64_t attnWorkspaceOffset = workspaceSizePerAttn_ * attnRankId;
    GM_ADDR dstGM = GetPeerAddrByRankId(moeRankId_);
    GlobalTensor<int64_t> dstBatchInfoGMTensor;
    dstBatchInfoGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + attnWorkspaceOffset));

    for (uint32_t i = 0; i < BATCH_INFO_VAL_NUM; ++i) {
        batchInfoTensor_(i) = 0;
    }
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyPad(dstBatchInfoGMTensor, batchInfoTensor_,
        {1U, sizeof(int64_t) * BATCH_INFO_VAL_NUM, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE3_S>();
}

} // MoeDistributeDispatchImpl
#endif // AFD_ASYNC_DISPATCH_RECV_H
