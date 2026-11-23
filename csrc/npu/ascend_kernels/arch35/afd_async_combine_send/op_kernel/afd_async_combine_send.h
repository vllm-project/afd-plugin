/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineSend operator kernel function header file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineSend operator kernel function header file
 */
#ifndef AFD_ASYNC_COMBINE_SEND_H
#define AFD_ASYNC_COMBINE_SEND_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "afd_async_combine_send_tiling.h"
#include "cam_async_comm_args.h"
#include "cam_async_window.h"
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
#include "cam_async_mte_pipeline.h"
#endif

namespace MoeDistributeCombineSendImpl {
constexpr uint64_t CAM_MAX_RANK_SIZE = 384;  // max NPUs supported by the Cam comm library
constexpr uint32_t UB_ALIGN = 32;            // UB aligned to 32 bytes
constexpr uint32_t MAX_AIV_NUM = 48;         // max AIV core count
constexpr uint32_t INFO_NUM = 5;  // number of valid batch-info fields; also start/end expert per chunk

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename ExpandXType
#define TemplateMC2TypeFunc ExpandXType
using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeClass>
class AfdAsyncCombineSend {
public:
    __aicore__ inline AfdAsyncCombineSend(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR workspaceGM, TPipe *pipe,
        const AfdAsyncCombineSendTilingData *tilingData, GM_ADDR commArgs, GM_ADDR batchInfo,
        int32_t isCamComm);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint64_t MathCeil(uint64_t n, uint64_t align)
    {
        return align * (n / align + (n % align == 0 ? 0 : 1));
    }

    __aicore__ inline GM_ADDR GetPeerAddrByRankId(const int32_t rankId)
    {
        return dataWindow_.GetDataWindow(rankId);
    }

    __aicore__ inline void CombineSend();

    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t moeRankId_{0};
    uint32_t worldSize_{0};
    uint32_t attnRankNum_{0};
    uint32_t moeRankNum_{0};
    uint32_t routeExpertNumPerMoe_{0};
    uint32_t expertNumPerMoe_{0};
    uint32_t expertNum_{0};
    uint32_t maxSeqLen_{0};
    uint32_t hiddenSize_{0};
    uint32_t topk_{0};
    uint64_t totalUbSize_{0};
    uint64_t totalWorkspaceSize_{0};
    uint64_t dispatchOffset_{0};
    uint16_t tpSize_{0};

    GlobalTensor<ExpandXType> xGMTensor_;
    GlobalTensor<int64_t> batchInfoGMTensor_;
    Cam::AsyncDataWindow dataWindow_;

    uint64_t ubReuseSize_{0};
    uint32_t maxUbTokenNum_{0};

    TPipe *tpipe_{nullptr};
    TBuf<> ubBuffer_;
    LocalTensor<uint32_t> tpExpertTokenNumTensor_;
    LocalTensor<uint32_t> tpExpertTokenSendCntTensor_;
    LocalTensor<uint32_t> moePrefixTokenNumTensor_;
    LocalTensor<int64_t>  tpBatchInfoTensor_;
    LocalTensor<ExpandXType> xTensor_;

    bool isError_{false};
};

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineSend<TemplateMC2TypeFunc>::Init(GM_ADDR expandX,
    GM_ADDR workspaceGM, TPipe *pipe, const AfdAsyncCombineSendTilingData *tilingData, GM_ADDR commArgs,
    GM_ADDR batchInfo, int32_t isCamComm)
{
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeRankId_ = tilingData->moeDistributeCombineInfo.moeRankId;
    worldSize_ = tilingData->moeDistributeCombineInfo.worldSize;
    attnRankNum_ = tilingData->moeDistributeCombineInfo.attnRankNum;
    moeRankNum_ = tilingData->moeDistributeCombineInfo.moeRankNum;
    routeExpertNumPerMoe_ = tilingData->moeDistributeCombineInfo.routeExpertNumPerMoe;
    // Compact protocol: local expert IDs are [0, R), with no extra slots.
    expertNumPerMoe_ = routeExpertNumPerMoe_;
    expertNum_ = expertNumPerMoe_ * moeRankNum_;
    maxSeqLen_ = tilingData->moeDistributeCombineInfo.maxSeqLen;
    hiddenSize_ = tilingData->moeDistributeCombineInfo.hiddenSize;
    topk_ = tilingData->moeDistributeCombineInfo.topk;
    totalUbSize_ = tilingData->moeDistributeCombineInfo.totalUbSize;
    totalWorkspaceSize_ = tilingData->moeDistributeCombineInfo.totalWorkspaceSize;
    tpSize_ = tilingData->moeDistributeCombineInfo.tpSize;
    dispatchOffset_ = MathCeil(sizeof(uint32_t) * (moeRankNum_ + expertNum_) * MAX_AIV_NUM, UB_ALIGN);

    xGMTensor_.SetGlobalBuffer((__gm__ ExpandXType*)expandX);
    batchInfoGMTensor_.SetGlobalBuffer((__gm__ int64_t *)batchInfo);
    // commArgs is an ABI placeholder; MC2 initializes the HCCL resource context.
    dataWindow_.Init(AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), moeRankId_);

    tpipe_ = pipe;
    tpipe_->Reset();
    tpipe_->InitBuffer(ubBuffer_, totalUbSize_);

    uint64_t bufOffset = 0;
    uint64_t bufSize;

    bufSize = MathCeil(sizeof(int64_t) * (INFO_NUM + tpSize_ + expertNumPerMoe_ *  tpSize_), UB_ALIGN);
    tpBatchInfoTensor_ = ubBuffer_.GetWithOffset<int64_t>(bufSize / sizeof(int64_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint32_t) * expertNumPerMoe_ * tpSize_, UB_ALIGN);
    tpExpertTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    // One send counter for each TP rank.
    bufSize = MathCeil(sizeof(uint32_t) * tpSize_, UB_ALIGN);
    tpExpertTokenSendCntTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // remaining reusable UB space
    ubReuseSize_ = totalUbSize_ - bufOffset;

    maxUbTokenNum_ = ubReuseSize_ / (sizeof(ExpandXType) * hiddenSize_);

    bufSize = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN);
    moePrefixTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);

    bufSize = MathCeil(sizeof(ExpandXType) * hiddenSize_ * maxUbTokenNum_, UB_ALIGN);
    xTensor_ = ubBuffer_.GetWithOffset<ExpandXType>(bufSize / sizeof(ExpandXType), bufOffset);
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineSend<TemplateMC2TypeFunc>::Process()
{
    if (isError_ == true) {
        return;
    }

    // load batch info: total token num, attn rank, layer index, start/end expert,
    // how many tokens attn i sent to prior moes, and how many tokens each expert in this
    // moe received from each attn
    DataCopyPad(tpBatchInfoTensor_, batchInfoGMTensor_,
        {1U, (uint32_t)(sizeof(int64_t) * (INFO_NUM + tpSize_ + expertNumPerMoe_ * tpSize_)), 0U, 0U, 0U},
        {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();

    CombineSend();
}

// Patch reason: the B0 GM-to-UB-to-peer loop waits on Scalar after each tile.
// Patch functionality: overlap two A5 MTE slots while retaining B0 fallback and final ready ordering.
// Signature: matches upstream/B0; no added parameters.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineSend<TemplateMC2TypeFunc>::CombineSend()
{
    uint32_t totalTokenNum = 0;
    uint32_t tpAttnRankId = tpBatchInfoTensor_(1);
    // Inclusive local routed expert IDs in [0, R).
    uint32_t startExpert = tpBatchInfoTensor_(3);
    uint32_t endExpert = tpBatchInfoTensor_(4);
    uint32_t tpInfoOffset = INFO_NUM + tpSize_;

    for (uint32_t i = 0; i < tpSize_; ++i) {
        for (uint32_t j = startExpert; j <= endExpert; ++j) {
            totalTokenNum += tpBatchInfoTensor_(tpInfoOffset + i * expertNumPerMoe_ + j);
        }
    }

    uint32_t totalTokenNumPerAiv = totalTokenNum / aivNum_;
    uint32_t totalTokenNumPerAivRemain = totalTokenNum % aivNum_;
    uint32_t totalTokenNumPerAivStart = totalTokenNumPerAiv * aivId_;
    if (aivId_ < totalTokenNumPerAivRemain) {
        totalTokenNumPerAiv += 1;
        totalTokenNumPerAivStart += aivId_;
    } else {
        totalTokenNumPerAivStart += totalTokenNumPerAivRemain;
    }
    uint32_t totalTokenNumPerAivEnd = totalTokenNumPerAivStart + totalTokenNumPerAiv;

    // ### PATCH START: A5 tile plan
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    const auto bufferPlan = Cam::AsyncMte::PlanCombineSendBuffers(
        ubReuseSize_, sizeof(ExpandXType) * hiddenSize_, true);
    // A single row cannot overlap another transfer, so keep its B0 path.
    const bool useMtePipeline = bufferPlan.doubleBuffered && totalTokenNumPerAiv > 1;
    const uint64_t tileRowCapacity = useMtePipeline ? bufferPlan.rowsPerSlot : maxUbTokenNum_;
    uint32_t bufferSlot = 0;
    Cam::AsyncMte::TwoSlotPipeline mtePipeline;
    if (useMtePipeline) {
        mtePipeline.Init(tpipe_);
    }
#endif
    // ### PATCH END: A5 tile plan

    Duplicate(tpExpertTokenSendCntTensor_, (uint32_t)0, tpSize_);
    SyncFunc<HardEvent::V_S>();

    for (uint32_t i = 0; i < tpSize_; ++i) {
        for (uint32_t j = 0; j < startExpert; ++j) {
            // across rounds: tokens already sent in prior rounds
            tpExpertTokenSendCntTensor_(i) += tpBatchInfoTensor_(tpInfoOffset + i * expertNumPerMoe_ + j);
        }
    }

    uint32_t currSendId = tpSize_ * startExpert;
    uint32_t tpIndex = currSendId % tpSize_;
    // Walk experts first, then TP ranks; count-table rows have exactly R entries.
    uint32_t expertId = (expertNumPerMoe_ * tpIndex) + (currSendId / tpSize_);
    uint32_t currTokenNumCnt = 0;
    uint32_t nextTokenNumCnt = tpBatchInfoTensor_(tpInfoOffset + expertId);

    GlobalTensor<ExpandXType> dstTokenStoreGM;
    uint32_t resolvedTpIndex = tpSize_;

    for (uint32_t i = 0; i < totalTokenNumPerAivEnd;) {
        // Skip all empty (expert, TP-rank) cells before scheduling a data copy.
        while (nextTokenNumCnt <= i) {
            currTokenNumCnt += tpBatchInfoTensor_(tpInfoOffset + expertId);
            ++currSendId;
            tpIndex = currSendId % tpSize_;
            expertId = (expertNumPerMoe_ * tpIndex) + (currSendId / tpSize_);
            nextTokenNumCnt += tpBatchInfoTensor_(tpInfoOffset + expertId);
        }

        if (i < totalTokenNumPerAivStart) {
            uint32_t tokenNum;
            if (nextTokenNumCnt < totalTokenNumPerAivStart) {
                tokenNum = nextTokenNumCnt - i;
            } else {
                tokenNum = totalTokenNumPerAivStart - i;
            }
            tpExpertTokenSendCntTensor_(tpIndex) += tokenNum;
            i += tokenNum;
        } else {
            uint32_t sendTokenNumRemain = totalTokenNumPerAivEnd - i;
            uint32_t expertTokenNumRemain = nextTokenNumCnt - i;
            uint32_t cnt = expertTokenNumRemain < sendTokenNumRemain ? expertTokenNumRemain : sendTokenNumRemain;
            // ### PATCH START: A5 tile capacity
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
            cnt = static_cast<uint32_t>(tileRowCapacity < cnt ? tileRowCapacity : cnt);
#else
            cnt = maxUbTokenNum_ < cnt ? maxUbTokenNum_ : cnt;
#endif
            // ### PATCH END: A5 tile capacity
            // offset on the original attn rank, computed from the total count
            uint32_t moeTokenSendIdx = tpExpertTokenSendCntTensor_(tpIndex);

            // Skipped cells need no peer lookup; a destination base depends only on its TP rank.
            if (resolvedTpIndex != tpIndex) {
                GM_ADDR dstRankWorkspaceGm = GetPeerAddrByRankId(tpAttnRankId + tpIndex) + dispatchOffset_;
                uint32_t moePrefixTokenNum = tpBatchInfoTensor_(INFO_NUM + tpIndex);
                uint64_t tokenStoreOffset = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN)
                    + (sizeof(ExpandXType) * hiddenSize_ * moePrefixTokenNum);
                dstTokenStoreGM.SetGlobalBuffer((__gm__ ExpandXType *)(dstRankWorkspaceGm + tokenStoreOffset));
                resolvedTpIndex = tpIndex;
            }

            // ### PATCH START: A5 DMA handoff
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
            if (useMtePipeline) {
                // Slot reuse waits on MTE2; data readiness waits on MTE3. Neither
                // tile handoff waits on Scalar, so the other slot can overlap.
                mtePipeline.WaitForReuse(bufferSlot);
                const uint64_t sourceElementOffset = static_cast<uint64_t>(hiddenSize_) * i;
                const uint64_t destinationElementOffset =
                    static_cast<uint64_t>(hiddenSize_) * moeTokenSendIdx;
                const uint32_t slotElementOffset = static_cast<uint32_t>(
                    bufferSlot * bufferPlan.bytesPerSlot / sizeof(ExpandXType));
                const uint32_t transferElementCount = static_cast<uint32_t>(
                    static_cast<uint64_t>(hiddenSize_) * cnt);
                LocalTensor<ExpandXType> tileTensor = xTensor_[slotElementOffset];
                DataCopy(tileTensor, xGMTensor_[sourceElementOffset], transferElementCount);
                mtePipeline.MarkLoaded(bufferSlot);
                mtePipeline.WaitLoaded(bufferSlot);
                DataCopy(dstTokenStoreGM[destinationElementOffset], tileTensor, transferElementCount);
                mtePipeline.MarkWritten(bufferSlot);
                bufferSlot = (bufferSlot + 1) % Cam::AsyncMte::BUFFER_COUNT;
            } else
#endif
            {
                // B0 remains the A3 path and the fallback when two complete rows do not fit.
                DataCopy(xTensor_, xGMTensor_[hiddenSize_ * i], hiddenSize_ * cnt);
                SyncFunc<HardEvent::MTE2_S>();
                SyncFunc<HardEvent::S_MTE2>();

                SyncFunc<HardEvent::S_MTE3>();
                DataCopy(dstTokenStoreGM[hiddenSize_ * moeTokenSendIdx], xTensor_, hiddenSize_ * cnt);
                SyncFunc<HardEvent::MTE3_S>();
            }
            // ### PATCH END: A5 DMA handoff

            tpExpertTokenSendCntTensor_(tpIndex) += cnt;
            i += cnt;
        }
    }
    // ### PATCH START: A5 transfer drain
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    if (useMtePipeline) {
        // Drain every peer write before the original cross-core completion and
        // before the token buffers are reused as completion-prefix scratch.
        mtePipeline.Drain();
    }
#endif
    // ### PATCH END: A5 transfer drain
    SyncAll<true>();

    if (endExpert == expertNumPerMoe_ - 1) {
        for (uint32_t i = 0; i < tpSize_; ++i) {
            if (i % aivNum_ != aivId_) {
                continue;
            }
            // how many tokens tp i sent to prior moes
            uint32_t moePrefixTokenNum = tpBatchInfoTensor_(INFO_NUM + i);
            uint32_t moeSelfRankId = moeRankId_;
            uint32_t dstRankId = tpAttnRankId + i;

            GM_ADDR dstRankWorkspaceGm = GetPeerAddrByRankId(dstRankId) + dispatchOffset_;

            GlobalTensor<uint32_t> tokenStatusGM;
            tokenStatusGM.SetGlobalBuffer((__gm__ uint32_t *)dstRankWorkspaceGm);

            moePrefixTokenNumTensor_(0) = moePrefixTokenNum + 1;
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyPad(tokenStatusGM[(moeSelfRankId - attnRankNum_)], moePrefixTokenNumTensor_,
                {1U, sizeof(uint32_t), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
        }
    }
}
}  // namespace MoeDistributeCombineSendImpl
#endif // AFD_ASYNC_COMBINE_SEND_H
