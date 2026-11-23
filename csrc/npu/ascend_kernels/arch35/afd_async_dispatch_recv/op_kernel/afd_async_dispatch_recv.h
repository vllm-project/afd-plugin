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
#include "cam_async_window.h"
// ### PATCH START: A5 MTE event helper
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
#include "cam_async_mte_pipeline.h"
#define CAM_ASYNC_RECEIVE_CACHE_INLINE __aicore__ inline
#include "cam_async_receive_cache.h"
#undef CAM_ASYNC_RECEIVE_CACHE_INLINE
#define CAM_ASYNC_RECEIVE_RUNS_INLINE __aicore__ inline
#include "cam_async_receive_runs.h"
#undef CAM_ASYNC_RECEIVE_RUNS_INLINE
#endif
// ### PATCH END: A5 MTE event helper

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
// ### PATCH START: Read the retained metadata for contiguous source runs
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
struct ReceiveRunBackend {
    LocalTensor<int32_t> *sourceCounts;
    LocalTensor<uint32_t> *tpCounts;
    LocalTensor<uint16_t> *wireIndices;
    uint32_t expertCount;

    __aicore__ inline uint32_t read_count(uint32_t expert) const
    {
        return static_cast<uint32_t>(sourceCounts->GetValue(expert));
    }
    __aicore__ inline uint32_t read_total(uint32_t expert) const
    {
        return tpCounts->GetValue(expert);
    }
    __aicore__ inline uint32_t read_prior(uint32_t expert) const
    {
        return tpCounts->GetValue(expertCount + expert);
    }
    __aicore__ inline uint32_t read_wire(uint32_t index) const
    {
        return wireIndices->GetValue(index);
    }
};
#endif
// ### PATCH END: Read the retained metadata for contiguous source runs
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
    __aicore__ inline uint64_t MathCeil(uint64_t n, uint64_t align)
    {
        return align * (n / align + (n % align == 0 ? 0 : 1));
    }

    __aicore__ inline GM_ADDR GetPeerAddrByRankId(const int32_t rankId)
    {
        return dataWindow_.GetDataWindow(rankId);
    }

    __aicore__ inline void DispatchRecv();
    __aicore__ inline void DispatchRecvRouted(uint32_t tpAttnRankId, uint32_t tpIndex,
                                               uint32_t startExpert, uint32_t endExpert);

    // ### PATCH START: A5 private row pipeline
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    __aicore__ inline void DispatchRecvRowsPipelined(GlobalTensor<ExpandXOutType> &tokenStore,
        uint32_t tokenStart, uint32_t tokenEnd, uint32_t startExpert, uint32_t endExpert,
        uint64_t secondBufferOffset);
    __aicore__ inline bool dispatch_recv_runs_pipelined(GlobalTensor<ExpandXOutType> &tokenStore,
        uint32_t tokenStart, uint32_t tokenEnd, uint32_t startExpert, uint32_t endExpert,
        const AsyncReceiveRuns::ReceiveRunPlan &plan, uint64_t firstTileOffset);
#endif
    // ### PATCH END: A5 private row pipeline

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
    Cam::AsyncDataWindow dataWindow_;

    uint64_t ubReuseSize_{0};
    // ### PATCH START: Preserve the address-slice UB base
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    uint64_t tokenAddrBufferOffset_{0};
#endif
    // ### PATCH END: Preserve the address-slice UB base

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

// Patch reason: B0 gives the address slice all remaining UB, so an optional
// second row must retain the original slice base and bounds.
// Patch functionality: record that UB base for the A5-only row pipeline.
// Signature: matches the original Init; no added parameters.
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
    dataWindow_.Init(AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), moeRankId_);

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

    // ### PATCH START: Record the existing address-slice UB base
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    tokenAddrBufferOffset_ = bufOffset;
#endif
    // ### PATCH END: Record the existing address-slice UB base
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

// Patch reason: B0 serializes each row through Scalar while MTE2/MTE3 can
// process separate UB rows concurrently on A5.
// Patch functionality: use independent contiguous-run tiles or the original
// two-row cache only when spare UB fits, then drain before original flag reuse.
// Signature: matches the original DispatchRecvRouted; no added parameters.
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

    // ### PATCH START: A5 optional pipelines with an unchanged address slice
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    uint64_t addressSliceBytes = MathCeil(sizeof(uint16_t) * tokenNumPerAiv, UB_ALIGN);
    uint64_t rowBytes = DynamicQuant
        ? sizeof(ExpandXOutType) * hiddenSizeQuant_ : sizeof(ExpandXOutType) * hiddenSize_;
    rowBytes = MathCeil(rowBytes, UB_ALIGN);
    // ### PATCH START: Prefer real contiguous runs using two new UB tiles
    const auto runPlan = AsyncReceiveRuns::plan_receive_runs(
        ubReuseSize_, addressSliceBytes, rowBytes, tokenNumPerAiv);
    if (runPlan.enabled && dispatch_recv_runs_pipelined(dstTokenStoreGMTensor, tokenNumPerAivStart,
        tokenNumPerAivEnd, startExpert, endExpert, runPlan, tokenAddrBufferOffset_ + addressSliceBytes)) {
        // The complete source slice has been copied and both run tiles drained.
    } else
    // ### PATCH END: Prefer real contiguous runs using two new UB tiles
    if (AsyncMte::CanUseReceiveSecondSlot(ubReuseSize_, addressSliceBytes,
        rowBytes, tokenNumPerAiv)) {
        DispatchRecvRowsPipelined(dstTokenStoreGMTensor, tokenNumPerAivStart,
            tokenNumPerAivEnd, startExpert, endExpert, tokenAddrBufferOffset_ + addressSliceBytes);
    } else
#endif
    {
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

    }
    // ### PATCH END: A5 optional pipelines with an unchanged address slice

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

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
// ### PATCH START: A5 contiguous receive runs
// Patch reason: row-at-a-time gather emits separate input DMA for adjacent wire
// rows, even when their output is contiguous inside the same selected expert.
// Patch functionality: a shared cursor preserves placement and combines only
// these runs; two independent optional tiles follow the mandatory address slice.
// New private parameters: plan owns each tile's capacity, firstTileOffset is
// its UB base. False means no contiguous candidate and retains cache/P0.
template <TemplateMC2TypeClass>
__aicore__ inline bool AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::dispatch_recv_runs_pipelined(
    GlobalTensor<ExpandXOutType> &tokenStore, uint32_t tokenStart, uint32_t tokenEnd,
    uint32_t startExpert, uint32_t endExpert, const AsyncReceiveRuns::ReceiveRunPlan &plan,
    uint64_t firstTileOffset)
{
    const ReceiveRunBackend metadata{
        &expertRecvTokenNumTensor_, &tpExpertRecvTokenNumTensor_, &tokenAddrOffsetTensor_, expertNumPerMoe_};
    if (!AsyncReceiveRuns::has_contiguous_run(metadata, tokenStart, tokenEnd,
        startExpert, endExpert, plan.rowsPerTile)) return false;
    static_assert(AsyncReceiveRuns::RUN_TILE_COUNT == AsyncMte::BUFFER_COUNT,
        "receive run tiles retain the original two-slot event ownership");
    const uint32_t rowElements = DynamicQuant ? hiddenSizeQuant_ : hiddenSize_;
    const uint64_t wireRowBytes = sizeof(ExpandXOutType) * static_cast<uint64_t>(rowElements);
    const uint64_t payloadBytes = sizeof(ExpandXOutType) * static_cast<uint64_t>(hiddenSize_);
    if (AsyncReceiveRuns::plan_payload_copy(payloadBytes, wireRowBytes, 1).blockCount == 0) return false;
    LocalTensor<ExpandXOutType> tiles[AsyncReceiveRuns::RUN_TILE_COUNT] = {
        ubBuffer_.GetWithOffset<ExpandXOutType>(plan.bytesPerTile / sizeof(ExpandXOutType), firstTileOffset),
        ubBuffer_.GetWithOffset<ExpandXOutType>(plan.bytesPerTile / sizeof(ExpandXOutType),
            firstTileOffset + plan.bytesPerTile)};
    AsyncReceiveRuns::ReceiveRunCursor<ReceiveRunBackend> cursor(metadata);
    cursor.init(tokenStart, tokenEnd, startExpert, endExpert, plan.rowsPerTile);
    AsyncMte::TwoSlotPipeline pipeline;
    pipeline.Init(tpipe_);
    AsyncReceiveRuns::ReceiveRun run;
    uint32_t slot = 0;
    while (cursor.next(run)) {
        pipeline.WaitForReuse(slot);
        DataCopyPad(tiles[slot], tokenStore[static_cast<uint64_t>(run.wireIndex) * rowElements],
            {1U, static_cast<uint32_t>(run.rows * wireRowBytes), 0U, 0U, 0U}, {false, 0U, 0U, 0});
        pipeline.MarkLoaded(slot);
        pipeline.WaitLoaded(slot);
        const auto payloadCopy = AsyncReceiveRuns::plan_payload_copy(payloadBytes, wireRowBytes, run.rows);
        DataCopyPad(xOutGMTensor_[static_cast<uint64_t>(run.outputRow) * hiddenSize_], tiles[slot],
            {payloadCopy.blockCount, payloadCopy.blockBytes, payloadCopy.sourceStrideBlocks, 0U, 0U});
        if constexpr (DynamicQuant) {
            // The payload copy strips one 32B padding block per row. Keep each
            // original exact 4B scale extraction under the same tile's lease.
            const uint64_t tileOffset = firstTileOffset + slot * plan.bytesPerTile;
            auto scales = ubBuffer_.GetWithOffset<float>(plan.bytesPerTile / sizeof(float), tileOffset);
            for (uint32_t row = 0; row < run.rows; ++row) {
                const uint64_t scaleIndex = (row * wireRowBytes + payloadBytes) / sizeof(float);
                DataCopyPad(quantScaleGMTensor_[run.outputRow + row], scales[scaleIndex],
                    {1U, sizeof(float), 0U, 0U, 0U});
            }
        }
        // A tile remains owned through ALL payload and scale source reads.
        pipeline.MarkWritten(slot);
        slot = (slot + 1) % AsyncReceiveRuns::RUN_TILE_COUNT;
    }
    pipeline.Drain();
    return true;
}
// ### PATCH END: A5 contiguous receive runs

// ### PATCH START: A5 MTE row handoff
// Both quantized payload and its scale stay in one leased row slot until MTE3
// finishes both output copies. A source-address/wire-index cache reuses real
// loads only within this routed call; every hit reacquires the prior MTE3 credit.
// Expert/TP/token placement and every assignment output follow the B0 loop.
// New private-helper parameters: tokenStore is the source wire-row view;
// [tokenStart, tokenEnd) is this AIV's source-expert row interval; startExpert
// seeds the original output-index walk; the new endExpert parameter disables
// cache lookup for a single selected expert. secondBufferOffset is the UB byte
// offset of row slot 1 after the preserved address slice. No public ABI changes.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchRecv<TemplateMC2TypeFunc>::DispatchRecvRowsPipelined(
    GlobalTensor<ExpandXOutType> &tokenStore, uint32_t tokenStart, uint32_t tokenEnd,
    uint32_t startExpert, uint32_t endExpert, uint64_t secondBufferOffset)
{
    uint32_t rowElements = DynamicQuant ? hiddenSizeQuant_ : hiddenSize_;
    uint64_t rowBytes = MathCeil(sizeof(ExpandXOutType) * rowElements, UB_ALIGN);
    LocalTensor<ExpandXOutType> rows[DOUBLE] = {
        xOutTensor_, ubBuffer_.GetWithOffset<ExpandXOutType>(rowBytes / sizeof(ExpandXOutType),
            secondBufferOffset)};
    AsyncMte::TwoSlotPipeline pipeline;
    pipeline.Init(tpipe_);
    static_assert(AsyncReceiveCache::CACHE_WAYS == DOUBLE,
        "receive cache must retain the existing two leased row slots");
    AsyncReceiveCache::TwoRowCache cache;

    uint32_t currExpertId = startExpert;
    uint32_t currExpertTokenNum = 0;
    uint32_t nextExpertTokenNum = expertRecvTokenNumTensor_(currExpertId);
    uint32_t preExpertTokenNum = 0;
    while (nextExpertTokenNum < tokenStart) {
        preExpertTokenNum += tpExpertRecvTokenNumTensor_(currExpertId);
        currExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
        ++currExpertId;
        nextExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
    }
    // A single selected expert, or an AIV slice wholly inside the initial
    // expert, retains P0's alternating slots without source-key/tag operations.
    // A slice starting exactly on a boundary conservatively keeps the cache.
    const bool useCache = AsyncReceiveCache::CanUseCache(
        startExpert, endExpert, tokenEnd, nextExpertTokenNum);
    for (uint32_t expertSlotOffset = tokenStart; expertSlotOffset < tokenEnd; ++expertSlotOffset) {
        while (expertSlotOffset >= nextExpertTokenNum) {
            preExpertTokenNum += tpExpertRecvTokenNumTensor_(currExpertId);
            currExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
            ++currExpertId;
            nextExpertTokenNum += expertRecvTokenNumTensor_(currExpertId);
        }
        uint32_t tpExpertSlotOffset = preExpertTokenNum
            + tpExpertRecvTokenNumTensor_(expertNumPerMoe_ + currExpertId)
            + (expertSlotOffset - currExpertTokenNum);
        uint16_t moeTokenSendIdx = tokenAddrOffsetTensor_(expertSlotOffset - tokenStart);
        const uint64_t sourceElementOffset = static_cast<uint64_t>(rowElements) * moeTokenSendIdx;
        uint32_t slot = (expertSlotOffset - tokenStart) % DOUBLE;
        bool cacheHit = false;
        AsyncReceiveCache::RowKey key{};
        if (useCache) {
            key = {reinterpret_cast<uint64_t>(tokenStore.GetPhyAddr())
                + sizeof(ExpandXOutType) * sourceElementOffset, moeTokenSendIdx};
            const AsyncReceiveCache::RowSelection selection = cache.Select(key);
            slot = selection.slot;
            cacheHit = selection.hit;
        }

        // A hit must also finish its earlier MTE3 payload/scale reads. For a
        // pending hit this credit proves the earlier real row load completed.
        pipeline.WaitForReuse(slot);
        if (!cacheHit) {
            DataCopy(rows[slot], tokenStore[sourceElementOffset], rowElements);
        }
        // On a hit this ready pair bridges the acquired MTE3->MTE2 credit back
        // to MTE3; it protects already-loaded bytes and never invents a new load.
        pipeline.MarkLoaded(slot);
        pipeline.WaitLoaded(slot);
        DataCopy(xOutGMTensor_[static_cast<uint64_t>(hiddenSize_) * tpExpertSlotOffset], rows[slot], hiddenSize_);
        if constexpr (DynamicQuant) {
            LocalTensor<float> scale = xOutFloatTensor_;
            if (slot != 0) {
                scale = ubBuffer_.GetWithOffset<float>(rowBytes / sizeof(float), secondBufferOffset);
            }
            DataCopyPad(quantScaleGMTensor_[tpExpertSlotOffset], scale[hiddenSize_ / sizeof(float)],
                {1U, sizeof(float), 0U, 0U, 0U});
        }
        pipeline.MarkWritten(slot);
        if (useCache) {
            cache.Remember(slot, key);
        }
    }
    // Clear all in-flight slot ownership and return event leases before the
    // original chunk flags and ClearFlags/SyncAll source-window lifetime logic.
    pipeline.Drain();
}
// ### PATCH END: A5 MTE row handoff
#endif

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
