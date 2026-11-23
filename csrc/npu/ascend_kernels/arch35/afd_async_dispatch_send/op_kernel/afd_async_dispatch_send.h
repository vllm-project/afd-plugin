/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncDispatchSend operator kernel function header file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncDispatchSend operator kernel function header file
 */
#ifndef AFD_ASYNC_DISPATCH_SEND_H
#define AFD_ASYNC_DISPATCH_SEND_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "afd_async_dispatch_send_tiling.h"
#include "cam_async_comm_args.h"
#include "cam_async_window.h"
// ### PATCH START: A5 sparse payload rank bookkeeping
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
#define AFD_ASYNC_DISPATCH_SPARSE_RANKS 1
#define CAM_ASYNC_SPARSE_RANKS_INLINE __aicore__ inline
#include "cam_async_sparse_ranks.h"
#undef CAM_ASYNC_SPARSE_RANKS_INLINE
#define AFD_ASYNC_DISPATCH_TOKEN_PIPELINE 1
#include "cam_async_dispatch_pipeline.h"
#endif
// ### PATCH END: A5 sparse payload rank bookkeeping
// ### PATCH START: Isolate SIMT declarations to their selected keys
#define CAM_ASYNC_ROUTING_BUDGET_INLINE __aicore__ inline
#include "cam_async_simt_routing_budget.h"
#undef CAM_ASYNC_ROUTING_BUDGET_INLINE
#if ((defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)) && defined(TILING_KEY_VAR) && (TILING_KEY_VAR == AFD_ASYNC_DISPATCH_SIMT_BF16_KEY || TILING_KEY_VAR == AFD_ASYNC_DISPATCH_SIMT_FP16_KEY)
#define AFD_ASYNC_DISPATCH_SIMT_ROUTING 1
#include "afd_async_dispatch_send_simt.h"
#endif
// ### PATCH END: Isolate SIMT declarations to their selected keys

namespace MoeDistributeDispatchImpl {
constexpr uint64_t CAM_MAX_RANK_SIZE = 384;  // max NPUs supported by the Cam comm library
constexpr uint32_t UB_ALIGN = 32;            // UB aligned to 32 bytes
constexpr uint32_t BATCH_INFO_VAL_NUM = 5;   // number of batch-related info fields
constexpr uint32_t IDS_PAGE_SIZE = 1024 * 64;  // routing table cache page size
constexpr uint32_t IDS_PAGE_ELEMENT_NUM = IDS_PAGE_SIZE / sizeof(int32_t); // elements per cache page
// ### PATCH START: SIMT tiles invalidate the scalar routing-page tag
constexpr int32_t INVALID_IDS_PAGE_INDEX = -1;
// ### PATCH END: SIMT tiles invalidate the scalar routing-page tag

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// ### PATCH START: Compile-time metadata backend selection
// Added template parameter UseSimtRouting selects only count/prefix metadata;
// false retains the scalar call graph and its complete original UB arena.
#define TemplateMC2TypeClass typename XType, typename ExpandXOutType, bool DynamicQuant, bool UseSimtRouting
#define TemplateMC2TypeFunc XType, ExpandXOutType, DynamicQuant, UseSimtRouting
// ### PATCH END: Compile-time metadata backend selection

using namespace AscendC;
using namespace Cam;
// ### PATCH START: Reuse the original flag arena for visited ranks
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
struct SparsePayloadRankBackend {
    static __aicore__ inline void WriteRank(LocalTensor<uint8_t> &ranks,
        uint32_t index, uint32_t rank)
    {
        ranks.SetValue(index, static_cast<uint8_t>(rank));
    }

    static __aicore__ inline uint32_t ReadRank(LocalTensor<uint8_t> &ranks, uint32_t index)
    {
        return ranks.GetValue(index);
    }

    static __aicore__ inline void IncrementOffset(LocalTensor<uint16_t> &offsets, uint32_t rank)
    {
        offsets.SetValue(rank, static_cast<uint16_t>(offsets.GetValue(rank) + 1));
    }
};
#endif
// ### PATCH END: Reuse the original flag arena for visited ranks
template <TemplateMC2TypeClass>
class AfdAsyncDispatchSend {
public:
    __aicore__ inline AfdAsyncDispatchSend() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR workspaceGM,
                                TPipe *pipe, const AfdAsyncDispatchSendTilingData *tilingData,
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

    __aicore__ inline int32_t GetIds(uint32_t idsIndex, bool forceLoad = false)
    {
        uint32_t idsPageIndex = idsIndex / IDS_PAGE_ELEMENT_NUM;

        // routing-table cache page miss
        if (forceLoad || idsPageIndex != idsPageIndex_) {
            // The final page may be shorter than 64 KiB (e.g. the 4-token smoke test).
            uint32_t pageStart = IDS_PAGE_ELEMENT_NUM * idsPageIndex;
            uint32_t remaining = batchSize_ * topk_ - pageStart;
            uint32_t count = remaining < IDS_PAGE_ELEMENT_NUM ? remaining : IDS_PAGE_ELEMENT_NUM;
            DataCopyPad(idsPageTensor_, idsGMTensor_[pageStart],
                {1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U},
                {false, 0U, 0U, 0});
            SyncFunc<HardEvent::MTE2_S>();
            idsPageIndex_ = idsPageIndex;
        }
        return idsPageTensor_(idsIndex % IDS_PAGE_ELEMENT_NUM);
    }

    __aicore__ inline void PreProcess(uint32_t bufOffset);
    // ### PATCH START: A5 UB-only metadata counter
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
    __aicore__ inline void count_routing_tokens(LocalTensor<uint32_t> &statistics);
#endif
    // ### PATCH END: A5 UB-only metadata counter

    __aicore__ inline void SetFlags();

    __aicore__ inline void DispatchSend();

    // A5-only private parameters tokenPipeline/tokenSlot select exclusive tail
    // ready events when tokens overlap. A3 retains the original two-parameter
    // signature; A5 defaults retain existing serial callers and arithmetic.
    __aicore__ inline void QuantProcess(LocalTensor<XType> &xTensor, LocalTensor<ExpandXOutType> &xOutTensor
    // ### PATCH START: A5 private ready-event parameters
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
        , AsyncDispatch::TwoSlotPipeline<DynamicQuant> *tokenPipeline = nullptr, uint32_t tokenSlot = 0
#endif
    // ### PATCH END: A5 private ready-event parameters
    );

    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t attnRankId_{0};
    uint32_t worldSize_{0};
    uint32_t attnRankNum_{0};
    uint32_t moeRankNum_{0};
    uint32_t routeExpertNumPerMoe_{0};
    uint32_t expertNumPerMoe_{0};
    uint32_t expertNum_{0};
    uint32_t maxBatchSizePerRank_{0};
    uint32_t batchSize_{0};
    uint32_t hiddenSize_{0};
    uint32_t topk_{0};
    uint64_t totalUbSize_{0};
    uint64_t totalWorkspaceSize_{0};
    uint64_t workspaceSizePerAttn_{0};
    uint32_t layerIndex_{0};
    int32_t idsPageIndex_{0};

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> idsGMTensor_;
    Cam::AsyncDataWindow dataWindow_;

    uint64_t ubReuseSize_{0};

    TPipe *tpipe_{nullptr};
    TBuf<> ubBuffer_;
    LocalTensor<int32_t> idsPageTensor_;  // routing table
    LocalTensor<int64_t> batchInfoTensor_;  // output
    // tokens received by each routed expert, grouped by MoE rank
    LocalTensor<int32_t> expertRecvTokenNumTensor_;  // output
    LocalTensor<XType> xTensor_;                   // output
    LocalTensor<ExpandXOutType> xOutTensor_;       // output
    LocalTensor<float> xOutFloatTensor_;           // output
    // ### PATCH START: Optional A5 in-place input/wire row slot
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    LocalTensor<XType> xPrefetchTensor_;
    LocalTensor<ExpandXOutType> xPrefetchOutTensor_;
    AsyncDispatch::DispatchBufferPlan dispatchBufferPlan_{};
#endif
    // ### PATCH END: Optional A5 in-place input/wire row slot

    // prefix token count before each routed expert within its MoE rank
    LocalTensor<uint32_t> expertPrefixOffsetTensor_;
    // starting token index for each routed expert on the receiving MoE rank
    LocalTensor<uint16_t> expertSendStartIdxTensor_;
    LocalTensor<uint32_t> moeRecvTokenNumTensor_;
    LocalTensor<uint16_t> moeSendStartOffsetTensor_;
    LocalTensor<uint8_t> tokenSentFlagTensor_;

    // for quantization
    uint32_t hiddenSizeQuant_{0};
    LocalTensor<float> quantFloatTensor_;
    LocalTensor<float> quantFloatAbsTensor_;
    LocalTensor<float> quantRowMaxTensor_;

    uint32_t batchSizePerAivStart_{0};
    uint32_t batchSizePerAivEnd_{0};

    // batch 16 entries, then flush to the peer once
    uint32_t tokenOffsetFlushBatchSize_{0};  // 16
    LocalTensor<uint16_t> tokenOffsetBatchTensor_;        // 16 * expertNum, holds the 16 entries during batching
    LocalTensor<uint32_t> tokenOffsetBatchInfoTensor_;   // 2 * expertNum

    bool isError_{false};
    // expertId | tokens already written to the target position | tokens sent to that expert
    __aicore__ inline void PushExpertTokenOffset(
        uint32_t expertId, uint32_t moeTokenSendIdx, uint32_t expertSlotOffset, uint64_t tokenStoreOffset)
    {
        uint32_t pendingOffsetCnt = tokenOffsetBatchInfoTensor_(2 * expertId);
        if (pendingOffsetCnt == 0) {
            tokenOffsetBatchInfoTensor_(2 * expertId + 1) = expertSlotOffset;
        }

        tokenOffsetBatchTensor_(tokenOffsetFlushBatchSize_ * expertId + pendingOffsetCnt) = moeTokenSendIdx;
        pendingOffsetCnt += 1;
        tokenOffsetBatchInfoTensor_(2 * expertId) = pendingOffsetCnt;

        if (pendingOffsetCnt == tokenOffsetFlushBatchSize_) {
            FlushExpertTokenOffset(expertId, tokenStoreOffset);
        }
    }

    __aicore__ inline void FlushExpertTokenOffset(uint32_t expertId, uint64_t tokenStoreOffset)
    {
        uint32_t pendingOffsetCnt = tokenOffsetBatchInfoTensor_(2 * expertId);
        uint32_t expertSlotOffset  = tokenOffsetBatchInfoTensor_(2 * expertId + 1);

        if (pendingOffsetCnt == 0) {
            return;
        }

        uint32_t dstRankId = attnRankNum_ + expertId / expertNumPerMoe_;
        GM_ADDR dstGM = GetPeerAddrByRankId(dstRankId);
        GlobalTensor<uint16_t> dstTokenAddrOffsetGMTensor;
        dstTokenAddrOffsetGMTensor.SetGlobalBuffer((__gm__ uint16_t*)(dstGM + tokenStoreOffset));

        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(dstTokenAddrOffsetGMTensor[expertSlotOffset],
            tokenOffsetBatchTensor_[tokenOffsetFlushBatchSize_ * expertId],
            {1U, (uint32_t)(sizeof(uint16_t) * pendingOffsetCnt), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
        tokenOffsetBatchInfoTensor_(2 * expertId) = 0;
    }
};

// Patch reason: SIMT resources must not overlap native UB or contaminate scalar keys.
// Patch functionality: reserve the documented SIMT upper bound only for the new
// metadata specialization and let payload routing reload its normal page cache.
// Optional A5 spare UB repeats the original in-place input/wire row layout;
// insufficient spare retains the original arena and minimum UB requirement.
// Signature: matches the original Init; no added method parameters. The added
// class template parameter UseSimtRouting is documented at its declaration.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR workspaceGM, TPipe *pipe,
    const AfdAsyncDispatchSendTilingData *tilingData, GM_ADDR commArgs,
    int32_t isCamComm)
{
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeDispatchInfo.aivNum;
    attnRankId_ = tilingData->moeDistributeDispatchInfo.attnRankId;
    worldSize_ = tilingData->moeDistributeDispatchInfo.worldSize;
    attnRankNum_ = tilingData->moeDistributeDispatchInfo.attnRankNum;
    moeRankNum_ = tilingData->moeDistributeDispatchInfo.moeRankNum;
    routeExpertNumPerMoe_ = tilingData->moeDistributeDispatchInfo.routeExpertNumPerMoe;
    expertNumPerMoe_ = routeExpertNumPerMoe_;
    expertNum_ = expertNumPerMoe_ * moeRankNum_;
    maxBatchSizePerRank_ = tilingData->moeDistributeDispatchInfo.maxBatchSize;
    batchSize_ = tilingData->moeDistributeDispatchInfo.batchSize;
    hiddenSize_ = tilingData->moeDistributeDispatchInfo.hiddenSize;
    topk_ = tilingData->moeDistributeDispatchInfo.topk;
    totalUbSize_ = tilingData->moeDistributeDispatchInfo.totalUbSize;
    // ### PATCH START: Keep SIMT resources outside the native UB arena
    if constexpr (UseSimtRouting) {
        totalUbSize_ -= AsyncRouting::SIMT_RESERVED_UB_BYTES;
    }
    // ### PATCH END: Keep SIMT resources outside the native UB arena
    totalWorkspaceSize_ = tilingData->moeDistributeDispatchInfo.totalWorkspaceSize;
    layerIndex_ = tilingData->moeDistributeDispatchInfo.layerIndex;

    uint64_t batchInfoWorkspaceSize = MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    uint64_t expertRecvTokenNumWorkspaceSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t expertChunkFlagWorkspaceSize = MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    uint64_t tokenWorkspaceSize = 0;
    if constexpr (DynamicQuant) {
        tokenWorkspaceSize = MathCeil(
            (sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) * maxBatchSizePerRank_, UB_ALIGN);
    } else {
        tokenWorkspaceSize = MathCeil(sizeof(XType) * hiddenSize_ * maxBatchSizePerRank_, UB_ALIGN);
    }
    uint64_t tokenAddrWorkspaceSize = MathCeil(sizeof(uint16_t) * maxBatchSizePerRank_ * topk_, UB_ALIGN);
    uint64_t totalWorkspaceSize = batchInfoWorkspaceSize + expertRecvTokenNumWorkspaceSize
                           + expertChunkFlagWorkspaceSize + tokenWorkspaceSize + tokenAddrWorkspaceSize;
    workspaceSizePerAttn_ = UB_ALIGN * (totalWorkspaceSize / UB_ALIGN);


    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    idsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    // commArgs is an ABI placeholder; MC2 initializes the HCCL resource context.
    dataWindow_.Init(AscendC::GetHcclContext<HCCL_GROUP_ID_0>(), attnRankId_);

    // split work across cores
    uint32_t batchSizePerAiv = batchSize_ / aivNum_;
    uint32_t batchSizePerAivRemain = batchSize_ % aivNum_;
    batchSizePerAivStart_ = batchSizePerAiv * aivId_;
    if (aivId_ < batchSizePerAivRemain) {
        batchSizePerAiv += 1;
        batchSizePerAivStart_ += aivId_;
    } else {
        batchSizePerAivStart_ += batchSizePerAivRemain;
    }
    batchSizePerAivEnd_ = batchSizePerAivStart_ + batchSizePerAiv;

    tpipe_ = pipe;
    tpipe_->Reset();
    tpipe_->InitBuffer(ubBuffer_, totalUbSize_);

    uint64_t bufOffset = 0;
    uint64_t bufSize;

    bufSize = MathCeil(IDS_PAGE_SIZE, UB_ALIGN); // 64 KB
    idsPageTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
    bufOffset += bufSize;

    // ### PATCH START: Keep the scalar page loader separate from SIMT tiles
    if constexpr (UseSimtRouting) {
        idsPageIndex_ = INVALID_IDS_PAGE_INDEX;
    } else {
        GetIds(0, true);
    }
    // ### PATCH END: Keep the scalar page loader separate from SIMT tiles

    bufSize = MathCeil(sizeof(int32_t) * expertNum_, UB_ALIGN);
    expertRecvTokenNumTensor_ = ubBuffer_.GetWithOffset<int32_t>(bufSize / sizeof(int32_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint8_t) * moeRankNum_, UB_ALIGN);
    tokenSentFlagTensor_ = ubBuffer_.GetWithOffset<uint8_t>(bufSize / sizeof(uint8_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint16_t) * moeRankNum_, UB_ALIGN);
    moeSendStartOffsetTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint16_t) * expertNum_, UB_ALIGN);
    expertSendStartIdxTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
    bufOffset += bufSize;

    PreProcess(bufOffset);

    bufSize = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN);
    moeRecvTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint32_t) * expertNum_, UB_ALIGN);
    expertPrefixOffsetTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN); // 32 B
    batchInfoTensor_ = ubBuffer_.GetWithOffset<int64_t>(bufSize / sizeof(int64_t), bufOffset);
    bufOffset += bufSize;

    if constexpr (DynamicQuant) {
        hiddenSizeQuant_ = (sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) / sizeof(ExpandXOutType);

        bufSize = MathCeil(sizeof(XType) * hiddenSize_, UB_ALIGN); // 14 KB
        xTensor_ = ubBuffer_.GetWithOffset<XType>(bufSize / sizeof(XType), bufOffset);
        xOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(bufSize / sizeof(ExpandXOutType), bufOffset);
        xOutFloatTensor_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
        bufOffset += bufSize;

        bufSize = MathCeil(sizeof(float) * hiddenSize_, UB_ALIGN); // 28 KB
        quantFloatTensor_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
        bufOffset += bufSize;

        bufSize = MathCeil(sizeof(float) * hiddenSize_, UB_ALIGN); // 28 KB
        quantFloatAbsTensor_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
        bufOffset += bufSize;

        bufSize = MathCeil(sizeof(float) * 8, UB_ALIGN); // 32 B
        quantRowMaxTensor_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
        bufOffset += bufSize;
    } else {
        bufSize = MathCeil(sizeof(XType) * hiddenSize_, UB_ALIGN); // 14 KB
        xTensor_ = ubBuffer_.GetWithOffset<XType>(bufSize / sizeof(XType), bufOffset);
        xOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(bufSize / sizeof(ExpandXOutType), bufOffset);
        bufOffset += bufSize;
    }

    // ----- mandatory buffers allocated above -----

    tokenOffsetFlushBatchSize_ = UB_ALIGN / sizeof(uint16_t);
    bufSize = MathCeil(sizeof(uint16_t) * tokenOffsetFlushBatchSize_ * expertNum_, UB_ALIGN); // 32 B or 64 B
    tokenOffsetBatchTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint32_t) * expertNum_ * 2, UB_ALIGN); // 32 B or 64 B
    tokenOffsetBatchInfoTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // remaining reusable UB space
    ubReuseSize_ = totalUbSize_ - bufOffset;
    // ### PATCH START: One spare row retains the original UB minimum
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    const uint32_t wireElements = DynamicQuant ? hiddenSizeQuant_ : hiddenSize_;
    dispatchBufferPlan_ = AsyncDispatch::plan_dispatch_buffers(ubReuseSize_, hiddenSize_,
        sizeof(XType), wireElements, sizeof(ExpandXOutType), batchSizePerAivEnd_ - batchSizePerAivStart_);
    if (dispatchBufferPlan_.enabled) {
        xPrefetchTensor_ = ubBuffer_.GetWithOffset<XType>(
            dispatchBufferPlan_.secondSlotBytes / sizeof(XType), bufOffset);
        xPrefetchOutTensor_ = ubBuffer_.GetWithOffset<ExpandXOutType>(
            dispatchBufferPlan_.secondSlotBytes / sizeof(ExpandXOutType), bufOffset);
        ubReuseSize_ -= dispatchBufferPlan_.secondSlotBytes;
    }
#endif
    // ### PATCH END: One spare row retains the original UB minimum
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::Process()
{
    if (isError_ == true) {
        return;
    }

    DispatchSend();
}

// Patch reason: the original payload loop clears and scans every MoE rank for
// each token even when routing visits only a few ranks.
// Patch functionality: A5 records visited ranks in the existing flag UB arena
// and advances their wire offsets only after all of a token's assignments.
// With spare UB, A5 also prefetches another in-place input/wire slot and returns
// its credit only after all peer reads; the original quantization is unchanged.
// Signature: unchanged; no added parameters. A3 keeps the original rank loops.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::DispatchSend()
{
    // initialize
    Duplicate(moeRecvTokenNumTensor_, (uint32_t)0, moeRankNum_);  // tokens this attn sends to each moe card
    // per-card-range prefix sum, counted per moe card; offset of tokens sent to each expert
    Duplicate(expertPrefixOffsetTensor_, (uint32_t)0, expertNum_);
    Duplicate(tokenOffsetBatchTensor_, (uint16_t)0, tokenOffsetFlushBatchSize_ * expertNum_); // 16 * expertNum
    Duplicate(tokenOffsetBatchInfoTensor_, (uint32_t)0, expertNum_ * 2);
    SyncFunc<HardEvent::V_S>();

    // accumulate send offsets at expert granularity
    for (uint32_t expertId = 0; expertId < expertNum_; ++expertId) {
        moeRecvTokenNumTensor_(expertId / expertNumPerMoe_) += expertRecvTokenNumTensor_(expertId);

        // Each MoE rank's first routed expert starts at token-address offset 0.
        if ((expertId % expertNumPerMoe_) != 0) {
            expertPrefixOffsetTensor_(expertId) =
                expertPrefixOffsetTensor_(expertId - 1) + expertRecvTokenNumTensor_(expertId - 1);
        }
    }


    uint32_t attnRankId = attnRankId_;
    // each moe card reserves a region for each attn
    uint64_t attnWorkspaceOffset = workspaceSizePerAttn_ * attnRankId;
    // sent token num, attn rank id, layer index
    uint64_t expertRecvTokenNumOffset =
        attnWorkspaceOffset + MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    // tokens received by each expert
    uint64_t expertChunkFlagOffset =
        expertRecvTokenNumOffset + MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    // reserved for moe-side chunking: token num, attn rank id, layer index
    uint64_t tokenStoreOffset =
        expertChunkFlagOffset + MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);
    // tokens come here; after them, expertNum * tokenNum position entries tell each moe expert where to fetch
    uint64_t tokenAddrStoreOffset = 0;
    // compute the max memory for storing tokens, by whether quantized
    if constexpr (DynamicQuant) {
        tokenAddrStoreOffset = tokenStoreOffset
            + MathCeil((sizeof(ExpandXOutType) * hiddenSize_ + UB_ALIGN) * maxBatchSizePerRank_, UB_ALIGN);
    } else {
        tokenAddrStoreOffset = tokenStoreOffset
            + MathCeil(sizeof(XType) * hiddenSize_ * maxBatchSizePerRank_, UB_ALIGN);
    }


    GlobalTensor<ExpandXOutType> dstTokenGMTensor;
    GlobalTensor<uint16_t> dstTokenAddrOffsetGMTensor;
    // ### PATCH START: A5 payload-only token rank tracker
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
    const bool useSparseRanks = AsyncSparseRanks::CanUseSparseRanks(worldSize_, attnRankNum_, moeRankNum_);
    AsyncSparseRanks::TokenRankTracker rankTracker(moeRankNum_);
#endif
    // ### PATCH END: A5 payload-only token rank tracker
    // ### PATCH START: Initialize optional A5 token input/wire leases
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    const bool useTokenPipeline = dispatchBufferPlan_.enabled;
    AsyncDispatch::TwoSlotPipeline<DynamicQuant> tokenPipeline;
    LocalTensor<XType> inputRows[AsyncDispatch::TOKEN_SLOT_COUNT] = {xTensor_, xPrefetchTensor_};
    LocalTensor<ExpandXOutType> wireRows[AsyncDispatch::TOKEN_SLOT_COUNT] = {xOutTensor_, xPrefetchOutTensor_};
    uint32_t tokenSlot = 0;
    if (useTokenPipeline) {
        tokenPipeline.init(tpipe_);
        DataCopy(inputRows[tokenSlot],
            xGMTensor_[static_cast<uint64_t>(hiddenSize_) * batchSizePerAivStart_], hiddenSize_);
        tokenPipeline.mark_loaded(tokenSlot);
    }
#endif
    // ### PATCH END: Initialize optional A5 token input/wire leases
    // Send one address entry per routed assignment and deduplicate payloads per rank.
    for (uint32_t topkId = (topk_ * batchSizePerAivStart_); topkId < (topk_ * batchSizePerAivEnd_); ++topkId) {
        uint32_t tokenId = topkId / topk_;  // corresponding token id
        if (topkId % topk_ == 0) {  // first read of the token
            // ### PATCH START: Prefetch the next input before current quantization
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
            if (useTokenPipeline) {
                tokenSlot = (tokenId - batchSizePerAivStart_) % AsyncDispatch::TOKEN_SLOT_COUNT;
                tokenPipeline.wait_loaded(tokenSlot);
                if (tokenId + 1 < batchSizePerAivEnd_) {
                    const uint32_t nextSlot = (tokenSlot + 1) % AsyncDispatch::TOKEN_SLOT_COUNT;
                    tokenPipeline.wait_for_reuse(nextSlot);
                    DataCopy(inputRows[nextSlot],
                        xGMTensor_[static_cast<uint64_t>(hiddenSize_) * (tokenId + 1)], hiddenSize_);
                    tokenPipeline.mark_loaded(nextSlot);
                }
                if constexpr (DynamicQuant) {
                    // Both original FP32 scratch rows stay on the ordered V
                    // pipeline; each token's result and scale use its own slot.
                    QuantProcess(inputRows[tokenSlot], wireRows[tokenSlot], &tokenPipeline, tokenSlot);
                }
            } else
#endif
            // ### PATCH END: Prefetch the next input before current quantization
            {
            // read the token
            if constexpr (DynamicQuant) {
                DataCopy(xTensor_, xGMTensor_[hiddenSize_ * tokenId], hiddenSize_);
                SyncFunc<HardEvent::MTE2_V>();
                SyncFunc<HardEvent::V_MTE2>();


                QuantProcess(xTensor_, xOutTensor_);
            } else {
                DataCopy(xTensor_, xGMTensor_[hiddenSize_ * tokenId], hiddenSize_);
                SyncFunc<HardEvent::MTE2_S>();
                SyncFunc<HardEvent::S_MTE2>();
                SyncFunc<HardEvent::S_MTE3>();
            }
            }

            // ### PATCH START: Reset scalar state instead of all rank flags
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
            if (useSparseRanks) {
                rankTracker.BeginToken();
            } else
#endif
            {
                // clear the send table
                for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
                    tokenSentFlagTensor_(moeRankId) = 0;  // whether sent to this moe
                }
            }
            // ### PATCH END: Reset scalar state instead of all rank flags

            // First routed destination performs the physical token send.
        }

        // send token to the routing expert
        uint32_t routeExpertId = GetIds(topkId);  // routing expert id
        uint32_t moeRankId = routeExpertId / routeExpertNumPerMoe_;  // which rank to send to
        // how many prior AIVs sent to this MoE; deduplicate routed destinations
        uint32_t moeTokenSendIdx = moeSendStartOffsetTensor_(moeRankId);
        uint32_t dstRankId = attnRankNum_ + moeRankId;
        GM_ADDR dstGM = GetPeerAddrByRankId(dstRankId);
        // The compact protocol uses the input's zero-based global expert ID.
        uint32_t expertId = routeExpertId;
        // tokens received by experts before expertId on its card
        uint32_t expertPrefixOffset = expertPrefixOffsetTensor_(expertId);
        // how many prior aivs sent to this expert; +1 after each send
        uint16_t expertTokenSendIdx = expertSendStartIdxTensor_(expertId);
        uint32_t expertSlotOffset = expertPrefixOffset + expertTokenSendIdx;

        // ### PATCH START: First encounter uses a token-local rank bitmap
        bool firstRankVisit;
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
        if (useSparseRanks) {
            firstRankVisit = rankTracker.RecordRank<SparsePayloadRankBackend>(
                tokenSentFlagTensor_, moeRankId) == AsyncSparseRanks::Visit::First;
        } else
#endif
        {
            firstRankVisit = tokenSentFlagTensor_(moeRankId) == 0;
        }
        // ### PATCH END: First encounter uses a token-local rank bitmap
        // send only if the token has not been sent to this moe
        if (firstRankVisit) {
            dstTokenGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType*)(dstGM + tokenStoreOffset));
            // ### PATCH START: Hold one wire row for every peer of this token
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
            LocalTensor<ExpandXOutType> &payloadRow = useTokenPipeline ? wireRows[tokenSlot] : xOutTensor_;
#else
            LocalTensor<ExpandXOutType> &payloadRow = xOutTensor_;
#endif
            if constexpr (DynamicQuant) {
                DataCopy(dstTokenGMTensor[hiddenSizeQuant_ * moeTokenSendIdx], payloadRow, hiddenSizeQuant_);
            } else {
                DataCopy(dstTokenGMTensor[hiddenSize_ * moeTokenSendIdx], payloadRow, hiddenSize_);
            }
            // ### PATCH END: Hold one wire row for every peer of this token
            // ### PATCH START: A5 replaces peer waits with one whole-token credit
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
            if (!useTokenPipeline)
#endif
            {
                SyncFunc<HardEvent::MTE3_S>();
            }
            // ### PATCH END: A5 replaces peer waits with one whole-token credit
            // ### PATCH START: Only the original path writes rank-indexed flags
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
            if (!useSparseRanks)
#endif
            {
                tokenSentFlagTensor_(moeRankId) = 1;
            }
            // ### PATCH END: Only the original path writes rank-indexed flags
        }


        PushExpertTokenOffset(expertId, moeTokenSendIdx, expertSlotOffset, tokenAddrStoreOffset);

        expertSendStartIdxTensor_(expertId) = expertTokenSendIdx + 1;

        if ((topkId + 1) % topk_ == 0) {
            // ### PATCH START: Advance only visited ranks at token end
#if defined(AFD_ASYNC_DISPATCH_SPARSE_RANKS)
            if (useSparseRanks) {
                rankTracker.AdvanceOffsets<SparsePayloadRankBackend>(
                    tokenSentFlagTensor_, moeSendStartOffsetTensor_);
            } else
#endif
            {
                // refresh offsets from the send table
                for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
                    if (tokenSentFlagTensor_(moeRankId) == 1) {
                        ++moeSendStartOffsetTensor_(moeRankId);
                    }
                }
            }
            // ### PATCH END: Advance only visited ranks at token end
            // ### PATCH START: Credit follows all peer reads, not each assignment
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
            if (useTokenPipeline) tokenPipeline.mark_token_sent(tokenSlot);
#endif
            // ### PATCH END: Credit follows all peer reads, not each assignment
        }
    }
    // ### PATCH START: Retire payload leases before metadata and completion flags
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    if (useTokenPipeline) tokenPipeline.drain();
#endif
    // ### PATCH END: Retire payload leases before metadata and completion flags
    for (uint32_t expertId = 0; expertId < expertNum_; ++expertId) {
        FlushExpertTokenOffset(expertId, tokenAddrStoreOffset);
    }

    SetFlags();
}

// Patch reason: the original Scalar metadata loop serializes count/prefix work.
// Patch functionality: selected A5 keys use UB-only SIMT while preserving native
// GM copies, both cross-AIV barriers, final gather, and the scalar staging fallback.
// Signature: matches the original PreProcess; no added method parameters.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::PreProcess(uint32_t bufOffset)
{
    uint32_t statEntriesPerAiv = moeRankNum_ + expertNum_;   // moe rank num + total expert num
    uint32_t bufSize = MathCeil(sizeof(uint32_t) * statEntriesPerAiv * aivNum_, UB_ALIGN); // 32 B or 64 B
    LocalTensor<uint32_t> aivStatTensor = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;
    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // initialize
    Duplicate(expertRecvTokenNumTensor_, (int32_t)0, expertNum_);  // tokens this attn sends to each expert
    // how many tokens prior aivs sent to each expert; i.e. where this aiv starts
    Duplicate(expertSendStartIdxTensor_, (uint16_t)0, expertNum_);
    // how many tokens prior aivs sent to each moe rank; i.e. where this aiv starts
    Duplicate(moeSendStartOffsetTensor_, (uint16_t)0, moeRankNum_);
    // prefix sum: how many each aiv sends to each moe and each expert
    Duplicate(aivStatTensor, (uint32_t)0, statEntriesPerAiv * aivNum_);
    SyncFunc<HardEvent::V_S>();

    // ### PATCH START: Preserve stable native-AIV counts through UB SIMT
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
    if constexpr (UseSimtRouting) {
        count_routing_tokens(aivStatTensor);
    } else
#endif
    {
        // each aiv counts the quantity it is responsible for sending
        for (uint32_t topkId = (topk_ * batchSizePerAivStart_); topkId < (topk_ * batchSizePerAivEnd_); ++topkId) {
            if (topkId % topk_ == 0) {  // a new token
                // clear the send table
                for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {  // routed destinations only
                    tokenSentFlagTensor_(moeRankId) = 0;
                }

                // Only routed destinations participate in the per-token send table.
            }

            {
                uint32_t routeExpertId = GetIds(topkId);
                uint32_t moeRankId = routeExpertId / routeExpertNumPerMoe_;
                aivStatTensor(moeRankNum_ + routeExpertId)++;

                if (tokenSentFlagTensor_(moeRankId) == 0) {
                    tokenSentFlagTensor_(moeRankId) = 1;
                }
            }

            if ((topkId + 1) % topk_ == 0) {
                // refresh offsets from the send table
                for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
                    if (tokenSentFlagTensor_(moeRankId) == 1) {
                        aivStatTensor(moeRankId)++;  // how many this aiv sends to each moe
                    }
                }
            }
        }
    }
    // ### PATCH END: Preserve stable native-AIV counts through UB SIMT

    // each aiv shares the quantity it is responsible for sending with the other aivs
    GM_ADDR srcGM = GetPeerAddrByRankId(attnRankId_);
    GlobalTensor<uint32_t> aivStatGMTensor;
    aivStatGMTensor.SetGlobalBuffer((__gm__ uint32_t *)srcGM);
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyPad(aivStatGMTensor[statEntriesPerAiv * aivId_], aivStatTensor,
        {1U, (uint32_t)(sizeof(uint32_t) * statEntriesPerAiv), 0U, 0U, 0U});  // moe rank num
    SyncFunc<HardEvent::MTE3_S>();

    SyncAll<true>();

    // gather the quantity every aiv is responsible for sending
    DataCopyPad(aivStatTensor, aivStatGMTensor,
        {1U, (uint32_t)(sizeof(uint32_t) * statEntriesPerAiv * aivNum_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();

    // Assign consecutive columns evenly, then stage their inclusive prefixes in aligned UB rows.
    // Use the same UB capacity decision on every core, including cores with one fewer column.
    uint32_t maxColumnsPerAiv = (statEntriesPerAiv + aivNum_ - 1) / aivNum_;
    uint64_t maxPrefixBufferSize = static_cast<uint64_t>(aivNum_ - 1)
        * MathCeil(maxColumnsPerAiv * sizeof(uint32_t), UB_ALIGN);
    // ### PATCH START: Use the same checked staging gate in native and CPU validation
    if (AsyncRouting::CanFitPrefixStaging(totalUbSize_, bufOffset, maxPrefixBufferSize)) {
    // ### PATCH END: Use the same checked staging gate in native and CPU validation
        uint32_t columnsPerAiv = statEntriesPerAiv / aivNum_;
        uint32_t extraColumns = statEntriesPerAiv % aivNum_;
        uint32_t columnStart = aivId_ * columnsPerAiv
            + (aivId_ < extraColumns ? aivId_ : extraColumns);
        uint32_t columnCount = columnsPerAiv + (aivId_ < extraColumns ? 1U : 0U);
        if (columnCount > 0 && aivNum_ > 1) {
            uint32_t prefixRowStride = MathCeil(columnCount * sizeof(uint32_t), UB_ALIGN) / sizeof(uint32_t);
            LocalTensor<uint32_t> prefixTensor = ubBuffer_.GetWithOffset<uint32_t>(
                (aivNum_ - 1) * prefixRowStride, bufOffset);
            // ### PATCH START: Fill aligned UB prefixes without SIMT GM writes
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
            if constexpr (UseSimtRouting) {
                SyncFunc<HardEvent::MTE2_V>();
                AscendC::Simt::VF_CALL<prefix_routing_columns_vf>(
                    AscendC::Simt::Dim3(ROUTING_VF_THREAD_COUNT),
                    (__ubuf__ const uint32_t *)aivStatTensor.GetPhyAddr(),
                    (__ubuf__ uint32_t *)prefixTensor.GetPhyAddr(), aivNum_,
                    statEntriesPerAiv, columnStart, columnCount, prefixRowStride);
                SyncFunc<HardEvent::V_MTE3>();
            } else
#endif
            {
                for (uint32_t col = 0; col < columnCount; ++col) {
                    uint32_t sum = aivStatTensor(columnStart + col);
                    for (uint32_t row = 1; row < aivNum_; ++row) {
                        sum += aivStatTensor(statEntriesPerAiv * row + columnStart + col);
                        prefixTensor((row - 1) * prefixRowStride + col) = sum;
                    }
                }
            }
            // ### PATCH END: Fill aligned UB prefixes without SIMT GM writes

            // GM keeps its original compact row layout; only the UB staging rows are padded.
            constexpr uint32_t PREFIX_COPY_ROWS = 32;
            SyncFunc<HardEvent::S_MTE3>();
            for (uint32_t row = 1; row < aivNum_; row += PREFIX_COPY_ROWS) {
                uint32_t rowsToCopy = aivNum_ - row;
                rowsToCopy = rowsToCopy < PREFIX_COPY_ROWS ? rowsToCopy : PREFIX_COPY_ROWS;
                DataCopyPad(aivStatGMTensor[statEntriesPerAiv * row + columnStart],
                    prefixTensor[(row - 1) * prefixRowStride],
                    {static_cast<uint16_t>(rowsToCopy), static_cast<uint32_t>(columnCount * sizeof(uint32_t)),
                     0U, static_cast<uint32_t>((statEntriesPerAiv - columnCount) * sizeof(uint32_t)), 0U});
            }
            SyncFunc<HardEvent::MTE3_S>();
        }
    } else {
        // Preserve support for shapes whose existing statistics buffer nearly fills UB.
        for (uint32_t col = 0; col < statEntriesPerAiv; ++col) {
            if (col % aivNum_ != aivId_) {
                continue;
            }
            for (uint32_t row = 1; row < aivNum_; ++row) {
                uint32_t sum = aivStatTensor(statEntriesPerAiv * (row - 1) + col)
                    + aivStatTensor(statEntriesPerAiv * row + col);
                aivStatTensor(statEntriesPerAiv * row + col) = sum;
                aivStatTensor(0) = sum;
                SyncFunc<HardEvent::S_MTE3>();
                DataCopyPad(aivStatGMTensor[statEntriesPerAiv * row + col], aivStatTensor,
                    {1U, sizeof(uint32_t), 0U, 0U, 0U});
                SyncFunc<HardEvent::MTE3_S>();
            }
        }
    }

    SyncAll<true>();

    // re-gather the quantity every aiv is responsible for sending
    DataCopyPad(aivStatTensor, aivStatGMTensor,
        {1U, (uint32_t)(sizeof(uint32_t) * statEntriesPerAiv * aivNum_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();

    // aiv updates its send offsets
    uint32_t lastRowIdx = aivNum_ - 1;
    if (aivId_ == 0) {
        for (uint32_t col = moeRankNum_; col < statEntriesPerAiv; ++col) {
            uint32_t expertId = col - moeRankNum_;
            expertRecvTokenNumTensor_(expertId) = aivStatTensor(statEntriesPerAiv * lastRowIdx + col);
        }
    } else {
        for (uint32_t col = 0; col < statEntriesPerAiv; ++col) {
            if (col < moeRankNum_) {
                moeSendStartOffsetTensor_(col) += aivStatTensor(statEntriesPerAiv * (aivId_ - 1) + col);
            } else {
                uint32_t expertId = col - moeRankNum_;
                expertRecvTokenNumTensor_(expertId) = aivStatTensor(statEntriesPerAiv * lastRowIdx + col);
                expertSendStartIdxTensor_(expertId) = aivStatTensor(statEntriesPerAiv * (aivId_ - 1) + col);
            }
        }
    }
}

// ### PATCH START: Token-aligned UB tiles with native pipeline ownership
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
// The host admits at least one complete token in the existing 64-KiB ID page.
// Each VF consumes an entire tile before MTE2 may overwrite it; statistics
// accumulate in the initialized first row, and native MTE3 publishes that row.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::count_routing_tokens(
    LocalTensor<uint32_t> &statistics)
{
    const uint32_t tokensPerTile = IDS_PAGE_ELEMENT_NUM / topk_;
    for (uint32_t tokenStart = batchSizePerAivStart_; tokenStart < batchSizePerAivEnd_;) {
        const uint32_t remainingTokens = batchSizePerAivEnd_ - tokenStart;
        const uint32_t tileTokens = tokensPerTile < remainingTokens ? tokensPerTile : remainingTokens;
        const uint32_t tileBytes = static_cast<uint32_t>(
            static_cast<uint64_t>(tileTokens) * topk_ * sizeof(int32_t));
        DataCopyPad(idsPageTensor_, idsGMTensor_[static_cast<uint64_t>(tokenStart) * topk_],
            {1U, tileBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        SyncFunc<HardEvent::MTE2_V>();
        AscendC::Simt::VF_CALL<count_routing_tokens_vf>(
            AscendC::Simt::Dim3(ROUTING_VF_THREAD_COUNT),
            (__ubuf__ const int32_t *)idsPageTensor_.GetPhyAddr(),
            (__ubuf__ uint32_t *)statistics.GetPhyAddr(), tileTokens, topk_,
            moeRankNum_, expertNumPerMoe_);
        SyncFunc<HardEvent::V_MTE2>();
        tokenStart += tileTokens;
    }
    SyncFunc<HardEvent::V_MTE3>();
    // Tiles begin at the native AIV's token offset, not a global routing page.
    // Force GetIds to reload before the unchanged payload loop reads that page.
    idsPageIndex_ = INVALID_IDS_PAGE_INDEX;
}
#endif
// ### PATCH END: Token-aligned UB tiles with native pipeline ownership

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::SetFlags()
{
    SyncAll<true>();

    uint64_t attnWorkspaceOffset = workspaceSizePerAttn_ * attnRankId_;
    uint64_t expertRecvTokenNumOffset =
        attnWorkspaceOffset + MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    uint64_t expertChunkFlagOffset =
        expertRecvTokenNumOffset + MathCeil(sizeof(int32_t) * expertNumPerMoe_, UB_ALIGN);

    GlobalTensor<int64_t> dstMoeTokenNumGMTensor;
    GlobalTensor<int32_t> dstExpertTokenCountGMTensor;
    GlobalTensor<int32_t> dstExpertTokenCountChunkGMTensor;
    for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
        if (moeRankId % aivNum_ == aivId_) {
            uint32_t dstRankId = attnRankNum_ + moeRankId;
            GM_ADDR dstGM = GetPeerAddrByRankId(dstRankId);
            dstMoeTokenNumGMTensor.SetGlobalBuffer((__gm__ int64_t *)(dstGM + attnWorkspaceOffset));
            dstExpertTokenCountGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertRecvTokenNumOffset));
            dstExpertTokenCountChunkGMTensor.SetGlobalBuffer((__gm__ int32_t *)(dstGM + expertChunkFlagOffset));

            // accumulate first
            uint64_t moePrefixTokenNum = 0;
            for (uint32_t i = 0; i < moeRankId; ++i) {
                moePrefixTokenNum += moeRecvTokenNumTensor_(i);
            }

            batchInfoTensor_(0) = 0;
            batchInfoTensor_(1) = attnRankId_;
            batchInfoTensor_(2) = layerIndex_;
            // tokens this attn sent to prior moes
            batchInfoTensor_(3) = moePrefixTokenNum;
            batchInfoTensor_(BATCH_INFO_VAL_NUM - 1) = 0;  // reserved for Moe
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyPad(dstMoeTokenNumGMTensor, batchInfoTensor_,
                {1U, sizeof(int64_t) * BATCH_INFO_VAL_NUM, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();

            for (uint32_t i = 0; i < expertNumPerMoe_; ++i) {
                // token count sent to an expert, computed during preprocessing
                expertRecvTokenNumTensor_(i) = expertRecvTokenNumTensor_(expertNumPerMoe_ * moeRankId + i);
            }
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyPad(dstExpertTokenCountGMTensor, expertRecvTokenNumTensor_,
                {1U, (uint32_t)(sizeof(int32_t) * expertNumPerMoe_), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();

            // whether each expert's tokens have been read
            Duplicate(expertRecvTokenNumTensor_, (int32_t)0, expertNumPerMoe_);
            SyncFunc<HardEvent::V_MTE3>();
            DataCopyPad(dstExpertTokenCountChunkGMTensor, expertRecvTokenNumTensor_,
                {1U, (uint32_t)(sizeof(int32_t) * expertNumPerMoe_), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();

            // write the flag last to signal data is ready
            batchInfoTensor_(0) = moeRecvTokenNumTensor_(moeRankId) + 1;
            SyncFunc<HardEvent::S_MTE3>();
            DataCopyPad(dstMoeTokenNumGMTensor, batchInfoTensor_,
                {1U, sizeof(int64_t), 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE3_S>();
        }
    }
}

// Patch reason: overlapping tokens cannot share a borrowed V/S -> MTE3 flag
// until the older token's consumer has executed its wait.
// Patch functionality: optional A5 slot-specific ready events preserve the
// original arithmetic and four-byte scale; the serial path remains unchanged.
// Added A5-only private parameters: tokenPipeline supplies exclusive ready IDs;
// tokenSlot selects the protected in-place row. Defaults keep existing callers.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::QuantProcess(
    LocalTensor<XType> &xTensor, LocalTensor<ExpandXOutType> &xOutTensor
    // ### PATCH START: A5 private ready-event parameters
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    , AsyncDispatch::TwoSlotPipeline<DynamicQuant> *tokenPipeline, uint32_t tokenSlot
#endif
    // ### PATCH END: A5 private ready-event parameters
    )
{
    float dynamicScale = 0.0;

    Cast(quantFloatTensor_, xTensor, RoundMode::CAST_NONE, hiddenSize_);
    AscendC::PipeBarrier<PIPE_V>();

    if constexpr (DynamicQuant) {
        Abs(quantFloatAbsTensor_, quantFloatTensor_, hiddenSize_);
        AscendC::PipeBarrier<PIPE_V>();

        ReduceMax(quantRowMaxTensor_, quantFloatAbsTensor_, quantFloatAbsTensor_, hiddenSize_, false);

        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(quantFloatAbsTensor_, 127.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();
        Div(quantFloatAbsTensor_, quantFloatAbsTensor_, quantRowMaxTensor_, 1);
        SyncFunc<HardEvent::V_S>();
        dynamicScale = quantFloatAbsTensor_.GetValue(0);

        SyncFunc<HardEvent::S_V>();

        Muls(quantFloatTensor_, quantFloatTensor_, dynamicScale, hiddenSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    LocalTensor<half> halfLocalTemp = quantFloatTensor_.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = quantFloatTensor_.ReinterpretCast<int32_t>();

    Cast(int32LocalTemp, quantFloatTensor_, RoundMode::CAST_RINT, hiddenSize_);
    AscendC::PipeBarrier<PIPE_V>();

    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, hiddenSize_);
    AscendC::PipeBarrier<PIPE_V>();

    Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, hiddenSize_);
    // ### PATCH START: Slot-exclusive quantized payload readiness
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
    if (tokenPipeline != nullptr) tokenPipeline->mark_quantized(tokenSlot);
    else
#endif
    SyncFunc<HardEvent::V_MTE3>();
    // ### PATCH END: Slot-exclusive quantized payload readiness

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalTemp = xOutTensor.template ReinterpretCast<float>();
        floatLocalTemp.SetValue(hiddenSize_ / sizeof(float), float(1.0) / dynamicScale);
        // ### PATCH START: Slot-exclusive scale readiness
#if defined(AFD_ASYNC_DISPATCH_TOKEN_PIPELINE)
        if (tokenPipeline != nullptr) tokenPipeline->mark_scale_ready(tokenSlot);
        else
#endif
        SyncFunc<HardEvent::S_MTE3>();
        // ### PATCH END: Slot-exclusive scale readiness
    }
}

} // MoeDistributeDispatchImpl
#endif // AFD_ASYNC_DISPATCH_SEND_H
