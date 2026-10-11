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
#include "cam_async_comm_group.h"

namespace MoeDistributeDispatchImpl {
constexpr uint64_t CAM_MAX_RANK_SIZE = 384;  // max NPUs supported by the Cam comm library
constexpr uint32_t UB_ALIGN = 32;            // UB aligned to 32 bytes
constexpr uint32_t BATCH_INFO_VAL_NUM = 5;   // number of batch-related info fields
constexpr uint32_t IDS_PAGE_SIZE = 1024 * 64;  // routing table cache page size
constexpr uint32_t IDS_PAGE_ELEMENT_NUM = IDS_PAGE_SIZE / sizeof(int32_t); // elements per cache page

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
class AfdAsyncDispatchSend {
public:
    __aicore__ inline AfdAsyncDispatchSend() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR workspaceGM,
                                TPipe *pipe, const AfdAsyncDispatchSendTilingData *tilingData,
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
        if (rankId == attnRankId_) {
            return (GM_ADDR)epWinContext_->localWindowsIn;
        }
        return (GM_ADDR)((HcclRankRelationResV2 *)
            epWinContext_->remoteRes[rankId].nextDevicePtr)->windowsIn;
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

    __aicore__ inline void SetFlags();

    __aicore__ inline void DispatchSend();

    __aicore__ inline void QuantProcess(LocalTensor<XType> &xTensor, LocalTensor<ExpandXOutType> &xOutTensor);

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
    __gm__ HcclOpResParam *epWinContext_{nullptr};

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
    epWinContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

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

    GetIds(0, true);

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
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::Process()
{
    if (isError_ == true) {
        return;
    }

    DispatchSend();
}

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
    // Send one address entry per routed assignment and deduplicate payloads per rank.
    for (uint32_t topkId = (topk_ * batchSizePerAivStart_); topkId < (topk_ * batchSizePerAivEnd_); ++topkId) {
        uint32_t tokenId = topkId / topk_;  // corresponding token id
        if (topkId % topk_ == 0) {  // first read of the token
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

            // clear the send table
            for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
                tokenSentFlagTensor_(moeRankId) = 0;  // whether sent to this moe
            }

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

        // send only if the token has not been sent to this moe
        if (tokenSentFlagTensor_(moeRankId) == 0) {
            dstTokenGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType*)(dstGM + tokenStoreOffset));
            if constexpr (DynamicQuant) {
                DataCopy(dstTokenGMTensor[hiddenSizeQuant_ * moeTokenSendIdx], xOutTensor_, hiddenSizeQuant_);
                SyncFunc<HardEvent::MTE3_S>();
            } else {
                DataCopy(dstTokenGMTensor[hiddenSize_ * moeTokenSendIdx], xOutTensor_, hiddenSize_);
                SyncFunc<HardEvent::MTE3_S>();
            }
            tokenSentFlagTensor_(moeRankId) = 1;
        }


        PushExpertTokenOffset(expertId, moeTokenSendIdx, expertSlotOffset, tokenAddrStoreOffset);

        expertSendStartIdxTensor_(expertId) = expertTokenSendIdx + 1;

        if ((topkId + 1) % topk_ == 0) {
            // refresh offsets from the send table
            for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
                if (tokenSentFlagTensor_(moeRankId) == 1) {
                    ++moeSendStartOffsetTensor_(moeRankId);
                }
            }
        }
    }
    for (uint32_t expertId = 0; expertId < expertNum_; ++expertId) {
        FlushExpertTokenOffset(expertId, tokenAddrStoreOffset);
    }

    SetFlags();
}

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
    if (maxPrefixBufferSize <= totalUbSize_ - bufOffset) {
        uint32_t columnsPerAiv = statEntriesPerAiv / aivNum_;
        uint32_t extraColumns = statEntriesPerAiv % aivNum_;
        uint32_t columnStart = aivId_ * columnsPerAiv
            + (aivId_ < extraColumns ? aivId_ : extraColumns);
        uint32_t columnCount = columnsPerAiv + (aivId_ < extraColumns ? 1U : 0U);
        if (columnCount > 0 && aivNum_ > 1) {
            uint32_t prefixRowStride = MathCeil(columnCount * sizeof(uint32_t), UB_ALIGN) / sizeof(uint32_t);
            LocalTensor<uint32_t> prefixTensor = ubBuffer_.GetWithOffset<uint32_t>(
                (aivNum_ - 1) * prefixRowStride, bufOffset);
            for (uint32_t col = 0; col < columnCount; ++col) {
                uint32_t sum = aivStatTensor(columnStart + col);
                for (uint32_t row = 1; row < aivNum_; ++row) {
                    sum += aivStatTensor(statEntriesPerAiv * row + columnStart + col);
                    prefixTensor((row - 1) * prefixRowStride + col) = sum;
                }
            }

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

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncDispatchSend<TemplateMC2TypeFunc>::QuantProcess(
    LocalTensor<XType> &xTensor, LocalTensor<ExpandXOutType> &xOutTensor)
{
    float dynamicScale = 0.0;

    Cast(quantFloatTensor_, xTensor, RoundMode::CAST_NONE, hiddenSize_);
    pipe_barrier(PIPE_V);

    if constexpr (DynamicQuant) {
        Abs(quantFloatAbsTensor_, quantFloatTensor_, hiddenSize_);
        pipe_barrier(PIPE_V);

        ReduceMax(quantRowMaxTensor_, quantFloatAbsTensor_, quantFloatAbsTensor_, hiddenSize_, false);

        pipe_barrier(PIPE_V);
        Duplicate(quantFloatAbsTensor_, 127.0f, 1);
        pipe_barrier(PIPE_V);
        Div(quantFloatAbsTensor_, quantFloatAbsTensor_, quantRowMaxTensor_, 1);
        SyncFunc<HardEvent::V_S>();
        dynamicScale = quantFloatAbsTensor_.GetValue(0);

        SyncFunc<HardEvent::S_V>();

        Muls(quantFloatTensor_, quantFloatTensor_, dynamicScale, hiddenSize_);
        pipe_barrier(PIPE_V);
    }

    LocalTensor<half> halfLocalTemp = quantFloatTensor_.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = quantFloatTensor_.ReinterpretCast<int32_t>();

    Cast(int32LocalTemp, quantFloatTensor_, RoundMode::CAST_RINT, hiddenSize_);
    pipe_barrier(PIPE_V);

    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, hiddenSize_);
    pipe_barrier(PIPE_V);

    Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, hiddenSize_);
    SyncFunc<HardEvent::V_MTE3>();

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalTemp = xOutTensor.template ReinterpretCast<float>();
        floatLocalTemp.SetValue(hiddenSize_ / sizeof(float), float(1.0) / dynamicScale);
        SyncFunc<HardEvent::S_MTE3>();
    }
}

} // MoeDistributeDispatchImpl
#endif // AFD_ASYNC_DISPATCH_SEND_H
