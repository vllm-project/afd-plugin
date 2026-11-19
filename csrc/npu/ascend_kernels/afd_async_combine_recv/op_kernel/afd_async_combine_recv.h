/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineRecv operator kernel function header file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineRecv operator kernel function header file
 */
#ifndef AFD_ASYNC_COMBINE_RECV_H
#define AFD_ASYNC_COMBINE_RECV_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "afd_async_combine_recv_tiling.h"
#include "cam_async_comm_args.h"
#include "cam_async_comm_group.h"

#define DATA_FLUSH(_gm_tensor, _type)                                                                      \
    do {                                                                                                   \
        Barrier();                                                                                         \
        DataCacheCleanAndInvalid<_type, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(_gm_tensor); \
        __asm__("NOP");                                                                                    \
        dsb(DSB_ALL);                                                                                      \
    } while (0)

namespace MoeDistributeCombineRecvImpl {
constexpr uint64_t CAM_MAX_RANK_SIZE = 384;  // max NPUs supported by the Cam comm library
constexpr uint32_t UB_ALIGN = 32;            // UB aligned to 32 bytes
constexpr uint32_t MAX_AIV_NUM = 48;         // max AIV core count
constexpr uint32_t IDS_PAGE_SIZE = 1024 * 64;  // routing table cache page size
constexpr uint32_t IDS_PAGE_ELEMENT_NUM = IDS_PAGE_SIZE / sizeof(int32_t); // elements per cache page

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename XOutType, bool UseDispatchPrefix
#define TemplateMC2TypeFunc XOutType, UseDispatchPrefix
using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeClass>
class AfdAsyncCombineRecv {
public:
    __aicore__ inline AfdAsyncCombineRecv(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR xOut,
        GM_ADDR workspaceGM, TPipe *pipe,
        const AfdAsyncCombineRecvTilingData *tilingData, GM_ADDR commArgs, int32_t isCamComm);
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

    __aicore__ inline void LoadPreProcess(uint32_t bufOffset);

    __aicore__ inline void PreProcess();

    __aicore__ inline void CombineRecv();

    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t attnRankId_{0};
    uint32_t worldSize_{0};
    uint32_t attnRankNum_{0};
    uint32_t moeRankNum_{0};
    uint32_t routeExpertNumPerMoe_{0};
    uint32_t expertNumPerMoe_{0};
    uint32_t expertNum_{0};
    uint32_t batchSize_{0};
    uint32_t hiddenSize_{0};
    uint32_t topk_{0};
    uint64_t totalUbSize_{0};
    uint64_t totalWorkspaceSize_{0};
    uint64_t dispatchOffset_{0};
    int32_t idsPageIndex_{0};

    GlobalTensor<XOutType> xOutGMTensor_;
    GlobalTensor<int32_t> idsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    __gm__ HcclOpResParam *epWinContext_{nullptr};

    uint64_t ubReuseSize_{0};

    TPipe *tpipe_{nullptr};
    TBuf<> ubBuffer_;
    LocalTensor<int32_t> idsPageTensor_;  // routing table
    // Token counts indexed by global routed expert ID.
    LocalTensor<uint16_t> expertRecvTokenNumTensor_;
    // Exclusive token prefix within each expert's MoE rank.
    LocalTensor<uint32_t> expertPrefixOffsetTensor_;
    // Occurrence index of the next token received from each routed expert.
    LocalTensor<uint16_t> expertRecvStartIdxTensor_;
    LocalTensor<XOutType> xOutTensor_;                 // staging token

    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;
    LocalTensor<uint32_t> moePrefixTokenNumTensor_;

    uint32_t batchSizePerAivStart_{0};
    uint32_t batchSizePerAivEnd_{0};

    bool isError_{false};
};

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::Init(GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expertScales, GM_ADDR xOut, GM_ADDR workspaceGM,
    TPipe *pipe, const AfdAsyncCombineRecvTilingData *tilingData, GM_ADDR commArgs, int32_t isCamComm)
{
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    attnRankId_ = tilingData->moeDistributeCombineInfo.attnRankId;
    worldSize_ = tilingData->moeDistributeCombineInfo.worldSize;
    attnRankNum_ = tilingData->moeDistributeCombineInfo.attnRankNum;
    moeRankNum_ = tilingData->moeDistributeCombineInfo.moeRankNum;
    routeExpertNumPerMoe_ = tilingData->moeDistributeCombineInfo.routeExpertNumPerMoe;
    // Compact protocol: local expert IDs are [0, R), with no extra slots.
    expertNumPerMoe_ = routeExpertNumPerMoe_;
    expertNum_ = expertNumPerMoe_ * moeRankNum_;
    batchSize_ = tilingData->moeDistributeCombineInfo.batchSize;
    hiddenSize_ = tilingData->moeDistributeCombineInfo.hiddenSize;
    topk_ = tilingData->moeDistributeCombineInfo.topk;
    totalUbSize_ = tilingData->moeDistributeCombineInfo.totalUbSize;
    totalWorkspaceSize_ = tilingData->moeDistributeCombineInfo.totalWorkspaceSize;
    dispatchOffset_ = MathCeil(sizeof(uint32_t) * (moeRankNum_ + expertNum_) * MAX_AIV_NUM, UB_ALIGN);

    xOutGMTensor_.SetGlobalBuffer((__gm__ XOutType *)xOut);
    idsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
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

    bufSize = MathCeil(sizeof(uint16_t) * expertNum_, UB_ALIGN); // 96 B or 544 B
    expertRecvTokenNumTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
    bufOffset += bufSize;

    LoadPreProcess(bufOffset);

    bufSize = MathCeil(sizeof(uint32_t) * expertNum_, UB_ALIGN); // 160 B or 1088 B
    expertPrefixOffsetTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint16_t) * expertNum_, UB_ALIGN); // 96 B or 544 B
    expertRecvStartIdxTensor_ = ubBuffer_.GetWithOffset<uint16_t>(bufSize / sizeof(uint16_t), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(float) * hiddenSize_, UB_ALIGN);
    rowTmpFloatLocal_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset); // 28 KB
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(float) * hiddenSize_, UB_ALIGN); // 28 KB
    sumFloatBufLocal_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(float) * hiddenSize_, UB_ALIGN); // 28 KB
    mulBufLocal_ = ubBuffer_.GetWithOffset<float>(bufSize / sizeof(float), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(XOutType) * hiddenSize_, UB_ALIGN); // 14 KB
    xOutTensor_ = ubBuffer_.GetWithOffset<XOutType>(bufSize / sizeof(XOutType), bufOffset);
    bufOffset += bufSize;

    bufSize = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN); // 64 B
    moePrefixTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;

    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // remaining reusable UB space
    ubReuseSize_ = totalUbSize_ - bufOffset;
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::LoadPreProcess(uint32_t bufOffset)
{
    Duplicate(expertRecvTokenNumTensor_, (uint16_t)0, expertNum_);
    SyncFunc<HardEvent::V_S>();

    uint32_t statEntriesPerAiv = moeRankNum_ + expertNum_;
    uint32_t bufSize = MathCeil(sizeof(uint32_t) * statEntriesPerAiv * aivNum_, UB_ALIGN); // 30720 30KB
    LocalTensor<uint32_t> aivStatTensor = ubBuffer_.GetWithOffset<uint32_t>(bufSize / sizeof(uint32_t), bufOffset);
    bufOffset += bufSize;
    if (bufOffset > totalUbSize_) {
        isError_ = true;
        return;
    }

    // gather the quantity every aiv is responsible for sending in the DispatchSend stage
    GM_ADDR srcGM = GetPeerAddrByRankId(attnRankId_);
    GlobalTensor<uint32_t> aivStatGMTensor;
    aivStatGMTensor.SetGlobalBuffer((__gm__ uint32_t *)srcGM);
    DataCopyPad(aivStatTensor, aivStatGMTensor,
        {1U, (uint32_t)(sizeof(uint32_t) * statEntriesPerAiv * aivNum_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SyncFunc<HardEvent::MTE2_S>();
    SyncFunc<HardEvent::S_MTE2>();

    // The final AIV row contains cumulative counts for every routed expert.
    uint32_t lastRowIdx = aivNum_ - 1;
    for (uint32_t expertId = 0; expertId < expertNum_; ++expertId) {
        uint32_t col = moeRankNum_ + expertId;
        expertRecvTokenNumTensor_(expertId) = aivStatTensor(statEntriesPerAiv * lastRowIdx + col);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::Process()
{
    if (isError_ == true) {
        return;
    }

    CombineRecv();
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::CombineRecv()
{
    PreProcess();

    // iterate the routing table
    DATA_FLUSH(scalesGMTensor_, float);
    GlobalTensor<XOutType> dstTokenStoreGM;
    uint32_t attnSelfRankId = attnRankId_;
    GM_ADDR dstRankWorkspaceGm = GetPeerAddrByRankId(attnSelfRankId) + dispatchOffset_;
    uint32_t topkStart = 0;
    if constexpr (UseDispatchPrefix) {
        topkStart = topk_ * batchSizePerAivStart_;
    }
    for (uint32_t topkId = topkStart; topkId < (topk_ * batchSizePerAivEnd_); ++topkId) {
        uint32_t routeExpertId = GetIds(topkId);
        uint32_t moeRankId = routeExpertId / routeExpertNumPerMoe_;
        uint32_t expertId = routeExpertId;
        uint16_t expertTokenSendIdx = expertRecvStartIdxTensor_(expertId);

        uint32_t tokenId = topkId / topk_;
        if (tokenId >= batchSizePerAivEnd_) {
            break;
        }

        if ((tokenId >= batchSizePerAivStart_) && (tokenId < batchSizePerAivEnd_)) {
            uint32_t topkIndex = topkId % topk_;
            if (topkIndex == 0) {
                Duplicate(sumFloatBufLocal_, (float)0, hiddenSize_);
                SyncFunc<HardEvent::V_S>();
            }

            uint32_t expertPrefixOffset = expertPrefixOffsetTensor_(expertId);
            float scaleVal = scalesGMTensor_(topkId);
            uint32_t moePrefixTokenNum = moePrefixTokenNumTensor_(moeRankId) - 1;
            uint64_t tokenStoreOffset = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN) +
                (sizeof(XOutType) * hiddenSize_ * moePrefixTokenNum);

            dstTokenStoreGM.SetGlobalBuffer((__gm__ XOutType *)(dstRankWorkspaceGm + tokenStoreOffset));
            DataCopy(xOutTensor_,
                dstTokenStoreGM[hiddenSize_ * (expertPrefixOffset + expertTokenSendIdx)], hiddenSize_);
            SyncFunc<HardEvent::MTE2_V>();


            Cast(rowTmpFloatLocal_, xOutTensor_, RoundMode::CAST_NONE, hiddenSize_);
            pipe_barrier(PIPE_V);
            Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, hiddenSize_);
            pipe_barrier(PIPE_V);
            Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, hiddenSize_);
            SyncFunc<HardEvent::V_S>();


            if (topkIndex == (topk_ - 1)) {
                // All routed contributions for this token have been accumulated.
                pipe_barrier(PIPE_V);

                // output
                Cast(xOutTensor_, sumFloatBufLocal_, RoundMode::CAST_RINT, hiddenSize_);
                SyncFunc<HardEvent::V_MTE3>();
                DataCopy(xOutGMTensor_[tokenId * hiddenSize_], xOutTensor_, hiddenSize_);
                SyncFunc<HardEvent::MTE3_V>();
            }
        }
        expertRecvStartIdxTensor_(expertId) = expertTokenSendIdx + 1;
    }

    SyncAll<true>();

    if (aivId_ == 0) {
        GlobalTensor<uint32_t> tokenStatusGM;
        tokenStatusGM.SetGlobalBuffer((__gm__ uint32_t *)dstRankWorkspaceGm);
        for (uint32_t i = 0; i < moeRankNum_; ++i) {
            moePrefixTokenNumTensor_(i) = 0;
        }
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyPad(tokenStatusGM, moePrefixTokenNumTensor_,
            {1U, (uint32_t)(sizeof(uint32_t) * moeRankNum_), 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE3_S>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::PreProcess()
{
    // Reset each routed expert's occurrence index.
    Duplicate(expertRecvStartIdxTensor_, (uint16_t)0, expertNum_);
    SyncFunc<HardEvent::V_S>();
    if constexpr (UseDispatchPrefix) {
        if (aivId_ > 0) {
            // DispatchSend uses the same token partition and stores inclusive per-AIV counts.
            // Seed this core's occurrence indices with the preceding core's expert counts.
            uint32_t statEntriesPerAiv = moeRankNum_ + expertNum_;
            GlobalTensor<uint32_t> aivStatGMTensor;
            aivStatGMTensor.SetGlobalBuffer((__gm__ uint32_t *)GetPeerAddrByRankId(attnRankId_));
            // This buffer is still unused; its final expert prefixes are built below.
            DataCopyPad(expertPrefixOffsetTensor_,
                aivStatGMTensor[statEntriesPerAiv * (aivId_ - 1) + moeRankNum_],
                {1U, static_cast<uint32_t>(sizeof(uint32_t) * expertNum_), 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
            SyncFunc<HardEvent::MTE2_S>();
            for (uint32_t expertId = 0; expertId < expertNum_; ++expertId) {
                expertRecvStartIdxTensor_(expertId) = expertPrefixOffsetTensor_(expertId);
            }
        }
    }

    // Prefixes restart at each MoE rank; local expert 0 contributes real tokens.
    for (uint32_t moeRankId = 0; moeRankId < moeRankNum_; ++moeRankId) {
        uint32_t tokenPrefix = 0;
        for (uint32_t localExpertId = 0; localExpertId < expertNumPerMoe_; ++localExpertId) {
            uint32_t expertId = moeRankId * expertNumPerMoe_ + localExpertId;
            expertPrefixOffsetTensor_(expertId) = tokenPrefix;
            tokenPrefix += expertRecvTokenNumTensor_(expertId);
        }
    }


    uint32_t attnSelfRankId = attnRankId_;
    GM_ADDR dstRankWorkspaceGm = GetPeerAddrByRankId(attnSelfRankId) + dispatchOffset_;
    GlobalTensor<uint32_t> dataInfoGMTensor;
    dataInfoGMTensor.SetGlobalBuffer((__gm__ uint32_t *)dstRankWorkspaceGm);
    // wait for all moe cards to finish CombineSend
    while (true) {
        DataCopyPad(moePrefixTokenNumTensor_, dataInfoGMTensor,
            {1U, (uint32_t)(sizeof(uint32_t) * moeRankNum_), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFunc<HardEvent::MTE2_S>();
        SyncFunc<HardEvent::S_MTE2>();

        bool isAllRecv = true;
        for (uint32_t i = 0; i < moeRankNum_; ++i) {
            if (moePrefixTokenNumTensor_(i) == 0) {
                isAllRecv = false;
                break;
            }
        }
        if (isAllRecv == true) {
            break;
        }
    }
}
}  // namespace MoeDistributeCombineRecvImpl
#endif // AFD_ASYNC_COMBINE_RECV_H
