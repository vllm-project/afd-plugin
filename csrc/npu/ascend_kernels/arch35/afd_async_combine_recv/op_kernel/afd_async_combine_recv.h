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
#include "cam_async_window.h"

// ### PATCH START: A5-only vector input prefetch
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
#include "cam_async_combine_prefetch.h"
#define CAM_ASYNC_COMBINE_TILE_INLINE __aicore__ inline
#include "cam_async_combine_tile_plan.h"
#undef CAM_ASYNC_COMBINE_TILE_INLINE
#endif
// ### PATCH END: A5-only vector input prefetch

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

    __aicore__ inline void LoadPreProcess(uint32_t bufOffset);

    __aicore__ inline void PreProcess();

    __aicore__ inline void CombineRecv();

    // ### PATCH START: CR private hidden-tile prefetch
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    __aicore__ inline void CombineRecvPipelined();
    __aicore__ inline void combine_recv_tiled();
    __aicore__ inline void load_tiled_input(LocalTensor<XOutType> &input,
        const AsyncCombineTiles::HiddenKTile &tile, float &weight);
    class TokenSourceBackend {
    public:
        __aicore__ inline explicit TokenSourceBackend(AfdAsyncCombineRecv *owner) : owner_(owner) {}
        __aicore__ inline uint32_t route_expert(uint32_t contribution)
        {
            return owner_->GetIds(contribution);
        }
        __aicore__ inline uint32_t take_occurrence(uint32_t expert)
        {
            const uint16_t occurrence = owner_->expertRecvStartIdxTensor_(expert);
            owner_->expertRecvStartIdxTensor_(expert) = occurrence + 1;
            return occurrence;
        }
        __aicore__ inline uint32_t rank_prefix(uint32_t expert)
        {
            return owner_->moePrefixTokenNumTensor_(expert / owner_->routeExpertNumPerMoe_) - 1;
        }
        __aicore__ inline uint32_t expert_prefix(uint32_t expert)
        {
            return owner_->expertPrefixOffsetTensor_(expert);
        }
        __aicore__ inline float weight(uint32_t contribution)
        {
            return owner_->scalesGMTensor_(contribution);
        }
        __aicore__ inline void store_descriptor(uint32_t logical, uint64_t address, float scale)
        {
            owner_->token_source_addresses_.SetValue(logical, address);
            owner_->token_source_weights_.SetValue(logical, scale);
        }
    private:
        AfdAsyncCombineRecv *owner_;
    };

    __aicore__ inline void LoadCombineTile(LocalTensor<XOutType> &input,
        const AsyncCombine::CombineTile &tile, GM_ADDR sourceWindow,
        float &cachedScale,
        GlobalTensor<XOutType> &cachedSource, float &scale);
#endif
    // ### PATCH END: CR private hidden-tile prefetch

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
    Cam::AsyncDataWindow dataWindow_;

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
    // ### PATCH START: Optional A5 second input tile
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    LocalTensor<XOutType> xPrefetchTensor_;
    AsyncCombine::CombinePrefetchPlan combinePrefetchPlan_{};
    // ### PATCH START: A5 token descriptor and tile-sized FP32 plan
    AsyncCombineTiles::TileScratchPlan tile_scratch_plan_{};
    LocalTensor<uint64_t> token_source_addresses_;
    LocalTensor<float> token_source_weights_;
    // ### PATCH END: A5 token descriptor and tile-sized FP32 plan
#endif
    // ### PATCH END: Optional A5 second input tile

    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;
    LocalTensor<uint32_t> moePrefixTokenNumTensor_;

    uint32_t batchSizePerAivStart_{0};
    uint32_t batchSizePerAivEnd_{0};

    bool isError_{false};
};

// Patch reason: the original input row is serialized with vector arithmetic.
// Patch functionality: A5 may use checked token descriptors and tile-sized FP32
// scratch; accepted shapes and the original minimum UB/fallback stay unchanged.
// Signature: unchanged; no added parameters.
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

    // ### PATCH START: Checked A5 tile scratch before the full-H fallback layout
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    if (bufOffset <= totalUbSize_) {
        const uint64_t prefix_bytes = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN);
        tile_scratch_plan_ = AsyncCombineTiles::plan_tile_scratch(totalUbSize_ - bufOffset,
            hiddenSize_, sizeof(XOutType), topk_, batchSizePerAivEnd_ - batchSizePerAivStart_, prefix_bytes);
        if (tile_scratch_plan_.enabled) {
            const uint64_t base = bufOffset;
            token_source_addresses_ = ubBuffer_.GetWithOffset<uint64_t>(
                tile_scratch_plan_.address_bytes / sizeof(uint64_t), base + tile_scratch_plan_.address_offset);
            token_source_weights_ = ubBuffer_.GetWithOffset<float>(
                tile_scratch_plan_.scale_bytes / sizeof(float), base + tile_scratch_plan_.scale_offset);
            moePrefixTokenNumTensor_ = ubBuffer_.GetWithOffset<uint32_t>(
                prefix_bytes / sizeof(uint32_t), base + tile_scratch_plan_.prefix_offset);
            rowTmpFloatLocal_ = ubBuffer_.GetWithOffset<float>(
                tile_scratch_plan_.float_row_bytes / sizeof(float), base + tile_scratch_plan_.row_offset);
            mulBufLocal_ = ubBuffer_.GetWithOffset<float>(
                tile_scratch_plan_.float_row_bytes / sizeof(float), base + tile_scratch_plan_.product_offset);
            sumFloatBufLocal_ = ubBuffer_.GetWithOffset<float>(
                tile_scratch_plan_.float_row_bytes / sizeof(float), base + tile_scratch_plan_.sum_offset);
            xOutTensor_ = ubBuffer_.GetWithOffset<XOutType>(
                tile_scratch_plan_.input_row_bytes / sizeof(XOutType), base + tile_scratch_plan_.input_offset[0]);
            xPrefetchTensor_ = ubBuffer_.GetWithOffset<XOutType>(
                tile_scratch_plan_.input_row_bytes / sizeof(XOutType), base + tile_scratch_plan_.input_offset[1]);
            ubReuseSize_ = totalUbSize_ - base - tile_scratch_plan_.total_bytes;
            return;
        }
    }
#endif
    // ### PATCH END: Checked A5 tile scratch before the full-H fallback layout

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
    // ### PATCH START: Preserve baseline UB and select a second hidden tile
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    const uint64_t ownedContributions = static_cast<uint64_t>(batchSizePerAivEnd_ - batchSizePerAivStart_) * topk_;
    combinePrefetchPlan_ = AsyncCombine::PlanCombinePrefetch(ubReuseSize_, hiddenSize_,
        sizeof(XOutType), ownedContributions);
    if (combinePrefetchPlan_.enabled) {
        xPrefetchTensor_ = ubBuffer_.GetWithOffset<XOutType>(
            combinePrefetchPlan_.secondSlotBytes / sizeof(XOutType), bufOffset);
        ubReuseSize_ -= combinePrefetchPlan_.secondSlotBytes;
    }
#endif
    // ### PATCH END: Preserve baseline UB and select a second hidden tile
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

// Patch reason: the native loop waits for each row before starting vector work.
// Patch functionality: A5 prefetches the next hidden tile while preserving each
// element's Cast -> Muls -> Add top-k order, output rounding, and flag reset.
// Signature: unchanged; no added parameters.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::CombineRecv()
{
    PreProcess();

    // ### PATCH START: A5 ordered hidden-tile path with native fallback
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    // ### PATCH START: Prefer the larger hidden-first A5 tile when its plan fits
    if (tile_scratch_plan_.enabled) {
        DATA_FLUSH(scalesGMTensor_, float);
        combine_recv_tiled();
    } else if (combinePrefetchPlan_.enabled) {
        DATA_FLUSH(scalesGMTensor_, float);
        CombineRecvPipelined();
    } else
    // ### PATCH END: Prefer the larger hidden-first A5 tile when its plan fits
#endif
    {
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
            AscendC::PipeBarrier<PIPE_V>();
            Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, hiddenSize_);
            AscendC::PipeBarrier<PIPE_V>();
            Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, hiddenSize_);
            SyncFunc<HardEvent::V_S>();


            if (topkIndex == (topk_ - 1)) {
                // All routed contributions for this token have been accumulated.
                AscendC::PipeBarrier<PIPE_V>();

                // output
                Cast(xOutTensor_, sumFloatBufLocal_, RoundMode::CAST_RINT, hiddenSize_);
                SyncFunc<HardEvent::V_MTE3>();
                DataCopy(xOutGMTensor_[tokenId * hiddenSize_], xOutTensor_, hiddenSize_);
                SyncFunc<HardEvent::MTE3_V>();
            }
        }
        expertRecvStartIdxTensor_(expertId) = expertTokenSendIdx + 1;
    }

    }
    // ### PATCH END: A5 ordered hidden-tile path with native fallback

    SyncAll<true>();

    if (aivId_ == 0) {
        // ### PATCH START: Shared completion base for both CR paths
        GM_ADDR dstRankWorkspaceGm = GetPeerAddrByRankId(attnRankId_) + dispatchOffset_;
        // ### PATCH END: Shared completion base for both CR paths
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


// ### PATCH START: Ordered A5 CR vector-input pipeline
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
// AFD private loader: advance the stable expert occurrence once per contribution
// and keep that row's physical address/scale while loading its hidden tiles.
// sourceWindow is the existing compact-v2 return allocation, already observed
// complete by PreProcess. This local event path adds no remote visibility claim.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::LoadCombineTile(
    LocalTensor<XOutType> &input, const AsyncCombine::CombineTile &tile, GM_ADDR sourceWindow,
    float &cachedScale,
    GlobalTensor<XOutType> &cachedSource, float &scale)
{
    if (tile.hiddenOffset == 0) {
        const uint32_t routeExpertId = GetIds(tile.contributionIndex);
        const uint32_t moeRankId = routeExpertId / routeExpertNumPerMoe_;
        const uint16_t occurrence = expertRecvStartIdxTensor_(routeExpertId);
        const uint32_t expertPrefixOffset = expertPrefixOffsetTensor_(routeExpertId);
        const uint32_t moePrefixTokenNum = moePrefixTokenNumTensor_(moeRankId) - 1;
        const uint64_t tokenStoreOffset = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN)
            + sizeof(XOutType) * static_cast<uint64_t>(hiddenSize_) * moePrefixTokenNum;
        const uint64_t rowOffset = sizeof(XOutType) * static_cast<uint64_t>(hiddenSize_)
            * (static_cast<uint64_t>(expertPrefixOffset) + occurrence);
        cachedSource.SetGlobalBuffer((__gm__ XOutType *)(sourceWindow + tokenStoreOffset + rowOffset));
        cachedScale = scalesGMTensor_(tile.contributionIndex);
        expertRecvStartIdxTensor_(routeExpertId) = occurrence + 1;
    }
    // The cursor emits the first tile before later tiles, retaining its row lease.
    scale = cachedScale;
    DataCopyPad(input, cachedSource[tile.hiddenOffset],
        {1U, static_cast<uint32_t>(sizeof(XOutType) * tile.hiddenElements), 0U, 0U, 0U},
        {false, 0U, 0U, 0});
}

// Patch reason: per-tile Duplicate/Add produce no values read by Scalar.
// Patch functionality: keep V work asynchronous, capture each scale by value,
// and retain PIPE_V, input credits, output MTE3_V, and the final Scalar drain.
// Signature: private helper unchanged; no added parameters.
// AFD private pipeline: SIMD remains over independent hidden elements. It never
// reassociates top-k reduction or fuses the separately rounded Muls and Add.
// Both input slots and all event IDs are drained before the existing SyncAll and
// compact-v2 completion flags are reset in CombineRecv.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::CombineRecvPipelined()
{
    const uint32_t contributionBegin = batchSizePerAivStart_ * topk_;
    const uint32_t contributionEnd = batchSizePerAivEnd_ * topk_;
    if constexpr (!UseDispatchPrefix) {
        // The non-prefix key must count preceding cores' route occurrences even
        // though it prefetches only this core's owned token slice.
        for (uint32_t contribution = 0; contribution < contributionBegin; ++contribution) {
            const uint32_t expert = GetIds(contribution);
            expertRecvStartIdxTensor_(expert) = expertRecvStartIdxTensor_(expert) + 1;
        }
    }
    AsyncCombine::CombineTileCursor cursor;
    cursor.Init(contributionBegin, contributionEnd, hiddenSize_, combinePrefetchPlan_.tileElements);
    if (!cursor.HasNext()) return;
    GM_ADDR sourceWindow = GetPeerAddrByRankId(attnRankId_) + dispatchOffset_;
    LocalTensor<XOutType> inputs[AsyncCombine::PREFETCH_SLOT_COUNT] = {xOutTensor_, xPrefetchTensor_};
    AsyncCombine::CombineTile tiles[AsyncCombine::PREFETCH_SLOT_COUNT];
    float scales[AsyncCombine::PREFETCH_SLOT_COUNT]{};
    GlobalTensor<XOutType> cachedSource;
    float cachedScale = 0.0F;
    AsyncCombine::TwoSlotPrefetch pipeline;
    pipeline.Init(tpipe_);
    uint32_t slot = 0;
    tiles[slot] = cursor.Current();
    LoadCombineTile(inputs[slot], tiles[slot], sourceWindow,
        cachedScale, cachedSource, scales[slot]);
    pipeline.MarkLoaded(slot);
    cursor.Advance();
    while (true) {
        pipeline.WaitLoaded(slot);
        const auto current = tiles[slot];
        // ### PATCH START: Capture this tile's scalar operand at issue time
        const float currentScale = scales[slot];
        // ### PATCH END: Capture this tile's scalar operand at issue time
        const bool nextReady = cursor.HasNext();
        const uint32_t nextSlot = (slot + 1) % AsyncCombine::PREFETCH_SLOT_COUNT;
        if (nextReady) {
            pipeline.WaitForReuse(nextSlot);
            tiles[nextSlot] = cursor.Current();
            LoadCombineTile(inputs[nextSlot], tiles[nextSlot], sourceWindow,
                cachedScale, cachedSource, scales[nextSlot]);
            pipeline.MarkLoaded(nextSlot);
            cursor.Advance();
        }
        const uint32_t topkIndex = current.contributionIndex % topk_;
        const uint32_t tokenId = current.contributionIndex / topk_;
        auto sum = sumFloatBufLocal_[current.hiddenOffset];
        auto row = rowTmpFloatLocal_[current.hiddenOffset];
        auto product = mulBufLocal_[current.hiddenOffset];
        if (topkIndex == 0) {
            Duplicate(sum, 0.0F, current.hiddenElements);
            // ### PATCH START: Keep FP32 scratch owned by the vector pipeline
            // No Scalar reads sum. The PIPE_V after Cast closes this Duplicate
            // and preceding vector work before product/sum scratch is reused.
            // ### PATCH END: Keep FP32 scratch owned by the vector pipeline
        }
        Cast(row, inputs[slot], RoundMode::CAST_NONE, current.hiddenElements);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(product, row, currentScale, current.hiddenElements);
        AscendC::PipeBarrier<PIPE_V>();
        Add(sum, sum, product, current.hiddenElements);
        // ### PATCH START: Return slot ownership without a per-tile Scalar wait
        // Scratch is read only by V. Existing PIPE_V fences and the V -> MTE2
        // credit keep its consumers ordered; final output retains MTE3_V.
        // ### PATCH END: Return slot ownership without a per-tile Scalar wait
        if (topkIndex == topk_ - 1) {
            AscendC::PipeBarrier<PIPE_V>();
            Cast(inputs[slot], sum, RoundMode::CAST_RINT, current.hiddenElements);
            SyncFunc<HardEvent::V_MTE3>();
            DataCopyPad(xOutGMTensor_[static_cast<uint64_t>(tokenId) * hiddenSize_ + current.hiddenOffset],
                inputs[slot], {1U, static_cast<uint32_t>(sizeof(XOutType) * current.hiddenElements), 0U, 0U, 0U});
            // Output reuses this input slot. MTE3 must finish all source reads
            // before its V -> MTE2 credit can permit another input transfer.
            SyncFunc<HardEvent::MTE3_V>();
        }
        pipeline.MarkSourceConsumed(slot);
        if (!nextReady) break;
        slot = nextSlot;
    }
    pipeline.Drain();
}

// ### PATCH START: Hidden-first CR with frozen per-token source descriptors
// Patch reason: hidden-first traversal must not advance expert occurrences for
// every hidden tile. The source row/weight were captured once in Scalar-owned UB.
// Patch functionality: evaluate each descriptor by value when issuing MTE2; no
// deferred packet retains a mutable address/weight reference. Private new API.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::load_tiled_input(
    LocalTensor<XOutType> &input, const AsyncCombineTiles::HiddenKTile &tile, float &weight)
{
    const uint64_t address = token_source_addresses_.GetValue(tile.logical_index);
    const float current_weight = token_source_weights_.GetValue(tile.logical_index);
    GlobalTensor<XOutType> source;
    source.SetGlobalBuffer((__gm__ XOutType *)address);
    DataCopyPad(input, source[tile.hidden_offset],
        {1U, static_cast<uint32_t>(sizeof(XOutType) * tile.hidden_elements), 0U, 0U, 0U},
        {false, 0U, 0U, 0});
    weight = current_weight;
}

// Patch reason: contribution-major traversal retains three full-H FP32 rows.
// Patch functionality: freeze K addresses/weights per token, then accumulate each
// hidden tile in logical K order using one shared tile-sized scratch triplet.
// Muls and Add remain separately rounded; output remains CAST_RINT. Private API.
template <TemplateMC2TypeClass>
__aicore__ inline void AfdAsyncCombineRecv<TemplateMC2TypeFunc>::combine_recv_tiled()
{
    if constexpr (!UseDispatchPrefix) {
        const uint32_t contribution_begin = batchSizePerAivStart_ * topk_;
        for (uint32_t contribution = 0; contribution < contribution_begin; ++contribution) {
            const uint32_t expert = GetIds(contribution);
            expertRecvStartIdxTensor_(expert) = expertRecvStartIdxTensor_(expert) + 1;
        }
    }
    const uint64_t source_window = reinterpret_cast<uint64_t>(GetPeerAddrByRankId(attnRankId_) + dispatchOffset_);
    const uint64_t wire_header_bytes = MathCeil(sizeof(uint32_t) * moeRankNum_, UB_ALIGN);
    LocalTensor<XOutType> inputs[AsyncCombineTiles::TILE_INPUT_SLOTS] = {xOutTensor_, xPrefetchTensor_};
    AsyncCombine::CannPrefetchBackend backend;
    backend.Init(tpipe_);
    AsyncCombine::TwoSlotPrefetch pipeline(backend);
    pipeline.Init(tpipe_);
    TokenSourceBackend routes(this);
    for (uint32_t token = batchSizePerAivStart_; token < batchSizePerAivEnd_; ++token) {
        AsyncCombineTiles::capture_token_sources(token, topk_, hiddenSize_, sizeof(XOutType),
            source_window, wire_header_bytes, routes);
        AsyncCombineTiles::HiddenKCursor cursor;
        cursor.init(hiddenSize_, tile_scratch_plan_.tile_elements, topk_);
        AsyncCombineTiles::HiddenKTile tiles[AsyncCombineTiles::TILE_INPUT_SLOTS];
        float weights[AsyncCombineTiles::TILE_INPUT_SLOTS]{};
        uint32_t slot = 0;
        tiles[slot] = cursor.current();
        load_tiled_input(inputs[slot], tiles[slot], weights[slot]);
        pipeline.MarkLoaded(slot);
        cursor.advance();
        while (true) {
            pipeline.WaitLoaded(slot);
            const auto current = tiles[slot];
            const float current_weight = weights[slot];
            const bool next_ready = cursor.has_next();
            const uint32_t next_slot = (slot + 1) % AsyncCombineTiles::TILE_INPUT_SLOTS;
            if (next_ready) {
                pipeline.WaitForReuse(next_slot);
                tiles[next_slot] = cursor.current();
                load_tiled_input(inputs[next_slot], tiles[next_slot], weights[next_slot]);
                pipeline.MarkLoaded(next_slot);
                cursor.advance();
            }
            if (current.logical_index == 0) {
                // Prior output's MTE3_V stays ahead of this Duplicate on V; the
                // new tile cannot overwrite sum while its old final Cast reads it.
                Duplicate(sumFloatBufLocal_, 0.0F, current.hidden_elements);
            }
            Cast(rowTmpFloatLocal_, inputs[slot], RoundMode::CAST_NONE, current.hidden_elements);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(mulBufLocal_, rowTmpFloatLocal_, current_weight, current.hidden_elements);
            AscendC::PipeBarrier<PIPE_V>();
            Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, current.hidden_elements);
            if (current.logical_index == topk_ - 1) {
                AscendC::PipeBarrier<PIPE_V>();
                Cast(inputs[slot], sumFloatBufLocal_, RoundMode::CAST_RINT, current.hidden_elements);
                SyncFunc<HardEvent::V_MTE3>();
                DataCopyPad(xOutGMTensor_[static_cast<uint64_t>(token) * hiddenSize_ + current.hidden_offset],
                    inputs[slot], {1U, static_cast<uint32_t>(sizeof(XOutType) * current.hidden_elements), 0U, 0U, 0U});
                SyncFunc<HardEvent::MTE3_V>();
            }
            pipeline.MarkSourceConsumed(slot);
            if (!next_ready) break;
            slot = next_slot;
        }
        // Close late MTE2/V/MTE3 reads before the next token rewrites descriptor
        // storage. This is a per-token boundary, not a per-K Scalar wait.
        AsyncCombineTiles::retire_token_descriptor_use(pipeline, backend);
    }
    pipeline.Drain();
}
// ### PATCH END: Hidden-first CR with frozen per-token source descriptors

#endif
// ### PATCH END: Ordered A5 CR vector-input pipeline

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
