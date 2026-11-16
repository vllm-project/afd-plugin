/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_matmul_bl1_full_load.h
 * \brief BlockMmad specialization for B full-load matmul.
 *
 * B is fully loaded into L1 once at the top. A uses pipeline buffering with l1Stages_ stages.
 * Supports both ON_THE_FLY (L0C -> GM) and fixpipe (L0C -> UB with cross-core sync) output modes.
 * Uses BufferManager for unified pipeline buffer and event synchronization.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t L0COutModel_, uint64_t FusedOpType_, class KernelSchedule_, class AType_, class LayoutA_,
          class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockBFullLoad<L0COutModel_, FusedOpType_, KernelSchedule_>, AType_, LayoutA_, BType_,
                LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockBFullLoad<L0COutModel_, FusedOpType_, KernelSchedule_>;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;

    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 4;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 6;
    constexpr static uint16_t FLAG_ID_MAX = 16;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;

    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t oriK{0};
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        uint64_t splitM{0};
        uint8_t ubDB{1};
        uint64_t rowStride{0};
    };

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = params.oriK;
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        splitM_ = params.splitM;
        ubDB_ = params.ubDB;
        enableL0cPingPong_ = params.l0cStages > 1;
        isBL1Loaded_ = false;
        kAlign_ = Blaze::Gemm::CeilAlign(k_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        aL1OneBuffer_ = mL1_ * kL1_ * sizeof(AType);
        bL1OneBuffer_ = nL1_ * kAlign_ * sizeof(BType);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        cvPingPong_ = 0;

        uint64_t aL1Total = aL1OneBuffer_ * l1Stages_;
        uint64_t bL1Offset = aL1Total;
        uint64_t biasL1Offset = bL1Offset + bL1OneBuffer_;
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            bufMgr_.InitAL1(i, aL1OneBuffer_ * i, i);
        }
        bufMgr_.InitBL1(0, bL1Offset, l1Stages_);
        bufMgr_.InitBias(0, biasL1Offset, l1Stages_);
        bufMgr_.InitBT(sizeof(float) * baseN_);
        bufMgr_.InitL0();
        bufMgr_.InitL0C();
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& tensorC,
                                      TupleShape& tileShape)
    {
        uint64_t curM = asc::te::get<MNK_M>(tileShape);
        uint64_t curN = asc::te::get<MNK_N>(tileShape);
        uint64_t curK = asc::te::get<MNK_K>(tileShape);

        auto l1FullLoadTensorTuple = CopyL1FromGMbFullLoad(gmB, gmBias, curK, curN);
        auto tensorB = asc::te::get<0>(l1FullLoadTensorTuple);
        auto tensorBias = asc::te::get<1>(l1FullLoadTensorTuple);

        curBaseN_ = Min(curN, baseN_);
        nL1Iter_ = CeilDiv(curN, curBaseN_);
        kL1_ = Min(k_, kL1_);
        kL1Iter_ = CeilDiv(k_, kL1_);
        for (uint64_t iterN = 0; iterN < nL1Iter_; ++iterN) {
            auto tileN = (iterN + 1 == nL1Iter_) ? (curN - curBaseN_ * iterN) : curBaseN_;
            const auto& l0cSlot = bufMgr_.GetL0CSlot(l0cPingPong_ & 0x1);
            auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<16>>{}(curM, tileN);
            auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cSlot.Addr()),
                                                  layoutL0C);

            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1_ * iter0) : kL1_;
                uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
                uint64_t btBufId = abL1LoopCnt_ & 0x1;

                const auto& aL1Slot = bufMgr_.GetL1ASlot(l1BufId);
                const auto& btSlot = bufMgr_.GetBTSlot(btBufId);

                TileShape l1Shape{curM, tileN, curKL1};
                auto tensorAL1 = CopyL1FromGM(gmA, aL1Slot, l1Shape, iter0);

                uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

                    uint64_t aL1MOffset = 0;
                    uint64_t aL1KOffset = iter1 * baseK_;
                    uint64_t bL1KOffset = iter0 * kL1_ + iter1 * baseK_;
                    uint64_t bL1NOffset = iterN * curBaseN_;

                    TileShape l0Shape{curM, tileN, curK0};
                    bool needBias = NeedProcessBias(iter0, iter1);
                    auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorB, tensorBias, l0Shape, l0Slot, aL1Slot,
                                                      aL1MOffset, aL1KOffset, bL1KOffset, bL1NOffset, needBias, btSlot);
                    auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                    auto tensorBL0 = asc::te::get<1>(l0TensorTuple);
                    auto tensorBiasL0 = asc::te::get<2>(l0TensorTuple);

                    {
                        auto l0Lock = l0Slot.LockM();
                        auto btLock = btSlot.LockM();
                        bool initCmatrix = iter0 == 0 && iter1 == 0 && !isBias_;
                        uint8_t unitFlag = ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION :
                                                                                              NON_FINAL_ACCUMULATION);
                        Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag,
                                initCmatrix);
                    }
                    l0PingPong_++;
                }
                abL1LoopCnt_++;
            }

            CopyOutFromL0C(tensorC, tensorL0C, tileN, curM, iterN);

            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    template <typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGMbFullLoad(TensorB& gmB, TensorBias& gmBias, uint64_t curK, uint64_t curN)
    {
        const auto& bL1Slot = bufMgr_.GetL1BSlot(0);
        const auto& biasL1Slot = bufMgr_.GetL1BiasSlot(0);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

        auto layoutBL1 = MakeLayoutBL1{}(curK, curN);
        auto tensorB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Slot.Addr()),
                                            layoutBL1);

        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curN);
        auto tensorBias = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Slot.Addr()), layoutBiasL1);

        if (!isBL1Loaded_) {
            auto lock = bL1Slot.LockMte2();
            asc::te::copy(copyGM2L1, tensorB, gmB);
            if (isBias_) {
                asc::te::copy(copyGM2L1, tensorBias, gmBias);
            }
            isBL1Loaded_ = true;
        }
        return AscendC::Std::make_tuple(tensorB, tensorBias);
    }

    template <typename TensorA>
    __aicore__ inline auto CopyL1FromGM(const TensorA& tensorA, const BufferSlot& aL1Slot, const TileShape& l1Shape,
                                        uint64_t kIdx)
    {
        uint64_t curM = asc::te::get<MNK_M>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);

        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Slot.Addr()),
                                              layoutAL1);
        {
            auto lock = aL1Slot.LockMte2();
            auto gmTileA = tensorA.slice(asc::te::make_coord(0, kIdx * kL1_), asc::te::make_shape(curM, curKL1));
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);
        }
        return tensorAL1;
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorA& tensorAL1, const TensorB& tensorBL1,
                                        const TensorBias& tensorBiasL1, const TileShape& l0Shape,
                                        const BufferSlot& l0Slot, const BufferSlot& aL1Slot, uint64_t aL1MOffset,
                                        uint64_t aL1KOffset, uint64_t bL1KOffset, uint64_t bL1NOffset, bool needBias,
                                        const BufferSlot& btSlot)
    {
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Slot.Addr()),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(aL1MOffset, aL1KOffset),
                                              asc::te::make_shape(curM, curK0));
        {
            auto l1LockA = aL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        }

        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK0,
                                                                                                                  curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Slot.Addr()),
                                              layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(bL1KOffset, bL1NOffset),
                                              asc::te::make_shape(curK0, curN));
        {
            auto l0Lock = l0Slot.LockMte1();
            asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        }

        uint64_t nL1Align = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nL1Align);
        auto tensorBiasL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(btSlot.Addr()),
                                                 layoutBiasL0);
        if (needBias) {
            auto tensorBiasL1Slice = tensorBiasL1.slice(asc::te::make_coord(0, bL1NOffset),
                                                        asc::te::make_shape(1, curN));
            auto btLock = btSlot.LockMte1();
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1Slice);
        }

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyOutFromL0C(TensorC& tensorC, TensorL0C& tensorL0C, uint64_t tileN, uint64_t curM,
                                          uint64_t iterN)
    {
        asc::te::l0c_to_gm_params fixpParams{asc::te::unit_flag_mode::enable_update};
        if constexpr (DispatchPolicy::L0C2OUT_MODEL != ON_THE_FLY) {
            uint16_t slot = (ubDB_ > 1) ? static_cast<uint16_t>(cvPingPong_ & 0x1) : 0U;
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + slot);
            if (splitM_) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + slot + FLAG_ID_MAX);
            }
            CopyOutFromL0C2UB(tensorC, tensorL0C, tileN, curM, slot);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + slot);
            if (splitM_) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + slot + FLAG_ID_MAX);
            }
            cvPingPong_++;
        } else {
            auto tensorGmC = tensorC.slice(asc::te::make_coord(0, iterN * curBaseN_), asc::te::make_shape(curM, tileN));
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            asc::te::copy(copyL0C2GM.with(fixpParams), tensorGmC, tensorL0C);
        }
    }

    template <typename TensorUB, typename TensorL0C>
    __aicore__ inline void CopyOutFromL0C2UB(TensorUB& tensorC, TensorL0C& tensorL0C, uint64_t tileN, uint64_t curM,
                                             uint16_t slotIdx)
    {
        asc::te::l0c_to_ub_params fixpParams{asc::te::unit_flag_mode::enable_update};
        uint64_t tileNAlign = Blaze::Gemm::CeilAlign(tileN, static_cast<uint64_t>(asc::te::c0_element<CType>));
        auto layoutUB = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN), tileNAlign);
        constexpr int64_t ubHalfElems = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(CType) /
                                                             DOUBLE_BUFFER_COUNT);
        auto ubTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub>(tensorC.data().get() + slotIdx * ubHalfElems), layoutUB);
        if (splitM_) {
            auto copyL0C2UBSplitM = asc::te::make_copy(asc::te::copy_l0c_to_ub{},
                                                       Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
            asc::te::copy(copyL0C2UBSplitM.with(fixpParams), ubTensor, tensorL0C);
        } else {
            auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
            asc::te::copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TileShape& l0Shape, bool needBias, uint8_t unitFlag,
                                   bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK0), static_cast<asc::te::unit_flag_mode>(unitFlag),
                                        initCmatrix};
        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint64_t SPLIT_M_ALIGN = 2;

    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t curBaseN_{16};
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t nL1Iter_{0};
    uint64_t kL1Iter_{0};
    uint64_t kAlign_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t ubDB_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool splitM_{false};
    bool isBL1Loaded_{false};
    uint64_t cvPingPong_{0};

    BufferManager<4, 2, 2> bufMgr_; // 4 = A 4-stage pipeline, 2 = B full-load double buffers, 2 = L0 ping-pong
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
