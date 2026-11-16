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
 * \file block_mmad_matmul_streamk_split_k.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <MatMulL0C2Out FixpOpti_, bool IsSplitSinglecoreK_, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockWithStreamKSplitK<FixpOpti_, IsSplitSinglecoreK_>, AType_, LayoutA_, BType_, LayoutB_,
                CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockWithStreamKSplitK<FixpOpti_, IsSplitSinglecoreK_>;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = asc::te::shape<int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;

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
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{2};
        uint16_t l0cStages{1};
    };

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;

        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;

        aL1OneBuffer_ = mL1_ * kL1_ * sizeof(AType);
        bL1OneBuffer_ = nL1_ * kL1_ * sizeof(BType);

        static constexpr uint64_t QUARTER_L1_SIZE = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        for (auto i = 0; i < l1Stages_; ++i) {
            aL1Buffer_[i] = QUARTER_L1_SIZE * (QUADRUPLE_BUFFER_COUNT / l1Stages_) * i;
            bL1Buffer_[i] = aL1Buffer_[i] + aL1OneBuffer_;
            biasL1Buffer_[i] = bL1Buffer_[i] + bL1OneBuffer_;
        }

        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC, typename TensorWorkspace>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& gmC,
                                      TensorWorkspace gmWorkspace, TupleShape tileShape, int64_t kCntIndex,
                                      bool checkIsSkScene)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        int64_t curML1 = asc::te::get<MNK_M>(tileShape);
        int64_t curNL1 = asc::te::get<MNK_N>(tileShape);
        uint64_t skSingleCoreK = asc::te::get<MNK_K>(tileShape);
        splitSingleCoreKRound_ = skSingleCoreK / splitSingleCoreK_;
        splitSingleCoreKTail_ = skSingleCoreK % splitSingleCoreK_ + splitSingleCoreK_;

        for (uint64_t splitSingleCoreKIdx = 0; splitSingleCoreKIdx < splitSingleCoreKRound_; splitSingleCoreKIdx++) {
            uint64_t coordK = splitSingleCoreKIdx * splitSingleCoreK_;
            blkK_ = splitSingleCoreKIdx == (splitSingleCoreKRound_ - 1) ? splitSingleCoreKTail_ : splitSingleCoreK_;
            auto gmBlockA = gmA.slice(AscendC::MakeCoord(0, coordK), AscendC::MakeShape(curML1, blkK_));
            auto gmBlockB = gmB.slice(AscendC::MakeCoord(coordK, 0), AscendC::MakeShape(blkK_, curNL1));
            uint64_t curKL1Iter = Blaze::Gemm::CeilDiv(blkK_, kL1_);
            uint64_t nL1Align = Blaze::Gemm::CeilAlign(curNL1, static_cast<int64_t>(AscendC::BLOCK_CUBE));

            auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
                curML1, curNL1);
            auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);

            for (uint64_t iter0 = 0; iter0 < curKL1Iter; ++iter0) {
                uint64_t curKL1 = (iter0 + 1 == curKL1Iter) ? (blkK_ - iter0 * kL1_) : kL1_;
                uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

                TileShape l1Shape{curML1, curNL1, static_cast<int64_t>(curKL1)};
                // Copy L1 From GM
                auto l1TensorTuple = CopyL1FromGM(gmBlockA, gmBlockB, gmBias, l1Shape, l1BufId, iter0, kCntIndex,
                                                  splitSingleCoreKIdx);
                auto tensorAL1 = asc::te::get<0>(l1TensorTuple);
                auto tensorBL1 = asc::te::get<1>(l1TensorTuple);
                auto tensorBiasL1 = asc::te::get<2>(l1TensorTuple);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId + L1_EVENT_ID_OFFSET);

                uint64_t kL0Iter = Blaze::Gemm::CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                    uint64_t mte1Flag = l0PingPong_ & 0x1;

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));

                    TileShape l0Shape{curML1, curNL1, static_cast<int64_t>(curK0)};
                    bool needBias = NeedProcessBias(iter0, iter1, kCntIndex, splitSingleCoreKIdx);
                    // Copy L0 From L1
                    auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0Offset,
                                                      baseK_ * iter1, needBias, l1BufId, iter1, kCntIndex);

                    auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                    auto tensorBL0 = asc::te::get<1>(l0TensorTuple);
                    auto tensorBiasL0 = asc::te::get<2>(l0TensorTuple);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));

                    asc::te::unit_flag_mode unitFlag = (iter0 + 1 == curKL1Iter && iter1 + 1 == kL0Iter) ?
                                                           asc::te::unit_flag_mode::enable_update :
                                                           asc::te::unit_flag_mode::enable_keep;
                    bool initCmatrix = iter0 == 0 && iter1 == 0 &&
                                       (!isBias_ || (isBias_ && !(kCntIndex == 0 && splitSingleCoreKIdx == 0)));
                    // Mmad
                    Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag, initCmatrix);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                    l0PingPong_++;
                }
                if (iter0 + 1 == curKL1Iter) {
                    AscendC::PipeBarrier<PIPE_FIX>();
                    // atomic需流水同步
                    if (splitSingleCoreKIdx != 0) {
                        AscendC::SetAtomicAdd<float>();
                    }
                    auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
                    if (checkIsSkScene) {
                        asc::te::copy(
                            CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)),
                            gmWorkspace, tensorL0C);
                    } else {
                        asc::te::copy(
                            CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), gmC,
                            tensorL0C);
                    }
                    if (splitSingleCoreKIdx == (splitSingleCoreKRound_ - 1)) {
                        AscendC::DisableDmaAtomic();
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId + L1_EVENT_ID_OFFSET);
                abL1LoopCnt_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1, int64_t kCntIndex,
                                           uint64_t splitSingleCoreKIdx)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kCntIndex == 0 && splitSingleCoreKIdx == 0;
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(const TensorA& tensorA, const TensorB& tensorB, const TensorBias& tensorBias,
                                        const TileShape& l1Shape, uint64_t l1BufId, uint64_t kIdx, int64_t kCntIndex,
                                        uint64_t splitSingleCoreKIdx)
    {
        uint64_t curM = asc::te::get<MNK_M>(l1Shape);
        uint64_t curN = asc::te::get<MNK_N>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);

        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});

        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Buffer_[l1BufId]),
                                              layoutAL1);
        auto gmTileA = tensorA.slice(asc::te::make_coord(0, kIdx * kL1_), asc::te::make_shape(curM, curKL1));
        asc::te::copy(copyGM2L1, tensorAL1, gmTileA);

        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Buffer_[l1BufId]),
                                              layoutBL1);
        auto gmTileB = tensorB.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curN));
        asc::te::copy(copyGM2L1, tensorBL1, gmTileB);

        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curN);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Buffer_[l1BufId]), layoutBiasL1);
        if (isBias_ && kIdx == 0 && kCntIndex == 0 && splitSingleCoreKIdx == 0) {
            asc::te::copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        return AscendC::Std::make_tuple(tensorAL1, tensorBL1, tensorBiasL1);
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorA& tensorAL1, const TensorB& tensorBL1,
                                        const TensorBias& tensorBiasL1, const TileShape& l0Shape, uint64_t l0Offset,
                                        uint64_t kIdx, bool needBias, uint64_t biasBufId, uint64_t iter1,
                                        int64_t kCntIndex)
    {
        uint64_t curM = asc::te::get<MNK_M>(l0Shape);
        uint64_t curN = asc::te::get<MNK_N>(l0Shape);
        uint64_t curK0 = asc::te::get<MNK_K>(l0Shape);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kIdx), asc::te::make_shape(curM, curK0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        if (iter1 == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(biasBufId + L1_EVENT_ID_OFFSET);
        }

        uint64_t nL1Align = Blaze::Gemm::CeilAlign(curN, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nL1Align);
        auto offsetBiasL0 = nL1_ * biasBufId * sizeof(float);
        auto tensorBiasL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(offsetBiasL0),
                                                 layoutBiasL0);
        if (needBias) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }

        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK0,
                                                                                                                  curN);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset),
                                              layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kIdx, 0), asc::te::make_shape(curK0, curN));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TileShape& l0Shape, bool needBias,
                                   asc::te::unit_flag_mode unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM = asc::te::get<MNK_M>(l0Shape);
        auto curN = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);

        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                        static_cast<uint16_t>(curK0), unitFlag, initCmatrix};

        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;
    static constexpr uint16_t L1_EVENT_ID_OFFSET = 2;

    uint64_t blkK_{1}; // actual value after spliting singcorek
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t splitSingleCoreK_{1024};
    uint64_t splitSingleCoreKTail_{1024};
    uint64_t splitSingleCoreKRound_{0};
    uint32_t l1Stages_{2};

    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    bool isBias_{false};

    uint64_t aL1Buffer_[4] = {0};
    uint64_t bL1Buffer_[4] = {0};
    uint64_t biasL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
