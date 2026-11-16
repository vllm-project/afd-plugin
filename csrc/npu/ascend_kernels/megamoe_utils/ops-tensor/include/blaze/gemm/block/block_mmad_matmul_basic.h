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
 * \file block_mmad_matmul_basic.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/datamove.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t FullLoadMode_, uint64_t FusedOpType_, class KernelSchedule_, uint64_t NonContiguousType_,
          MatmulOutputMode OutputMode_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockBasic<FullLoadMode_, FusedOpType_, KernelSchedule_, NonContiguousType_, OutputMode_>,
                AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockBasic<FullLoadMode_, FusedOpType_, KernelSchedule_, NonContiguousType_,
                                                 OutputMode_>;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = DispatchPolicy::NON_CONTIGUOUS_TYPE;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using TripleShape = asc::te::shape<int64_t, int64_t, int64_t>;

    // TRANS_A and TRANS_B
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;
    static constexpr bool IS_INT8_OUT = AscendC::Std::is_same_v<CType, signed char>;
    // AL1 Layout
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<AType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>>;
    // BL1 Layout
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::layout_trait_default<BType>>>;

    // kernel params
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
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        GM_ADDR scaleGmAddr{nullptr};
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
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        enableL0cPingPong_ = params.l0cStages > 1;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        uint64_t aL1OneSize = mL1_ * kL1_ * sizeof(AType);
        uint64_t bL1OneSize = nL1_ * kL1_ * sizeof(BType);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        uint64_t stride = QUADRUPLE_BUFFER_COUNT / l1Stages_;
        // 2buffer时：|APing,BPing,BiasPing------|APong,BPong,BiasPong---|
        // 4buffer时：|A0,B0,Bias0 |A1,B1,Bias1 |A2,B2,Bias2 |A3,B3,Bias3 |
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            uint64_t base = slotSize * stride * i;
            aL1Buffer_[i] = base;
            bL1Buffer_[i] = base + aL1OneSize;
            biasL1Buffer_[i] = bL1Buffer_[i] + bL1OneSize;
        }

        // Scale L1 buffer for int8 output fixpipe quantization
        uint64_t biasL1OneSize = nL1_ * sizeof(BiasType);
        uint64_t lastSlotBase = slotSize * stride * (l1Stages_ - 1);
        scaleL1Buffer_ = lastSlotBase + aL1OneSize + bL1OneSize + biasL1OneSize;
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& gmC,
                                      TupleShape& blockShape, __gm__ uint64_t* gmScalePtr = nullptr)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        // m0 == m1 && n0 == n1
        int64_t curM = Blaze::Gemm::Min(asc::te::get<MNK_M>(blockShape), static_cast<int64_t>(baseM_));
        int64_t curN = Blaze::Gemm::Min(asc::te::get<MNK_N>(blockShape), static_cast<int64_t>(baseN_));
        uint64_t oriK = asc::te::get<MNK_K>(blockShape); // 非全载blockShape的K维度固定返回原始K
        uint64_t l0cOffset = HALF_L0C_SIZE * (l0cPingPong_ & 0x1);
        // LoC搬出
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, asc::te::_16>{}(curM, curN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                              layoutL0C);

        gmScalePtr_ = gmScalePtr;
        curNOut_ = static_cast<uint64_t>(curN);

        kL1Iter_ = Blaze::Gemm::CeilDiv(oriK, kL1_);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (oriK - kL1_ * iter0) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
            uint64_t btBufId = abL1LoopCnt_ & 0x1;

            // GM->L1
            TripleShape l1Shape{curM, curN, static_cast<int64_t>(curKL1)};
            auto l1TensorTuple = CopyL1FromGM(gmA, gmB, gmBias, l1Shape, l1BufId, iter0);
            auto tensorAL1 = asc::te::get<0>(l1TensorTuple);
            auto tensorBL1 = asc::te::get<1>(l1TensorTuple);
            auto tensorBiasL1 = asc::te::get<2>(l1TensorTuple);

            uint64_t kL0Iter = Blaze::Gemm::CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                uint16_t l0BufferId = l0PingPong_ & 0x1;

                // A L1->L0
                TripleShape l0Shape{curM, curN, static_cast<int64_t>(curK0)};
                bool needBias = NeedProcessBias(iter0, iter1);
                auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0BufferId,
                                                  baseK_ * iter1, needBias, l1BufId, btBufId);
                auto tensorAL0 = asc::te::get<0>(l0TensorTuple);
                auto tensorBL0 = asc::te::get<1>(l0TensorTuple);
                auto tensorBiasL0 = asc::te::get<2>(l0TensorTuple);

                bool initCmatrix = iter0 == 0 && iter1 == 0 && !isBias_;
                asc::te::unit_flag_mode unitFlag = ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ?
                                                        asc::te::unit_flag_mode::enable_update :
                                                        asc::te::unit_flag_mode::enable_keep);

                {
                    Gemm::LockM(l0BufferId + L0_BUFFER_ID_BASE);
                    Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag, initCmatrix);
                    Gemm::UnLockM(l0BufferId + L0_BUFFER_ID_BASE);
                }
                l0PingPong_++;
            }
            abL1LoopCnt_++;
        }

        // 数据搬出到GM
        CopyL0CToGM(gmC, tensorL0C);

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(const TensorA& tensorA, const TensorB& tensorB, const TensorBias& tensorBias,
                                        const TripleShape& l1Shape, uint16_t l1BufferId, uint64_t kIdx)
    {
        constexpr bool IS_SLICE_POLICY = NON_CONTIGUOUS_TYPE ==
                                         static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE);
        constexpr bool IS_BATCHED_B_POLICY = NON_CONTIGUOUS_TYPE ==
                                             static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_BATCHED_B);
        uint64_t curML1 = asc::te::get<MNK_M>(l1Shape);
        uint64_t curNL1 = asc::te::get<MNK_N>(l1Shape);
        uint64_t curKL1 = asc::te::get<MNK_K>(l1Shape);

        Gemm::LockMte2(l1BufferId + L1_BUFFER_ID_BASE);
        // A GM->L1
        auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(aL1Buffer_[l1BufferId]), layoutAL1);

        if constexpr (IS_SLICE_POLICY) {
            auto layoutGmA = tensorA.layout();
            auto sliceM = asc::te::get<0>(asc::te::get<1>(layoutGmA.shape()));
            auto gmTileASlice = tensorA.slice(
                asc::te::make_coord(0, asc::te::make_coord(0, kIdx * kL1_)),
                asc::te::make_shape(curML1 / sliceM, asc::te::make_shape(sliceM, curKL1)));
            auto copyGM2L1Slice = asc::te::make_copy(Blaze::Gemm::Tile::CopySliceGM2L1{});
            asc::te::copy(copyGM2L1Slice, tensorAL1, gmTileASlice);
        } else {
            auto gmTileA = tensorA.slice(asc::te::make_coord(0, kIdx * kL1_), asc::te::make_shape(curML1, curKL1));
            asc::te::copy(copyGM2L1, tensorAL1, gmTileA);
        }

        // B GM->L1
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Buffer_[l1BufferId]), layoutBL1);
        if constexpr (IS_BATCHED_B_POLICY) {
            CopyBatchedBToConcatL1(copyGM2L1, tensorBL1, tensorB, bL1Buffer_[l1BufferId], curKL1, kIdx);
        } else {
            auto gmTileB = tensorB.slice(asc::te::make_coord(kIdx * kL1_, 0), asc::te::make_shape(curKL1, curNL1));
            asc::te::copy(copyGM2L1, tensorBL1, gmTileB);
        }

        // Bias GM->L1
        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curNL1);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(biasL1Buffer_[l1BufferId]), layoutBiasL1);
        if (isBias_ && kIdx == 0) {
            asc::te::copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        // Scale GM->L1 (int8 output fixpipe quantization)
        if (kIdx == 0) {
            if constexpr (IS_INT8_OUT) {
                auto layoutScaleL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curNOut_);
                auto tensorScaleL1 = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::l1, uint64_t>(scaleL1Buffer_), layoutScaleL1);
                auto layoutScaleGM = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curNOut_);
                auto gmScale = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(gmScalePtr_),
                                                    layoutScaleGM);
                Gemm::LockMte2(SCALEL1_BUFFER_ID_BASE);
                asc::te::copy(copyGM2L1, tensorScaleL1, gmScale);
                Gemm::UnLockMte2(SCALEL1_BUFFER_ID_BASE);
            }
        }
        Gemm::UnLockMte2(l1BufferId + L1_BUFFER_ID_BASE);
        return AscendC::Std::make_tuple(tensorAL1, tensorBL1, tensorBiasL1);
    }

    template <typename CopyGM2L1, typename TensorBL1, typename TensorB>
    __aicore__ inline void CopyBatchedBToConcatL1(const CopyGM2L1& copyGM2L1, const TensorBL1& tensorBL1,
                                                  const TensorB& tensorB, uint64_t bL1Address, uint64_t curKL1,
                                                  uint64_t kIdx)
    {
        using TensorBElementType = asc::te::get_attribute_element_type<typename TensorB::element_type*>;
        using TensorBLayoutPattern = asc::te::get_layout_pattern<typename TensorB::layout_type>;
        using TensorBL1ElementType = asc::te::get_attribute_element_type<typename TensorBL1::element_type*>;
        using TensorBL1LayoutPattern = asc::te::get_layout_pattern<typename TensorBL1::layout_type>;
        using ExpectedBL1LayoutPattern = AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn,
                                                                     asc::te::nz_layout_ptn>;
        static_assert(AscendC::Std::is_same_v<TensorBElementType, BType>,
                      "Batched-B GM tensor element type must match BlockMmad BType.");
        static_assert(AscendC::Std::is_same_v<TensorBL1ElementType, BType>,
                      "Batched-B L1 tensor element type must match BlockMmad BType.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<TensorB>, asc::te::location::gm>,
                      "Batched-B source tensor must reside in GM.");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<TensorBL1>, asc::te::location::l1>,
                      "Batched-B destination tensor must reside in L1.");
        static_assert(AscendC::Std::is_same_v<TensorBLayoutPattern, LayoutB>,
                      "Batched-B GM tensor layout pattern must match BlockMmad LayoutB.");
        static_assert(AscendC::Std::is_same_v<TensorBL1LayoutPattern, ExpectedBL1LayoutPattern>,
                      "Batched-B L1 tensor layout must match the BlockMmad transpose mode.");
        static_assert(TensorB::layout_type::depth == asc::te::five_dim_data,
                      "Batched-B source tensor must contain an outer batch and a two-dimensional matrix layout.");

        auto gmBLayout = tensorB.layout();
        const uint64_t batchCount = static_cast<uint64_t>(asc::te::get<0>(gmBLayout.shape()));
        using GmBLayoutPattern = asc::te::get_layout_pattern<typename TensorB::layout_type>;
        using BLayoutTrait = asc::te::layout_trait_default<BType>;
        auto singleGmBLayout = asc::te::make_pattern_layout<GmBLayoutPattern, BLayoutTrait>(
            asc::te::get<1>(gmBLayout.shape()), asc::te::get<1>(gmBLayout.stride()));
        const uint64_t singleN = asc::te::get_total_column_shape(singleGmBLayout);
        auto gmTileB = tensorB.slice(asc::te::make_coord(0UL, asc::te::make_coord(kIdx * kL1_, 0UL)),
                                     asc::te::make_shape(batchCount, asc::te::make_shape(curKL1, singleN)));

        // Re-view the same BL1 storage as a batch so one GM2L1 copy packs the matrices into adjacent N ranges.
        // The original two-dimensional tensorBL1 remains the compute view used by L1-to-L0 and MMAD.
        auto concatBL1Layout = tensorBL1.layout();
        auto singleBL1 = tensorBL1.slice(asc::te::make_coord(0UL, 0UL), asc::te::make_shape(curKL1, singleN));
        auto singleBL1Layout = singleBL1.layout();
        const uint64_t batchStride = concatBL1Layout(asc::te::make_coord(0UL, singleN));
        using BL1Layout = typename TensorBL1::layout_type;
        using BL1LayoutPattern = asc::te::get_layout_pattern<BL1Layout>;
        auto batchedBL1Layout = asc::te::make_pattern_layout<BL1LayoutPattern, BLayoutTrait>(
            asc::te::make_shape(batchCount, singleBL1Layout.shape()),
            asc::te::make_stride(batchStride, singleBL1Layout.stride()));
        auto batchedBL1 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(bL1Address),
                                               batchedBL1Layout);
        asc::te::copy(copyGM2L1, batchedBL1, gmTileB);
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(const TensorA& tensorAL1, const TensorB& tensorBL1,
                                        const TensorBias& tensorBiasL1, const TripleShape& l0Shape,
                                        const uint16_t l0BufferId, uint64_t kIdx, bool needBias,
                                        const uint16_t l1BufferId, const uint16_t btBufferId)
    {
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0BaseOffset = HALF_L0_SIZE * l0BufferId;
        auto curM0 = asc::te::get<MNK_M>(l0Shape);
        auto curN0 = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        Gemm::LockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::LockMte1(l0BufferId + L0_BUFFER_ID_BASE);
        // A L1->L0A
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto layoutAL0 = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            curM0, curK0);
        auto tensorAL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BaseOffset),
                                              layoutAL0);
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kIdx), asc::te::make_shape(curM0, curK0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        // B L1->L0B
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto layoutBL0 = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            curK0, curN0);
        auto tensorBL0 = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0BaseOffset),
                                              layoutBL0);
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kIdx, 0), asc::te::make_shape(curK0, curN0));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        // Bias L1->L0
        uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN0, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl1Align);
        auto tensorBiasL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::bias, float>(baseN_ * btBufferId * sizeof(float)), layoutBiasL0);
        if (needBias) {
            auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }
        Gemm::UnLockMte1(l1BufferId + L1_BUFFER_ID_BASE);
        Gemm::UnLockMte1(l0BufferId + L0_BUFFER_ID_BASE);
        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyL0CToGM(TensorC& gmC, TensorL0C& tensorL0C)
    {
        asc::te::l0c_to_gm_params fixpParams{asc::te::unit_flag_mode::enable_update};
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        if constexpr (IS_INT8_OUT) {
            auto layoutScaleL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, curNOut_);
            auto tensorScaleL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, uint64_t>(scaleL1Buffer_), layoutScaleL1);

            Gemm::LockFix(SCALEL1_BUFFER_ID_BASE);
            asc::te::copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C, tensorScaleL1);
            Gemm::UnLockFix(SCALEL1_BUFFER_ID_BASE);
        } else {
            asc::te::copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0,
                                   TensorC& tensorL0C, const TripleShape& l0Shape, bool needBias,
                                   asc::te::unit_flag_mode unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});
        auto curM0 = asc::te::get<MNK_M>(l0Shape);
        auto curN0 = asc::te::get<MNK_N>(l0Shape);
        auto curK0 = asc::te::get<MNK_K>(l0Shape);
        // Mmad参数
        asc::te::mmad_params mmadParams{static_cast<uint16_t>(curM0), static_cast<uint16_t>(curN0),
                                        static_cast<uint16_t>(curK0), unitFlag, initCmatrix};
        // 传入自定义Trait类型
        if (needBias) {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint16_t L1_BUFFER_ID_BASE = 0;
    static constexpr uint16_t L0_BUFFER_ID_BASE = 4;
    static constexpr uint16_t SCALEL1_BUFFER_ID_BASE = 6;
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    uint64_t kL1Iter_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    uint64_t curNOut_{0};
    __gm__ uint64_t* gmScalePtr_{nullptr};
    uint64_t scaleL1Buffer_{0};
    uint64_t biasL1Buffer_[4] = {0};
    uint64_t aL1Buffer_[4] = {0};
    uint64_t bL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
