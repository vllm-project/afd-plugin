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
 * \file block_mmad_iterbatch_broadcast.h
 * \brief MMAD block for IterBatch-Broadcast: batched L1/L0 pipelining + broadcast data sharing
 *        Unified batched mode with MNK tiling support:
 *        - 3D L0 layouts with batched copy (tensor-api BatchLoadDataImpl supports 5D batch layout)
 *        - MNK tiling within each batch step (baseM/baseN/baseK may be smaller than M/N/K)
 *        - L0A/L0B ping-pong on K axis, L0C ping-pong on MN axis (independent double buffering)
 *        - Batched fixpipe for multiple batches per MNK tile
 *        When tiling guarantees baseM>=M, baseN>=N, baseK>=K, MNK loops degenerate to single step
 *        GM→L1: kernel layer does gmA/gmB Slice for broadcast mapping before calling MMAD
 *        MMAD CopyGM2L1A/B receives pre-sliced GM tensors, no broadcast index logic needed
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

template <bool ABc, bool BBc, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_>
class BlockMmad<MatmulIterBatchBroadcast<ABc, BBc>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
                LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulIterBatchBroadcast<ABc, BBc>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t m{0};
        uint64_t n{0};
        uint64_t k{0};
        uint64_t baseM{0};
        uint64_t baseN{0};
        uint64_t baseK{0};
        uint64_t iterBatchL1{1};
        uint64_t iterBatchL0{1};
        bool aBroadcastSingleBatch{false};
        bool bBroadcastSingleBatch{false};
    };

private:
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool A_BROADCAST = ABc;
    static constexpr bool B_BROADCAST = BBc;
    static constexpr uint64_t BUFFER_NUM = 2;
    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        m_ = params.m;
        n_ = params.n;
        k_ = params.k;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        isBias_ = params.biasGmAddr != nullptr;
        mainIterBatchL1_ = params.iterBatchL1;
        mainIterBatchL0_ = params.iterBatchL0;
        abL1BufId_ = 0;
        aBroadcastSingleBatch_ = params.aBroadcastSingleBatch;
        bBroadcastSingleBatch_ = params.bBroadcastSingleBatch;
        uint64_t al1BatchCount;
        if constexpr (A_BROADCAST) {
            al1BatchCount = (aBroadcastSingleBatch_) ? 1 : mainIterBatchL1_;
        } else {
            al1BatchCount = mainIterBatchL1_;
        }
        uint64_t bl1BatchCount;
        if constexpr (B_BROADCAST) {
            bl1BatchCount = (bBroadcastSingleBatch_) ? 1 : mainIterBatchL1_;
        } else {
            bl1BatchCount = mainIterBatchL1_;
        }
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(AType);
        if constexpr (!TRANS_A) {
            aL1BatchStrideElems_ = CeilAlign(m_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(k_, c0Size);
        } else {
            aL1BatchStrideElems_ = CeilAlign(m_, c0Size) * CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        if constexpr (!TRANS_B) {
            bL1BatchStrideElems_ = CeilAlign(k_, static_cast<uint64_t>(BLOCK_CUBE)) * CeilAlign(n_, c0Size);
        } else {
            bL1BatchStrideElems_ = CeilAlign(k_, c0Size) * CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        }
        aL1OneBuffer_ = aL1BatchStrideElems_ * sizeof(AType) * al1BatchCount;
        bL1OneBuffer_ = bL1BatchStrideElems_ * sizeof(BType) * bl1BatchCount;
    }

    template <typename TensorC, typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmB, const TensorBias& gmBias, TensorC& gmC,
                                      uint64_t curIterBatchL1)
    {
        uint64_t curAl1Count = GetAl1Count(curIterBatchL1);
        uint64_t curBl1Count = GetBl1Count(curIterBatchL1);
        uint64_t l1BufId = abL1BufId_ & 0x1;
        static constexpr uint64_t HALF_L1_SIZE = AscendC::TOTAL_L1_SIZE / BUFFER_NUM;
        uint64_t offsetAl1 = HALF_L1_SIZE * l1BufId;
        uint64_t offsetBl1 = offsetAl1 + aL1OneBuffer_;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
        auto al1Tensor = CopyGM2L1A(gmA, offsetAl1, curAl1Count);
        auto bl1Tensor = CopyGM2L1B(gmB, offsetBl1, curBl1Count);
        auto biasL1Tensor = CopyGM2L1Bias(gmBias, offsetAl1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{}, asc::te::l1_to_l0a_trait_default{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{}, asc::te::l1_to_l0b_trait_default{});
        auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{}, asc::te::l1_to_biastable_trait_default{});
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{}, asc::te::l0c_to_gm_trait_default{});
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, asc::te::mmad_trait_default{});

        BatchedMmadLoop(al1Tensor, bl1Tensor, gmC, biasL1Tensor, curIterBatchL1, copyL12L0A, copyL12L0B, copyL12BT,
                        copyL0C2GM, mmadAtom);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
        abL1BufId_++;
    }

private:
    struct L0TileInfo {
        uint64_t curAl0Cnt;
        uint64_t curBl0Cnt;
        uint64_t curM;
        uint64_t curN;
        uint64_t curK;
        uint64_t al1BatchOffset;
        uint64_t bl1BatchOffset;
        uint64_t iterK;
        uint64_t iterN;
        uint64_t iterM;
    };
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t baseM_{1};
    uint64_t baseN_{1};
    uint64_t baseK_{1};
    uint64_t mainIterBatchL1_{1};
    uint64_t mainIterBatchL0_{1};
    bool isBias_{false};
    uint64_t abL1BufId_{0};
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t aL1BatchStrideElems_{1};
    uint64_t bL1BatchStrideElems_{1};
    bool aBroadcastSingleBatch_{false};
    bool bBroadcastSingleBatch_{false};

    template <typename TensorA>
    __aicore__ inline auto CopyGM2L1A(const TensorA& gmA, uint64_t offsetAl1, uint64_t al1Count)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        using LayoutAl1Ptn = AscendC::Std::conditional_t<TRANS_A, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        auto al1Layout = asc::te::make_frame_layout<LayoutAl1Ptn, asc::te::layout_trait_default<AType>>(al1Count, m_,
                                                                                                        k_);
        auto al1Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(offsetAl1),
                                              al1Layout);
        asc::te::copy(copyGM2L1, al1Tensor, gmA);
        return al1Tensor;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyGM2L1B(const TensorB& gmB, uint64_t offsetBl1, uint64_t bl1Count)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        using LayoutBl1Ptn = AscendC::Std::conditional_t<TRANS_B, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
        auto bl1Layout = asc::te::make_frame_layout<LayoutBl1Ptn, asc::te::layout_trait_default<BType>>(bl1Count, k_,
                                                                                                        n_);
        auto bl1Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, BType>(offsetBl1),
                                              bl1Layout);
        asc::te::copy(copyGM2L1, bl1Tensor, gmB);
        return bl1Tensor;
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyGM2L1Bias(const TensorBias& gmBias, uint64_t offsetAl1)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        uint64_t nAlign = CeilAlign(n_, static_cast<uint64_t>(BLOCK_CUBE));
        auto biasL1Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nAlign);
        auto biasL1Tensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(offsetAl1 + aL1OneBuffer_ + bL1OneBuffer_),
            biasL1Layout);
        if (isBias_) {
            asc::te::copy(copyGM2L1, biasL1Tensor, gmBias);
        }
        return biasL1Tensor;
    }

    template <typename TensorAl1, typename TensorBl1, typename TensorC, typename TensorBiasL1, typename CopyL12L0AT,
              typename CopyL12L0BT, typename CopyL12BTT, typename CopyL0C2GMT, typename MmadAtomT>
    __aicore__ inline void BatchedMmadLoop(const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor, TensorC gmC,
                                           const TensorBiasL1& biasL1Tensor, uint64_t curIterBatchL1,
                                           CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT,
                                           CopyL0C2GMT copyL0C2GM, MmadAtomT mmadAtom)
    {
        uint64_t batchStepCnt = CeilDiv(curIterBatchL1, mainIterBatchL0_);
        const uint64_t c0Size = BLOCK_BYTE_SIZE / sizeof(AType);
        uint64_t ml0Cnt = CeilDiv(m_, baseM_);
        uint64_t nl0Cnt = CeilDiv(n_, baseN_);
        uint64_t kl0Cnt = CeilDiv(k_, baseK_);
        uint64_t l0cBufIdx = 0;
        for (uint64_t iter1 = 0; iter1 < batchStepCnt; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == batchStepCnt) ? (curIterBatchL1 - mainIterBatchL0_ * iter1) :
                                                                    mainIterBatchL0_;
            uint64_t curAl0Cnt = GetAl0Count(curIterBatchL0);
            uint64_t curBl0Cnt = GetBl0Count(curIterBatchL0);
            uint64_t al1BatchOffset = GetAl1BatchOffset(iter1);
            uint64_t bl1BatchOffset = GetBl1BatchOffset(iter1);
            for (uint64_t iterNl0 = 0; iterNl0 < nl0Cnt; ++iterNl0) {
                uint64_t curN = (iterNl0 == nl0Cnt - 1) ? (n_ - iterNl0 * baseN_) : baseN_;
                for (uint64_t iterMl0 = 0; iterMl0 < ml0Cnt; ++iterMl0) {
                    uint64_t curM = (iterMl0 == ml0Cnt - 1) ? (m_ - iterMl0 * baseM_) : baseM_;
                    uint64_t l0cBufId = l0cBufIdx % BUFFER_NUM;
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0cBufId));
                    for (uint64_t iterKl0 = 0; iterKl0 < kl0Cnt; ++iterKl0) {
                        uint64_t curK = (iterKl0 == kl0Cnt - 1) ? (k_ - iterKl0 * baseK_) : baseK_;
                        uint64_t l0BufId = iterKl0 % BUFFER_NUM;
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
                        L0TileInfo tileInfo{curAl0Cnt,      curBl0Cnt,      curM,    curN,    curK,
                                            al1BatchOffset, bl1BatchOffset, iterKl0, iterNl0, iterMl0};
                        CopyL1ToL0(al1Tensor, bl1Tensor, biasL1Tensor, l0BufId, tileInfo, copyL12L0A, copyL12L0B,
                                   copyL12BT);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(l0BufId));
                        uint64_t l0aPerBatchBytes = CeilAlign(curM, static_cast<uint64_t>(BLOCK_CUBE)) *
                                                    CeilAlign(curK, c0Size) * sizeof(AType);
                        uint64_t l0bPerBatchBytes = CeilAlign(curK, c0Size) *
                                                    CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(BType);
                        uint64_t l0cPerBatchBytes = CeilAlign(curM, static_cast<uint64_t>(BLOCK_CUBE)) *
                                                    CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(float);
                        for (uint64_t batchL0Idx = 0; batchL0Idx < curIterBatchL0; ++batchL0Idx) {
                            uint64_t al0BatchIdx = GetABatchIdx(batchL0Idx);
                            uint64_t bl0BatchIdx = GetBBatchIdx(batchL0Idx);
                            ComputeMmad(
                                curM, curN, curK, iterKl0, mmadAtom,
                                AscendC::TOTAL_L0A_SIZE / BUFFER_NUM * l0BufId + al0BatchIdx * l0aPerBatchBytes,
                                AscendC::TOTAL_L0A_SIZE / BUFFER_NUM * l0BufId + bl0BatchIdx * l0bPerBatchBytes,
                                AscendC::TOTAL_L0C_SIZE / BUFFER_NUM * l0cBufId + batchL0Idx * l0cPerBatchBytes);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(l0BufId) + SIXTH_FLAG);
                    }
                    FixL0CToGM(gmC, l0cBufId, curIterBatchL0, curM, curN, iter1, iterMl0, iterNl0, copyL0C2GM);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(static_cast<uint16_t>(l0cBufId));
                    l0cBufIdx++;
                }
            }
        }
    }

    template <typename TensorAl1, typename TensorBl1, typename TensorBiasL1, typename CopyL12L0AT, typename CopyL12L0BT,
              typename CopyL12BTT>
    __aicore__ inline void CopyL1ToL0(const TensorAl1& al1Tensor, const TensorBl1& bl1Tensor,
                                      const TensorBiasL1& biasL1Tensor, uint64_t l0BufId, const L0TileInfo& tileInfo,
                                      CopyL12L0AT copyL12L0A, CopyL12L0BT copyL12L0B, CopyL12BTT copyL12BT)
    {
        auto al0KLayout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(
            tileInfo.curAl0Cnt, tileInfo.curM, tileInfo.curK);
        auto al0KTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0a, AType>(AscendC::TOTAL_L0A_SIZE / BUFFER_NUM * l0BufId),
            al0KLayout);
        auto al1Slice = al1Tensor.slice(
            asc::te::make_coord(tileInfo.al1BatchOffset,
                                asc::te::make_coord(tileInfo.iterM * baseM_, tileInfo.iterK * baseK_)),
            asc::te::make_shape(tileInfo.curAl0Cnt, asc::te::make_shape(tileInfo.curM, tileInfo.curK)));
        asc::te::copy(copyL12L0A, al0KTensor, al1Slice);

        auto bl0KLayout = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(
            tileInfo.curBl0Cnt, tileInfo.curK, tileInfo.curN);
        auto bl0KTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, BType>(AscendC::TOTAL_L0A_SIZE / BUFFER_NUM * l0BufId),
            bl0KLayout);
        auto bl1Slice = bl1Tensor.slice(
            asc::te::make_coord(tileInfo.bl1BatchOffset,
                                asc::te::make_coord(tileInfo.iterK * baseK_, tileInfo.iterN * baseN_)),
            asc::te::make_shape(tileInfo.curBl0Cnt, asc::te::make_shape(tileInfo.curK, tileInfo.curN)));
        asc::te::copy(copyL12L0B, bl0KTensor, bl1Slice);

        if (isBias_ && tileInfo.iterK == 0) {
            uint64_t nl0Align = CeilAlign(tileInfo.curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl0Align);
            auto biasL0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(0),
                                                     biasL0Layout);
            auto biasL1Slice = biasL1Tensor.slice(asc::te::make_coord(0, tileInfo.iterN * baseN_),
                                                  asc::te::make_shape(1UL, tileInfo.curN));
            asc::te::copy(copyL12BT, biasL0Tensor, biasL1Slice);
        }
    }

    template <typename MmadAtomT>
    __aicore__ inline void ComputeMmad(uint64_t curM, uint64_t curN, uint64_t curK, uint64_t iterK, MmadAtomT mmadAtom,
                                       uint64_t al0ByteOff, uint64_t bl0ByteOff, uint64_t l0cByteOff)
    {
        auto al0Layout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, asc::te::layout_trait_default<AType>>(curM,
                                                                                                                  curK);
        auto al0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(al0ByteOff),
                                              al0Layout);
        auto bl0Layout = asc::te::make_frame_layout<asc::te::zn_layout_ptn, asc::te::layout_trait_default<BType>>(curK,
                                                                                                                  curN);
        auto bl0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(bl0ByteOff),
                                              bl0Layout);
        auto l0cLayout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<16>>(curM, curN);
        auto l0cTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cByteOff),
                                              l0cLayout);
        bool cmatrixInitVal = (iterK == 0 && !isBias_);
        asc::te::mmad_params mmadParams(curM, curN, curK, asc::te::unit_flag_mode::disable, cmatrixInitVal);
        if (isBias_ && iterK == 0) {
            uint64_t nl0Align = CeilAlign(curN, static_cast<uint64_t>(BLOCK_CUBE));
            auto biasL0Layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, nl0Align);
            auto biasL0Tensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(0),
                                                     biasL0Layout);
            asc::te::mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor, biasL0Tensor);
        } else {
            asc::te::mmad(mmadAtom.with(mmadParams), l0cTensor, al0Tensor, bl0Tensor);
        }
    }

    template <typename TensorC, typename CopyL0C2GMT>
    __aicore__ inline void FixL0CToGM(TensorC& gmC, uint64_t l0cBufId, uint64_t curIterBatchL0, uint64_t curM,
                                      uint64_t curN, uint64_t iter1, uint64_t iterMl0, uint64_t iterNl0,
                                      CopyL0C2GMT copyL0C2GM)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(static_cast<uint16_t>(l0cBufId));

        auto l0cOutLayout = asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<16>>(curIterBatchL0,
                                                                                                      curM, curN);
        auto l0cOutTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0c, float>(AscendC::TOTAL_L0C_SIZE / BUFFER_NUM * l0cBufId),
            l0cOutLayout);
        auto gmCSlice = gmC.slice(
            asc::te::make_coord(iter1 * mainIterBatchL0_, asc::te::make_coord(iterMl0 * baseM_, iterNl0 * baseN_)),
            asc::te::make_shape(curIterBatchL0, asc::te::make_shape(curM, curN)));

        asc::te::l0c_to_gm_params fixpParams(asc::te::unit_flag_mode::disable);
        asc::te::copy(copyL0C2GM.with(fixpParams), gmCSlice, l0cOutTensor);
    }

    __aicore__ inline uint64_t GetAl1Count(uint64_t iterBatchL1) const
    {
        if constexpr (A_BROADCAST) {
            if (aBroadcastSingleBatch_) {
                return 1;
            }
        }
        return iterBatchL1;
    }

    __aicore__ inline uint64_t GetBl1Count(uint64_t iterBatchL1) const
    {
        if constexpr (B_BROADCAST) {
            if (bBroadcastSingleBatch_) {
                return 1;
            }
        }
        return iterBatchL1;
    }

    __aicore__ inline uint64_t GetAl0Count(uint64_t iterBatchL0) const
    {
        if constexpr (A_BROADCAST) {
            if (aBroadcastSingleBatch_) {
                return 1;
            }
        }
        return iterBatchL0;
    }

    __aicore__ inline uint64_t GetBl0Count(uint64_t iterBatchL0) const
    {
        if constexpr (B_BROADCAST) {
            if (bBroadcastSingleBatch_) {
                return 1;
            }
        }
        return iterBatchL0;
    }

    __aicore__ inline uint64_t GetAl1BatchOffset(uint64_t iterIdx) const
    {
        if constexpr (A_BROADCAST) {
            if (aBroadcastSingleBatch_) {
                return 0;
            }
        }
        return iterIdx * mainIterBatchL0_;
    }

    __aicore__ inline uint64_t GetBl1BatchOffset(uint64_t iterIdx) const
    {
        if constexpr (B_BROADCAST) {
            if (bBroadcastSingleBatch_) {
                return 0;
            }
        }
        return iterIdx * mainIterBatchL0_;
    }

    __aicore__ inline uint64_t GetABatchIdx(uint64_t batchL1Idx) const
    {
        if constexpr (A_BROADCAST) {
            if (aBroadcastSingleBatch_) {
                return 0;
            }
        }
        return batchL1Idx;
    }

    __aicore__ inline uint64_t GetBBatchIdx(uint64_t batchL1Idx) const
    {
        if constexpr (B_BROADCAST) {
            if (bBroadcastSingleBatch_) {
                return 0;
            }
        }
        return batchL1Idx;
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
