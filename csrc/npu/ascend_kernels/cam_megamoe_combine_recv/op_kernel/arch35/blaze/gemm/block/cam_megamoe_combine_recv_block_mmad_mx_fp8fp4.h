/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Provide shared primitives for quantized MegaMoE GEMM execution.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file block_mmad_mx_fp8fp4.h
 * \brief MMAD block implementation for weight-quant grouped matmul split-M pipeline.
 */
#pragma once

#include "../policy/dispatch_policy_megamoe_combine_recv.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "kernel_basic_intf.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;
using Blaze::Gemm::DOUBLE_BUFFER;
using Blaze::Gemm::FINAL_ACCUMULATION;
using Blaze::Gemm::MX_FP8FP4_L1_K_CONFIG_256;
using Blaze::Gemm::MX_FP8FP4_L1_K_CONFIG_512;
using Blaze::Gemm::MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD;
using Blaze::Gemm::MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD;
using Blaze::Gemm::MX_FP8FP4_SCALE_K_L1_SIZE;
using Blaze::Gemm::MXFP_DIVISOR_SIZE;
using Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
using Blaze::Gemm::NON_FINAL_ACCUMULATION;
using Blaze::Gemm::SYNC_MODE4;

// Macro aliases keep the specialization declaration compact for this dispatch-policy binding.
#define BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS \
    template <class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_, \
              class LayoutC_, class BiasType_, class LayoutBias_>

#define BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION \
    BlockMmad<MatmulMxFp8Fp4DynamicKL1TailResplit, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, \
              LayoutC_, BiasType_, LayoutBias_>

/*!
 * \brief AIC tile-compute unit for one tile inside one group in the weight-quant grouped matmul pipeline.
 *
 * Design reason:
 * - This class handles MMAD compute for a single-group tile only.
 * - AIC does not support direct FP4E2M1 -> FP8E4M3 conversion in the MMAD path.
 * - Therefore B must be preprocessed by prologue first, and this class consumes the converted B tiles.
 *
 * Distinctive behaviors:
 * 1) It synchronizes with prologue through cross-core flags, and the sync semantics must match prologue exactly.
 * 2) It uses dynamic kL1 splitting (kaL1/kbL1) and this policy is aligned with prologue's K-window organization.
 * 3) It uses dynamic kL0 splitting inside each kL1 tile to balance compute and memory movement.
 * 4) A/scaleA/scaleB/B use different transfer granularities by design:
 *    - scaleA/scaleB are moved with MX_FP8FP4_SCALE_K_L1_SIZE = 4096 K-window,
 *    - typical kaL1/kbL1 are 256/512,
 *    - this larger 4096 window is used to organize scale transfer at 128B cacheline-friendly granularity,
 *      improving effective bandwidth and reuse.
 *
 * Key constraints:
 * 1) A must satisfy ND format.
 * 2) B must be prologue-converted ZN format and use the same compute type as AType.
 * 3) Scale layout must satisfy MXFP_DIVISOR_SIZE = 64.
 *
 * When to use:
 * - Use this block on the mxfp8fp4 path when dynamic kL1/kL0 blocking is needed to increase per-tile data workload
 *   and overall throughput.
 */
BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
class BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION {
public:
    using DispatchPolicy = MatmulMxFp8Fp4DynamicKL1TailResplit;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using CType = CType_;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;

    static_assert(asc::te::is_satisfied_ptn_format_v<
                  decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>((__gm__ AType *)0),
                                                LayoutA{}(0UL, 0UL))),
                  asc::te::nd_ext_layout_ptn>);

    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
    };

    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params);

    __aicore__ inline BlockMmad();
    // When copyUbToV1 is true, L0C->UB copies the result to V1 core instead of V0.
    // Only meaningful when C tensor resides in UB; ignored for GM output.
    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(const TensorA &tensorA, const TensorScaleA &tensorScaleA,
                                      const TensorScaleB &tensorScaleB, const TensorC &tensorC,
                                      bool copyUbToV1 = false);
    __aicore__ inline ~BlockMmad();

private:
    struct BlockMmadOffsetParam {
        uint64_t mL1Size;
        uint64_t kaL1Size;
        uint64_t kbL1Size;
        uint64_t nL1Size;
        uint64_t kSize;
        uint64_t scaleKL1Size;
        uint64_t l0KSize;
    };

    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t configuredKbL1Size,
                                             uint64_t &kaL1Size, uint64_t &kbL1Size) const;
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam &param);
    __aicore__ inline void CopyAAndScaleAL1ToL0(uint64_t l1KOffset, uint64_t kbOffset, uint64_t realL0K,
                                                uint64_t realL0ScaleK, const BlockMmadOffsetParam &param);
    template <typename TensorBL1Type>
    __aicore__ inline void CopyBAndScaleBL1ToL0(const TensorBL1Type &tensorBL1, uint64_t l1KOffset, uint64_t kbOffset,
                                                uint64_t realL0K, uint64_t realL0ScaleK,
                                                const BlockMmadOffsetParam &param);
    __aicore__ inline void WaitAMTE1ToMTE2();
    __aicore__ inline void SetMTE1ToMTE2();
    __aicore__ inline void WaitScaleMTE1ToMTE2();
    __aicore__ inline void SetScaleMTE1ToMTE2();
    template <typename TensorA>
    __aicore__ inline void CopyAGmToL1(const TensorA &tensorA, const BlockMmadOffsetParam &param, int64_t kaGmOffset);
    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA &tensorScaleA, const TensorScaleB &tensorScaleB,
                                             const BlockMmadOffsetParam &param, uint64_t kbL1Offset);
    template <typename TensorC>
    __aicore__ inline void CopyCL0c2GmOrUb(const TensorC &tensorC, const BlockMmadOffsetParam &param, bool copyUbToV1);

    using MakeLayoutAL1 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 =
        typename asc::te::frame_layout_format<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 =
        typename asc::te::frame_layout_format<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>;

    using MakeLayoutAL0 = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL0 = asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL0 =
        typename asc::te::frame_layout_format<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL0 =
        typename asc::te::frame_layout_format<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>;

    // Init state used by the Tensor-based operator() path.
    uint64_t k_{1};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{MX_FP8FP4_SCALE_K_L1_SIZE};
    uint64_t baseM_{256};
    uint64_t baseN_{256};
    uint64_t baseK_{128};
    uint64_t kaL1Size_{MX_FP8FP4_L1_K_CONFIG_256};
    uint64_t kbL1Size_{MX_FP8FP4_L1_K_CONFIG_256};
    bool hasInitConfig_{false};

    uint64_t aL1BufIdx_ = 0;
    uint64_t bL1BufIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    template <typename DType, typename Layout>
    using L1TensorType =
        decltype(asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, DType>(0UL), Layout{}(16UL, 16UL)));

    using TensorAL1 = L1TensorType<AType, MakeLayoutAL1>;
    using TensorScaleAL1 = L1TensorType<ScaleAType, MakeLayoutScaleAL1>;
    using TensorScaleBL1 = L1TensorType<ScaleBType, MakeLayoutScaleBL1>;

    TensorAL1 tensorAL1_;
    TensorScaleAL1 tensorScaleAL1_;
    TensorScaleBL1 tensorScaleBL1_;

    // --- sync ---
    // 2 buffer
    static constexpr TEventID eventIdsMte1ToMte2_ = 0;
    // 2 buffer
    static constexpr TEventID eventIdsMxScaleMte1ToMte2_ = 2;
    static constexpr TEventID eventIdMte1ToMte2_ = 4;
    static constexpr TEventID eventIdMte2ToMte1_ = 0;
    static constexpr TEventID eventIdMToMte1_ = 0;
    static constexpr TEventID eventIdMte1ToM_ = 0;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;

    /**
     * L1 512KB Memory Map
     * * Segment 1 [0KB - 256KB]:
     * [0k]    [64k]      [96k]    [128k]                        [256k]
     * |--B B0---|--scA0---|--scB0---|----------- A (Part 1) -------|
     *     (64KB)     (32KB)    (32KB)            (128KB)
     * * Segment 2 [256KB - 512KB]:
     * [256k]                    [384k]        [448k]   [480k]   [512k]
     * |-------- A (Part 2) -------|---- B B1 ----|--scA1--|--scB1--|
     *          (128KB)               (64KB)        (32KB)    (32KB)
     * * Note: A is a contiguous 256KB block spanning the middle of the buffer.
     */
    static constexpr uint64_t SCALE_AL1_OFFSET = 64 * 1024;
    static constexpr uint64_t SCALE_BL1_OFFSET = 96 * 1024;
    static constexpr uint64_t A_L1_OFFSET = 128 * 1024;
    static constexpr uint64_t L1_BUF_OFFSET = 384 * 1024;
    static constexpr uint64_t A_L1_BUF_OFFSET = 128 * 1024;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128 * 1024;

    /**
     * L0 64KB Memory Map (double-buffered B tiles)
     * [0k]      [32KB]    [64KB]
     * |--- B0 ---|--- B1 ---|
     *    (32KB)     (32KB)
     */
    static constexpr uint64_t L0_BUF_OFFSET = 32 * 1024;
};

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyAAndScaleAL1ToL0(uint64_t l1KOffset, uint64_t kbOffset,
                                                                                 uint64_t realL0K,
                                                                                 uint64_t realL0ScaleK,
                                                                                 const BlockMmadOffsetParam &param)
{
    auto layoutAL0 = MakeLayoutAL0{}(param.mL1Size, realL0K);
    auto tensorAL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutAL0);
    auto tensorBlockAL1 = tensorAL1_.slice(asc::te::make_coord(0, (l1KOffset + kbOffset) % param.kaL1Size),
                                           asc::te::make_shape(param.mL1Size, realL0K));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0a{}), tensorAL0, tensorBlockAL1);

    auto layoutScaleAL0 = MakeLayoutScaleAL0{}(param.mL1Size, realL0ScaleK);
    auto tensorScaleAL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>((l0BufIdx_ * L0_BUF_OFFSET) >> 4),
        layoutScaleAL0);
    auto tensorBlockScaleAL1 = tensorScaleAL1_.slice(
        asc::te::make_coord(
            0, CeilDiv(((l1KOffset + kbOffset) % param.scaleKL1Size), MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
        asc::te::make_shape(param.mL1Size, realL0ScaleK));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0scalea{}), tensorScaleAL0, tensorBlockScaleAL1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorBL1Type>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyBAndScaleBL1ToL0(const TensorBL1Type &tensorBL1,
                                                                                 uint64_t l1KOffset, uint64_t kbOffset,
                                                                                 uint64_t realL0K,
                                                                                 uint64_t realL0ScaleK,
                                                                                 const BlockMmadOffsetParam &param)
{
    auto layoutBL0 = MakeLayoutBL0{}(realL0K, param.nL1Size);
    auto tensorBL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutBL0);
    auto tensorBlockBL1 =
        tensorBL1.slice(asc::te::make_coord(l1KOffset, 0), asc::te::make_shape(realL0K, param.nL1Size));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0b{}), tensorBL0, tensorBlockBL1);

    auto layoutScaleBL0 = MakeLayoutScaleBL0{}(realL0ScaleK, param.nL1Size);
    auto tensorScaleBL0 = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>((l0BufIdx_ * L0_BUF_OFFSET) >> 4),
        layoutScaleBL0);
    auto tensorBlockScaleBL1 = tensorScaleBL1_.slice(
        asc::te::make_coord(
            CeilDiv(((l1KOffset + kbOffset) % param.scaleKL1Size), MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
        asc::te::make_shape(realL0ScaleK, param.nL1Size));
    asc::te::copy(asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{}), tensorScaleBL0, tensorBlockScaleBL1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize,
                                                                          const BlockMmadOffsetParam &param)
{
    auto tensorBL1 =
        asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l1, AType>(bL1BufIdx_ * L1_BUF_OFFSET),
                             asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>{}(
                                 kbL1RealSize, param.nL1Size));
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_);
    // Decide the MMAD accumulation mode for this K-tile.
    bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    bool isFirstGmK = kbOffset == 0;
    uint64_t l0KSize = param.l0KSize;
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += l0KSize) {
        bool isLastL1K = l1KOffset + l0KSize >= kbL1RealSize;
        uint64_t realL0K = isLastL1K ? kbL1RealSize - l1KOffset : l0KSize;
        uint64_t realL0ScaleK = CeilDiv(realL0K, MXFP_DIVISOR_SIZE) * 2;
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + l0BufIdx_);

        CopyAAndScaleAL1ToL0(l1KOffset, kbOffset, realL0K, realL0ScaleK, param);
        CopyBAndScaleBL1ToL0(tensorBL1, l1KOffset, kbOffset, realL0K, realL0ScaleK, param);
        auto tensorAL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0BufIdx_ * L0_BUF_OFFSET),
                                 MakeLayoutAL0{}(param.mL1Size, realL0K));
        auto tensorBL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, AType>(l0BufIdx_ * L0_BUF_OFFSET),
                                 MakeLayoutBL0{}(realL0K, param.nL1Size));

        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);

        auto layoutL0C =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(param.mL1Size, param.nL1Size);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);
        asc::te::mmad_params params;
        params.m = static_cast<uint16_t>(param.mL1Size);
        params.k = static_cast<uint16_t>(realL0K);
        params.n = static_cast<uint16_t>(param.nL1Size);
        params.unit_flag =
            (isLastGmK && isLastL1K) ? asc::te::unit_flag_mode::enable_update : asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = isFirstGmK && l1KOffset == 0;
        asc::te::mmad(
            asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}.with(
                params),
            tensorL0C, tensorAL0, tensorBL0);

        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + l0BufIdx_);
        l0BufIdx_ ^= 1;
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitAMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitScaleMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetScaleMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
    scaleL1BufIdx_ ^= 1;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
    aL1BufIdx_ ^= 1;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorA>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyAGmToL1(const TensorA &tensorA,
                                                                        const BlockMmadOffsetParam &param,
                                                                        int64_t kaGmOffset)
{
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
    auto gmBlockA = tensorA.slice(asc::te::make_coord(0, kaGmOffset), asc::te::make_shape(param.mL1Size, kaL1RealSize));
    tensorAL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, AType>(A_L1_OFFSET + aL1BufIdx_ * A_L1_BUF_OFFSET), layoutAL1);
    asc::te::copy(copyGM2L1, tensorAL1_, gmBlockA);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyMxScaleGmToL1(const TensorScaleA &tensorScaleA,
                                                                              const TensorScaleB &tensorScaleB,
                                                                              const BlockMmadOffsetParam &param,
                                                                              uint64_t kbL1Offset)
{
    uint64_t scaleKGmSize = CeilDiv(param.kSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleKL1StandardLen = CeilDiv(param.scaleKL1Size, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + param.scaleKL1Size) > param.kSize ?
                                    CeilDiv(param.kSize - kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE :
                                    scaleKL1StandardLen;
    auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    tensorScaleAL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(SCALE_AL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET),
        layoutScaleAL1);
    auto gmBlockScaleA =
        tensorScaleA.slice(asc::te::make_coord(0, CeilDiv(kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                           asc::te::make_shape(param.mL1Size, scaleKL1RealSize));
    asc::te::copy(CopyScaleGM2L1, tensorScaleAL1_, gmBlockScaleA);

    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    tensorScaleBL1_ = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(SCALE_BL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET),
        layoutScaleBL1);
    auto gmBlockScaleB =
        tensorScaleB.slice(asc::te::make_coord(CeilDiv(kbL1Offset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
                           asc::te::make_shape(scaleKL1RealSize, param.nL1Size));
    asc::te::copy(CopyScaleGM2L1, tensorScaleBL1_, gmBlockScaleB);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CopyCL0c2GmOrUb(const TensorC &tensorC,
                                                                            const BlockMmadOffsetParam &param,
                                                                            bool copyUbToV1)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    auto layoutL0C =
        asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<L0C_C0>>(param.mL1Size, param.nL1Size);
    auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(0), layoutL0C);
    if constexpr (AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>) {
        // C L0C->UB
        auto CopyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{});
        asc::te::copy(CopyL0C2UB.with(asc::te::l0c_to_ub_params(asc::te::unit_flag_mode::enable_update, copyUbToV1)),
                      tensorC, tensorL0C, FP32_64_AS_UINT64);
    } else {
        // C L0C->GM
        auto CopyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        asc::te::copy(CopyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), tensorC,
                      tensorL0C, FP32_64_AS_UINT64);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        SetAicToAiv();
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::~BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::Init(const ProblemShape &problemShape,
                                                                 const BlockShape &l0TileShape,
                                                                 const L1Params &l1Params)
{
    k_ = asc::te::get<IDX_K_IDX>(problemShape);
    kL1_ = l1Params.kL1;
    scaleKL1_ = l1Params.scaleKL1;
    baseM_ = asc::te::get<IDX_M_IDX>(l0TileShape);
    baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
    baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
    CalcDynamicKBlock(baseM_, baseN_, kL1_, kaL1Size_, kbL1Size_);

    aL1BufIdx_ = 0;
    bL1BufIdx_ = 0;
    scaleL1BufIdx_ = 0;
    l0BufIdx_ = 0;
    hasInitConfig_ = true;
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size,
                                                                              uint64_t configuredKbL1Size,
                                                                              uint64_t &kaL1Size,
                                                                              uint64_t &kbL1Size) const
{
    kbL1Size = configuredKbL1Size != 0 ? configuredKbL1Size :
                                         (mL1Size <= MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD &&
                                                  nL1Size <= MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD ?
                                              MX_FP8FP4_L1_K_CONFIG_512 :
                                              MX_FP8FP4_L1_K_CONFIG_256);
    if (mL1Size < nL1Size) {
        uint64_t mL1Align = CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
        /*
         * 小 M 时若直接填满 128 KiB A 槽位，首个 MMAD 会等待一个过深的 A 搬运窗口。
         * 按 N / (2M) 选择 ka 深度，使 A 搬运更细地与连续的 B/prologue 窗口重叠；物理容量仅作为上限。
         */
        uint64_t balancedKaDepth = CeilDiv(nL1Size, mL1Align * DOUBLE_BUFFER);
        uint64_t maxKaDepth = (A_L1_SINGLE_BUF_SIZE / sizeof(AType)) / (mL1Align * kbL1Size);
        uint64_t kaDepth = balancedKaDepth < maxKaDepth ? balancedKaDepth : maxKaDepth;
        kaL1Size = kaDepth * kbL1Size;
    } else {
        kaL1Size = kbL1Size;
    }
}

/*
 * kaL1Size % kbL1Size == 0
 * scaleL1Size % kbL1Size == 0
 */
BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::operator()(const TensorA &tensorA,
                                                                       const TensorScaleA &tensorScaleA,
                                                                       const TensorScaleB &tensorScaleB,
                                                                       const TensorC &tensorC, bool copyUbToV1)
{
    BlockMmadOffsetParam blockParam = {};
    blockParam.mL1Size = asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::row, 1>(tensorC.layout());
    blockParam.nL1Size =
        asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorC.layout());
    if (hasInitConfig_) {
        blockParam.kSize = k_;
        blockParam.scaleKL1Size = scaleKL1_;
        blockParam.l0KSize = baseK_;
        if (blockParam.mL1Size == baseM_ && blockParam.nL1Size == baseN_) {
            blockParam.kaL1Size = kaL1Size_;
            blockParam.kbL1Size = kbL1Size_;
        } else {
            CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, kL1_, blockParam.kaL1Size, blockParam.kbL1Size);
        }
    } else {
        blockParam.kSize =
            asc::te::get_element<asc::te::attr_info::shape, asc::te::attr_info::column, 1>(tensorA.layout());
        blockParam.scaleKL1Size = MX_FP8FP4_SCALE_K_L1_SIZE;
        blockParam.l0KSize = (blockParam.mL1Size <= 128 && blockParam.nL1Size <= 128) ? 256 : 128;
        CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, 0, blockParam.kaL1Size, blockParam.kbL1Size);
    }

    for (uint64_t kbGmOffset = 0; kbGmOffset < blockParam.kSize; kbGmOffset += blockParam.kbL1Size, bL1BufIdx_ ^= 1) {
        uint64_t kbL1RealSize = (kbGmOffset + blockParam.kbL1Size) >= blockParam.kSize ? blockParam.kSize - kbGmOffset :
                                                                                         blockParam.kbL1Size;
        if (kbGmOffset % blockParam.scaleKL1Size == 0) {
            WaitScaleMTE1ToMTE2();
            CopyMxScaleGmToL1(tensorScaleA, tensorScaleB, blockParam, kbGmOffset);
        }

        if (kbGmOffset % blockParam.kaL1Size == 0) {
            WaitAMTE1ToMTE2();
            CopyAGmToL1(tensorA, blockParam, kbGmOffset);
        }

        WaitAivToAic();
        ProcessTileL1(kbGmOffset, kbL1RealSize, blockParam);
        uint64_t nextKbGmOffset = kbGmOffset + blockParam.kbL1Size;
        if (nextKbGmOffset % blockParam.kaL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetMTE1ToMTE2();
        }
        if (nextKbGmOffset % blockParam.scaleKL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetScaleMTE1ToMTE2();
        }
        SetAicToAiv();
    }
    CopyCL0c2GmOrUb(tensorC, blockParam, copyUbToV1);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::WaitAivToAic()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
__aicore__ inline void BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION::SetAicToAiv()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef BLOCK_MMAD_MX_FP8FP4_TEMPLATE_PARAMS
#undef BLOCK_MMAD_MX_FP8FP4_SPECIALIZATION
} // namespace Block
} // namespace Gemm
} // namespace Blaze
