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
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {

/* block schedule policies */
struct KernelMmadWithScaleMx {};                       // Multi-block with Mx scale
struct KernelGroupedMmadWithScaleMx {};                // Grouped multi-block with Mx scale
struct KernelMmadWithScaleMxWithoutBatch {};           // Multi-block with Mx scale, without batch broadcast
struct KernelGroupedMmadWithScaleMxActivationQuant {}; // Grouped Mx matmul with AIV activation and quantization
struct KernelMmadWithScaleMxActivationQuant {};        // Multi-block with Mx scale, AIC+AIV fusion (gelu + mx quant)
struct KernelMmadWithScaleFixpipeQuant {};             // Multi-block with fixpipe quant scale (A8W8 fixpipe)
struct KernelGroupedMmadWithScaleFixpipeQuant {};      // Grouped S8S4 with fixpipe per-channel/per-group scale
struct KernelMmadWithScaleMix {};                      // Multi-block with fixpipe mix scale (WeightNZ)
struct KernelMmadWithScaleMixWithoutBatch {};          // Multi-block with fixpipe mix scale (WeightNZ), without batch
struct KernelMultiBlockStreamK {};                     // Multi-tile transfer with K-axis spliting and caching
struct KernelQbmmMultiBlockStreamK {};                 // QBMM MX StreamK schedule
struct KernelQbmmPertensorMultiBlockStreamK {};        // QBMM per-tensor StreamK schedule
struct KernelMmadMultiBlockBasic {};                   // Multi-tile basic
struct KernelMmadFmmWithScaleAdd {};                   // Fused matmul with scale/add epilogue
struct KernelIterBatchBroadcast {};                    // Multi-tile batchMatmul broadcast + iterbatch
struct KernelMmadMultiBlockBmmBroadcast {};            // Multi-tile batchMatmul broadcast
struct KernelMmadMultiBlockAFullLoad {};               // Multi-tile aFullLoad
struct KernelMmadMultiBlockBFullLoad {};               // Multi-tile fullLoad
struct KernelMmadMultiBlockFixpipeOpti {};             // Multi-tile FixpipeOpti
struct KernelMmadMultiBlockTBMM {};                    // tbmm schedule
struct KernelMmadMultiBlockTQBMM {};                   // tqbmm schedule
struct KernelMixWithWeightPrologue {};                 // Mix matmul with AIV weight preprocessing
struct KernelWqgmmMxMix {};                            // Grouped MX mix kernel with AIV weight preprocessing
struct KernelGmmSwiGluMixMx {};                        // MIX AIC+AIV schedule for GroupedMatmul + SwiGLU + MX quant
struct KernelMatmulEmuSplitWeight {};                  // Double bf16 matmul to simulate fp32 (AIC+AIV)
struct KernelMmadWithScaleMxMix {};                    // Multi-block with Mx scale, epilogue after block mmad
struct KernelGroupedMmadNoQuant {};                    // Grouped multi-block without quantization
enum class MatmulOutputMode : std::uint8_t { OVERWRITE = 0, INPLACE_ADD = 1 };
enum class MatMulL0C2Out : std::uint8_t { ON_THE_FLY = 0, ND_FIXPIPE_1_1 = 1, ND_FIXPIPE_1_2 = 2 };

/**
 * @struct MatmulWithScaleFixpipeQuant
 * @brief Quantized fixpipe matmul with scale and fixpipe dequant (Tensor API / Blaze)
 * @param [in] FullLoadMode_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] AtomicAdd_: whether to enable atomic add on output
 * @param [in] ScheduleType_: kernel schedule
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleFixpipeQuant>
struct MatmulWithScaleFixpipeQuant {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulWithScaleMix
 * @brief Mix fixpipe matmul with scale and fixpipe dequant for WeightNZ (Tensor API / Blaze)
 * @param [in] FullLoadMode_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] AtomicAdd_: whether to enable atomic add on output
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMix>
struct MatmulWithScaleMix {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulWithScaleMx
 * @brief Mx Matrix multiplication with scaleA and scaleB
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMx,
          uint64_t L0C2UBMode_ = L0C2UB_MODE_NONE, uint64_t NonContiguousType_ = 0>
struct MatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
    static constexpr uint64_t L0C2UB_MODE = L0C2UBMode_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulWithWeightQuantMx
 * @brief Weight-only MX matrix multiplication with AIV weight conversion.
 */
struct MatmulWithWeightQuantMx {
    using ScheduleType = KernelMixWithWeightPrologue;
};

/**
 * @struct GroupedMatmulWithWeightQuantMx
 * @brief Dispatch policy for grouped MX weight-quant matrix
 * multiplication.
 */
struct GroupedMatmulWithWeightQuantMx {
    using ScheduleType = KernelWqgmmMxMix;
};

/**
 * @struct GroupedMatmulWithScaleMx
 * @brief Grouped Mx matrix multiplication with scaleA and scaleB
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelGroupedMmadWithScaleMx>
struct GroupedMatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulWithScaleMxL0CPingpong
 * @brief Mx matrix multiplication with L0C ping-pong perf schedule.
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMx>
struct MatmulWithScaleMxL0CPingpong {
    using ScheduleType = ScheduleType_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ATOMIC_ADD = AtomicAdd_;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr MatMulL0C2Out FIXP_OPTI = FixpOpti_;
};

/**
 * @struct MatmulMultiBlockWithStreamKSplitK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 *  @param [in] KernelSchedule_: mmad dispatch policy
 */
template <MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, bool IsSplitSinglecoreK_ = true,
          class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamKSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr MatMulL0C2Out FIXP_OPTI = FixpOpti_;
    static constexpr bool IS_SPLIT_SINGLECORE_K = IsSplitSinglecoreK_;
};

/**
 * @struct MatmulMultiBlockBasic
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: matmul support non-contiguous scene such as: slice, transpose
 * @param [in] OutputMode_: grouped matmul output mode, default is overwrite
 */
template <uint64_t FullLoadMode_ = 0, uint64_t FusedOpType_ = 0, class KernelSchedule_ = KernelMmadMultiBlockBasic,
          uint64_t NonContiguousType_ = 0, MatmulOutputMode OutputMode_ = MatmulOutputMode::OVERWRITE>
struct MatmulMultiBlockBasic {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
    static constexpr MatmulOutputMode OUTPUT_MODE = OutputMode_;
};

/**
 * @struct MatmulEmuSplitWeightPolicy
 * @brief Dual matmul add dispatch policy for AIC+AIV fused dual-weight matmul (Tensor API / Blaze)
 */
struct MatmulEmuSplitWeightPolicy {
    using ScheduleType = KernelMatmulEmuSplitWeight;
};

/**
 * @struct BatchMatmulIterbatchBroadcast
 * @brief Matrix multiplication with batch broadcast, no quant, implemented based on Layout
 * @param [in] ABroadcast: A tensor needs broadcast
 * @param [in] BBroadcast: B tensor needs broadcast
 */
template <bool ABroadcast_, bool BBroadcast_>
struct MatmulIterBatchBroadcast {
    using ScheduleType = KernelIterBatchBroadcast;
    static constexpr bool A_BROADCAST = ABroadcast_;
    static constexpr bool B_BROADCAST = BBroadcast_;
};

/**
 * @struct MatmulMultiBlockBasicSplitK
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: 0 indicates support for continuity
 */
template <uint64_t FullLoadMode_ = 0, bool IsSplitSinglecoreK_ = true,
          class KernelSchedule_ = KernelMmadMultiBlockBasic, uint64_t NonContiguousType_ = 0>
struct MatmulMultiBlockBasicSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_SPLIT_SINGLECORE_K = IsSplitSinglecoreK_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockAFullLoad
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(A_FULL_LOAD_MODE)
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <uint64_t FullLoadMode_ = A_FULL_LOAD_MODE, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockAFullLoad, uint64_t NonContiguousType_ = 0>

struct MatmulMultiBlockAFullLoad {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockBFullLoad
 * @brief Matrix multiplication B full-load structure, with optional fixpipe output mode
 * @param [in] L0C2OutModel_: mode of L0C out mode, default is ON_THE_FLY(L0C2GM)
 * @param [in] FusedOpType_: execute fusion after mmad, default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */

template <uint64_t L0C2OutModel_ = ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockBFullLoad>
struct MatmulMultiBlockBFullLoad {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = B_FULL_LOAD_MODE;
    static constexpr uint64_t L0C2OUT_MODEL = L0C2OutModel_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
};

/**
 * @struct MatmulMultiBlockFixpipeOpti
 * @brief Matrix multiplication fixpipe optimization without B full-load
 * @param [in] FusedOpType_: execute fusion after mmad, default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 */
template <uint64_t L0C2OutModel_ = ON_THE_FLY, uint64_t FusedOpType_ = 0,
          class KernelSchedule_ = KernelMmadMultiBlockFixpipeOpti>
struct MatmulMultiBlockFixpipeOpti {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = NONE_FULL_LOAD_MODE;
    static constexpr uint64_t L0C2OUT_MODEL = L0C2OutModel_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
};

} // namespace Gemm
} // namespace Blaze
