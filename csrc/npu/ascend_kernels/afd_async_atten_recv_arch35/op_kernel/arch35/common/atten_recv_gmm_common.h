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
 * Description: Provide shared gmm common definitions for split Mega execution.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#ifndef MEGA_COMBINE_RECV_GMM_COMMON_H
#define MEGA_COMBINE_RECV_GMM_COMMON_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "atten_recv_constants.h"
#include "atten_recv_types.h"
#include "atten_recv_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
// ### PATCH START: AFD merged source layout
#include "../blaze/gemm/block/afd_async_atten_recv_block_scheduler_swizzle.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "../blaze/gemm/block/afd_async_atten_recv_block_mmad_mx_fp8fp4.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "../blaze/prologue/afd_async_atten_recv_block_prologue_mx_fp8fp4.h"
// ### PATCH END: AFD merged source layout

namespace MegaCombineRecvImpl {

using namespace AscendC;

namespace GmmKernel {

constexpr uint32_t L1_TILE_K = 256U;
constexpr uint32_t SCALE_K_L1_RATE = 2U;

using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 0>; // 3: SwizzleOffset

// 根据数据类型路径选择对应的 BlockMmad 实现。
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy, typename C::ElementAType, typename C::LayoutA,
                                      typename C::ElementBType, typename C::LayoutB, typename C::ElementCType,
                                      typename C::LayoutC, typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type =
        Blaze::Gemm::Block::BlockMmad<typename C::DispatchPolicy,
                                      AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
                                      AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
                                      AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
                                      typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

// 汇总 GMM1/GMM2 在不同量化路径下共用的类型、shape 和 layout 配置。
template <bool IsA8W4, uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false, bool TopkWeightsPrefetch = false,
          bool IsShared = false, bool IsLayered = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
struct Config {
    static constexpr bool IS_SHARED = IsShared;
    static constexpr bool IS_WEIGHT_NZ = IsWeightNZ;
    static constexpr bool TOPK_WEIGHTS_PREFETCH = TopkWeightsPrefetch;
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;
    using BiasType = float;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;
    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();
    static constexpr uint32_t C0_SIZE_BIAS = AuxGetC0Size<BiasType>();

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;

    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy =
        Std::conditional_t<IsA8W4, Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit, Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB =
        Std::conditional_t<IsA8W4, asc::te::zn_layout_ptn,
                           Std::conditional_t<IsWeightNZ, asc::te::zn_layout_ptn, asc::te::dn_ext_layout_ptn>>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, Std::Int<C0_SIZE_C>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, Std::Int<C0_SIZE_BIAS>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t{}, uint32_t{}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t{}, uint32_t{}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t{}, uint32_t{}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t{}, uint32_t{}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t{}, uint32_t{}));
    using LayoutBiasType = decltype(MakeLayoutBias{}(uint32_t{}, uint32_t{}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;
    using BlockPrologue =
        Std::conditional_t<IsA8W4, Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>, void>;

    static __aicore__ inline typename BlockMmad::L1Params MakeL1Params(uint64_t kL1, uint64_t scaleKL1)
    {
        if constexpr (IsA8W4) {
            return typename BlockMmad::L1Params{.kL1 = kL1, .scaleKL1 = scaleKL1};
        } else {
            return typename BlockMmad::L1Params{.kL1 = kL1, .scaleKL1 = scaleKL1, .l1BufNum = 2};
        }
    }

    static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
    {
        if constexpr (IsA8W4) {
            return MakeL1Params(L1_TILE_K, Blaze::Gemm::MX_FP8FP4_SCALE_K_L1_SIZE);
        } else {
            return MakeL1Params(L1_TILE_K, L1_TILE_K * SCALE_K_L1_RATE);
        }
    }

    // 外层 scheduler 继续使用固定的 256 x 256 分块；该配置只控制 BlockMmad 内部基本块和 L1 搬运窗口。
    struct BlockMmadTilingConfig {
        uint32_t tileM;
        uint32_t tileN;
        typename BlockMmad::L1Params l1Params;
    };

    static __aicore__ inline BlockMmadTilingConfig MakeBaselineBlockMmadTilingConfig(uint32_t tileM)
    {
        return BlockMmadTilingConfig{tileM, L1_TILE_N, DefaultL1Params()};
    }

    /*
     * 容量模型（只表示占用关系，不表示 BlockMmad 的实际地址顺序）：
     *
     * +----------------------------------------+----------------------------------------+
     * | buffer 0: A(kL1) + B(kL1)             | buffer 1: A(kL1) + B(kL1)             |
     * |           + scaleA/B(scaleKL1) + free |           + scaleA/B(scaleKL1) + free |
     * +----------------------------------------+----------------------------------------+
     * |<-------------- halfL1Size ------------>|<-------------- halfL1Size ------------>|
     *
     * K 轴覆盖关系：
     * A/B data: |--- kL1 ---|--- kL1 ---|--- kL1 ---| ... | tail |
     * MX scale: |<------------ scaleKL1 ------------>| ... | tail |
     *                         scaleKL1 = scaleFactor * kL1
     *
     * 在 A/B 数据占用之外，用半片 L1 的剩余空间尽可能放大 scaleFactor，从而减少 MX scale 搬运轮数。
     * K 尾块不计入可复用的完整窗口，避免 scaleKL1 跨过实际 K 后再从大窗口内部读取 tail scale。
     */
    static __aicore__ inline typename BlockMmad::L1Params CalcAdaptiveL1Params(uint32_t blockM, uint32_t blockN,
                                                                               uint32_t k)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / 2U;
        constexpr uint64_t scaleGroupK = 64U;
        constexpr uint64_t scaleBytesPerGroup = MXFP_MULTI_BASE_SIZE;
        constexpr uint64_t scaleTransferBytes = 64U * 1024U;
        constexpr uint64_t maxScaleFactor = 127U;
        constexpr uint64_t maxKL1Units = 2U;
        constexpr uint64_t aElementsPerByte = Blaze::Gemm::IsFp4<ElementA>() ? 2U : 1U;
        constexpr uint64_t bElementsPerByte = Blaze::Gemm::IsFp4<ElementB>() ? 2U : 1U;

        const uint64_t blockM64 = blockM;
        const uint64_t blockN64 = blockN;
        /*
         * 先计算一个 256K 基础窗口的字节数。当前只允许 kL1=256/512，且 256 同时是
         * FP4 打包和 MX scale group 的整数倍，因此 512K 窗口的占用严格等于两倍基础窗口。
         */
        const uint64_t dataBytesPerUnit = Ops::Base::CeilDiv(blockM64 * L1_TILE_K, aElementsPerByte) +
                                          Ops::Base::CeilDiv(blockN64 * L1_TILE_K, bElementsPerByte);
        const uint64_t scaleKBytesPerUnit =
            Ops::Base::CeilDiv(static_cast<uint64_t>(L1_TILE_K), scaleGroupK) * scaleBytesPerGroup;
        const uint64_t scaleABytesPerUnit = blockM64 * scaleKBytesPerUnit;
        const uint64_t scaleBBytesPerUnit = blockN64 * scaleKBytesPerUnit;
        const uint64_t scaleBytesPerUnit = scaleABytesPerUnit + scaleBBytesPerUnit;

        /*
         * K>256 时，若两个基础窗口的 A/B 和至少一个 scale 窗口可同时放入半片 L1，
         * 且单侧 scale 不超过 64 KiB 搬运限制，则直接选择 512K；否则使用 256K。
         */
        const uint64_t doubleDataBytes = dataBytesPerUnit * maxKL1Units;
        const uint64_t doubleScaleABytes = scaleABytesPerUnit * maxKL1Units;
        const uint64_t doubleScaleBBytes = scaleBBytesPerUnit * maxKL1Units;
        const uint64_t doubleScaleBytes = scaleBytesPerUnit * maxKL1Units;
        const bool canUseDoubleKL1 = doubleDataBytes + doubleScaleBytes <= halfL1Size &&
                                     doubleScaleABytes <= scaleTransferBytes && doubleScaleBBytes <= scaleTransferBytes;
        const uint64_t kL1Units = canUseDoubleKL1 ? maxKL1Units : 1U;
        const uint64_t kL1 = static_cast<uint64_t>(L1_TILE_K) * kL1Units;
        const uint64_t dataBytes = dataBytesPerUnit * kL1Units;
        const uint64_t scaleABytes = scaleABytesPerUnit * kL1Units;
        const uint64_t scaleBBytes = scaleBBytesPerUnit * kL1Units;
        const uint64_t scaleBytes = scaleBytesPerUnit * kL1Units;

        /*
         * scaleFactor = min(floor((halfL1Size - dataBytes) / scaleBytes),
         *                   floor(64 KiB / scaleABytes), floor(64 KiB / scaleBBytes),
         *                   max(1, floor(K / kL1)), 127)
         * K 约束必须使用 floor 而不是 ceil：尾 K 不足一个完整窗口，必须在下一次 scale
         * 搬运中从窗口起点处理，不能挂在前一个 scaleKL1 窗口的末尾复用。
         */
        uint64_t scaleFactor = (halfL1Size - dataBytes) / scaleBytes;
        const uint64_t scaleFactorByA = scaleTransferBytes / scaleABytes;
        const uint64_t scaleFactorByB = scaleTransferBytes / scaleBBytes;
        const uint64_t fullKWindowCount = static_cast<uint64_t>(k) / kL1;
        const uint64_t scaleFactorByK = fullKWindowCount == 0U ? 1U : fullKWindowCount;
        scaleFactor = Blaze::Gemm::Min(scaleFactor, scaleFactorByA);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, scaleFactorByB);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, scaleFactorByK);
        scaleFactor = Blaze::Gemm::Min(scaleFactor, maxScaleFactor);
        return MakeL1Params(kL1, scaleFactor * kL1);
    }

    static __aicore__ inline BlockMmadTilingConfig MakeAdaptiveBlockMmadTilingConfig(uint32_t m, uint32_t k)
    {
        constexpr uint32_t blockMAlign = 16U;
        // blockM 是 BlockMmad 的容量上限；实际 MMAD 仍使用 actualShape 中的有效 M，不会补算对齐行。
        const uint32_t blockM = Ops::Base::CeilAlign(m, blockMAlign);
        return BlockMmadTilingConfig{blockM, L1_TILE_N, CalcAdaptiveL1Params(blockM, L1_TILE_N, k)};
    }

    static __aicore__ inline BlockMmadTilingConfig SelectBlockMmadTilingConfig(uint32_t m, uint32_t k, uint32_t tileM)
    {
        BlockMmadTilingConfig baselineConfig = MakeBaselineBlockMmadTilingConfig(tileM);
        if (m == 0U || m >= tileM) {
            return baselineConfig;
        }

        if constexpr (IsA8W4) {
            /*
             * kL1=0 让 BlockMmad 与 AIV prologue 按相同规则、基于实际 tile M/N 自动选择 kbL1；
             * BlockMmad 还会独立按实际 M/N 选择 kaL1。scale 使用专用实现固定的 4096K 窗口。
             */
            return BlockMmadTilingConfig{
                tileM, L1_TILE_N,
                typename BlockMmad::L1Params{.kL1 = 0U, .scaleKL1 = Blaze::Gemm::MX_FP8FP4_SCALE_K_L1_SIZE}};
        }

        if constexpr (g_coreType == AscendC::AIC) {
            // K<=256 时所有配置都只搬运一轮，保留基线可避免无收益的 Init。
            if (k <= L1_TILE_K) {
                return baselineConfig;
            }

            BlockMmadTilingConfig adaptiveConfig = MakeAdaptiveBlockMmadTilingConfig(m, k);
            // 512K 一旦可用，必然比 256K 减少 A/B 的 K 搬运轮数。
            if (adaptiveConfig.l1Params.kL1 > baselineConfig.l1Params.kL1) {
                return adaptiveConfig;
            }

            // kL1 不能增大时，仅在扩大 scale 窗口能减少搬运轮数时采用动态配置。
            const uint64_t adaptiveScaleLoops =
                Ops::Base::CeilDiv(static_cast<uint64_t>(k), adaptiveConfig.l1Params.scaleKL1);
            const uint64_t baselineScaleLoops =
                Ops::Base::CeilDiv(static_cast<uint64_t>(k), baselineConfig.l1Params.scaleKL1);
            return adaptiveScaleLoops < baselineScaleLoops ? adaptiveConfig : baselineConfig;
        }
        return baselineConfig;
    }

    struct ProblemConfig {
        using KernelConfig = Config;
        static constexpr bool SOURCE_GMM1_INTERLEAVED = IsGmm1Interleaved;
        static constexpr bool IS_WAVE_FLAG_GRAINED = IsWaveFlagGrained;

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t schedulerN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;
        uint32_t activationTileM = L1_TILE_M_256;
        BlockMmadTilingConfig blockMmadTiling{};
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;
    };

    __aicore__ static inline void FinalizeProblemConfig(ProblemConfig &config, const BlockJobContext &blockJob,
                                                        uint32_t gmm1TileM)
    {
        config.blockNum = blockJob.totalJobs;
        config.blockIdx = blockJob.jobIndex;
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        config.activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : gmm1TileM;
        config.blockMmadTiling = SelectBlockMmadTilingConfig(config.m, config.k, config.tileM);
    }

    __aicore__ static inline ProblemConfig BuildGmm1ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob, uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<N_VALUE>(problemShape);
        config.k = Get<K_VALUE>(problemShape);
        config.outputN = config.n / ACTIVATION_N_HALF;
        config.schedulerN = IsGmm1Interleaved ? config.n : config.outputN;
        config.tileM = gmm1TileM;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline ProblemConfig BuildGmm2ProblemConfig(const ProblemShape &problemShape,
                                                                  const BlockJobContext &blockJob, uint32_t gmm1TileM)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        config.n = Get<K_VALUE>(problemShape);
        config.k = Get<N_VALUE>(problemShape) / ACTIVATION_N_HALF;
        config.outputN = config.n;
        config.schedulerN = config.outputN;
        config.tileM = L1_TILE_M_256;
        FinalizeProblemConfig(config, blockJob, gmm1TileM);
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig &config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        layouts.c = MakeLayoutC{}(config.m, config.n);
        layouts.bias = MakeLayoutBias{}(1U, config.n);
        return layouts;
    }
};

// 统一描述一次 BlockMmad::Init 的可变配置，避免 Context 分散保存并逐字段比较。
// 通用 MX BlockMmad 的 l1BufNum 始终固定为 2，A8W4 L1Params 则没有该字段，因此不纳入运行时状态。
struct BlockMmadInitConfig {
    uint32_t problemK = 0U;
    uint32_t tileM = 0U;
    uint32_t tileN = 0U;
    uint64_t kL1 = 0U;
    uint64_t scaleKL1 = 0U;

    __aicore__ inline bool operator!=(const BlockMmadInitConfig &other) const
    {
        return problemK != other.problemK || tileM != other.tileM || tileN != other.tileN || kL1 != other.kL1 ||
               scaleKL1 != other.scaleKL1;
    }
};

// Wave 流程可持有同一个 BlockMmad；普通路径也复用该结构完成一致的初始化。
template <typename BlockMmad>
struct BlockMmadContext {
    BlockMmad blockMmad;
    BlockMmadInitConfig initConfig{};
    bool initialized = false;
};

// 仅当 BlockMmad::Init 消费的配置发生变化时刷新，problem 的实际 M/N 由每次 MMAD 调用传入。
template <typename MmadContext, typename ProblemConfig>
__aicore__ inline void InitBlockMmad(MmadContext &context, const ProblemConfig &config)
{
    using BlockMmad = decltype(context.blockMmad);
    const auto &tilingConfig = config.blockMmadTiling;
    const BlockMmadInitConfig requestedConfig{config.k, tilingConfig.tileM, tilingConfig.tileN,
                                              tilingConfig.l1Params.kL1, tilingConfig.l1Params.scaleKL1};
    if (!context.initialized || context.initConfig != requestedConfig) {
        typename BlockMmad::BlockShape l0TileShape{tilingConfig.tileM, tilingConfig.tileN, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        constexpr bool enableL0CPingPong = false;
        context.blockMmad.Init(matmulShape, l0TileShape, tilingConfig.l1Params, false, enableL0CPingPong);
        context.initConfig = requestedConfig;
        context.initialized = true;
    }
}

/*
 * CACHE_MODE_DISABLE 会使读取不填入 L2，仅适用于后续不会复用的数据。
 * 当前 problem 必须覆盖完整专家且 M 方向只有一个 tile，B/scaleB 才不会被后续 M tile 或 Wave 再读。
 * 热点专家被切到多个 Wave 时必须保留正常 L2 cache。
 * B 与 scaleB 的物理行跨度不同，二者必须独立判定 128B cache-line 对齐，不能共用一个结果。
 */
template <bool IsWeightNz, typename MatmulConfig, typename TensorB, typename TensorScaleB>
__aicore__ inline void SetWaveWeightL2CacheHint(const typename MatmulConfig::ProblemConfig &config,
                                                bool coversWholeExpert, TensorB &gmB, TensorScaleB &gmScaleB)
{
    constexpr uint64_t cacheLineBytes = 128U;
    // coversWholeExpert 防止跨 Wave 复用；m <= tileM 防止同一 problem 内跨 M tile 复用。
    bool hasNoLaterWeightReuse = coversWholeExpert && config.m <= config.tileM;

    /*
     * NZ 已由分形布局保证物理块对齐。ND/DN(transB) 的连续维是 K，每一行的起点只有在
     * K 方向物理字节数为 128B 整数倍时才允许绕过 L2。统一要求 256 个 logical element：
     * packed FP4 恰为 128B，FP8 为 256B；对 FP8 比最低 128-element 要求更保守，但不会误开 bypass。
     */
    bool bypassWeightL2 = hasNoLaterWeightReuse;
    if constexpr (!IsWeightNz) {
        bypassWeightL2 = bypassWeightL2 && config.k % 256U == 0U;
    }
    gmB.set_l2_cache_hint(bypassWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);

    /*
     * ScaleBDN 的连续维是 N，每个 logical N 含 C0_SIZE_SCALE 个 scale。这里检查完整 N 行跨度；
     * tile-N 固定为 256，其 tile 行跨度天然满足 128B 对齐，因此无需再做动态 baseN 检查。
     */
    constexpr uint64_t scaleTileNStrideBytes = static_cast<uint64_t>(L1_TILE_N) * MatmulConfig::C0_SIZE_SCALE *
                                               sizeof(typename MatmulConfig::ElementMxScaleBType);
    static_assert(scaleTileNStrideBytes % cacheLineBytes == 0U, "ScaleB tile-N stride must be cache-line aligned");
    uint64_t scaleNStrideBytes = static_cast<uint64_t>(config.n) * MatmulConfig::C0_SIZE_SCALE *
                                 sizeof(typename MatmulConfig::ElementMxScaleBType);
    bool bypassScaleL2 = hasNoLaterWeightReuse && scaleNStrideBytes % cacheLineBytes == 0U;
    gmScaleB.set_l2_cache_hint(bypassScaleL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

// 保存 GMM 执行所需的全部 tensor；当 bias 或 C 无实际存储时，调用方传入零地址占位 tensor。
template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
struct GroupMatmulWorkSet {
    Scheduler &scheduler;
    TensorA &gmA;
    TensorB &gmB;
    TensorScaleA &gmScaleA;
    TensorScaleB &gmScaleB;
    TensorBias &gmBias;
    TensorC &gmC;
};

} // namespace GmmKernel
} // namespace MegaCombineRecvImpl

#endif // MEGA_COMBINE_RECV_GMM_COMMON_H
