// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
/**
 * W4A8 execution plan for the native MegaMoE pull operator.
 *
 * Everything the pipeline needs that depends on the weight/activation dtype
 * lives here, so afd_mega_moe_pull_pipeline.h stays dtype-agnostic and a new
 * quantization only adds a sibling plan header.
 *
 * Include order: this header assumes kernel_operator.h and the installed MC2
 * Catlass headers are already included, plus the shared AfdNativeMega aliases
 * (Arch, RowMajor, Nz, VectorLayout) in afd_mega_moe_pull_native.cpp.
 */
#ifndef AFD_MEGA_MOE_PLAN_W4A8_H
#define AFD_MEGA_MOE_PLAN_W4A8_H

namespace AfdNativeMega {

struct W4A8Plan {
    static constexpr bool kPackedInt4 = true;
    static constexpr uint32_t kRows = 2;
    using OutputElement = half;
    using OutputScale = float;
    // Packed INT4 weights. An INT8 activation row of h elements is MSD-split
    // into 2*h INT4 halves, so GMM1 rows are twice the logical row count.
    using Element = AscendC::int4b_t;
    // Per-channel weight scales arrive as INT64-encoded floats.
    using WeightScale = uint64_t;

    using A = Catlass::Gemm::GemmType<Element, RowMajor>;
    using B = Catlass::Gemm::GemmType<Element, Nz>;
    using C = Catlass::Gemm::GemmType<half, RowMajor>;

    using Copy = Catlass::Gemm::Tile::QuantTileCopy<Arch, A, B, C, void,
        Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
    using Policy = Catlass::Gemm::MmadAtlasA2W4A4MatmulPerChannelDequant<
        1, 2, 2, 2, 1, false, true>;
    using Mmad = Catlass::Gemm::Block::BlockMmad<Policy, Catlass::GemmShape<128, 256, 1024>,
        Catlass::GemmShape<128, 256, 256>, A, B, C, void, Copy>;
    using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;

    // Installed MC2 W4A8 epilogue, patched to match Layered's rounding.
    using ActOutput = Catlass::Gemm::GemmType<int8_t, RowMajor>;
    using ActScale = Catlass::Gemm::GemmType<float, VectorLayout>;
    using ActCopy = Catlass::Epilogue::Tile::TileCopy<Arch, C,
        Catlass::Gemm::GemmType<uint64_t, VectorLayout>, ActScale, ActOutput>;
    using ActMuls = Catlass::Epilogue::Tile::TileElemWiseMuls<Arch,
        Catlass::Gemm::GemmType<float, RowMajor>, 0>;
    using NativeActivation = Catlass::Epilogue::Block::BlockEpilogue<
        Catlass::Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<2>,
        C, ActScale, ActOutput, ActMuls, ActCopy>;
};

} // namespace AfdNativeMega
#endif // AFD_MEGA_MOE_PLAN_W4A8_H
