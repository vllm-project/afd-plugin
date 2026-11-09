// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_PLAN_W8A8_H
#define AFD_MEGA_MOE_PLAN_W8A8_H

namespace AfdNativeMega {
struct W8A8Plan {
    static constexpr bool kPackedInt4 = false;
    static constexpr uint32_t kRows = 1;
    using Element = int8_t;
    using OutputElement = int32_t;
    using WeightScale = float;
    using OutputScale = bfloat16_t;
    using A = Catlass::Gemm::GemmType<Element, RowMajor>;
    using B = Catlass::Gemm::GemmType<Element, Nz>;
    using C = Catlass::Gemm::GemmType<OutputElement, RowMajor>;
    // Keep the integer accumulator until Vector applies Layered's FP32 scales.
    using Policy = Catlass::Gemm::MmadAtlasA2PreloadAsyncWithCallback<
        1, 2, 2, 2, 1, false, true>;
    using Mmad = Catlass::Gemm::Block::BlockMmad<Policy, Catlass::GemmShape<128, 256, 512>,
        Catlass::GemmShape<128, 256, 128>, A, B, C>;
    using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
};
} // namespace AfdNativeMega
#endif
