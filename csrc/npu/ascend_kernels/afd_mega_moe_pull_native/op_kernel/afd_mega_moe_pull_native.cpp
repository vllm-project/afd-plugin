// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "lib/matmul_intf.h" // CANN generated MIX wrapper calls clearWorkspace.
#include "afd_mega_moe_pull_native_tiling.h"
#include "afd_pull_protocol.h"
#include "layered/layered_tensor_addr.h"

// Use the installed MC2 implementation; do not vendor its template dependency tree.
#include "template_linear_algebra_v2/arch/resource.hpp"
#include "template_linear_algebra_v2/gemm/block/block_mmad.hpp"
#include "template_linear_algebra_v2/gemm/block/block_swizzle.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_muls.hpp"
#include "template_linear_algebra_v2/epilogue/block/block_epilogue.hpp"
#include "utils/block_mmad_w4a4.hpp"
// The installed MC2 epilogue uses unqualified AscendC names inside this namespace.
namespace Catlass::Epilogue::Block { using namespace AscendC; }
#include "utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"

namespace AfdNativeMega {
using namespace AscendC;
using Arch = Catlass::Arch::AtlasA2;
using Resource = Catlass::Arch::Resource<Arch>;
using RowMajor = Catlass::layout::RowMajor;
using Nz = Catlass::layout::zN;
using VectorLayout = Catlass::layout::VectorLayout;
using Coord = Catlass::GemmCoord;
using A = Catlass::Gemm::GemmType<int4b_t, RowMajor>;
using B = Catlass::Gemm::GemmType<int4b_t, Nz>;
using C = Catlass::Gemm::GemmType<half, RowMajor>;
using ActOutput = Catlass::Gemm::GemmType<int8_t, RowMajor>;
using ActScale = Catlass::Gemm::GemmType<float, VectorLayout>;
using ActCopy = Catlass::Epilogue::Tile::TileCopy<Arch, C,
    Catlass::Gemm::GemmType<uint64_t, VectorLayout>, ActScale, ActOutput>;
using ActMuls = Catlass::Epilogue::Tile::TileElemWiseMuls<Arch,
    Catlass::Gemm::GemmType<float, RowMajor>, 0>;
using NativeActivation = Catlass::Epilogue::Block::BlockEpilogue<
    Catlass::Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<2>,
    C, ActScale, ActOutput, ActMuls, ActCopy>;
using Copy = Catlass::Gemm::Tile::QuantTileCopy<Arch, A, B, C, void,
    Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
using Policy = Catlass::Gemm::MmadAtlasA2W4A4MatmulPerChannelDequant<1, 2, 2, 2, 1, false, true>;
using Mmad = Catlass::Gemm::Block::BlockMmad<Policy, Catlass::GemmShape<128, 256, 1024>,
    Catlass::GemmShape<128, 256, 256>, A, B, C, void, Copy>;
using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
constexpr uint32_t CUBE_M = 128, CUBE_N = 256;
constexpr uint32_t FLAG_SETS = 15, GMM1_READY = 10, ACT_READY = 11;
constexpr uint32_t MAX_EXPERTS = 128;
constexpr uint32_t ROUTE_UB = 184 * 1024;

template <HardEvent event> __aicore__ inline void Fence() {
    constexpr auto id = EVENT_ID7;
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
__aicore__ inline int64_t Load(GM_ADDR address, uint32_t index = 0) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address) + index);
    __asm__ __volatile__("" ::: "memory");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    __asm__ __volatile__("" ::: "memory");
    return *(reinterpret_cast<__gm__ volatile int64_t *>(address) + index);
}
__aicore__ inline Cam::Pull::Shape PullShape(const Cam::AfdMegaMoePullNativeTilingData &t) {
    const auto c = t.config;
    return {uint64_t(c[0] / c[9]), uint64_t(c[1]), uint64_t(c[3]), uint64_t(c[4]),
            uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[9])};
}
// MC2 W4A8 emits activation groups at the exponentially shrinking expert tail.
__aicore__ inline bool SyncExpert(uint32_t expert, uint32_t experts) {
    const uint32_t tail = experts - expert;
    return (tail & (tail - 1)) == 0;
}
struct ReceiveRound {
    uint32_t begin[MAX_EXPERTS + 1];
    uint32_t source, offset, total;
    __aicore__ ReceiveRound(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w, uint32_t experts) {
        source = Load(scratch + w.batch, 1);
        offset = Load(scratch + w.batch, 3);
        total = Load(scratch + w.batch, 4) - offset;
        begin[0] = 0;
        auto counts = reinterpret_cast<__gm__ volatile int64_t *>(scratch + w.counts);
        for (uint32_t e = 0; e < experts; ++e) {
            // Invalidate once per cache line after the preceding receive round reused it.
            if (e % 8 == 0) Load(scratch + w.counts, e);
            begin[e + 1] = begin[e] + counts[e];
        }
    }
    __aicore__ uint32_t Rows(uint32_t e) const { return begin[e + 1] - begin[e]; }
};

// Same Catlass block implementation, preload depth, K shuffle and expert swizzle as MC2.
__aicore__ inline void Compute(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR weights,
    GM_ADDR scales, int64_t layer, bool second, Resource &resource) {
    const uint32_t k = second ? t.config[2] : t.config[1];
    const uint32_t n = second ? t.config[1] : 2 * t.config[2];
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<uint64_t> scale;
    GlobalTensor<half> out;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(scratch + (second ? w.a2 : w.a1)));
    b.SetGlobalBuffer(GetLayerTensorAddr<int4b_t>(layer, t.config[13], weights));
    scale.SetGlobalBuffer(GetLayerTensorAddr<uint64_t>(layer, t.config[13], scales));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + (second ? w.c2 : w.c1)));
    const auto layoutB = Nz::MakeLayout<int4b_t>(k, n);
    Mmad mm(resource);
    Scheduler scheduler;
    uint32_t startCore = 0, syncGroup = 0;
    for (uint32_t e = 0; e < t.config[6]; ++e) {
        const uint32_t rows = 2 * round.Rows(e);
        if (second) {
            if (e == 0 || SyncExpert(e - 1, t.config[6])) CrossCoreWaitFlag<2>(ACT_READY);
        } else {
            CrossCoreWaitFlag<2>(e / FLAG_SETS);
        }
        scheduler.Update(Coord{rows, n, k}, Catlass::MatrixCoord{CUBE_M, CUBE_N});
        const auto jobs = scheduler.GetCoreLoops();
        const auto first = (get_block_idx() + t.aicNum - startCore) % t.aicNum;
        const RowMajor layoutA{rows, k}, layoutC{rows, n};
        auto expertB = b[uint64_t(e) * k * n];
        expertB.SetL2CacheHint(rows <= CUBE_M ? CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);
        for (uint32_t job = first; job < jobs; job += t.aicNum) {
            const auto coord = scheduler.GetBlockCoord(job);
            const auto actual = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * CUBE_M, col = coord.n() * CUBE_N;
            const int32_t done = second && job + t.aicNum >= jobs ? int32_t(e) : -1;
            mm(scale[e * n + col], VectorLayout{n},
               a[(2ULL * round.begin[e] + m) * k], layoutA,
               expertB[layoutB.GetOffset(Catlass::MatrixCoord{0, col})], layoutB,
               out[(2ULL * round.begin[e] + m) * n + col], layoutC, actual,
               ::Callback{}, ::Callback{}, done, 0);
        }
        if (!second && SyncExpert(e, t.config[6])) {
            mm.SynchronizeBlock();
            mm.Finalize(syncGroup++, GMM1_READY);
        }
        startCore = (startCore + jobs) % t.aicNum;
    }
    mm.SynchronizeBlock();
    if (second) mm.Finalize(t.config[6] - 1, 0);
}
} // namespace AfdNativeMega

#include "afd_mega_moe_pull_pipeline.h"

extern "C" __global__ __aicore__ void afd_mega_moe_pull_native(
    GM_ADDR anchor, GM_ADDR state, GM_ADDR layerMap, GM_ADDR scratch, GM_ADDR w13, GM_ADDR w2,
    GM_ADDR s13, GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, GM_ADDR completion,
    GM_ADDR workspace, GM_ADDR tiling) {
    using namespace AfdNativeMega;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(Cam::AfdMegaMoePullNativeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", Cam::AfdMegaMoePullNativeTilingData);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(Cam::AfdMegaMoePullNativeTilingData, t, tiling);
        const auto c = t.config;
        const Cam::NativeMegaWorkspace w(c[10], c[1], c[2], c[6], c[9]);
        Resource resource;
        TPipe pipe;
        // TPipe seeds M_MTE1 0..2; MC2 must own these events exclusively on AIC.
        if ASCEND_IS_AIC { pipe.Destroy(); }
        const uint32_t id = GetBlockIdx();
        // A fixed workspace drains one selected A transaction, including all receive rounds.
        do {
            if ASCEND_IS_AIV {
                if (id == 0) {
                    Cam::Pull::Transport rx(state, PullShape(t), c[7], t.aivNum, pipe, 0);
                    rx.Receive(nullptr, nullptr, scratch + w.batch, scratch + w.counts, c[10], false, false);
                    const auto modelLayer = rx.Read(scratch + w.batch + 16);
                    int64_t slot = -1;
                    if (rx.ReadState(Cam::Pull::SOURCE) >= 0 && !rx.ReadState(Cam::Pull::FAILURE)) {
                        if (modelLayer >= 0 && modelLayer < t.layerMapSize)
                            slot = rx.Read(layerMap + modelLayer * 8);
                        if (slot < 0 || slot >= c[13]) {
                            const auto source = rx.ReadState(Cam::Pull::SOURCE);
                            rx.Fail(source >= 0 && source + c[9] <= c[5] ? source : 0, -4);
                            slot = -1;
                        }
                    }
                    rx.Write(scratch + w.layer, slot);
                }
                pipe.Reset();
            }
            SyncAll<false>();
            if ASCEND_IS_AIV {
                Cam::Pull::Transport check(state, PullShape(t), c[7], t.aivNum, pipe, id);
                check.ValidateReturn(scratch + w.batch, c[10]);
            }
            SyncAll<false>();
            const auto layer = Load(state, Cam::Pull::FAILURE) ? -1 : Load(scratch + w.layer);
            if ASCEND_IS_AIV { pipe.Reset(); }
            if (layer >= 0 && layer < c[13]) {
                const ReceiveRound round(scratch, w, c[6]);
                if ASCEND_IS_AIC {
                    Compute(scratch, w, t, round, w13, s13, layer, false, resource);
                    Compute(scratch, w, t, round, w2, s2, layer, true, resource);
                }
                if ASCEND_IS_AIV {
                    PullAndPrepare(scratch, w, t, round, resource);
                    Activate(scratch, w, t, round, b13, layer, resource);
                    Combine(scratch, w, t, round, b2, layer, resource);
                }
            }
            SyncAll<false>();
            if ASCEND_IS_AIV { pipe.Reset(); }
            if ASCEND_IS_AIV {
                Cam::Pull::Transport tx(state, PullShape(t), c[7], t.aivNum, pipe, id);
                if (layer >= 0 && layer < c[13]) tx.Return(nullptr, scratch + w.batch, c[10], false);
                if (id == 0) tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));
            }
            // All GMM/epilogue events have drained before scratch and flag slots are reused.
            SyncAll<false>();
            if ASCEND_IS_AIV { pipe.Reset(); }
        } while (!Load(state, Cam::Pull::FAILURE) && Load(state, Cam::Pull::SOURCE) >= 0);
    }
}
