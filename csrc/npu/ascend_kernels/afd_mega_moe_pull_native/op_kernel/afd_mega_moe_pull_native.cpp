// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "lib/matmul_intf.h" // CANN generated MIX wrapper calls clearWorkspace.
#include "afd_mega_moe_pull_native_tiling.h"
#include "afd_pull_protocol.h"
#include "layered/layered_tensor_addr.h"

// Use the installed MC2 implementation; do not vendor its template dependency tree.
#if defined(AFD_CANN_CATLASS_92)
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#else
#include "template_linear_algebra_v2/arch/resource.hpp"
#include "template_linear_algebra_v2/gemm/block/block_mmad.hpp"
#include "template_linear_algebra_v2/gemm/block/block_swizzle.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_muls.hpp"
#include "template_linear_algebra_v2/epilogue/block/block_epilogue.hpp"
#endif
#include "utils/block_mmad_w4a4.hpp"
// The installed MC2 epilogue uses unqualified AscendC names inside this namespace.
namespace Catlass::Epilogue::Block { using namespace AscendC; }
#include "afd_mega_moe_activation.h"
#include "afd_swiglu_limit.h"

namespace AfdNativeMega {
using namespace AscendC;
// Shared Catlass aliases. The per-quantization plan header below consumes these.
using Arch = Catlass::Arch::AtlasA2;
using Resource = Catlass::Arch::Resource<Arch>;
using RowMajor = Catlass::layout::RowMajor;
using Nz = Catlass::layout::zN;
using VectorLayout = Catlass::layout::VectorLayout;
using Coord = Catlass::GemmCoord;
} // namespace AfdNativeMega

#include "afd_mega_moe_plan_w4a8.h"
#include "afd_mega_moe_plan_w8a8.h"

namespace AfdNativeMega {
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
    uint32_t active[MAX_EXPERTS];
    uint32_t activeCount = 0, expertCount;
    uint32_t source, offset, total;
    __aicore__ ReceiveRound(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w, uint32_t experts) {
        expertCount = experts;
        source = Load(scratch + w.batch, 1);
        offset = Load(scratch + w.batch, 3);
        total = Load(scratch + w.batch, 4) - offset;
        begin[0] = 0;
        auto counts = reinterpret_cast<__gm__ volatile int64_t *>(scratch + w.counts);
        for (uint32_t e = 0; e < experts; ++e) {
            // Invalidate once per cache line after the preceding receive round reused it.
            if (e % 8 == 0) Load(scratch + w.counts, e);
            begin[e + 1] = begin[e] + counts[e];
            activeCount += begin[e + 1] != begin[e];
        }
        // Keep dense rounds on the original expert order without writing an index list.
        if (activeCount != experts) {
            uint32_t index = 0;
            for (uint32_t e = 0; e < experts; ++e) {
                if (Rows(e)) active[index++] = e;
            }
        }
    }
    __aicore__ uint32_t Rows(uint32_t e) const { return begin[e + 1] - begin[e]; }
    __aicore__ uint32_t Expert(uint32_t index) const {
        return activeCount == expertCount ? index : active[index];
    }
};

// Same Catlass block implementation, preload depth, K shuffle and expert swizzle as MC2.
// The plan supplies the operand element type, GEMM tiling and scheduler; the
// receive/round/flag structure below is identical for every quantization.
template <class Plan>
__aicore__ inline void Compute(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR weights,
    GM_ADDR scales, int64_t layer, bool second, Resource &resource) {
    using Element = typename Plan::Element;
    const uint32_t k = second ? t.config[2] : t.config[1];
    const uint32_t n = second ? t.config[1] : 2 * t.config[2];
    GlobalTensor<Element> a, b;
    GlobalTensor<typename Plan::WeightScale> scale;
    GlobalTensor<typename Plan::OutputElement> out;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(scratch + (second ? w.a2 : w.a1)));
    b.SetGlobalBuffer(GetLayerTensorAddr<Element>(layer, t.config[13], weights));
    scale.SetGlobalBuffer(
        GetLayerTensorAddr<typename Plan::WeightScale>(layer, t.config[13], scales));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ typename Plan::OutputElement *>(scratch + (second ? w.c2 : w.c1)));
    const auto layoutB = Nz::MakeLayout<Element>(k, n);
    typename Plan::Mmad mm(resource);
    typename Plan::Scheduler scheduler;
    uint32_t startCore = 0, syncGroup = 0;
    for (uint32_t index = 0; index < round.activeCount; ++index) {
        const uint32_t e = round.Expert(index);
        const uint32_t rows = Plan::kRows * round.Rows(e);
        if (second) {
            if (index == 0 || SyncExpert(index - 1, round.activeCount)) CrossCoreWaitFlag<2>(ACT_READY);
        } else {
            CrossCoreWaitFlag<2>(index / FLAG_SETS);
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
            const int32_t done = second && job + t.aicNum >= jobs ? int32_t(index) : -1;
            const uint64_t row = uint64_t(Plan::kRows) * round.begin[e] + m;
            if constexpr (Plan::kPackedInt4) {
                mm(scale[e * n + col], VectorLayout{n}, a[row * k], layoutA,
                   expertB[layoutB.GetOffset(Catlass::MatrixCoord{0, col})], layoutB,
                   out[row * n + col], layoutC, actual, ::Callback{}, ::Callback{}, done, 0);
            } else {
                mm(a[row * k], layoutA,
                   expertB[layoutB.GetOffset(Catlass::MatrixCoord{0, col})], layoutB,
                   out[row * n + col], layoutC, actual, ::Callback{}, ::Callback{});
            }
        }
        if (!second && SyncExpert(index, round.activeCount)) {
            mm.SynchronizeBlock();
            if constexpr (Plan::kPackedInt4) mm.Finalize(syncGroup++, GMM1_READY);
            else CrossCoreSetFlag<2, PIPE_FIX>(GMM1_READY);
        }
        if constexpr (!Plan::kPackedInt4) {
            if (second) {
                mm.SynchronizeBlock();
                CrossCoreSetFlag<2, PIPE_FIX>(index / FLAG_SETS);
            }
        }
        startCore = (startCore + jobs) % t.aicNum;
    }
    mm.SynchronizeBlock();
    if constexpr (Plan::kPackedInt4) {
        if (second) mm.Finalize(int32_t(round.activeCount) - 1, 0);
    }
}
} // namespace AfdNativeMega

#include "afd_mega_moe_pull_pipeline.h"

namespace AfdNativeMega {
template <class Plan>
__aicore__ inline void Process(GM_ADDR state, GM_ADDR layerMap, GM_ADDR scratch,
    GM_ADDR w13, GM_ADDR w2, GM_ADDR s13, GM_ADDR s2, GM_ADDR b13, GM_ADDR b2,
    GM_ADDR completion, const Cam::AfdMegaMoePullNativeTilingData &t) {
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
                rx.PrepareRound(scratch + w.batch, scratch + w.counts, c[10]);
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
        const auto layer = Load(state, Cam::Pull::FAILURE) ? -1 : Load(scratch + w.layer);
        if (layer >= 0 && layer < c[13]) {
            const ReceiveRound round(scratch, w, c[6]);
            if (round.total) {
                if ASCEND_IS_AIC {
                    Compute<Plan>(scratch, w, t, round, w13, s13, layer, false, resource);
                    Compute<Plan>(scratch, w, t, round, w2, s2, layer, true, resource);
                }
                if ASCEND_IS_AIV {
                    PullAndPrepare<Plan>(scratch, w, t, round, resource);
                    Activate<Plan>(scratch, w, t, round, Plan::kPackedInt4 ? b13 : s13, layer, resource);
                    Combine<Plan>(scratch, w, t, round, Plan::kPackedInt4 ? b2 : s2, layer, resource);
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV { pipe.Reset(); }
        if ASCEND_IS_AIV {
            if (id == 0) {
                Cam::Pull::Transport tx(state, PullShape(t), c[7], t.aivNum, pipe, 0);
                if (layer >= 0 && layer < c[13]) tx.CompleteRound(scratch + w.batch, c[10]);
                tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));
            }
        }
        // All GMM/epilogue events have drained before scratch and flag slots are reused.
        SyncAll<false>();
        if ASCEND_IS_AIV { pipe.Reset(); }
    } while (!Load(state, Cam::Pull::FAILURE) && Load(state, Cam::Pull::SOURCE) >= 0);
}
} // namespace AfdNativeMega

extern "C" __global__ __aicore__ void afd_mega_moe_pull_native(
    GM_ADDR anchor, GM_ADDR state, GM_ADDR layerMap, GM_ADDR scratch, GM_ADDR w13, GM_ADDR w2,
    GM_ADDR s13, GM_ADDR s2, GM_ADDR b13, GM_ADDR b2, GM_ADDR completion,
    GM_ADDR workspace, GM_ADDR tiling) {
    using namespace AfdNativeMega;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(Cam::AfdMegaMoePullNativeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", Cam::AfdMegaMoePullNativeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", Cam::AfdMegaMoePullNativeTilingData);
    GET_TILING_DATA_WITH_STRUCT(Cam::AfdMegaMoePullNativeTilingData, t, tiling);
    if (TILING_KEY_IS(1)) Process<W4A8Plan>(state, layerMap, scratch, w13, w2, s13, s2, b13, b2, completion, t);
    if (TILING_KEY_IS(2)) Process<W8A8Plan>(state, layerMap, scratch, w13, w2, s13, s2, b13, b2, completion, t);
}
