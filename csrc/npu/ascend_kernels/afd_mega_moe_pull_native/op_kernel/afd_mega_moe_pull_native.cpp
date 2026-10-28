// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_mega_moe_pull_native_tiling.h"
#include "afd_pull_protocol.h"

#include "grouped_matmul_swiglu_quant_v2_layered_utils.h"
#include "gmm_infra/gemm/block/block_swizzle.hpp"

namespace AfdNativeMega {
using namespace AscendC;
using namespace GroupedMatmulDequantSwigluQuant;
using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
using B = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t, false>;
using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using MM = MMImplTypeCustom<A, B, C>;
using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
constexpr uint32_t CUBE_M = 128, CUBE_N = 256;
enum Stage : uint32_t { INPUT, PRE1, G1, ACT, G2 };

template <HardEvent event> __aicore__ inline void Fence() {
    const auto id = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
__aicore__ inline int64_t Load(GM_ADDR address, uint32_t index = 0) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address) + index);
    // Like MC2's gm_dcci, keep every polling load after a fresh cache invalidation.
    __asm__ __volatile__("" ::: "memory");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    __asm__ __volatile__("" ::: "memory");
    return *(reinterpret_cast<__gm__ volatile int64_t *>(address) + index);
}
__aicore__ inline void Store(GM_ADDR address, uint32_t index, int64_t value) {
    GlobalTensor<int64_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address) + index);
    gm.SetValue(0, value);
    __asm__ __volatile__("" ::: "memory");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
    __asm__ __volatile__("" ::: "memory");
}

// One cache line per writer. Empty groups participate in exactly the same sequence.
class Flags {
public:
    __aicore__ explicit Flags(GM_ADDR base) : base_(base) {}
    __aicore__ void Clear() {
        for (uint32_t i = 0; i < Cam::NATIVE_STAGES * (Cam::NATIVE_MAX_AIVS + 1); ++i)
            Store(base_ + 64 * i, 0, 0);
    }
    __aicore__ void Wait(Stage stage, uint32_t seq) const {
        while (Load(Address(stage, 0)) < seq) {}
    }
    __aicore__ void Arrive(Stage stage, uint32_t seq, uint32_t id, uint32_t lanes) const {
        if ASCEND_IS_AIC { Fence<HardEvent::FIX_S>(); }
        if ASCEND_IS_AIV { Fence<HardEvent::MTE3_S>(); }
        Store(Address(stage, id + 1), 0, seq);
        if (id == 0) {
            for (uint32_t i = 1; i < lanes; ++i)
                while (Load(Address(stage, i + 1)) < seq) {}
            Store(Address(stage, 0), 0, seq);
        }
    }
private:
    __aicore__ GM_ADDR Address(Stage stage, uint32_t lane) const {
        return base_ + (stage * (Cam::NATIVE_MAX_AIVS + 1) + lane) * 64;
    }
    GM_ADDR base_;
};

__aicore__ inline Cam::Pull::Shape PullShape(const Cam::AfdMegaMoePullNativeTilingData &t) {
    const auto c = t.config;
    return {uint64_t(c[0] / c[9]), uint64_t(c[1]), uint64_t(c[3]), uint64_t(c[4]),
            uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[9])};
}


struct Tile {
    uint32_t begin, rows, sequence;

};
__aicore__ inline Tile GetTile(uint32_t begin, uint32_t total) {
    const uint32_t left = total - begin;
    return {begin, left < Cam::NATIVE_TILE_ROWS ? left : Cam::NATIVE_TILE_ROWS,
            begin / Cam::NATIVE_TILE_ROWS + 1};
}
__aicore__ inline GM_ADDR Counts(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                  const Cam::AfdMegaMoePullNativeTilingData &t, const Tile &tile) {
    return scratch + w.tileCounts + (tile.sequence - 1) * t.config[6] * sizeof(int64_t);
}
__aicore__ inline void Output(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                             const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                             GM_ADDR biases, int64_t layer, uint32_t id, uint32_t vectors,
                             const Tile &tile, TPipe &pipe) {
    pipe.Reset();
    const uint32_t h = t.config[1];
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, h * 18 + 32);
    auto hi = buf.Get<half>();
    auto lo = buf.GetWithOffset<half>(h, h * 2);
    auto v = buf.GetWithOffset<float>(h, h * 4);
    auto tmp = buf.GetWithOffset<float>(h, h * 8);
    auto bias = buf.GetWithOffset<float>(h, h * 12);
    auto y = buf.GetWithOffset<bfloat16_t>(h, h * 16);
    auto scale = buf.GetWithOffset<float>(1, h * 18);
    GlobalTensor<half> raw;
    GlobalTensor<float> b, s;
    Cam::Pull::Layout layout{};
    Cam::Pull::MakeLayout(PullShape(t), layout);
    const auto source = Load(scratch + w.batch, 1);
    const auto roundBegin = Load(scratch + w.batch, 3);
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + w.c2));
    b.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.config[13], biases));
    s.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale2));
    uint32_t prefix = tile.begin;
    flags.Wait(G2, tile.sequence);
    uint64_t expertBegin = 0;
    for (uint32_t e = 0; e < t.config[6]; ++e) {
        const uint32_t rows = Load(Counts(scratch, w, t, tile), e);
        for (uint32_t r = prefix + rows * id / vectors;
             r < prefix + rows * (id + 1) / vectors; ++r) {
            DataCopy(bias, b[e * h], h);
            DataCopy(hi, raw[2ULL * r * h], h);
            DataCopy(lo, raw[(2ULL * r + 1) * h], h);
            DataCopyPad(scale, s[r], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Fence<HardEvent::MTE2_S>();
            const auto factor = scale.GetValue(0);
            Fence<HardEvent::MTE2_V>();
            Cast(v, hi, RoundMode::CAST_NONE, h);
            Cast(tmp, lo, RoundMode::CAST_NONE, h);
            PipeBarrier<PIPE_V>();
            Muls(v, v, 16.0f, h);
            PipeBarrier<PIPE_V>();
            Add(v, v, tmp, h);
            PipeBarrier<PIPE_V>();
            Add(v, v, bias, h);
            PipeBarrier<PIPE_V>();
            Muls(v, v, factor, h);
            PipeBarrier<PIPE_V>();
            Cast(y, v, RoundMode::CAST_RINT, h);
            if (t.routedScale != 1.0f) {
                PipeBarrier<PIPE_V>();
                Cast(v, y, RoundMode::CAST_NONE, h);
                PipeBarrier<PIPE_V>();
                Muls(v, v, t.routedScale, h);
                PipeBarrier<PIPE_V>();
                Cast(y, v, RoundMode::CAST_RINT, h);
            }
            Fence<HardEvent::V_MTE3>();
            uint64_t block = expertBegin;
            for (uint32_t tp = 0; tp < t.config[9]; ++tp) {
                const uint64_t count = Load(scratch + w.batch, 5 + t.config[9] + tp * t.config[6] + e);
                const uint64_t assignment = roundBegin + r;
                if (assignment >= block && assignment < block + count) {
                    uint64_t offset = Load(scratch + w.batch, 5 + tp);
                    for (uint32_t g = 0; g < e; ++g)
                        offset += Load(scratch + w.batch, 5 + t.config[9] + tp * t.config[6] + g);
                    GlobalTensor<bfloat16_t> remote;
                    remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                        Cam::Pull::Window(source + tp, t.config[7]) + layout.returned));
                    DataCopy(remote[(offset + assignment - block) * h], y, h);
                    break;
                }
                block += count;
            }
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
        prefix += rows;
        for (uint32_t tp = 0; tp < t.config[9]; ++tp)
            expertBegin += Load(scratch + w.batch, 5 + t.config[9] + tp * t.config[6] + e);
    }
    pipe.Reset();
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
        TPipe pipe;
        Flags flags(scratch + w.flags);
        uint32_t id = get_block_idx();
        if ASCEND_IS_AIV {
            id = GetBlockIdx();
            if (id == 0) {
                Cam::Pull::Transport rx(state, PullShape(t), c[7], t.aivNum, pipe, 0);
                rx.Receive(nullptr, nullptr, scratch + w.batch, scratch + w.counts, c[10], false, false);
                const auto modelLayer = rx.Read(scratch + w.batch + 16);
                int64_t slot = -1;
                if (!rx.ReadState(Cam::Pull::FAILURE) && modelLayer >= 0 && modelLayer < t.layerMapSize)
                    slot = rx.Read(layerMap + modelLayer * 8);
                if (slot < 0 || slot >= c[13]) {
                    const auto source = rx.ReadState(Cam::Pull::SOURCE);
                    rx.Fail(source >= 0 && source + c[9] <= c[5] ? source : 0, -4);
                    slot = -1;
                }
                rx.Write(scratch + w.layer, slot);
                const uint32_t total = rx.Read(scratch + w.batch + 32) - rx.Read(scratch + w.batch + 24);
                for (uint32_t begin = 0; begin < total; begin += Cam::NATIVE_TILE_ROWS) {
                    const auto tile = GetTile(begin, total);
                    uint64_t prefix = 0;
                    for (uint32_t e = 0; e < c[6]; ++e) {
                        const auto rows = rx.Read(scratch + w.counts + e * 8);
                        rx.Write(Counts(scratch, w, t, tile) + e * 8,
                                 Cam::Pull::Intersect(prefix, rows, tile.begin, tile.rows).count);
                        prefix += rows;
                    }
                }
                pipe.Reset();
                flags.Clear();
            }
        }
        SyncAll<false>();
        // Validate the frozen return transaction before any fused epilogue writes peer memory.
        if ASCEND_IS_AIV {
            Cam::Pull::Transport check(state, PullShape(t), c[7], t.aivNum, pipe, id);
            check.ValidateReturn(scratch + w.batch, c[10]);
            pipe.Reset();
        }
        SyncAll<false>();
        const auto layer = Load(state, Cam::Pull::FAILURE) ? -1 : Load(scratch + w.layer);
        const uint32_t total = Load(scratch + w.batch, 4) - Load(scratch + w.batch, 3);
        if (layer >= 0 && layer < c[13]) {
            if ASCEND_IS_AIC {
                ComputePipeline(scratch, w, t, flags, w13, w2, s13, s2, layer, total, pipe);
            }
            if ASCEND_IS_AIV {
                if (id < t.commAivNum) PullTiles(scratch, w, t, flags, total, id, pipe);
                else VectorPipeline(scratch, w, t, flags, b13, b2, layer, total, id - t.commAivNum, pipe);
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            Cam::Pull::Transport tx(state, PullShape(t), c[7], t.aivNum, pipe, id);
            if (layer >= 0 && layer < c[13]) tx.Return(nullptr, scratch + w.batch, c[10], false);
            if (id == 0) tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));
        }
    }
}
