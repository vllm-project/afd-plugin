// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "afd_mega_moe_comm_tiling.h"
#include "afd_pull_protocol.h"
#include "afd_async_combine_recv.h"

namespace AfdMegaComm {
using namespace AscendC;
using Cam::Pull::Fence;

// Publish only to the local A window. Statistics retain the existing Combine prefix layout.
__aicore__ inline void Publish(GM_ADDR x, GM_ADDR ids, GM_ADDR state,
                               const Cam::AfdMegaMoeCommTilingData &t, TPipe &pipe,
                               Cam::Pull::Transport &control, const Cam::Pull::Layout &layout) {
    const auto c = t.config;
    const uint32_t id = GetBlockIdx(), cores = t.aivNum, h = c[2], k = c[3], experts = c[4] * c[6];
    const uint32_t columns = c[4] + experts, rows = c[9];
    const uint32_t begin = rows / cores * id + (id < rows % cores ? id : rows % cores);
    const uint32_t end = begin + rows / cores + (id < rows % cores ? 1 : 0);
    const auto local = Cam::Pull::Window(c[7], c[7]);
    TBuf<TPosition::VECCALC> statsBuf, offsetsBuf, idsBuf, mathBuf;
    pipe.InitBuffer(statsBuf, Cam::Pull::Align(4ULL * columns * cores));
    pipe.InitBuffer(offsetsBuf, 2 * Cam::Pull::Align(4ULL * experts));
    pipe.InitBuffer(idsBuf, Cam::Pull::Align(4ULL * k));
    pipe.InitBuffer(mathBuf, 11ULL * h + 128);
    auto stats = statsBuf.Get<uint32_t>();
    auto starts = offsetsBuf.Get<uint32_t>();
    auto prefixes = offsetsBuf.GetWithOffset<uint32_t>(experts, Cam::Pull::Align(4ULL * experts));
    auto routes = idsBuf.Get<int32_t>();
    auto input = mathBuf.Get<bfloat16_t>();
    auto value = mathBuf.GetWithOffset<float>(h, 2 * h);
    auto temp = mathBuf.GetWithOffset<float>(h, 6 * h);
    auto quant = mathBuf.GetWithOffset<int8_t>(h + 32, 10 * h);
    auto maximum = mathBuf.GetWithOffset<float>(8, 11 * h + 32);
    GlobalTensor<uint32_t> globalStats;
    GlobalTensor<int32_t> globalIds;
    GlobalTensor<bfloat16_t> globalX;
    GlobalTensor<int8_t> payload;
    globalStats.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(local));
    globalIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
    globalX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
    payload.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(local + layout.payload));
    Duplicate(stats, uint32_t(0), columns * cores);
    Fence<HardEvent::V_S>();
    int64_t error = 0;
    for (uint32_t token = begin; token < end; ++token) {
        DataCopyPad(routes, globalIds[token * k], {1U, 4U * k, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        for (uint32_t j = 0; j < k; ++j) {
            const auto expert = routes.GetValue(j);
            if (expert < 0 || uint32_t(expert) >= experts) error = -1;
            else stats.SetValue(c[4] + expert, stats.GetValue(c[4] + expert) + 1);
        }
        Fence<HardEvent::S_MTE2>();
    }
    const auto lanes = Cam::Pull::LaneErrorsOffset(c[5] / c[8], control.ReadState(Cam::Pull::MAP_SIZE));
    control.Write(state + (lanes + id * Cam::Pull::WORDS_PER_LINE) * 8, error);
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(globalStats[id * columns], stats, {1U, 4U * columns, 0U, 0U, 0U});
    Fence<HardEvent::MTE3_S>();
    SyncAll<true>();
    if (id == 0) {
        DataCopyPad(stats, globalStats, {1U, 4U * columns * cores, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        for (uint32_t lane = 1; lane < cores; ++lane)
            for (uint32_t e = 0; e < experts; ++e) {
                const auto pos = lane * columns + c[4] + e;
                stats.SetValue(pos, stats.GetValue(pos) + stats.GetValue(pos - columns));
            }
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(globalStats, stats, {1U, 4U * columns * cores, 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
        for (uint32_t lane = 0; lane < cores; ++lane) {
            const auto failure = control.Read(state + (lanes + lane * Cam::Pull::WORDS_PER_LINE) * 8);
            if (!error && failure) error = failure;
        }
        if (error) control.WriteState(Cam::Pull::FAILURE, error);
    }
    SyncAll<true>();
    error = control.ReadState(Cam::Pull::FAILURE);
    DataCopyPad(stats, globalStats, {1U, 4U * columns * cores, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    Fence<HardEvent::MTE2_S>();
    uint32_t prefix = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        prefixes.SetValue(e, prefix);
        starts.SetValue(e, id ? stats.GetValue((id - 1) * columns + c[4] + e) : 0);
        prefix += stats.GetValue((cores - 1) * columns + c[4] + e);
    }
    for (uint32_t token = begin; !error && token < end; ++token) {
        DataCopyPad(routes, globalIds[token * k], {1U, 4U * k, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopy(input, globalX[token * h], h);
        Fence<HardEvent::MTE2_V>();
        // Preserve the original Dispatch's FP32 -> INT32 -> FP16 -> INT8 rounding.
        Cast(value, input, RoundMode::CAST_NONE, h);
        PipeBarrier<PIPE_V>();
        Abs(temp, value, h);
        PipeBarrier<PIPE_V>();
        ReduceMax(maximum, temp, temp, h, false);
        PipeBarrier<PIPE_V>();
        Duplicate(temp, 127.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(temp, temp, maximum, 1);
        Fence<HardEvent::V_S>();
        const float multiplier = temp.GetValue(0);
        Fence<HardEvent::S_V>();
        Muls(value, value, multiplier, h);
        PipeBarrier<PIPE_V>();
        Cast(value.ReinterpretCast<int32_t>(), value, RoundMode::CAST_RINT, h);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0f));
        PipeBarrier<PIPE_V>();
        Cast(value.ReinterpretCast<half>(), value.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, h);
        PipeBarrier<PIPE_V>();
        Cast(quant, value.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, h);
        Fence<HardEvent::V_MTE3>();
        quant.ReinterpretCast<float>().SetValue(h / sizeof(float), 1.0f / multiplier);
        Fence<HardEvent::S_MTE3>();
        Fence<HardEvent::MTE2_S>();
        for (uint32_t j = 0; j < k; ++j) {
            const uint32_t expert = routes.GetValue(j);
            const uint32_t row = prefixes.GetValue(expert) + starts.GetValue(expert);
            DataCopy(payload[row * layout.rowStride], quant, h + 32);
            Fence<HardEvent::MTE3_S>();
            starts.SetValue(expert, starts.GetValue(expert) + 1);
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
    SyncAll<true>();
    if (id == 0) {
        const auto seq = control.ReadState(Cam::Pull::ACTIVE_SEQUENCE) + 1;
        control.WriteState(Cam::Pull::ACTIVE_SEQUENCE, seq);
        for (uint32_t e = 0; e < experts; ++e)
            control.Write(local + layout.counts + 8ULL * e, stats.GetValue((cores - 1) * columns + c[4] + e));
        const int64_t values[] = {seq, c[10], rows, prefix, seq, error};
        for (uint32_t field = Cam::Pull::SEQUENCE; field <= Cam::Pull::ERROR; ++field)
            if (field != Cam::Pull::COMMIT)
                control.Write(local + layout.header + field * Cam::Pull::ALIGNMENT, values[field - 1]);
        control.Write(local + layout.header + Cam::Pull::COMMIT * Cam::Pull::ALIGNMENT, seq);
    }
}

__aicore__ inline bool WaitCombine(GM_ADDR state, const Cam::AfdMegaMoeCommTilingData &t,
                                   const Cam::Pull::Layout &layout, Cam::Pull::Transport &control) {
    const auto c = t.config;
    const auto local = Cam::Pull::Window(c[7], c[7]);
    if (GetBlockIdx() == 0) while (true) {
        auto error = control.ReadState(Cam::Pull::FAILURE);
        if (!error) error = control.Read(local + layout.header + Cam::Pull::ERROR * Cam::Pull::ALIGNMENT);
        bool ready = true;
        for (uint32_t f = 0; f < c[4]; ++f) {
            if (!error) error = control.Read(local + layout.errors + f * Cam::Pull::ALIGNMENT);
            if (!uint32_t(control.Read(local + layout.flags + 4ULL * f))) ready = false;
        }
        if (error) {
            control.WriteState(Cam::Pull::FAILURE, error);
            break;
        }
        if (ready) break;
    }
    SyncAll<true>();
    return !control.ReadState(Cam::Pull::FAILURE);
}
} // namespace AfdMegaComm

extern "C" __global__ __aicore__ void afd_mega_moe_comm(
    GM_ADDR x, GM_ADDR ids, GM_ADDR probabilities, GM_ADDR state, GM_ADDR batch,
    GM_ADDR packed, GM_ADDR scales, GM_ADDR metadata, GM_ADDR counts, GM_ADDR combined,
    GM_ADDR completion, GM_ADDR workspace, GM_ADDR tiling) {
    using namespace AfdMegaComm;
    REGISTER_TILING_DEFAULT(Cam::AfdMegaMoeCommTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", Cam::AfdMegaMoeCommTilingData);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(Cam::AfdMegaMoeCommTilingData, t, tiling);
        const auto c = t.config;
        const Cam::Pull::Shape shape{uint64_t(c[1] / c[8]), uint64_t(c[2]), uint64_t(c[3]),
            uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[8])};
        Cam::Pull::Layout layout{};
        Cam::Pull::MakeLayout(shape, layout);
        TPipe pipe;
        {
            Cam::Pull::Transport transport(state, shape, c[7], t.aivNum, pipe);
            SyncAll<true>();
            if (transport.ReadState(Cam::Pull::FAILURE)) {}
            else if (c[0] == 0) transport.Initialize();
            else if (c[0] == 1) Publish(x, ids, state, t, pipe, transport, layout);
            else if (c[0] == 2) transport.Receive(packed, scales, metadata, counts, c[11]);
            else if (c[0] == 3) transport.Return(x, batch, c[9]);
            else if (c[0] == 4 && WaitCombine(state, t, layout, transport)) {
                SyncAll<true>();
                if (c[9]) {
                    Cam::AfdAsyncCombineRecvTilingData combine{};
                    combine.moeDistributeCombineInfo = {0, uint32_t(c[9]), uint32_t(c[2]), uint32_t(c[3]),
                        uint32_t(c[4]), uint32_t(c[5]), uint32_t(c[6]), uint32_t(c[7]),
                        uint32_t(c[4] + c[5]), t.aivNum, t.ubSize, t.windowSize};
                    MoeDistributeCombineRecvImpl::AfdAsyncCombineRecv<bfloat16_t, true> op;
                    op.Init(x, ids, probabilities, combined, workspace, &pipe, &combine, state, 1);
                    op.Process();
                } else if (GetBlockIdx() == 0) {
                    GlobalTensor<uint32_t> flags;
                    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(Cam::Pull::Window(c[7], c[7]) + layout.flags));
                    for (uint32_t f = 0; f < c[4]; ++f) flags.SetValue(f, 0);
                    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(flags);
                }
            }
        }
        SyncAll<true>();
        // Release phase buffers before allocating the final status transport.
        pipe.Reset();
        Cam::Pull::Transport final(state, shape, c[7], t.aivNum, pipe);
        if (GetBlockIdx() == 0) final.Write(completion, final.ReadState(Cam::Pull::FAILURE));
    }
}
