// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "kernel_operator.h"
#include "afd_mega_moe_comm_tiling.h"
#include "afd_pull_protocol.h"

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
    pipe.InitBuffer(statsBuf, Cam::Pull::PublisherStatsBytes(
        {uint64_t(c[1] / c[8]), h, k, uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[8])}, cores));
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
    Duplicate(stats, uint32_t(0), columns);
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
    // Each lane owns adjacent expert columns; UB rows are padded independently of GM rows.
    const uint32_t firstExpert = id * (experts / cores) + (id < experts % cores ? id : experts % cores);
    const uint32_t expertCount = experts / cores + (id < experts % cores);
    if (expertCount) {
        const uint32_t stride = Cam::Pull::Align32(4U * expertCount) / 4;
        DataCopyPad(stats, globalStats[c[4] + firstExpert],
            {uint16_t(cores), 4U * expertCount, 4U * (columns - expertCount), 0U, 0U},
            {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        for (uint32_t lane = 1; lane < cores; ++lane)
            for (uint32_t e = 0; e < expertCount; ++e) {
                const auto pos = lane * stride + e;
                stats.SetValue(pos, stats.GetValue(pos) + stats.GetValue(pos - stride));
            }
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(globalStats[c[4] + firstExpert], stats,
            {uint16_t(cores), 4U * expertCount, 0U, 4U * (columns - expertCount), 0U});
        Fence<HardEvent::MTE3_S>();
    }
    if (id == 0) {
        GlobalTensor<int64_t> errors;
        errors.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(state) + lanes);
        DataCopy(stats.ReinterpretCast<int64_t>(), errors, cores * Cam::Pull::WORDS_PER_LINE);
        Fence<HardEvent::MTE2_S>();
        for (uint32_t lane = 0; lane < cores; ++lane) {
            const auto failure = stats.ReinterpretCast<int64_t>().GetValue(lane * Cam::Pull::WORDS_PER_LINE);
            if (!error && failure) error = failure;
        }
        if (error) control.WriteState(Cam::Pull::FAILURE, error);
    }
    SyncAll<true>();
    error = control.ReadState(Cam::Pull::FAILURE);
    DataCopyPad(stats, globalStats[(cores - 1) * columns + c[4]],
        {1U, 4U * experts, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    if (id) DataCopyPad(starts, globalStats[(id - 1) * columns + c[4]],
        {1U, 4U * experts, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    Fence<HardEvent::MTE2_S>();
    uint32_t prefix = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        prefixes.SetValue(e, prefix);
        if (!id) starts.SetValue(e, 0);
        prefix += stats.GetValue(e);
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
        auto counts = offsetsBuf.Get<int64_t>();
        for (uint32_t e = 0; e < experts; ++e) counts.SetValue(e, stats.GetValue(e));
        GlobalTensor<int64_t> publishedCounts;
        publishedCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(local + layout.counts));
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(publishedCounts, counts, {1U, 8U * experts, 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
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

// Publish and Combine partition tokens identically; only the last and preceding
// statistics rows are needed to recover each route's returned-row position.
__aicore__ inline void Combine(GM_ADDR ids, GM_ADDR probabilities, GM_ADDR combined,
                               const Cam::AfdMegaMoeCommTilingData &t,
                               const Cam::Pull::Layout &layout, TPipe &pipe) {
    const auto c = t.config;
    const uint32_t id = GetBlockIdx(), cores = t.aivNum, h = c[2], k = c[3], rows = c[9];
    const uint32_t experts = c[4] * c[6], columns = c[4] + experts;
    const uint32_t begin = rows / cores * id + (id < rows % cores ? id : rows % cores);
    const uint32_t end = begin + rows / cores + (id < rows % cores);
    const auto local = Cam::Pull::Window(c[7], c[7]);
    pipe.Reset();
    TBuf<TPosition::VECCALC> offsetsBuf, routesBuf, mathBuf;
    pipe.InitBuffer(offsetsBuf, 2 * Cam::Pull::Align(4ULL * experts));
    pipe.InitBuffer(routesBuf, 2 * Cam::Pull::Align(4ULL * k));
    pipe.InitBuffer(mathBuf, 14ULL * h);
    auto offsets = offsetsBuf.Get<uint32_t>();
    auto counts = offsetsBuf.GetWithOffset<uint32_t>(experts, Cam::Pull::Align(4ULL * experts));
    auto routes = routesBuf.Get<int32_t>();
    auto weights = routesBuf.GetWithOffset<float>(k, Cam::Pull::Align(4ULL * k));
    auto input = mathBuf.Get<bfloat16_t>();
    auto value = mathBuf.GetWithOffset<float>(h, 2 * h);
    auto product = mathBuf.GetWithOffset<float>(h, 6 * h);
    auto sum = mathBuf.GetWithOffset<float>(h, 10 * h);
    GlobalTensor<uint32_t> stats;
    GlobalTensor<int32_t> globalIds;
    GlobalTensor<float> globalWeights;
    GlobalTensor<bfloat16_t> returned, output;
    stats.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(local));
    globalIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
    globalWeights.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(probabilities));
    returned.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(local + layout.returned));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(combined));
    if (begin < end) {
        DataCopyPad(counts, stats[(cores - 1) * columns + c[4]],
            {1U, 4U * experts, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        if (id) DataCopyPad(offsets, stats[(id - 1) * columns + c[4]],
            {1U, 4U * experts, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        uint32_t prefix = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            offsets.SetValue(e, prefix + (id ? offsets.GetValue(e) : 0));
            prefix += counts.GetValue(e);
        }
        for (uint32_t token = begin; token < end; ++token) {
            DataCopyPad(routes, globalIds[token * k],
                {1U, 4U * k, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            DataCopyPad(weights, globalWeights[token * k],
                {1U, 4U * k, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Duplicate(sum, 0.0f, h);
            Fence<HardEvent::MTE2_S>();
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t expert = routes.GetValue(j), row = offsets.GetValue(expert);
                const float weight = weights.GetValue(j);
                DataCopy(input, returned[uint64_t(row) * h], h);
                Fence<HardEvent::MTE2_V>();
                // Keep the existing FP32 multiply/add and top-k accumulation order.
                Cast(value, input, RoundMode::CAST_NONE, h);
                PipeBarrier<PIPE_V>();
                Muls(product, value, weight, h);
                PipeBarrier<PIPE_V>();
                Add(sum, sum, product, h);
                Fence<HardEvent::V_MTE2>();
                offsets.SetValue(expert, row + 1);
            }
            PipeBarrier<PIPE_V>();
            Cast(input, sum, RoundMode::CAST_RINT, h);
            Fence<HardEvent::V_MTE3>();
            DataCopy(output[uint64_t(token) * h], input, h);
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
            Fence<HardEvent::S_MTE2>();
        }
    }
    // All returned rows have been consumed before the single source window is reusable.
    SyncAll<true>();
    if (id == 0) {
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(local + layout.flags));
        Duplicate(offsets, uint32_t(0), uint32_t(c[4]));
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(flags, offsets, {1U, uint32_t(4 * c[4]), 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
    }
}
} // namespace AfdMegaComm

extern "C" __global__ __aicore__ void afd_mega_moe_comm(
    GM_ADDR x, GM_ADDR ids, GM_ADDR probabilities, GM_ADDR state, GM_ADDR combined,
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
            else if (c[0] == 4 && WaitCombine(state, t, layout, transport)) {
                Combine(ids, probabilities, combined, t, layout, pipe);
            }
        }
        SyncAll<true>();
        // Release phase buffers before allocating the final status transport.
        pipe.Reset();
        if (GetBlockIdx() == 0) {
            Cam::Pull::Transport final(state, shape, c[7], t.aivNum, pipe);
            final.Write(completion, final.ReadState(Cam::Pull::FAILURE));
        }
    }
}
