/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Modifications Copyright contributors to the AFD plugin project.
 */
#ifndef AFD_MEGA_MOE_PULL_PIPELINE_H
#define AFD_MEGA_MOE_PULL_PIPELINE_H

// Port of MC2 arch22/a2_a3_copy_helpers.hpp and W4A8 block epilogues.
// AFD changes: frozen source routes, layer descriptors, direct A return windows,
// and the existing BF16 rounding before routed_scale. No all-to-all prologue.
namespace AfdNativeMega {
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

// MC2's signed high nibble and low nibble minus eight; bias compensates the latter.
__aicore__ inline void Msd(LocalTensor<int8_t> q, LocalTensor<half> work,
                           LocalTensor<int4b_t> packed, uint32_t n) {
    auto high = work, lowBits = work[n], low = work[2 * n];
    auto mask = work[3 * n].ReinterpretCast<int16_t>();
    Duplicate(mask, static_cast<int16_t>(0x0f0f), 128);
    Cast(high, q, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Muls(high, high, static_cast<half>(0.0625f), n);
    PipeBarrier<PIPE_V>();
    Cast(packed, high, RoundMode::CAST_FLOOR, n);
    And(lowBits.ReinterpretCast<int16_t>(), q.ReinterpretCast<int16_t>(), mask,
        128, n / 256, {1, 1, 1, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    Cast(low, lowBits.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Adds(low, low, static_cast<half>(-8), n);
    PipeBarrier<PIPE_V>();
    Cast(packed[n], low, RoundMode::CAST_NONE, n);
}

__aicore__ inline void FetchRow(LocalTensor<int8_t> input, GlobalTensor<int8_t> remote,
                               uint32_t hidden, uint32_t stage) {
    WaitFlag<HardEvent::V_MTE2>(stage);
    WaitFlag<HardEvent::MTE3_MTE2>(stage);
    DataCopyPad(input, remote, {1U, hidden + uint32_t(sizeof(float)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(stage);
}

__aicore__ inline void PullAndPrepare(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, Resource &resource) {
    const uint32_t h = t.config[1], experts = t.config[6], tp = t.config[9];
    const auto shape = PullShape(t);
    Cam::Pull::Layout layout{};
    Cam::Pull::MakeLayout(shape, layout);
    const uint32_t metadataBytes = 8 * Cam::Pull::MetadataWords(tp, experts);
    auto meta = resource.ubBuf.GetBufferByByte<int64_t>(0);
    GlobalTensor<int64_t> batch;
    batch.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + w.batch));
    DataCopyPad(meta, batch, {1U, metadataBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    Fence<HardEvent::MTE2_S>();
    uint32_t ub = Cam::Pull::Align(metadataBytes);
    LocalTensor<int8_t> input[2];
    LocalTensor<int4b_t> packed[2];
    for (uint32_t b = 0; b < 2; ++b) {
        input[b] = resource.ubBuf.GetBufferByByte<int8_t>(ub);
        ub += h + Cam::Pull::ROW_PADDING;
        packed[b] = resource.ubBuf.GetBufferByByte<int4b_t>(ub);
        ub += h;
        SetFlag<HardEvent::V_MTE2>(b);
        SetFlag<HardEvent::MTE3_MTE2>(b);
        SetFlag<HardEvent::MTE3_V>(b);
        SetFlag<HardEvent::MTE3_S>(b);
    }
    auto work = resource.ubBuf.GetBufferByByte<half>(ub);
    GlobalTensor<int8_t> out;
    GlobalTensor<float> scales;
    GlobalTensor<uint64_t> routes;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.a1));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale1));
    // The old INT8 staging allocation now holds the per-row return address cache.
    routes.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scratch + w.x));
    uint64_t block = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        const uint32_t rows = round.Rows(e), id = GetBlockIdx();
        const uint32_t perCore = rows / t.aivNum, tail = rows % t.aivNum;
        const uint32_t first = round.begin[e] + id * perCore + Min(id, tail);
        const uint32_t count = perCore + (id < tail);
        for (uint32_t s = 0; s < tp; ++s) {
            const uint32_t index = 5 + tp + s * experts + e;
            const uint64_t sourceRows = meta.GetValue(index);
            const uint64_t prefix = meta.GetValue(5 + s);
            const auto slice = Cam::Pull::Intersect(block, sourceRows, round.offset + first, count);
            if (slice.count) {
                GlobalTensor<int8_t> remote;
                const auto window = Cam::Pull::Window(round.source + s, t.config[7]);
                remote.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(window + layout.payload));
                FetchRow(input[0], remote[(prefix + slice.skip) * layout.rowStride], h, 0);
                for (uint32_t r = 0; r < slice.count; ++r) {
                    const uint32_t stage = r % 2;
                    if (r + 1 < slice.count)
                        FetchRow(input[1 - stage], remote[(prefix + slice.skip + r + 1) * layout.rowStride], h, 1 - stage);
                    WaitFlag<HardEvent::MTE2_V>(stage);
                    WaitFlag<HardEvent::MTE3_V>(stage);
                    Msd(input[stage], work, packed[stage], h);
                    SetFlag<HardEvent::V_MTE2>(stage);
                    SetFlag<HardEvent::V_MTE3>(stage);
                    WaitFlag<HardEvent::V_MTE3>(stage);
                    const uint32_t row = first + slice.local + r;
                    DataCopy(out[uint64_t(row) * h], packed[stage].ReinterpretCast<int8_t>(), h);
                    SetFlag<HardEvent::MTE3_V>(stage);
                    DataCopyPad(scales[row], input[stage][h].ReinterpretCast<float>(), {1U, 4U, 0U, 0U, 0U});
                    auto address = input[stage][h + 32].ReinterpretCast<uint64_t>();
                    // The next source may reuse this scalar address before its DMA drains.
                    WaitFlag<HardEvent::MTE3_S>(stage);
                    address.SetValue(0, reinterpret_cast<uint64_t>(window + layout.returned) +
                        (prefix + slice.skip + r) * h * sizeof(bfloat16_t));
                    Fence<HardEvent::S_MTE3>();
                    DataCopyPad(routes[row], address, {1U, 8U, 0U, 0U, 0U});
                    SetFlag<HardEvent::MTE3_S>(stage);
                    SetFlag<HardEvent::MTE3_MTE2>(stage);
                }
            }
            meta.SetValue(5 + s, prefix + sourceRows);
            block += sourceRows;
        }
        SyncAll<true>();
        CrossCoreSetFlag<2, PIPE_MTE3>(e / FLAG_SETS);
    }
    for (uint32_t b = 0; b < 2; ++b) {
        WaitFlag<HardEvent::V_MTE2>(b);
        WaitFlag<HardEvent::MTE3_MTE2>(b);
        WaitFlag<HardEvent::MTE3_V>(b);
        WaitFlag<HardEvent::MTE3_S>(b);
    }
}

// The current MC2 epilogue's bounded column tiles also support I > UB capacity.
// For a multi-tile row, pass one reduces a global maximum and pass two quantizes.
class Activation {
public:
    static constexpr uint32_t TILE = 3072, SCALE_BATCH = 256;
    __aicore__ Activation(Resource &resource, uint32_t intermediate) : i_(intermediate) {
        uint32_t ub = 0;
        for (uint32_t b = 0; b < 2; ++b) {
            raw_[b] = resource.ubBuf.GetBufferByByte<half>(ub); ub += 8 * TILE;
            bias_[b] = resource.ubBuf.GetBufferByByte<float>(ub); ub += 8 * TILE;
            packed_[b] = resource.ubBuf.GetBufferByByte<int4b_t>(ub); ub += TILE;
            SetFlag<HardEvent::V_MTE2>(b);
            SetFlag<HardEvent::MTE3_V>(b);
        }
        gate_ = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * TILE;
        up_ = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * TILE;
        value_ = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * TILE;
        quant_ = resource.ubBuf.GetBufferByByte<int8_t>(ub); ub += TILE;
        // Msd reuses gate/up/value after the activation has been quantized.
        msdWork_ = gate_.ReinterpretCast<half>();
        maximum_ = resource.ubBuf.GetBufferByByte<float>(ub); ub += 32;
        scales_ = resource.ubBuf.GetBufferByByte<float>(ROUTE_UB);
        for (uint32_t b = 0; b < 2; ++b) {
            scaleOut_[b] = resource.ubBuf.GetBufferByByte<float>(ub); ub += SCALE_BATCH * 4;
            SetFlag<HardEvent::MTE3_S>(b);
        }
    }
    __aicore__ void LoadScales(GlobalTensor<float> source, uint32_t rows) {
        if (rows) DataCopyPad(scales_, source, {1U, 4 * rows, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ void Row(GlobalTensor<half> raw, GlobalTensor<float> bias,
                       GlobalTensor<int8_t> out, LocalTensor<float> scaleOut,
                       uint32_t scaleIndex) {
        const float scale = scales_.GetValue(scaleIndex);
        Duplicate(maximum_, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        Fetch(raw, bias, 0, 0);
        for (uint32_t col = 0, stage = 0; col < i_; col += TILE, stage ^= 1) {
            const auto length = Min(TILE, i_ - col);
            if (col + TILE < i_) Fetch(raw, bias, col + TILE, stage ^ 1);
            Evaluate(stage, length, scale);
            Abs(gate_, value_, length);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(up_, gate_, up_, length, false);
            PipeBarrier<PIPE_V>();
            Max(maximum_, maximum_, up_, 1);
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_S>();
        const float absMax = maximum_.GetValue(0);
        scaleOut.SetValue(scaleIndex, absMax / 127.0f);
        const float multiplier = absMax > 0 ? 127.0f / absMax : 0.0f;
        Fence<HardEvent::S_V>();
        if (i_ <= TILE) {
            Quantize(out, 0, i_, 0, multiplier);
        } else {
            Fetch(raw, bias, 0, 0);
            for (uint32_t col = 0, stage = 0; col < i_; col += TILE, stage ^= 1) {
                const auto length = Min(TILE, i_ - col);
                if (col + TILE < i_) Fetch(raw, bias, col + TILE, stage ^ 1);
                Evaluate(stage, length, scale);
                Quantize(out, col, length, stage, multiplier);
            }
        }
    }
    __aicore__ LocalTensor<float> BeginScale(uint32_t stage) {
        WaitFlag<HardEvent::MTE3_S>(stage);
        return scaleOut_[stage];
    }
    __aicore__ void EndScale(GlobalTensor<float> output, uint32_t stage, uint32_t count) {
        Fence<HardEvent::S_MTE3>();
        if (count) DataCopyPad(output, scaleOut_[stage], {1U, 4 * count, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_S>(stage);
    }
    __aicore__ void Finalize() {
        for (uint32_t b = 0; b < 2; ++b) {
            WaitFlag<HardEvent::V_MTE2>(b);
            WaitFlag<HardEvent::MTE3_V>(b);
            WaitFlag<HardEvent::MTE3_S>(b);
        }
    }
private:
    __aicore__ void Fetch(GlobalTensor<half> raw, GlobalTensor<float> bias, uint32_t col, uint32_t b) {
        WaitFlag<HardEvent::V_MTE2>(b);
        const uint32_t n = Min(TILE, i_ - col);
        for (uint32_t part = 0; part < 4; ++part)
            DataCopy(raw_[b][part * TILE], raw[part * i_ + col], n);
        DataCopy(bias_[b], bias[col], n);
        DataCopy(bias_[b][TILE], bias[i_ + col], n);
        SetFlag<HardEvent::MTE2_V>(b);
    }
    __aicore__ void Evaluate(uint32_t b, uint32_t n, float scale) {
        WaitFlag<HardEvent::MTE2_V>(b);
        Cast(gate_, raw_[b], RoundMode::CAST_NONE, n);
        Cast(value_, raw_[b][2 * TILE], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Muls(gate_, gate_, 16.0f, n);
        PipeBarrier<PIPE_V>();
        Add(gate_, gate_, value_, n);
        PipeBarrier<PIPE_V>();
        Cast(up_, raw_[b][TILE], RoundMode::CAST_NONE, n);
        Cast(value_, raw_[b][3 * TILE], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Muls(up_, up_, 16.0f, n);
        PipeBarrier<PIPE_V>();
        Add(up_, up_, value_, n);
        PipeBarrier<PIPE_V>();
        Add(gate_, gate_, bias_[b], n);
        Add(up_, up_, bias_[b][TILE], n);
        SetFlag<HardEvent::V_MTE2>(b);
        PipeBarrier<PIPE_V>();
        Muls(gate_, gate_, scale, n);
        Muls(up_, up_, scale, n);
        PipeBarrier<PIPE_V>();
        Muls(value_, gate_, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(value_, value_, n);
        PipeBarrier<PIPE_V>();
        Adds(value_, value_, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(value_, gate_, value_, n);
        PipeBarrier<PIPE_V>();
        Mul(value_, value_, up_, n);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ void Quantize(GlobalTensor<int8_t> out, uint32_t col, uint32_t n,
                             uint32_t b, float multiplier) {
        Muls(gate_, value_, multiplier, n);
        PipeBarrier<PIPE_V>();
        auto integer = gate_.ReinterpretCast<int32_t>();
        auto fp16 = gate_.ReinterpretCast<half>();
        Cast(integer, gate_, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1));
        Cast(fp16, integer, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        Cast(quant_, fp16, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(b);
        Msd(quant_, msdWork_, packed_[b], n);
        SetFlag<HardEvent::V_MTE3>(b);
        WaitFlag<HardEvent::V_MTE3>(b);
        DataCopy(out[col / 2], packed_[b].ReinterpretCast<int8_t>(), n / 2);
        DataCopy(out[i_ / 2 + col / 2], packed_[b][n].ReinterpretCast<int8_t>(), n / 2);
        SetFlag<HardEvent::MTE3_V>(b);
    }
    uint32_t i_;
    LocalTensor<half> raw_[2], msdWork_;
    LocalTensor<int4b_t> packed_[2];
    LocalTensor<int8_t> quant_;
    LocalTensor<float> bias_[2], gate_, up_, value_, maximum_, scales_, scaleOut_[2];
};

__aicore__ inline void ActivateNative(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR biases,
    int64_t layer, Resource &resource) {
    const uint32_t i = t.config[2], experts = t.config[6], id = GetBlockIdx();
    // Adapt the selected layer to MC2's one-tensor list and int32 expert counts.
    auto metadata = resource.ubBuf.GetBufferByByte<uint64_t>(0);
    metadata.SetValue(0, 8);
    metadata.SetValue(1, reinterpret_cast<uint64_t>(GetLayerTensorAddr<float>(layer, t.config[13], biases)));
    GM_ADDR descriptor = scratch + w.flags + id * 64;
    GlobalTensor<uint64_t> desc;
    desc.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(descriptor));
    Fence<HardEvent::S_MTE3>();
    DataCopy(desc, metadata, 4);
    GlobalTensor<int32_t> counts;
    counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratch + w.tileCounts));
    if (id == 0) {
        auto localCounts = resource.ubBuf.GetBufferByByte<int32_t>(64);
        for (uint32_t e = 0; e < experts; ++e) localCounts.SetValue(e, round.Rows(e));
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(counts, localCounts, {1U, 4 * experts, 0U, 0U, 0U});
    }
    Fence<HardEvent::MTE3_S>();
    Load(descriptor);
    GlobalTensor<half> raw;
    GlobalTensor<float> s1, s2;
    GlobalTensor<int8_t> out;
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + w.c1));
    s1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale1));
    s2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale2));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.a2));
    NativeActivation activation(resource, 2 * i);
    uint32_t begin = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        if (!SyncExpert(e, experts)) continue;
        CrossCoreWaitFlag<2>(GMM1_READY);
        SyncAll<true>();
        if (begin == 0) {
            for (uint32_t j = 0; j < experts; j += 16) Load(scratch + w.tileCounts + 4 * j);
        }
        const uint32_t rows = round.begin[e + 1] - begin;
        if (rows) activation(raw[4ULL * begin * i], Catlass::MatrixCoord{rows, 2 * i},
            s1[begin], reinterpret_cast<__gm__ float *>(descriptor), out[uint64_t(begin) * i],
            counts, begin, s2[begin], experts, 1, 0, 1, resource, t.aivNum, 0.0f, 2 * i);
        SyncAll<true>();
        CrossCoreSetFlag<2, PIPE_MTE3>(ACT_READY);
        begin = round.begin[e + 1];
    }
    activation.Finalize();
}

__aicore__ inline void Activate(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR biases,
    int64_t layer, Resource &resource) {
    if (t.config[2] <= Activation::TILE) {
        ActivateNative(scratch, w, t, round, biases, layer, resource);
        return;
    }
    const uint32_t i = t.config[2], experts = t.config[6], id = GetBlockIdx();
    GlobalTensor<half> raw;
    GlobalTensor<float> bias, s1, s2;
    GlobalTensor<int8_t> out;
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + w.c1));
    bias.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.config[13], biases));
    s1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale1));
    s2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale2));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.a2));
    Activation activation(resource, i);
    uint32_t groupBegin = 0, expert = 0, group = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        if (!SyncExpert(e, experts)) continue;
        CrossCoreWaitFlag<2>(GMM1_READY);
        SyncAll<true>();
        const uint32_t rows = round.begin[e + 1] - groupBegin;
        const uint32_t each = rows / t.aivNum, tail = rows % t.aivNum;
        const uint32_t first = groupBegin + id * each + Min(id, tail);
        const uint32_t count = each + (id < tail);
        // Bound UB usage even when one expert consumes the entire receive capacity.
        for (uint32_t offset = 0; offset < count; offset += Activation::SCALE_BATCH) {
            const uint32_t length = Min(Activation::SCALE_BATCH, count - offset);
            const uint32_t stage = group++ % 2;
            auto scaleOut = activation.BeginScale(stage);
            activation.LoadScales(s1[first + offset], length);
            for (uint32_t r = 0; r < length; ++r) {
                const uint32_t row = first + offset + r;
                while (expert < e && row >= round.begin[expert + 1]) ++expert;
                activation.Row(raw[4ULL * row * i], bias[2ULL * expert * i],
                               out[uint64_t(row) * i], scaleOut, r);
            }
            activation.EndScale(s2[first + offset], stage, length);
        }
        SyncAll<true>();
        CrossCoreSetFlag<2, PIPE_MTE3>(ACT_READY);
        groupBegin = round.begin[e + 1];
    }
    activation.Finalize();
}

class Output {
public:
    static constexpr uint32_t ROWS = 16, COLS = CUBE_N, ELEMENTS = ROWS * COLS;
    __aicore__ Output(Resource &resource, GM_ADDR routeBase, uint32_t rows) : totalRows_(rows) {
        uint32_t ub = 0;
        for (uint32_t b = 0; b < 2; ++b) {
            hi_[b] = resource.ubBuf.GetBufferByByte<half>(ub); ub += 2 * ELEMENTS;
            lo_[b] = resource.ubBuf.GetBufferByByte<half>(ub); ub += 2 * ELEMENTS;
            bias_[b] = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * COLS;
            y_[b] = resource.ubBuf.GetBufferByByte<bfloat16_t>(ub); ub += 2 * ELEMENTS;
            scale_[b] = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * ROWS;
            scaleRow_[b] = UINT32_MAX;
            SetFlag<HardEvent::V_MTE2>(b);
            SetFlag<HardEvent::V_MTE2>(b + 2);
            SetFlag<HardEvent::MTE3_V>(b);
        }
        value_ = resource.ubBuf.GetBufferByByte<float>(ub); ub += 4 * ELEMENTS;
        temp_ = resource.ubBuf.GetBufferByByte<float>(ub);
        routes_ = resource.ubBuf.GetBufferByByte<uint64_t>(ROUTE_UB);
        routeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(routeBase));
    }
    __aicore__ void Tile(GlobalTensor<half> raw, GlobalTensor<float> bias, GlobalTensor<float> scale,
                         uint32_t row, uint32_t rows, uint32_t col, uint32_t hidden, float routedScale) {
        const uint32_t b = stage_;
        WaitFlag<HardEvent::V_MTE2>(b);
        DataCopyPad(hi_[b], raw[2ULL * row * hidden + col],
            {uint16_t(rows), COLS * 2, (2 * hidden - COLS) * 2, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopyPad(lo_[b], raw[(2ULL * row + 1) * hidden + col],
            {uint16_t(rows), COLS * 2, (2 * hidden - COLS) * 2, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopy(bias_[b], bias[col], COLS);
        SetFlag<HardEvent::MTE2_V>(b);
        WaitFlag<HardEvent::V_MTE2>(b + 2);
        if (scaleRow_[b] != row || scaleRows_[b] != rows) {
            DataCopyPad(scale_[b], scale[row], {1U, 4 * rows, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            scaleRow_[b] = row;
            scaleRows_[b] = rows;
        }
        if (row < routeBegin_ || row + rows > routeBegin_ + routeCount_) {
            routeBegin_ = row;
            routeCount_ = Min(Cam::NATIVE_METADATA_ROWS, totalRows_ - row);
            DataCopyPad(routes_, routeGm_[row], {1U, 8 * routeCount_, 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
        }
        Fence<HardEvent::MTE2_S>();
        WaitFlag<HardEvent::MTE2_V>(b);
        const uint32_t length = rows * COLS;
        Cast(value_, hi_[b], RoundMode::CAST_NONE, length);
        Cast(temp_, lo_[b], RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        Muls(value_, value_, 16.0f, length);
        PipeBarrier<PIPE_V>();
        Add(value_, value_, temp_, length);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) Add(value_[r * COLS], value_[r * COLS], bias_[b], COLS);
        SetFlag<HardEvent::V_MTE2>(b);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r) Muls(value_[r * COLS], value_[r * COLS], scale_[b].GetValue(r), COLS);
        SetFlag<HardEvent::V_MTE2>(b + 2);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(b);
        Cast(y_[b], value_, RoundMode::CAST_RINT, length);
        if (routedScale != 1.0f) {
            PipeBarrier<PIPE_V>();
            Cast(value_, y_[b], RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Muls(value_, value_, routedScale, length);
            PipeBarrier<PIPE_V>();
            Cast(y_[b], value_, RoundMode::CAST_RINT, length);
        }
        SetFlag<HardEvent::V_MTE3>(b);
        WaitFlag<HardEvent::V_MTE3>(b);
        for (uint32_t r = 0; r < rows;) {
            const uint64_t address = routes_.GetValue(row - routeBegin_ + r);
            uint32_t contiguous = 1;
            while (r + contiguous < rows && routes_.GetValue(row - routeBegin_ + r + contiguous) ==
                   address + uint64_t(contiguous) * hidden * sizeof(bfloat16_t)) ++contiguous;
            GlobalTensor<bfloat16_t> remote;
            remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(address));
            DataCopyPad(remote[col], y_[b][r * COLS],
                {uint16_t(contiguous), COLS * 2, 0U, (hidden - COLS) * 2, 0U});
            r += contiguous;
        }
        SetFlag<HardEvent::MTE3_V>(b);
        stage_ ^= 1;
    }
    __aicore__ void Finalize() {
        for (uint32_t b = 0; b < 2; ++b) {
            WaitFlag<HardEvent::V_MTE2>(b);
            WaitFlag<HardEvent::V_MTE2>(b + 2);
            WaitFlag<HardEvent::MTE3_V>(b);
        }
    }
private:
    uint32_t stage_ = 0, scaleRow_[2], scaleRows_[2];
    uint32_t totalRows_, routeBegin_ = 0, routeCount_ = 0;
    LocalTensor<half> hi_[2], lo_[2];
    LocalTensor<float> bias_[2], scale_[2], value_, temp_;
    LocalTensor<bfloat16_t> y_[2];
    LocalTensor<uint64_t> routes_;
    GlobalTensor<uint64_t> routeGm_;
};

__aicore__ inline void Combine(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR biases,
    int64_t layer, Resource &resource) {
    const uint32_t h = t.config[1];
    GlobalTensor<half> raw;
    GlobalTensor<float> bias, scales;
    raw.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + w.c2));
    bias.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.config[13], biases));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale2));
    Output output(resource, scratch + w.x, round.total);
    Scheduler scheduler;
    uint32_t startCore = 0, sync = 0;
    for (uint32_t e = 0; e < t.config[6]; ++e) {
        scheduler.Update(Coord{2 * round.Rows(e), h, uint32_t(t.config[2])},
                         Catlass::MatrixCoord{CUBE_M, CUBE_N});
        const auto jobs = scheduler.GetCoreLoops();
        const auto first = (get_block_idx() + t.aicNum - startCore) % t.aicNum;
        for (uint32_t job = first; job < jobs; job += t.aicNum) {
            const auto coord = scheduler.GetBlockCoord(job);
            const auto actual = scheduler.GetActualBlockShape(coord);
            for (; sync <= e; ++sync) CrossCoreWaitFlag<2>(sync / FLAG_SETS);
            // Identical AIC job ownership; each paired AIV handles 32 expanded MSD rows.
            const uint32_t blocks = (actual.m() + 31) / 32;
            const uint32_t half = blocks / 2;
            const uint32_t begin = get_subblockid() == 0 ? 0 : half;
            const uint32_t end = get_subblockid() == 0 ? half : blocks;
            for (uint32_t b = begin; b < end; ++b) {
                const uint32_t rows = Min(32, actual.m() - b * 32) / 2;
                const uint32_t row = round.begin[e] + coord.m() * (CUBE_M / 2) + b * 16;
                output.Tile(raw, bias[e * h], scales, row, rows, coord.n() * CUBE_N, h, t.routedScale);
            }
        }
        startCore = (startCore + jobs) % t.aicNum;
    }
    for (; sync < t.config[6]; ++sync) CrossCoreWaitFlag<2>(sync / FLAG_SETS);
    output.Finalize();
}
} // namespace AfdNativeMega
#endif
