// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_PULL_PIPELINE_H
#define AFD_MEGA_MOE_PULL_PIPELINE_H

namespace AfdNativeMega {
// Dedicated AIVs pull future tiles while the other AIVs and AICs consume published tiles.
__aicore__ inline void PullTiles(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                                uint32_t total, uint32_t id, TPipe &pipe) {
    const auto shape = PullShape(t);
    Cam::Pull::Layout layout{};
    Cam::Pull::MakeLayout(shape, layout);
    pipe.Reset();
    // Receive has frozen this round's metadata; DMA avoids per-field GM cache flushes.
    TBuf<TPosition::VECCALC> metadataBuffer;
    const uint32_t metadataBytes = 8 * Cam::Pull::MetadataWords(shape.tp, shape.experts);
    pipe.InitBuffer(metadataBuffer, Cam::Pull::Align(metadataBytes));
    auto metadata = metadataBuffer.Get<int64_t>();
    GlobalTensor<int64_t> batch;
    batch.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(scratch + w.batch));
    DataCopyPad(metadata, batch, {1U, metadataBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    Fence<HardEvent::MTE2_S>();
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, Cam::NativeAlign(shape.hidden + sizeof(float)));
    auto row = buffer.Get<int8_t>();
    auto scale = buffer.GetWithOffset<float>(1, shape.hidden);
    GlobalTensor<int8_t> x;
    GlobalTensor<float> scales;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.x));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale1));
    const uint32_t source = metadata.GetValue(1);
    const uint64_t roundBegin = metadata.GetValue(3);
    // Keep source-major exclusive payload prefixes, including each source's terminal prefix.
    for (uint32_t s = 0; s < shape.tp; ++s) {
        uint64_t prefix = metadata.GetValue(5 + s);
        for (uint32_t e = 0; e < shape.experts; ++e) {
            const auto index = 5 + shape.tp + s * shape.experts + e;
            const auto count = metadata.GetValue(index);
            metadata.SetValue(index, prefix);
            prefix += count;
        }
        metadata.SetValue(5 + s, prefix);
    }
    uint32_t e = 0, s = 0;
    uint64_t block = 0;
    for (uint32_t begin = 0; begin < total; begin += Cam::NATIVE_TILE_ROWS) {
        const auto tile = GetTile(begin, total);
        const uint64_t end = roundBegin + begin + tile.rows;
        // Walk the expert-major receive order once; retain a partially consumed entry.
        while (e < shape.experts && block < end) {
            const auto index = 5 + shape.tp + s * shape.experts + e;
            const uint64_t prefix = metadata.GetValue(index);
            const uint64_t next = metadata.GetValue(e + 1 < shape.experts ? index + 1 : 5 + s);
            const uint64_t count = next - prefix;
            const auto slice = Cam::Pull::Intersect(block, count, roundBegin + begin, tile.rows);
            if (id < slice.count) {
                GlobalTensor<int8_t> remote;
                remote.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(
                    Cam::Pull::Window(source + s, t.config[7]) + layout.payload));
                for (uint32_t r = id; r < slice.count; r += t.commAivNum) {
                    DataCopyPad(row, remote[(prefix + slice.skip + r) * layout.rowStride],
                        {1U, uint32_t(shape.hidden + sizeof(float)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                    Fence<HardEvent::MTE2_MTE3>();
                    DataCopy(x[(begin + slice.local + r) * shape.hidden], row, shape.hidden);
                    DataCopyPad(scales[begin + slice.local + r], scale, {1U, 4U, 0U, 0U, 0U});
                    Fence<HardEvent::MTE3_MTE2>();
                }
            }
            if (block + count > end) break;
            block += count;
            if (++s == shape.tp) { s = 0; ++e; }
        }
        flags.Arrive(INPUT, tile.sequence, id, t.commAivNum);
    }
    pipe.Reset();
}

// MC2's M128/N256 grid, Nz(9,1) order, and rotating first core across expert jobs.
// The SDK Matmul owns the L1/L0 double buffers; host tiling fixes K1024/K256.
__aicore__ inline void ComputeTile(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                  const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                                  const Tile &tile, GM_ADDR weights, GM_ADDR scales, int64_t layer,
                                  bool second, uint32_t &startCore, MM::MT &mm) {
    flags.Wait(second ? ACT : PRE1, tile.sequence);
    const uint32_t k = second ? t.config[2] : t.config[1];
    const uint32_t n = second ? t.config[1] : 2 * t.config[2];
    GlobalTensor<int4b_t> a, b;
    GlobalTensor<uint64_t> scale;
    GlobalTensor<half> out;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(scratch + (second ? w.a2 : w.a1)));
    b.SetGlobalBuffer(GetLayerTensorAddr<int4b_t>(layer, t.config[13], weights));
    scale.SetGlobalBuffer(GetLayerTensorAddr<uint64_t>(layer, t.config[13], scales));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + (second ? w.c2 : w.c1)));
    Scheduler scheduler;
    uint32_t prefix = tile.begin;
    for (uint32_t e = 0; e < t.config[6]; ++e) {
        const uint32_t rows = Load(Counts(scratch, w, t, tile), e);
        if (!rows) continue;
        scheduler.Update(Catlass::GemmCoord{2 * rows, n, k}, Catlass::MatrixCoord{CUBE_M, CUBE_N});
        const auto jobs = scheduler.GetCoreLoops();
        const auto first = (GetBlockIdx() + t.aicNum - startCore) % t.aicNum;
        mm.SetOrgShape(2 * rows, n, k);
        for (uint32_t job = first; job < jobs; job += t.aicNum) {
            const auto coord = scheduler.GetBlockCoord(job);
            const auto actual = scheduler.GetActualBlockShape(coord);
            const uint32_t m = coord.m() * CUBE_M, col = coord.n() * CUBE_N;
            mm.SetSingleShape(actual.m(), actual.n(), k);
            mm.SetTensorA(a[(2ULL * prefix + m) * k]);
            auto weight = b[uint64_t(e) * k * n + uint64_t(col) * k];
            if (2 * rows <= CUBE_M) weight.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            mm.SetTensorB(weight);
            mm.SetQuantVector(scale[e * n + col]);
            mm.Iterate();
            mm.GetTensorC(out[(2ULL * prefix + m) * n + col]);
        }
        startCore = (startCore + jobs) % t.aicNum;
        prefix += rows;
    }
    flags.Arrive(second ? G2 : G1, tile.sequence, GetBlockIdx(), t.aicNum);
}

__aicore__ inline void ComputePipeline(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                      const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                                      GM_ADDR w13, GM_ADDR w2, GM_ADDR s13, GM_ADDR s2,
                                      int64_t layer, uint32_t total, TPipe &pipe) {
    if (!total) return;
    MM::MT mm;
    TCubeTiling cube = *reinterpret_cast<const TCubeTiling *>(t.cube);
    mm.SetSubBlockIdx(0);
    mm.Init(&cube);
    uint32_t firstCore = 0, secondCore = 0;
    ComputeTile(scratch, w, t, flags, GetTile(0, total), w13, s13, layer, false, firstCore, mm);
    for (uint32_t begin = 0; begin < total; begin += Cam::NATIVE_TILE_ROWS) {
        if (begin + Cam::NATIVE_TILE_ROWS < total)
            ComputeTile(scratch, w, t, flags, GetTile(begin + Cam::NATIVE_TILE_ROWS, total),
                        w13, s13, layer, false, firstCore, mm);
        ComputeTile(scratch, w, t, flags, GetTile(begin, total), w2, s2, layer, true, secondCore, mm);
    }
    mm.End();
    pipe.Reset();
}

// The same signed MSD split as layered: x = 16*high + low + 8.
__aicore__ inline void PackMsd(const LocalTensor<int8_t> &q, const LocalTensor<half> &temp,
                              GlobalTensor<int8_t> &out, uint64_t row, uint32_t k) {
    PipeBarrier<PIPE_V>();
    auto high = temp;
    auto lowBits = temp[k];
    auto low = temp[k + k / 2];
    auto packed = temp[5 * k / 2].ReinterpretCast<int4b_t>();
    auto mask = temp[3 * k].ReinterpretCast<int16_t>();
    Duplicate(mask, static_cast<int16_t>(0x0F0F), 128);
    Cast(high, q, RoundMode::CAST_NONE, k);
    PipeBarrier<PIPE_V>();
    Muls(high, high, static_cast<half>(0.0625f), k);
    PipeBarrier<PIPE_V>();
    Cast(packed, high, RoundMode::CAST_FLOOR, k);
    And(lowBits.ReinterpretCast<int16_t>(), q.ReinterpretCast<int16_t>(), mask,
        128, k / 256, {1, 1, 1, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    Cast(low, lowBits.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, k);
    PipeBarrier<PIPE_V>();
    Adds(low, low, static_cast<half>(-8), k);
    PipeBarrier<PIPE_V>();
    Cast(packed[k], low, RoundMode::CAST_NONE, k);
    Fence<HardEvent::V_MTE3>();
    DataCopy(out[row * k], packed.ReinterpretCast<int8_t>(), k);
    Fence<HardEvent::MTE3_V>();
}

__aicore__ inline void PrepareTile(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                  const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                                  const Tile &tile, uint32_t id, TPipe &pipe) {
    flags.Wait(INPUT, tile.sequence);
    const uint32_t h = t.config[1], vectors = t.aivNum - t.commAivNum;
    pipe.Reset();
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 7 * h + 256);
    auto q = buffer.Get<int8_t>();
    auto temp = buffer.GetWithOffset<half>(3 * h + 128, h);
    GlobalTensor<int8_t> x, out;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.x));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.a1));
    for (uint32_t row = tile.begin + id; row < tile.begin + tile.rows; row += vectors) {
        DataCopy(q, x[row * h], h);
        Fence<HardEvent::MTE2_V>();
        PackMsd(q, temp, out, row, h);
        Fence<HardEvent::V_MTE2>();
    }
    pipe.Reset();
    flags.Arrive(PRE1, tile.sequence, id, vectors);
}

__aicore__ inline void ActivateTile(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                   const Cam::AfdMegaMoePullNativeTilingData &t,
                                   const Tile &tile, GM_ADDR biases, int64_t layer,
                                   uint32_t id, TPipe &pipe) {
    const uint32_t i = t.config[2], n = 2 * i, vectors = t.aivNum - t.commAivNum;
    pipe.Reset();
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 25 * i + 512);
    auto raw = buffer.Get<half>();
    auto value = buffer.GetWithOffset<float>(n, 8 * i);
    auto temp = buffer.GetWithOffset<float>(n, 16 * i);
    auto control = buffer.GetWithOffset<float>(32, 24 * i);
    auto quant = buffer.GetWithOffset<int8_t>(i, 24 * i + 128);
    GlobalTensor<half> input;
    GlobalTensor<float> bias, scales, outputScales;
    GlobalTensor<int8_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scratch + w.c1));
    bias.SetGlobalBuffer(GetLayerTensorAddr<float>(layer, t.config[13], biases));
    scales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale1));
    outputScales.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch + w.scale2));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(scratch + w.a2));
    uint32_t prefix = tile.begin;
    for (uint32_t e = 0; e < t.config[6]; ++e) {
        const uint32_t rows = Load(Counts(scratch, w, t, tile), e);
        for (uint32_t row = prefix + id; row < prefix + rows; row += vectors) {
            DataCopy(raw, input[2ULL * row * n], 2 * n);
            DataCopyPad(control, scales[row], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            Fence<HardEvent::MTE2_S>();
            const auto scale = control.GetValue(0);
            Fence<HardEvent::MTE2_V>();
            Cast(value, raw, RoundMode::CAST_NONE, n);
            Cast(temp, raw[n], RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
            Muls(value, value, 16.0f, n);
            PipeBarrier<PIPE_V>();
            Add(value, value, temp, n);
            Fence<HardEvent::V_MTE2>();
            DataCopy(temp, bias[e * n], n);
            Fence<HardEvent::MTE2_V>();
            Add(value, value, temp, n);
            PipeBarrier<PIPE_V>();
            Muls(value, value, scale, n);
            PipeBarrier<PIPE_V>();
            SwiGLU<float, false>(temp, value[i], value, 1.0f, i);
            PipeBarrier<PIPE_V>();
            DataCopy(value, temp, i);
            PipeBarrier<PIPE_V>();
            Abs(value[i], value, i);
            PipeBarrier<PIPE_V>();
            auto reduce = control[8], reduceTmp = control[16];
            ReduceMaxTemplate(reduce, temp, value[i], reduceTmp, i);
            Fence<HardEvent::V_S>();
            const float quantScale = reduce.GetValue(0) / 127.0f;
            control.SetValue(0, quantScale);
            const float multiplier = quantScale == 0.0f ? 0.0f : 1.0f / quantScale;
            Fence<HardEvent::S_V>();
            Muls(value, value, multiplier, i);
            PipeBarrier<PIPE_V>();
            auto castSpace = temp.ReinterpretCast<int8_t>();
            CastFp32ToInt8Template(quant, value, castSpace, 0, 0, i);
            PackMsd(quant, raw, output, row, i);
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(outputScales[row], control, {1U, 4U, 0U, 0U, 0U});
            Fence<HardEvent::MTE3_MTE2>();
            Fence<HardEvent::V_MTE2>();
        }
        prefix += rows;
    }
    pipe.Reset();
}

__aicore__ inline void VectorPipeline(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
                                     const Cam::AfdMegaMoePullNativeTilingData &t, const Flags &flags,
                                     GM_ADDR b13, GM_ADDR b2, int64_t layer, uint32_t total,
                                     uint32_t id, TPipe &pipe) {
    if (!total) return;
    const auto vectors = t.aivNum - t.commAivNum;
    PrepareTile(scratch, w, t, flags, GetTile(0, total), id, pipe);
    if (total > Cam::NATIVE_TILE_ROWS)
        PrepareTile(scratch, w, t, flags, GetTile(Cam::NATIVE_TILE_ROWS, total), id, pipe);
    for (uint32_t begin = 0; begin < total; begin += Cam::NATIVE_TILE_ROWS) {
        const auto tile = GetTile(begin, total);
        flags.Wait(G1, tile.sequence);
        ActivateTile(scratch, w, t, tile, b13, layer, id, pipe);
        flags.Arrive(ACT, tile.sequence, id, vectors);
        if (begin + 2 * Cam::NATIVE_TILE_ROWS < total)
            PrepareTile(scratch, w, t, flags, GetTile(begin + 2 * Cam::NATIVE_TILE_ROWS, total), id, pipe);
        Output(scratch, w, t, flags, b2, layer, id, vectors, tile, pipe);
    }
}
} // namespace AfdNativeMega
#endif
