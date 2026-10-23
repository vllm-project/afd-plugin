// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// Experimental one-work primitive. No HCCL, production registration or Dot fallback.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "probe_layout.h"

using namespace AscendC;
using namespace FaaSProbe;

namespace {
template <HardEvent Event>
__aicore__ inline void Fence(TPipe &pipe)
{
    const auto id = pipe.FetchEventID(Event);
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

__aicore__ inline void Flush(const GlobalTensor<int32_t> &tensor)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(tensor);
}

__aicore__ inline void Mark(GM_ADDR status, uint32_t pairs, uint32_t phase)
{
    GlobalTensor<int32_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(status));
    const uint32_t kind = g_coreType == AscendC::AIC ? 0 : 1;
    const uint32_t offset = (kind * pairs + GetBlockIdx()) * LINE_WORDS;
    out.SetValue(offset, kind);
    out.SetValue(offset + 1, GetBlockIdx());
    out.SetValue(offset + 2, GetSubBlockIdx());
    out.SetValue(offset + 3, GetSubBlockNum());
    out.SetValue(offset + 4, GetBlockNum());
    out.SetValue(offset + 5, phase);
    Flush(out);
}

__aicore__ inline void Group(GM_ADDR controls, GM_ADDR ids, GlobalTensor<int32_t> &meta)
{
    GlobalTensor<int32_t> control, expertIds;
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(controls));
    expertIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
    const int32_t tokens = control.GetValue(1);
    const int32_t experts = control.GetValue(2);
    int32_t code = control.GetValue(0);
    if (tokens < 0 || tokens > MAX_TOKENS || experts <= 0 || experts > MAX_EXPERTS) {
        code = INVALID;
    }
    meta.SetValue(CODE, code);
    meta.SetValue(ROWS, 0);
    meta.SetValue(EXPERTS, experts);
    meta.SetValue(TOKENS, tokens);
    if (code != OK) {
        Flush(meta);
        return;
    }
    for (uint32_t e = 0; e < experts; ++e) {
        meta.SetValue(COUNTS + e, 0);
    }
    for (uint32_t route = 0; route < tokens * TOP_K; ++route) {
        const int32_t expert = expertIds.GetValue(route);
        if (expert >= 0 && expert < experts) {
            meta.SetValue(COUNTS + expert, meta.GetValue(COUNTS + expert) + 1);
        }
    }
    uint32_t prefix = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        meta.SetValue(PREFIX + e, prefix);
        meta.SetValue(CURSORS + e, prefix);
        prefix += ((meta.GetValue(COUNTS + e) + TILE_M - 1) / TILE_M) * TILE_M;
    }
    meta.SetValue(PREFIX + experts, prefix);
    meta.SetValue(ROWS, prefix);
    for (uint32_t row = 0; row < prefix; ++row) {
        meta.SetValue(ROUTES + row, -1);
    }
    for (uint32_t route = 0; route < tokens * TOP_K; ++route) {
        const int32_t expert = expertIds.GetValue(route);
        if (expert >= 0 && expert < experts) {
            const uint32_t cursor = meta.GetValue(CURSORS + expert);
            meta.SetValue(ROUTES + cursor, route);
            meta.SetValue(CURSORS + expert, cursor + 1);
        }
    }
    Flush(meta);
}

class VectorStages {
public:
    __aicore__ inline VectorStages(TPipe &pipe, GM_ADDR scratch) : pipe_(pipe)
    {
        pipe_.InitBuffer(copyBuf_, 2 * I * sizeof(bfloat16_t));
        pipe_.InitBuffer(gateBuf_, I * sizeof(float));
        pipe_.InitBuffer(upBuf_, I * sizeof(float));
        pipe_.InitBuffer(tempBuf_, I * sizeof(float));
        copy_ = copyBuf_.Get<bfloat16_t>();
        gate_ = gateBuf_.Get<float>();
        up_ = upBuf_.Get<float>();
        temp_ = tempBuf_.Get<float>();
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + PACKED));
        gateUp_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + GATE_UP));
        activation_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + ACTIVATION));
        down_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch + DOWN));
    }

    __aicore__ inline void Pack(GM_ADDR x, GM_ADDR y, const GlobalTensor<int32_t> &meta, uint32_t pairs)
    {
        GlobalTensor<bfloat16_t> input, output;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y));
        for (uint32_t row = GetBlockIdx(); row < meta.GetValue(ROWS); row += pairs * GetSubBlockNum()) {
            const int32_t route = meta.GetValue(ROUTES + row);
            if (route >= 0) {
                DataCopy(copy_, input[(route / TOP_K) * H], H);
                Fence<HardEvent::MTE2_MTE3>(pipe_);
            } else {
                Duplicate(copy_, static_cast<bfloat16_t>(0), H);
                Fence<HardEvent::V_MTE3>(pipe_);
            }
            DataCopy(packed_[row * H], copy_, H);
            Fence<HardEvent::MTE3_MTE2>(pipe_);
            Fence<HardEvent::MTE3_V>(pipe_);
        }
        Duplicate(copy_, static_cast<bfloat16_t>(0), H);
        Fence<HardEvent::V_MTE3>(pipe_);
        for (uint32_t route = GetBlockIdx(); route < meta.GetValue(TOKENS) * TOP_K; route += pairs * GetSubBlockNum()) {
            DataCopy(output[route * H], copy_, H);
        }
        Fence<HardEvent::MTE3_S>(pipe_);
    }

    __aicore__ inline void Activate(const GlobalTensor<int32_t> &meta, uint32_t pairs)
    {
        for (uint32_t row = GetBlockIdx(); row < meta.GetValue(ROWS); row += pairs * GetSubBlockNum()) {
            DataCopy(copy_, gateUp_[row * 2 * I], 2 * I);
            Fence<HardEvent::MTE2_V>(pipe_);
            Cast(gate_, copy_, RoundMode::CAST_NONE, I);
            Cast(up_, copy_[I], RoundMode::CAST_NONE, I);
            PipeBarrier<PIPE_V>();
            Muls(temp_, gate_, -1.0f, I);
            PipeBarrier<PIPE_V>();
            Exp(temp_, temp_, I);
            PipeBarrier<PIPE_V>();
            Adds(temp_, temp_, 1.0f, I);
            PipeBarrier<PIPE_V>();
            Div(gate_, gate_, temp_, I);
            PipeBarrier<PIPE_V>();
            Mul(gate_, gate_, up_, I);
            PipeBarrier<PIPE_V>();
            Cast(copy_, gate_, RoundMode::CAST_RINT, I);
            Fence<HardEvent::V_MTE3>(pipe_);
            DataCopy(activation_[row * I], copy_, I);
            Fence<HardEvent::MTE3_MTE2>(pipe_);
        }
        Fence<HardEvent::MTE3_S>(pipe_);
    }

    __aicore__ inline void Scatter(GM_ADDR y, const GlobalTensor<int32_t> &meta, uint32_t pairs)
    {
        GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y));
        for (uint32_t row = GetBlockIdx(); row < meta.GetValue(ROWS); row += pairs * GetSubBlockNum()) {
            const int32_t route = meta.GetValue(ROUTES + row);
            if (route >= 0) {
                DataCopy(copy_, down_[row * H], H);
                Fence<HardEvent::MTE2_MTE3>(pipe_);
                DataCopy(output[route * H], copy_, H);
                Fence<HardEvent::MTE3_MTE2>(pipe_);
            }
        }
        Fence<HardEvent::MTE3_S>(pipe_);
    }

private:
    TPipe &pipe_;
    TBuf<TPosition::VECCALC> copyBuf_, gateBuf_, upBuf_, tempBuf_;
    LocalTensor<bfloat16_t> copy_;
    LocalTensor<float> gate_, up_, temp_;
    GlobalTensor<bfloat16_t> packed_, gateUp_, activation_, down_;
};

using InputType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using WeightType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using OutputType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
constexpr auto CUBE_CONFIG = matmul::GetMatmulApiTiling<
    InputType, WeightType, OutputType, OutputType>(GetNormalConfig());

__aicore__ inline void Project(GM_ADDR input, GM_ADDR weights, GM_ADDR output,
                              GM_ADDR tiling, const GlobalTensor<int32_t> &meta,
                              uint32_t fullN, uint32_t fullK, uint32_t pairs)
{
    TPipe cubePipe;
    AscendC::MatmulImpl<InputType, WeightType, OutputType, OutputType, CUBE_CONFIG> mm;
    mm.Init(reinterpret_cast<__gm__ TCubeTiling *>(tiling), &cubePipe);
    GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weights));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
    const uint32_t nTiles = fullN / TILE_N;
    const uint32_t tasks = meta.GetValue(ROWS) / TILE_M * nTiles;
    for (uint32_t task = GetBlockIdx(); task < tasks; task += pairs) {
        uint32_t mBlock = task / nTiles;
        uint32_t expert = 0;
        for (; expert < meta.GetValue(EXPERTS); ++expert) {
            const uint32_t blocks = (meta.GetValue(PREFIX + expert + 1) - meta.GetValue(PREFIX + expert)) / TILE_M;
            if (mBlock < blocks) {
                break;
            }
            mBlock -= blocks;
        }
        const uint32_t rowBase = meta.GetValue(PREFIX + expert) + TILE_M * mBlock;
        const uint32_t nStart = (task % nTiles) * TILE_N;
        mm.SetOrgShape(TILE_M, fullN, fullK);
        mm.SetSingleShape(TILE_M, TILE_N, fullK);
        mm.SetTail(TILE_M, TILE_N, fullK);
        mm.SetTensorA(a[rowBase * fullK]);
        mm.SetTensorB(b[(expert * fullN + nStart) * fullK], true);
        mm.IterateAll(c[rowBase * fullN + nStart]);
        mm.End();
    }
}
}  // namespace

extern "C" __global__ __aicore__ void faas_probe_geometry(GM_ADDR status)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Mark(status, GetBlockNum(), 0);
}

extern "C" __global__ __aicore__ void faas_probe_compute(
    GM_ADDR x, GM_ADDR ids, GM_ADDR w13, GM_ADDR w2, GM_ADDR controls,
    GM_ADDR scratch, GM_ADDR y, GM_ADDR status, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t pairs = GetBlockNum();
    GlobalTensor<int32_t> meta;
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratch));
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(COMMAND);
        Flush(meta);
        if (meta.GetValue(CODE) != OK) {
            Mark(status, pairs, 1);
            return;
        }
        Project(scratch + PACKED, w13, scratch + GATE_UP, tiling, meta, 2 * I, H, pairs);
        CrossCoreSetFlag<0x2, PIPE_FIX>(GATE_DONE);
        CrossCoreWaitFlag(DOWN_READY);
        Project(scratch + ACTIVATION, w2, scratch + DOWN,
                tiling + sizeof(TCubeTiling), meta, H, I, pairs);
        CrossCoreSetFlag<0x2, PIPE_FIX>(DOWN_DONE);
        Mark(status, pairs, 4);
    } else {
        TPipe vectorPipe;
        VectorStages vector(vectorPipe, scratch);
        if (GetBlockIdx() == 0) {
            Group(controls, ids, meta);
        }
        SyncAll<true>();
        Flush(meta);
        if (meta.GetValue(CODE) == OK) {
            vector.Pack(x, y, meta, pairs);
        }
        SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(COMMAND);
        if (meta.GetValue(CODE) != OK) {
            Mark(status, pairs, 1);
            return;
        }
        CrossCoreWaitFlag(GATE_DONE);
        SyncAll<true>();
        vector.Activate(meta, pairs);
        SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(DOWN_READY);
        CrossCoreWaitFlag(DOWN_DONE);
        SyncAll<true>();
        vector.Scatter(y, meta, pairs);
        SyncAll<true>();
        Mark(status, pairs, 4);
    }
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void faas_probe_geometry_do(uint32_t blocks, void *stream, uint8_t *status)
{
    faas_probe_geometry<<<blocks, nullptr, stream>>>(status);
}

extern "C" void faas_probe_compute_do(uint32_t blocks, void *stream, uint8_t *x,
    uint8_t *ids, uint8_t *w13, uint8_t *w2, uint8_t *controls, uint8_t *scratch,
    uint8_t *y, uint8_t *status, uint8_t *tiling)
{
    faas_probe_compute<<<blocks, nullptr, stream>>>(
        x, ids, w13, w2, controls, scratch, y, status, tiling);
}
#endif
