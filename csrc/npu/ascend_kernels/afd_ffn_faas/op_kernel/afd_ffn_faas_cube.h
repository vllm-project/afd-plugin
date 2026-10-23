// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_FFN_FAAS_CUBE_H
#define AFD_FFN_FAAS_CUBE_H

#include "lib/matmul_intf.h"
#include "afd_ffn_faas_tiling.h"

namespace AfdFaas {
using InputType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using WeightType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using OutputType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t>;
constexpr auto CUBE_CONFIG = matmul::GetMatmulApiTiling<
    InputType, WeightType, OutputType, OutputType>(GetNormalConfig());

__aicore__ inline void FlushScratch(const GlobalTensor<int32_t> &meta)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(meta);
}

__aicore__ inline void Project(GM_ADDR input, GM_ADDR weights, GM_ADDR output,
                              GM_ADDR tiling, const GlobalTensor<int32_t> &meta,
                              const FaaSInfo &info, const ScratchLayout &scratch,
                              uint32_t fullN, uint32_t fullK)
{
    TPipe cubePipe;
    AscendC::MatmulImpl<InputType, WeightType, OutputType, OutputType, CUBE_CONFIG> mm;
    mm.Init(reinterpret_cast<__gm__ TCubeTiling *>(tiling), &cubePipe);
    GlobalTensor<bfloat16_t> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weights));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
    const uint32_t nTiles = (fullN + TILE_N - 1) / TILE_N;
    const uint32_t tasks = meta.GetValue(WORK_ROWS) / TILE_M * nTiles;
    const uint32_t experts = info.config[EXPERTS] / info.config[FFN_RANKS];
    for (uint32_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        uint32_t mBlock = task / nTiles;
        uint32_t expert = 0;
        for (; expert < experts; ++expert) {
            const uint32_t blocks = (meta.GetValue(scratch.prefix + expert + 1) -
                                    meta.GetValue(scratch.prefix + expert)) / TILE_M;
            if (mBlock < blocks) {
                break;
            }
            mBlock -= blocks;
        }
        const uint32_t rowBase = meta.GetValue(scratch.prefix + expert) + TILE_M * mBlock;
        const uint32_t nStart = (task % nTiles) * TILE_N;
        const uint32_t n = fullN - nStart < TILE_N ? fullN - nStart : TILE_N;
        const uint64_t weight = static_cast<uint64_t>(meta.GetValue(WEIGHT_BASE)) + expert;
        mm.SetOrgShape(TILE_M, fullN, fullK);
        mm.SetSingleShape(TILE_M, n, fullK);
        mm.SetTail(TILE_M, n, fullK);
        mm.SetTensorA(a[rowBase * fullK]);
        mm.SetTensorB(b[(weight * fullN + nStart) * fullK], true);
        mm.IterateAll(c[rowBase * fullN + nStart]);
        mm.End();
    }
}

__aicore__ inline bool CubeExecute(const FaaSInfo &info, GM_ADDR scratchAddress,
                                  GM_ADDR w13, GM_ADDR w2, GM_ADDR tiling)
{
    const ScratchLayout scratch(info);
    GlobalTensor<int32_t> meta;
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratchAddress));
    CrossCoreWaitFlag(COMMAND_READY);
    FlushScratch(meta);
    if (meta.GetValue(WORK_CODE) != OK) {
        return false;
    }
    Project(scratchAddress + scratch.packed, w13, scratchAddress + scratch.gateUp,
            tiling + offsetof(AfdFfnFaasTilingData, gateUpTiling), meta, info, scratch,
            2 * info.config[INTERMEDIATE], info.config[HIDDEN]);
    CrossCoreSetFlag<0x2, PIPE_FIX>(GATE_DONE);
    CrossCoreWaitFlag(DOWN_READY);
    Project(scratchAddress + scratch.activation, w2, scratchAddress + scratch.down,
            tiling + offsetof(AfdFfnFaasTilingData, downTiling), meta, info, scratch,
            info.config[HIDDEN], info.config[INTERMEDIATE]);
    CrossCoreSetFlag<0x2, PIPE_FIX>(DOWN_DONE);
    return true;
}
}  // namespace AfdFaas
#endif
