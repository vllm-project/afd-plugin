// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
/**
 * Shared DSV4 clipped-SwiGLU clamp used by every AFD GMM/epilogue path.
 *
 * The model's swiglu_limit applies to the dequantized gate/up branches before
 * SiLU: gate is upper-clamped at L, up is clamped to [-L, L]. Zero disables
 * clipping, matching the ACLNN and PyTorch surfaces that default to 0.
 *
 * This lives in utils/op_kernel so the layered W4A8/A4W4 postprocessors and
 * the native MegaMoE activation share one definition instead of each carrying
 * its own Mins/Maxs sequence.
 */
#ifndef AFD_SWIGLU_LIMIT_H
#define AFD_SWIGLU_LIMIT_H

#include "kernel_operator.h"

__aicore__ inline void ClampSwiGLUInputs(AscendC::LocalTensor<float> gate,
                                         AscendC::LocalTensor<float> up,
                                         float limit, uint32_t count)
{
    if (limit > 0.0f) {
        AscendC::Mins(gate, gate, limit, count);
        AscendC::Mins(up, up, limit, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(up, up, -limit, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

#endif // AFD_SWIGLU_LIMIT_H
