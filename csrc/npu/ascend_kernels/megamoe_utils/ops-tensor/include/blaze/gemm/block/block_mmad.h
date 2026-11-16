/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad.h
 * \brief
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {
/**
 * @class BlockMmad
 * @brief Block matrix multiplication class for performing block matrix multiplication operations
 */
template <class DispatchPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad {
    static_assert(Blaze::Gemm::always_false_v<DispatchPolicy_>, "BlockMmad is not implemented for this DispatchPolicy");
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
#include "blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "blaze/gemm/block/block_mmad_matmul_basic_split_k.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_mmad_matmul_bl1_full_load.h"
#include "blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk_split_k.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
