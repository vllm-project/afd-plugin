/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Prepare and load quantized operands for MegaMoE GEMM.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file block_prologue.h
 * \brief block prologue interface.
 */
#pragma once

#include "../gemm/policy/dispatch_policy_megamoe_combine_recv.h"

namespace Blaze {
namespace Gemm {
namespace Prologue {

template <class DispatchPolicy, class... Args>
class BlockPrologue {
    static_assert(!AscendC::Std::is_same_v<DispatchPolicy, DispatchPolicy>,
                  "BlockPrologue is not implemented for this DispatchPolicy");
};

} // namespace Prologue
} // namespace Gemm
} // namespace Blaze
