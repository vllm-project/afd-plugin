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
 * Description: Select execution policies for MegaMoE GEMM kernels.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file dispatch_policy_megamoe_combine_recv.h
 * \brief Dispatch policy for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 *        Extracted from blaze/gemm/policy/dispatch_policy.h for megamoe_combine_recv.
 */
#pragma once

#include "../utils/common_utils_megamoe_combine_recv.h"

namespace Blaze {
namespace Gemm {

/**
 * @struct MatmulMxFp8Fp4DynamicKL1TailResplit
 * @brief Dispatch policy for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 */
struct MatmulMxFp8Fp4DynamicKL1TailResplit {};

} // namespace Gemm
} // namespace Blaze
