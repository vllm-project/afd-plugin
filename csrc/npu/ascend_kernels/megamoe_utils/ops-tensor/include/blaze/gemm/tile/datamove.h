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
 * \file datamove.h
 * \brief Architecture-specific data movement primitives for tile-level tensors.
 */
#pragma once

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
#include "blaze/gemm/tile/arch35/copy_gm_to_l1.h"
#include "blaze/gemm/tile/arch35/copy_gm_to_ub.h"
#include "blaze/gemm/tile/arch35/copy_mx_scale.h"
#include "blaze/gemm/tile/arch35/copy_weight_ub_to_l1.h"
#endif
