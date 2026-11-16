// Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
#pragma once
#include <type_traits>
#include "op_common/op_host/util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef OPS_UTILS_LOG_SUB_MOD_NAME
#define OPS_UTILS_LOG_SUB_MOD_NAME "CAM_MEGAMOE"
#endif
#ifndef OPS_UTILS_LOG_PACKAGE_TYPE
#define OPS_UTILS_LOG_PACKAGE_TYPE "[CAM]"
#endif
#include "ops_log.h"
// Split tiling uses only these header-only math helpers from the upstream
// aggregate MC2 host headers. Keep all call sites and calculations unchanged.
namespace ops {
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
}
#ifndef OP_LOGE
#define OP_LOGE(node, ...) OPS_LOG_E(node, __VA_ARGS__)
#endif
