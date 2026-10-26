/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_utils.h"
#include "grouped_matmul_antiquant.h"
#include "grouped_matmul_vector.h"
#include "grouped_matmul_tiling_key.h"
#include "grouped_matmul.h"

#include "kernel_operator.h"
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))

#include "grouped_matmul_antiquant_a16w8_msd.h"
#include "grouped_matmul_antiquant_a8w4_msd_pre.h"
#include "grouped_matmul_antiquant_a8w4_msd.h"
#include "grouped_matmul_antiquant_a8w4_pre.h"
#include "grouped_matmul_antiquant_a8w4.h"
#include "grouped_matmul_antiquant_a8w4_msd_new.h"
#include "grouped_matmul_quant_mixcore.h"
#include "grouped_matmul_pre_tiling.h"
#include "grouped_matmul_a4w4.h"
#include "grouped_matmul_autotiling_a8w4.h"
#include "a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_controller.h"
#ifndef __CCE_KT_TEST__
#include "grouped_matmul_fixaxismove_interface.cpp"
#endif
#endif

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_ALL_WEIGHT) && FORMAT_ALL_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
constexpr MatmulConfig matmulCFG = NZ_CFG_MDL;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
constexpr MatmulConfig matmulCFG = CFG_MDL;
#endif

#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
// ### PATCH START: old API dequant MDL scheduling
// Patch reason: the old API dequant route uses normal, small-block loading.
// Patch functionality: test MDL loading with the existing unit-flag contract.
// Signature: no function or operator ABI changes. Remove when upstream adopts
// the selected scheduling policy for this route.
constexpr MatmulConfig A8W4_API_DEQUANT_CFG_MDL =
    GetMDLConfig(false, false, 0, false, false, false, true);
// ### PATCH END: old API dequant MDL scheduling
constexpr auto GetMmCFG() {
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using xTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_ALL_WEIGHT, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

template <bool trans = false>
using weightTypeMSD = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_ALL_WEIGHT, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using yTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_ALL_BIAS>;

namespace {
    __aicore__ inline static constexpr MatmulApiStaticTiling GetGmmMatmulApiTiling(bool isND2NZ, bool transB) {
        MatmulConfig conf = GenGmmConf(isND2NZ);
        MatmulApiStaticTiling staticTilingTmp;
        if (transB) {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<true>, yType, biasType>(conf);
        } else {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<false>, yType, biasType>(conf);
        }
        staticTilingTmp.depthA1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.depthB1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.stepM = 1;
        staticTilingTmp.stepN = 1;
        staticTilingTmp.stepKa = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.stepKb = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.dbL0A = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0B = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0C = 1;
        return staticTilingTmp;
    }
#if defined(FORMAT_ALL_WEIGHT) && FORMAT_ALL_WEIGHT == FORMAT_FRACTAL_NZ
    constexpr bool isWeightNZ = true;
#else
    constexpr bool isWeightNZ = false;
#endif
    constexpr static auto staticCFG = GetGmmMatmulApiTiling(isWeightNZ, false);
    constexpr static auto staticCFGtransB = GetGmmMatmulApiTiling(isWeightNZ, true);
} // namespace


#define GMM_IMP(computeClass, processClass, transA, transB, sync, cfg)                                             \
    do {                                                                                                           \
        using matmulType = MMType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                        \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);                                       \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_STATIC_TILING_IMP(processClass, transA, transB, sync, cfg)                                        \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                    \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, nullptr, &tPipe); \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_STATIC_TILING_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType) \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init((TCubeTiling*)nullptr, &tPipe);                                                                \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, nullptr, &tPipe); \
        computeOp.InitStaticTiling(&gmmBaseParams_, user1, (cfg).baseM, (cfg).baseN);                              \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, nullptr, 0, groupList, tiling);                                                   \
        op.InitStaticTiling((cfg).baseM, (cfg).baseN);                                                             \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#if defined(CONST_TILING)
#define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)               \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GMMTilingData gmmTilingData;                                                                               \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#else
    #define GMM_CV_SPLIT_IMP(computeClass, processClass, transA, transB, sync, cfg, aType, bType, cType)           \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        GMMPreTilingProcess preTiling;                                                                             \
        preTiling.Init(groupList, gmmBaseParams_, mmTilingData_, &tPipe);                                          \
        preTiling.Process(gmmBaseParams_, mmTilingData_);                                                          \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType, sync> computeOp(mm);                                                              \
        computeOp.Init(x, weight, bias, scale, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,  \
                       y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)
#endif

#define GMM_A4W4_IMP(computeClass, transA, transB, cfg, aType, bType, cType)                                       \
    do {                                                                                                           \
        using matmulType = MMImplType<aType<transA>, bType<transB>, cType, biasType, cfg>;                         \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> computeOp(mm);                                                                    \
        computeOp.Init(x, weight, scale, groupList, perTokenScale,                                                 \
                    y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe);                                            \
        computeOp.Process();                                                                                       \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_MSD(computeClass, cfg)                                                               \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4PreProcess op1;                                                                                 \
            op1.Init(x, x, groupList, user1, gmmBaseParams_, &tPipe);                                              \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X_DEV_A8W4MSD, false>;                          \
        using bT = MatmulType<TPosition::GM, wFormat, DTYPE_WEIGHT_DEV_A8W4MSD, false>;                            \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, (GM_ADDR)0, nullptr, nullptr, nullptr,               \
                y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4(computeClass, cfg)                                                                   \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, scale, user1, curLayer, gmmBaseParams_.layerNum, gmmBaseParams_, &tPipe);           \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;                                         \
        using matmulType = MMImplType<aT, bT, cT, biasT, cfg>;                                                     \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        computeClass<matmulType> op(mm);                                                                           \
        op.Init(x, weight, bias, groupList, scale, perTokenScale, (GM_ADDR)0, nullptr, nullptr, nullptr,               \
                    y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(computeClass, cfg)                                                          \
    do {                                                                                                           \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        if ASCEND_IS_AIV {                                                                                         \
            GMMA8W4FakeQuantPreProcess<wFormat> op1;                                                               \
            op1.Init(weight, y, scale, user1, curLayer, gmmBaseParams_.layerNum, gmmBaseParams_, &tPipe);             \
            op1.Process();                                                                                         \
            tPipe.Reset();                                                                                         \
            tPipe.Destroy();                                                                                       \
            tPipe.Init();                                                                                          \
        }                                                                                                          \
        SyncAll<false>();                                                                                          \
        GlobalTensor<int8_t> yGm;                                                                                  \
        yGm.SetGlobalBuffer((__gm__ int8_t *)workspace);                                                           \
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;                                       \
        using bT = MatmulType<TPosition::GM, wFormat, int8_t, false>;                                              \
        using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                   \
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;                                      \
        using matmulType = MMImplType<aT, bT, cT, biasT, matmulCFG>;                                               \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        if ASCEND_IS_AIC {                                                                                         \
            mm.SetSubBlockIdx(0);                                                                                  \
            mm.Init(&mmTilingData_, &tPipe);                                                                       \
        }                                                                                                          \
        GMMQuantMixCoreCompute<matmulType, false> computeOp(mm);                                                   \
        computeOp.isA8W4FakeQuant = true;                                                                          \
        computeOp.Init(x, user1, bias, user1, (GM_ADDR)0, (GM_ADDR)0, (GM_ADDR)0, groupList, perTokenScale,   \
                       y, user1, curLayer, &gmmBaseParams_, &mmTilingData_, &tPipe); \
        GMMProcess<decltype(computeOp)> op(computeOp);                                                             \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

#define GMM_CV_SPLIT_IMP_A16W4_MSD(computeClass, ...)                                 \
    do {                                                                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling); \
        computeClass<DTYPE_X, DTYPE_ALL_WEIGHT, DTYPE_ALL_BIAS, GROUP_LIST_TYPE> op;          \
        op.Init(x, weight, (GM_ADDR)0, bias, groupList, y, &gmmBaseParams_);        \
        op.Process(workspace, &tPipe);                                                                 \
    } while (0)

// Patch reason: profile shows sustained MTE2/scalar activity in old API dequant.
// Patch functionality: select MDL loading only for that route; retain unit flags.
// Signature: unchanged. Remove this specialization when upstream adopts the policy.
template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE,
          int IS_STATIC_TILING_API, int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO, bool IS_ENABLE_FIXED_AXIS>
__global__ __aicore__ void grouped_matmul_layered(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR layerIndex, GM_ADDR groupList, GM_ADDR perTokenScale,
                                                     GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

    // layered: consume layer_index at Init stage (one 8B GM read per core); all_weight/all_bias/
    // all_scale are per-layer full-expert tensor lists resolved through this index.
    int64_t curLayer = LayeredReadLayerIndex(layerIndex);

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#if defined(GMM_ANTI_QUANT_A8W4_MSD)
    // ANTIQUANT_A8W4
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT4) {
        if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT) {
            // ### PATCH START: old API dequant MDL scheduling
            GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDCompute, A8W4_API_DEQUANT_CFG_MDL);
            // ### PATCH END: old API dequant MDL scheduling
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT) {
            GMM_CV_SPLIT_IMP_A8W4_MSD(GMMA8W4MSDComputeNew, A8W4_GMM_CFG_MDL_NEW);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT) {
            GMM_CV_SPLIT_IMP_A8W4_FAKEA8W8(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT) {
            GMM_CV_SPLIT_IMP_A8W4(GMMA8W4Compute, A8W4_GMM_CFG_MDL);
        } else if constexpr (A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING) {
            GET_TILING_DATA_MEMBER(GMMTilingData, hpTilingData, tilingData, tiling);
            GM_ADDR A = x;
            GM_ADDR B = weight;
            GM_ADDR C = y;
            GM_ADDR groupListOptional = groupList;
            GM_ADDR bias_ = bias;
            GM_ADDR offset_ = (GM_ADDR)0;
            GM_ADDR sa = perTokenScale;
            GM_ADDR sw = scale;
            GM_ADDR workspaceDevice = user1;

            GMMA4W8AutotilingCompute op(A, B, C, groupListOptional, bias_, offset_, sa, sw, workspaceDevice,
                                        curLayer, const_cast<A8W4HPTiling *>(&tilingData), &tPipe);
            op.Init();
            op.Process();
        }
    }
#elif defined(GMM_ANTI_QUANT)
    // ANTIQUANT
    if constexpr ((D_T_A == GMM_TPL_BF16) &&
                  A16W8_KERNEL_TEMPLATE == GROUPED_MATMUL_A16W4_KERNEL_TEMPLATE_MSD_ANTIQUANT_GS32) {
        GMM_CV_SPLIT_IMP_A16W4_MSD(A16W4Msd::GMMWeightQuantA16W4MsdController, false);
    } else if constexpr ((D_T_A == GMM_TPL_FLOAT16 || D_T_A == GMM_TPL_BF16) &&
                         A16W8_KERNEL_TEMPLATE != GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD) {
        // ANTIQUANT_A16W4 & ANTIQUANT_A16W8_NOT_MSD
        if constexpr (TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
            GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, false, false, matmulCFG);
        } else if constexpr (TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
            GMM_IMP(GMMAntiquantComputeNorm, GMMAntiquantProcess, false, true, false, matmulCFG);
        } else if constexpr (TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_2) {
            GMM_IMP(GMMAntiquantComputePerformance, GMMAntiquantProcess, false, false, false, matmulCFG);
        }
    }
#if defined(ORIG_DTYPE_ALL_WEIGHT) && defined(DT_INT8) && ORIG_DTYPE_ALL_WEIGHT == DT_INT8
    // ANTIQUANT_A16W8_MSD
    if constexpr ((D_T_A == GMM_TPL_FLOAT16 || D_T_A == GMM_TPL_BF16) && D_T_B == GMM_TPL_INT8 &&
                  A16W8_KERNEL_TEMPLATE == GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD) {
        if constexpr (TRANS_B == 0) {
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, false, false, matmulCFG,
                             xTypeMSD, weightTypeMSD, yTypeMSD);
        } else if constexpr (TRANS_B == 1) {
            GMM_CV_SPLIT_IMP(GMMA16W8MSDCompute, GMMA16W8MSDProcess, false, true, false, matmulCFG,
                             xTypeMSD, weightTypeMSD, yTypeMSD);
        }
    }
#endif

#elif defined(GMM_QUANT_BF16) || defined(GMM_QUANT_FLOAT16)
    // QUANT_A8W8O16
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT8 && (D_T_Y == GMM_TPL_BF16 || D_T_Y == GMM_TPL_FLOAT16) &&
                  TRANS_A == 0 && A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
        if constexpr (IS_STATIC_TILING_API == 0) {
            if constexpr (AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
                if constexpr(IS_ENABLE_FIXED_AXIS == 0) {
                    if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
                    } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
                    } else if constexpr(TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, false, false, matmulCFG, xType,
                                    weightType, yType);
                    } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                        GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess, false, true, false, matmulCFG, xType,
                                    weightType, yType);
                    }
                } else if constexpr(IS_ENABLE_FIXED_AXIS == 1 && TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_CUMSUM) {
                    tPipe.Destroy();
                    AscendC::SetMMLayoutTransform(true);
                    GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling)
                    using XDType = int8_t;
                    using WeightDType = int8_t;
                    using CDType = int32_t;
                    using ScaleDType = float;
                    using GrouplistDType = int64_t;
                    using PerTokenScaleDType = float;
                    using YDType = half;
#ifndef __CCE_KT_TEST__
                    Catlass::grouped_matmul_fixaxismove<XDType, WeightDType, CDType, ScaleDType, GrouplistDType, PerTokenScaleDType, YDType>(
                        gmmBaseParams_.m, gmmBaseParams_.k, gmmBaseParams_.n, gmmBaseParams_.groupNum,
                        x, weight, scale, groupList, perTokenScale, y, user1, curLayer, gmmBaseParams_.coreNum,
                        gmmBaseParams_.layerNum);
#endif
                }
            } else if constexpr (AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_2) {
                if constexpr (TRANS_B == 0) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, false, false, matmulCFG, xType, weightType, yType);
                } else if constexpr (TRANS_B == 1) {
                    GMM_CV_SPLIT_IMP(GMMQuantMixCoreCompute, GMMProcess, false, true, false, matmulCFG, xType, weightType, yType);
                }
            }
        } else if (IS_STATIC_TILING_API == 1) {
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                            false, false, false, staticCFG, xType, weightType, yType);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMProcess,
                                            false, true, false, staticCFGtransB, xType, weightType, yType);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                            false, false, false, staticCFG, xType, weightType, yType);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CV_SPLIT_STATIC_TILING_IMP(GMMQuantMixCoreCompute, GMMGroupMSparseProcess,
                                            false, true, false, staticCFGtransB, xType, weightType, yType);
            }
        }
    }
#elif defined(GMM_A4W4)
    // QUANT_A4W4
    if constexpr (D_T_A == GMM_TPL_INT4 && D_T_B == GMM_TPL_INT4) {
        if constexpr (TRANS_B == 0){
            GMM_A4W4_IMP(GMMA4W4Compute, false, false, matmulCFG, xType, weightType, yType);
        }else{
            GMM_A4W4_IMP(GMMA4W4Compute, false, true, matmulCFG, xType, weightType, yType);
        }
    }
#elif defined(GMM_QUANT_INT8) || defined(GMM_QUANT_INT32)
    // QUANT_A8W8O8 & QUANT_A8W8O32
    if constexpr (D_T_A == GMM_TPL_INT8 && D_T_B == GMM_TPL_INT8 && (D_T_Y == GMM_TPL_INT8 || D_T_Y == GMM_TPL_INT32) &&
                  TRANS_A == 0 && A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE &&
                  AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
        if constexpr (IS_STATIC_TILING_API == 0) {
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMGroupMSparseProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_IMP(GMMGroupMSparseProcess, false, true, false, matmulCFGUnitFlag);
            }
        } else if constexpr (IS_STATIC_TILING_API == 1){
            if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, false, false, staticCFG);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMProcess, false, true, false, staticCFGtransB);
            } else if constexpr (TRANS_B == 0 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, false, false, staticCFG);
            } else if constexpr (TRANS_B == 1 && GROUP_LIST_TYPE == GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM) {
                GMM_CUBE_STATIC_TILING_IMP(GMMGroupMSparseProcess, false, true, false, staticCFGtransB);
            }
        }
    }
#elif defined(GMM_FLOAT)
    // NO_QUANT
    if (GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
        IS_STATIC_TILING_API == 0 &&
        A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
            if constexpr (TRANS_A == 0 && TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 0 && TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
                    GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
            } else if constexpr (TRANS_A == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
                if ASCEND_IS_AIV {
                    GET_TILING_DATA(tilingData, tiling);
                    EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
                }
                if ASCEND_IS_AIC {
                    GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
                }
            }
    }

#endif
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
#if defined(GMM_FLOAT)
    if constexpr (TRANS_A == 0 && TRANS_B == 0) {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFG);
    } else if constexpr (TRANS_A == 0 && TRANS_B == 1) {
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFG);
    } else if constexpr (TRANS_A == 1 && TRANS_B == 0) {
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);
            EmptyTensorCompute<DTYPE_Y>(groupList, y, &tilingData);
        }
        if ASCEND_IS_AIC {
            GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
        }
    }
#endif
#endif
}
