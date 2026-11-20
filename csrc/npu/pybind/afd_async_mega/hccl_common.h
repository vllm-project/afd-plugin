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
 * \file hccl_common.h
 * \brief
 */

#ifndef CANN_OPS_TRANSFORMER_HCCL_COMMON_H
#define CANN_OPS_TRANSFORMER_HCCL_COMMON_H

#include "aclnn_common.h"
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_res_defs.h"
#include "hccl/hccl_rank_graph.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
#include "torch_npu/csrc/flopcount/FlopCount.h"
#endif
#define NPU_NAME_SPACE at_npu::native

using _HcclKfcAllocOpArgs = HcclResult (*)(void **);                                      // 通信配置对象创建
using _HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void *, char *);                        // 设置通信类型
using _HcclKfcOpArgsSetCommEngine = HcclResult (*)(void *, uint8_t);                      // 设置通信方式
using _HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void *, void **);           // 创建HcclContext
using _HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm, uint64_t, void **, uint64_t *); // 获取远端地址
using _HcclKfcFreeOpArgs = HcclResult (*)(void *);                                        // 释放通信配置对象
using _HcclCommGetHandleWithName = HcclResult (*)(const char *, HcclComm *); // 通过groupName获取groupHandle
using _HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t *);               // 获取通信域大小
using _HcclGetRankId = HcclResult (*)(HcclComm, uint32_t *);                 // 获取卡号
using _HcclGetHcclBuffer = HcclResult (*)(HcclComm, void **, uint64_t *);    // 获取本卡地址

// 通过group名称获取comm handle
using _HcomGetCommHandleByGroup = HcclResult (*)(const char *, HcclComm *);
// 获取rank间通信链路
using _HcclRankGraphGetLinks = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *);
// 获取网络层信息
using _HcclRankGraphGetLayers = HcclResult (*)(HcclComm, uint32_t **, uint32_t *);
// 获取每层硬件连接拓扑类型
using _HcclRankGraphGetTopoTypeByLayer = HcclResult (*)(HcclComm, uint32_t, CommTopo *);
// Query topology instances and their local endpoints within CUSTOM layers.
using _HcclRankGraphGetTopoInstsByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t **, uint32_t *);
using _HcclRankGraphGetTopoType = HcclResult (*)(HcclComm, uint32_t, uint32_t, CommTopo *);
using _HcclRankGraphGetRanksByTopoInst = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t **, uint32_t *);
using _HcclRankGraphGetEndpointNum = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t *);
using _HcclRankGraphGetEndpointDesc = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t *, EndpointDesc *);
// 获取每层rank数量
using _HcclRankGraphGetRankSizeByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
// 初始化channel描述符
using _HcclRankGraphGetRanksByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t **, uint32_t *);
// 获取channel句柄
using _HcclChannelAcquire = HcclResult (*)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *);
// 通过channel获取buffer
using _HcclChannelGetHcclBuffer = HcclResult (*)(HcclComm, ChannelHandle, void **, uint64_t *);
// 创建引擎context
using _HcclEngineCtxCreate = HcclResult (*)(HcclComm, const char *, CommEngine, uint64_t, void **);
// 获取引擎context
using _HcclEngineCtxGet = HcclResult (*)(HcclComm, const char *, CommEngine, void **, uint64_t *);
// 拷贝context
using _HcclEngineCtxCopy = HcclResult (*)(HcclComm, CommEngine, const char *, void *, uint64_t, uint64_t);
// HcclBarrier
using _HcclBarrier = HcclResult (*)(HcclComm, aclrtStream);
// 注册内存到通信域
using _HcclCommMemReg = HcclResult (*)(HcclComm, const char *, const CommMem *, HcclMemHandle *);
// 通过channel获取远端注册的内存
using _HcclChannelGetRemoteMems = HcclResult (*)(HcclComm, ChannelHandle, uint32_t *, CommMem **, char ***);

static _HcclKfcAllocOpArgs HcclKfcAllocOpArgsFunc = nullptr;
static _HcclKfcOpArgsSetAlgConfig HcclKfcOpArgsSetAlgConfigFunc = nullptr;
static _HcclKfcOpArgsSetCommEngine HcclKfcOpArgsSetCommEngineFunc = nullptr;
static _HcclCreateOpResCtx HcclCreateOpResCtxFunc = nullptr;
static _HcclGetRemoteIpcHcclBuf HcclGetRemoteIpcHcclBufFunc = nullptr;
static _HcclKfcFreeOpArgs HcclKfcFreeOpArgsFunc = nullptr;
static _HcclCommGetHandleWithName HcclCommGetHandleWithNameFunc = nullptr;
static _HcclGetRankSize HcclGetRankSizeFunc = nullptr;
static _HcclGetRankId HcclGetRankIdFunc = nullptr;
static _HcclGetHcclBuffer HcclGetHcclBufferFunc = nullptr;

// 新HCCL EngineCtx API函数指针
static _HcomGetCommHandleByGroup HcomGetCommHandleByGroupFunc = nullptr;
static _HcclRankGraphGetLinks HcclRankGraphGetLinksFunc = nullptr;
static _HcclRankGraphGetLayers HcclRankGraphGetLayersFunc = nullptr;
static _HcclRankGraphGetTopoTypeByLayer HcclRankGraphGetTopoTypeByLayerFunc = nullptr;
static _HcclRankGraphGetTopoInstsByLayer HcclRankGraphGetTopoInstsByLayerFunc = nullptr;
static _HcclRankGraphGetTopoType HcclRankGraphGetTopoTypeFunc = nullptr;
static _HcclRankGraphGetRanksByTopoInst HcclRankGraphGetRanksByTopoInstFunc = nullptr;
static _HcclRankGraphGetEndpointNum HcclRankGraphGetEndpointNumFunc = nullptr;
static _HcclRankGraphGetEndpointDesc HcclRankGraphGetEndpointDescFunc = nullptr;
static _HcclRankGraphGetRankSizeByLayer HcclRankGraphGetRankSizeByLayerFunc = nullptr;
static _HcclRankGraphGetRanksByLayer HcclRankGraphGetRanksByLayerFunc = nullptr;
static _HcclChannelAcquire HcclChannelAcquireFunc = nullptr;
static _HcclChannelGetHcclBuffer HcclChannelGetHcclBufferFunc = nullptr;
static _HcclEngineCtxCreate HcclEngineCtxCreateFunc = nullptr;
static _HcclEngineCtxGet HcclEngineCtxGetFunc = nullptr;
static _HcclEngineCtxCopy HcclEngineCtxCopyFunc = nullptr;
static _HcclBarrier HcclBarrierFunc = nullptr;
static _HcclCommMemReg HcclCommMemRegFunc = nullptr;
static _HcclChannelGetRemoteMems HcclChannelGetRemoteMemsFunc = nullptr;

inline const char *GetHcclLibName(void)
{
    return "libhccl.so";
}

inline const char *GetHcclFwkLibName(void)
{
    return "libhccl_fwk.so";
}

template <typename T>
inline T GetFuncAddr(void *opApiHandler, const char *libName, const char *apiName)
{
    auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetHcclLibName(), apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s", apiName, GetHcclLibName(), dlerror());
        return nullptr;
    }
    T func = reinterpret_cast<T>(funcAddr);
    return func;
}

template <typename T>
inline T GetHcclFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclLibName(), apiName);
}

template <typename T>
inline T GetHcclFwkFuncAddr(const char *apiName)
{
    static auto opApiHandler = GetOpApiLibHandler(GetHcclFwkLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetFuncAddr<T>(opApiHandler, GetHcclFwkLibName(), apiName);
}

inline void InitHcclFunctions()
{
    HcclKfcAllocOpArgsFunc = GetHcclFuncAddr<_HcclKfcAllocOpArgs>("HcclKfcAllocOpArgs"); // 通信配置对象创建
    TORCH_CHECK(HcclKfcAllocOpArgsFunc != nullptr, "getHcclKfcAllocOpArgs failed.");
    HcclKfcFreeOpArgsFunc = GetHcclFuncAddr<_HcclKfcFreeOpArgs>("HcclKfcFreeOpArgs"); // 释放通信配置对象
    TORCH_CHECK(HcclKfcFreeOpArgsFunc != nullptr, "getHcclKfcFreeOpArgs failed.");
    HcclKfcOpArgsSetCommEngineFunc =
        GetHcclFuncAddr<_HcclKfcOpArgsSetCommEngine>("HcclKfcOpArgsSetCommEngine"); // 设置通信方式
    TORCH_CHECK(HcclKfcOpArgsSetCommEngineFunc != nullptr, "getHcclKfcOpArgsSetCommEngine failed.");
    HcclGetRankIdFunc = GetHcclFuncAddr<_HcclGetRankId>("HcclGetRankId"); // 获取本卡卡号
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "getFuncHcclGetRankId failed.");
    HcclGetHcclBufferFunc = GetHcclFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer"); // 获取本卡地址
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "getFuncHcclGetHcclBuffer failed.");
    HcclGetRemoteIpcHcclBufFunc =
        GetHcclFwkFuncAddr<_HcclGetRemoteIpcHcclBuf>("HcclGetRemoteIpcHcclBuf"); // 获取远端地址
    TORCH_CHECK(HcclGetRemoteIpcHcclBufFunc != nullptr, "getFuncHcclGetRemoteIpcHcclBuf failed.");
    HcclKfcOpArgsSetAlgConfigFunc =
        GetHcclFuncAddr<_HcclKfcOpArgsSetAlgConfig>("HcclKfcOpArgsSetAlgConfig"); // 设置通信类型
    TORCH_CHECK(HcclKfcOpArgsSetAlgConfigFunc != nullptr, "getFuncHcclKfcOpArgsSetAlgConfig failed.");
    HcclCommGetHandleWithNameFunc =
        GetHcclFwkFuncAddr<_HcclCommGetHandleWithName>("HcclCommGetHandleWithName"); // 通过groupName获取groupHandle
    TORCH_CHECK(HcclCommGetHandleWithNameFunc != nullptr, "getFuncHcclCommGetHandleWithName failed.");
    HcclCreateOpResCtxFunc = GetHcclFuncAddr<_HcclCreateOpResCtx>("HcclCreateOpResCtx"); // 创建HcclContext
    TORCH_CHECK(HcclCreateOpResCtxFunc != nullptr, "getFuncHcclCreateOpResCtx failed.");
    HcclGetRankSizeFunc = GetHcclFuncAddr<_HcclGetRankSize>("HcclGetRankSize"); // 获取通信域大小
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "getFuncHcclGetRankSize failed.");
    HcclBarrierFunc = GetHcclFuncAddr<_HcclBarrier>("HcclBarrier"); // 执行Barrier同步
    TORCH_CHECK(HcclBarrierFunc != nullptr, "getFuncHcclBarrier failed.");
}

inline void InitHcclRankGraphFunctions()
{
    HcclRankGraphGetLinksFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLinks>("HcclRankGraphGetLinks");
    TORCH_CHECK(HcclRankGraphGetLinksFunc != nullptr, "getHcclRankGraphGetLinks failed.");
    HcclRankGraphGetLayersFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetLayers>("HcclRankGraphGetLayers");
    TORCH_CHECK(HcclRankGraphGetLayersFunc != nullptr, "getHcclRankGraphGetLayers failed.");
    HcclRankGraphGetTopoTypeByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetTopoTypeByLayer>("HcclRankGraphGetTopoTypeByLayer");
    TORCH_CHECK(HcclRankGraphGetTopoTypeByLayerFunc != nullptr, "getHcclRankGraphGetTopoTypeByLayer failed.");
    HcclRankGraphGetTopoInstsByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetTopoInstsByLayer>("HcclRankGraphGetTopoInstsByLayer");
    TORCH_CHECK(HcclRankGraphGetTopoInstsByLayerFunc != nullptr, "getHcclRankGraphGetTopoInstsByLayer failed.");
    HcclRankGraphGetTopoTypeFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetTopoType>("HcclRankGraphGetTopoType");
    TORCH_CHECK(HcclRankGraphGetTopoTypeFunc != nullptr, "getHcclRankGraphGetTopoType failed.");
    HcclRankGraphGetRanksByTopoInstFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetRanksByTopoInst>("HcclRankGraphGetRanksByTopoInst");
    TORCH_CHECK(HcclRankGraphGetRanksByTopoInstFunc != nullptr, "getHcclRankGraphGetRanksByTopoInst failed.");
    HcclRankGraphGetEndpointNumFunc = GetHcclFwkFuncAddr<_HcclRankGraphGetEndpointNum>("HcclRankGraphGetEndpointNum");
    TORCH_CHECK(HcclRankGraphGetEndpointNumFunc != nullptr, "getHcclRankGraphGetEndpointNum failed.");
    HcclRankGraphGetEndpointDescFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetEndpointDesc>("HcclRankGraphGetEndpointDesc");
    TORCH_CHECK(HcclRankGraphGetEndpointDescFunc != nullptr, "getHcclRankGraphGetEndpointDesc failed.");
    HcclRankGraphGetRankSizeByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetRankSizeByLayer>("HcclRankGraphGetRankSizeByLayer");
    TORCH_CHECK(HcclRankGraphGetRankSizeByLayerFunc != nullptr, "getHcclRankGraphGetRankSizeByLayer failed.");
    HcclRankGraphGetRanksByLayerFunc =
        GetHcclFwkFuncAddr<_HcclRankGraphGetRanksByLayer>("HcclRankGraphGetRanksByLayer");
    TORCH_CHECK(HcclRankGraphGetRanksByLayerFunc != nullptr, "getHcclRankGraphGetRanksByLayer failed.");
}

// 初始化新的EngineCtx API (从libhccl_fwk.so加载)
inline void InitHcclEngineCtxFunctions()
{
    HcomGetCommHandleByGroupFunc = GetHcclFwkFuncAddr<_HcomGetCommHandleByGroup>("HcomGetCommHandleByGroup");
    TORCH_CHECK(HcomGetCommHandleByGroupFunc != nullptr, "getHcomGetCommHandleByGroup failed.");
    InitHcclRankGraphFunctions();
    HcclChannelAcquireFunc = GetHcclFwkFuncAddr<_HcclChannelAcquire>("HcclChannelAcquire");
    TORCH_CHECK(HcclChannelAcquireFunc != nullptr, "getHcclChannelAcquire failed.");
    HcclChannelGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclChannelGetHcclBuffer>("HcclChannelGetHcclBuffer");
    TORCH_CHECK(HcclChannelGetHcclBufferFunc != nullptr, "getHcclChannelGetHcclBuffer failed.");
    HcclEngineCtxCreateFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCreate>("HcclEngineCtxCreate");
    TORCH_CHECK(HcclEngineCtxCreateFunc != nullptr, "getHcclEngineCtxCreate failed.");
    HcclEngineCtxGetFunc = GetHcclFwkFuncAddr<_HcclEngineCtxGet>("HcclEngineCtxGet");
    TORCH_CHECK(HcclEngineCtxGetFunc != nullptr, "getHcclEngineCtxGet failed.");
    HcclEngineCtxCopyFunc = GetHcclFwkFuncAddr<_HcclEngineCtxCopy>("HcclEngineCtxCopy");
    TORCH_CHECK(HcclEngineCtxCopyFunc != nullptr, "getHcclEngineCtxCopy failed.");
    HcclGetHcclBufferFunc = GetHcclFwkFuncAddr<_HcclGetHcclBuffer>("HcclGetHcclBuffer"); // 获取本卡地址
    TORCH_CHECK(HcclGetHcclBufferFunc != nullptr, "getFuncHcclGetHcclBuffer failed.");
    HcclGetRankIdFunc = GetHcclFwkFuncAddr<_HcclGetRankId>("HcclGetRankId"); // 获取本卡卡号
    TORCH_CHECK(HcclGetRankIdFunc != nullptr, "getFuncHcclGetRankId failed.");
    HcclGetRankSizeFunc = GetHcclFwkFuncAddr<_HcclGetRankSize>("HcclGetRankSize"); // 获取通信域大小
    TORCH_CHECK(HcclGetRankSizeFunc != nullptr, "getFuncHcclGetRankSize failed.");
    HcclCommMemRegFunc = GetHcclFwkFuncAddr<_HcclCommMemReg>("HcclCommMemReg");
    TORCH_CHECK(HcclCommMemRegFunc != nullptr, "getHcclCommMemReg failed.");
    HcclChannelGetRemoteMemsFunc = GetHcclFwkFuncAddr<_HcclChannelGetRemoteMems>("HcclChannelGetRemoteMems");
    TORCH_CHECK(HcclChannelGetRemoteMemsFunc != nullptr, "getHcclChannelGetRemoteMems failed.");
}

#endif // CANN_OPS_TRANSFORMER_HCCL_COMMON_H
