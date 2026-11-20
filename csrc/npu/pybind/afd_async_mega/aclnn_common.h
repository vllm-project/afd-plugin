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
 * \file aclnn_common.h
 * \brief
 */

#ifndef CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
#define CANN_OPS_TRANSFORMER_ACLNN_COMMON_H

// 必须先于任何 torch_npu 头文件引入 acl 基础头文件：torch_npu 自带 third_party/acl 版本较旧，
// 缺少 ACL_FLOAT8_E5M2/ACL_FLOAT4_E2M1 等新 dtype 常量。若在其之后 include，旧头文件已设置
// 同名 include guard（INC_EXTERNAL_ACL_ACL_BASE_H_），新版 CANN 头文件会被跳过导致编译失败。
// 注意：本文件的 include 顺序只是必要条件之一；<acl/acl_base.h> 能否解析到环境上实际安装的
// CANN 头文件，还取决于编译期 -I 搜索顺序。OpBuilder.include_paths() 已将 CANN 路径排在
// torch_npu 路径之前，两者缺一不可——仅调整 include 顺序而 -I 顺序未变时，仍会命中旧副本。
#include <acl/acl_base.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <sstream>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <set>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
#include "torch_npu/csrc/flopcount/FlopCount.h"
#endif
#define NPU_NAME_SPACE at_npu::native

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;
using aclTensorList = struct aclTensorList;

using _aclCreateTensor = aclTensor *(*)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                        const int64_t *stride, int64_t offset, aclFormat format,
                                        const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
using _aclCreateScalar = aclScalar *(*)(void *value, aclDataType data_type);
using _aclCreateIntArray = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArray = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArray = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorList = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);

using _aclDestroyTensor = int (*)(const aclTensor *tensor);
using _aclDestroyScalar = int (*)(const aclScalar *scalar);
using _aclDestroyIntArray = int (*)(const aclIntArray *array);
using _aclDestroyFloatArray = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArray = int (*)(const aclBoolArray *array);
using _aclDestroyTensorList = int (*)(const aclTensorList *array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;
constexpr int64_t MAX_DIM_NUM = 5;
constexpr int64_t FP4_IN_INT8 = 2;
constexpr int64_t PENULTIMATE_DIM = 2;
const int g_toAclOffset = 256;

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
    _(at::ScalarType::Byte, ACL_UINT8) \
    _(at::ScalarType::Char, ACL_INT8) \
    _(at::ScalarType::Short, ACL_INT16) \
    _(at::ScalarType::Int, ACL_INT32) \
    _(at::ScalarType::Long, ACL_INT64) \
    _(at::ScalarType::Half, ACL_FLOAT16) \
    _(at::ScalarType::Float, ACL_FLOAT) \
    _(at::ScalarType::Double, ACL_DOUBLE) \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32) \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64) \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL) \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED) \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED) \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED) \
    _(at::ScalarType::BFloat16, ACL_BF16) \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED) \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2) \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN) \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt16, ACL_UINT16) \
    _(at::ScalarType::UInt32, ACL_UINT32) \
    _(at::ScalarType::UInt64, ACL_UINT64) \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Float8_e8m0fnu, ACL_FLOAT8_E8M0) \
    _(at::ScalarType::Float4_e2m1fn_x2, ACL_FLOAT4_E2M1) \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED) \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#define ENUM_OFFSET(new_name, old_name) new_name = static_cast<int>(old_name) + g_toAclOffset,

enum class DType {
    UNDEFINED = -1,
    ENUM_OFFSET(FLOAT, ACL_FLOAT)
    ENUM_OFFSET(FLOAT16, ACL_FLOAT16) ENUM_OFFSET(INT8, ACL_INT8) ENUM_OFFSET(INT32, ACL_INT32)
        ENUM_OFFSET(UINT8, ACL_UINT8) ENUM_OFFSET(INT16, ACL_INT16) ENUM_OFFSET(UINT16, ACL_UINT16)
            ENUM_OFFSET(UINT32, ACL_UINT32) ENUM_OFFSET(INT64, ACL_INT64) ENUM_OFFSET(UINT64, ACL_UINT64)
                ENUM_OFFSET(DOUBLE, ACL_DOUBLE) ENUM_OFFSET(BOOL, ACL_BOOL) ENUM_OFFSET(STRING, ACL_STRING)
                    ENUM_OFFSET(COMPLEX64, ACL_COMPLEX64) ENUM_OFFSET(COMPLEX128, ACL_COMPLEX128)
                        ENUM_OFFSET(BF16, ACL_BF16) ENUM_OFFSET(INT4, ACL_INT4) ENUM_OFFSET(UINT1, ACL_UINT1)
                            ENUM_OFFSET(COMPLEX32, ACL_COMPLEX32) ENUM_OFFSET(HIFLOAT8, ACL_HIFLOAT8)
                                ENUM_OFFSET(FLOAT8_E5M2, ACL_FLOAT8_E5M2) ENUM_OFFSET(FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN)
                                    ENUM_OFFSET(FLOAT8_E8M0, ACL_FLOAT8_E8M0) ENUM_OFFSET(FLOAT6_E3M2, ACL_FLOAT6_E3M2)
                                        ENUM_OFFSET(FLOAT6_E2M3, ACL_FLOAT6_E2M3)
                                            ENUM_OFFSET(FLOAT4_E2M1, ACL_FLOAT4_E2M1)
                                                ENUM_OFFSET(FLOAT4_E1M2, ACL_FLOAT4_E1M2)
};

namespace op_infer {
const int N = 32;
const int SIZE = 8;

inline c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}
} // namespace op_infer

typedef struct {
    const at::Tensor &tensor_;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList &tensor_list_;
    aclDataType dtype;
} TensorListWrapper;

inline bool Is4BitDtype(const aclDataType acl_data_type)
{
    return acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4;
}

inline int64_t ConvertToAclStorageOffset(const at::Tensor &at_tensor, const aclDataType acl_data_type)
{
    int64_t storageOffset = at_tensor.storage_offset();
    if (Is4BitDtype(acl_data_type)) {
        // ACL 4-bit descriptors express offsets in logical nibble elements.
        TORCH_CHECK(storageOffset <= INT64_MAX / FP4_IN_INT8,
                    "4-bit tensor storage offset is too large to convert to logical elements: ", storageOffset);
        storageOffset *= FP4_IN_INT8;
    }
    return storageOffset;
}

static std::unordered_map<aclFormat, aclFormat> FORMAT_FAKE_TO_REAL{
    {ACL_FORMAT_FRACTAL_NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_32}, {ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ}};

static inline bool IsOpInputBaseFormat(const at::Tensor &at_tensor)
{
    if (!torch_npu::utils::is_npu(at_tensor)) {
        return true;
    }
    const auto format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_NHWC) ||
           (format == ACL_FORMAT_NCDHW);
}

inline void CollectB4ShapeInfo(const at::Tensor &at_tensor, c10::SmallVector<int64_t, MAX_DIM_NUM> &wrapperStride,
                               c10::SmallVector<int64_t, MAX_DIM_NUM> &wrapperShape)
{
    int64_t nDim = at_tensor.sizes().size();
    if (nDim == 1) {
        wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
    } else if (nDim > 1) {
        if (wrapperStride[nDim - 1] == 1 && wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            if (wrapperShape[nDim - PENULTIMATE_DIM] == 1) {
                wrapperStride[nDim - 1] = wrapperStride[nDim - 1] * FP4_IN_INT8;
                wrapperShape[nDim - PENULTIMATE_DIM] = wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            } else if (wrapperShape[nDim - 1] == 1) {
                wrapperStride[nDim - PENULTIMATE_DIM] = wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
                wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
            }
        } else if (wrapperStride[nDim - 1] == 1) {
            wrapperStride[nDim - PENULTIMATE_DIM] = wrapperStride[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
            wrapperShape[nDim - 1] = wrapperShape[nDim - 1] * FP4_IN_INT8;
        } else if (wrapperStride[nDim - PENULTIMATE_DIM] == 1) {
            wrapperStride[nDim - 1] = wrapperStride[nDim - 1] * FP4_IN_INT8;
            wrapperShape[nDim - PENULTIMATE_DIM] = wrapperShape[nDim - PENULTIMATE_DIM] * FP4_IN_INT8;
        }

        for (auto i = 0; i < nDim - PENULTIMATE_DIM; i++) {
            wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
        }
    } else {
        TORCH_CHECK(false, "unsupported tensor size() in 4-bit dtype.");
    }
}

enum QuantMode {
    QUANT_MODE_NO_QUANT = 0,
    QUANT_MODE_STATIC = 1,
    QUANT_MODE_PERTOKEN = 2,
    QUANT_MODE_PERGROUP = 3,
    QUANT_MODE_MX = 4,
    QUANT_MODE_MX_CLIP = 5,
};

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

#define MEMCPY_TO_BUF(data_expression, size_expression) \
    if (g_hashOffset + (size_expression) > kHashBufSize) { \
        g_hashOffset = kHashBufMaxSize; \
        return; \
    } \
    auto ret = memcpy_s(g_hashBuf + g_hashOffset, size_expression, data_expression, size_expression); \
    TORCH_CHECK(ret == 0, "memcpy_s failed, error:", ret); \
    g_hashOffset += size_expression;

aclDataType ConvertToAclDataType(const at::ScalarType &data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (", dtype_index, ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions), "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

inline aclDataType GetAclDataType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return static_cast<aclDataType>(t - g_toAclOffset);
    }
    return ConvertToAclDataType(static_cast<at::ScalarType>(t));
}

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetTransformerOpApiLibName(void)
{
    return "libopapi_transformer.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

static inline void TryLoadCustOpApiFromEnv(const char *envName, const std::string &subDir,
                                           std::vector<void *> &handlers, std::set<std::string> &loadedPaths)
{
    const char *env = std::getenv(envName);
    if (env == nullptr) {
        return;
    }
    std::istringstream stream(env);
    std::string dir;
    while (std::getline(stream, dir, ':')) {
        if (dir.empty()) {
            continue;
        }
        std::string soPathStr = dir + subDir + GetCustOpApiLibName();
        char soPath[PATH_MAX] = {0};
        if (realpath(soPathStr.c_str(), soPath) == nullptr) {
            int errnoCopy = errno;
            ASCEND_LOGW("realpath failed for %s, errno=%d(%s).", soPathStr.c_str(), errnoCopy, strerror(errnoCopy));
            continue;
        }
        if (!loadedPaths.insert(soPath).second) {
            continue;
        }
        auto handler = dlopen(soPath, RTLD_LAZY);
        if (handler != nullptr) {
            handlers.push_back(handler);
        } else {
            ASCEND_LOGW("dlopen %s failed, error:%s.", soPath, dlerror());
        }
    }
}

// ### PATCH START: resolve custom operators only from AFD's private OPP
// An external CAM libcust_opapi.so may expose incompatible Mega schemas.
inline std::vector<void *> GetCustOpApiHandlers()
{
    const char *path = std::getenv("AFD_CUST_OPAPI_LIB_PATH");
    TORCH_CHECK(path != nullptr && path[0] == '/',
                "AFD_CUST_OPAPI_LIB_PATH must identify AFD's private libcust_opapi.so");
    auto handler = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(handler != nullptr, "Cannot load AFD custom operators: ", path, ": ", dlerror());
    return {handler};
}
// ### PATCH END: resolve custom operators only from AFD's private OPP

inline void *GetOpApiFuncAddr(const char *apiName)
{
    static auto custHandlers = GetCustOpApiHandlers();
    for (auto handler : custHandlers) {
        auto funcAddr = GetOpApiFuncAddrInLib(handler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    // ### PATCH START: reject missing AFD Mega entry points
    TORCH_CHECK(std::strncmp(apiName, "aclnnAfdAsync", sizeof("aclnnAfdAsync") - 1) != 0,
                "AFD custom operator is missing: ", apiName, "; rebuild the Ascend 950 OPP package");
    // ### PATCH END: reject missing AFD Mega entry points
    static auto transformerOpApiHandler = GetOpApiLibHandler(GetTransformerOpApiLibName());
    if (transformerOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(transformerOpApiHandler, GetTransformerOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor)
{
    c10::Scalar expScalar;
    const at::Tensor *aclInput = &tensor;
    if (aclInput->scalar_type() == at::ScalarType::Double) {
        double value = *(double *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Long) {
        int64_t value = *(int64_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Float) {
        float value = *(float *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Int) {
        int value = *(int *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Half) {
        c10::Half value = *(c10::Half *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
        int8_t value = *(int8_t *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
        c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
        c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
        c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
        c10::Scalar scalar(value);
        expScalar = scalar;
    } else {
        ASCEND_LOGE("unsupported scalar type! ");
    }
    return expScalar;
}

inline at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    return cpuPinMemTensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), deviceIndex),
                              cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    std::vector<int64_t> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    const bool isBaseFormat = IsOpInputBaseFormat(at_tensor);
    if (!isBaseFormat) {
        format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = at_npu::native::get_npu_storage_sizes(at_tensor);
        }
    } else {
        switch (dimNum) {
            case 3:
                format = ACL_FORMAT_NCL;
                break;
            case 4:
                format = ACL_FORMAT_NCHW;
                break;
            case 5:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (acl_data_type != ACL_STRING && Is4BitDtype(acl_data_type)) {
        CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
        storageDims.back() *= FP4_IN_INT8;
        if (!isBaseFormat) {
            auto realFormat = FORMAT_FAKE_TO_REAL.find(format);
            TORCH_CHECK(realFormat != FORMAT_FAKE_TO_REAL.end(), "not support convert ", format, ".");
            format = realFormat->second;
        }
    }

    if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        c10::Scalar expScalar = ConvertTensorToScalar(at_tensor);
        at::Tensor aclInput = CopyScalarToDevice(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        ConvertToAclStorageOffset(at_tensor, acl_data_type), format, storageDims.data(),
                        storageDims.size(), const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

typedef struct {
    const at::Tensor &tensor_;
} StorageShapeTensor;

inline aclTensor *ConvertType(const StorageShapeTensor &storage_shape_tensor)
{
    const at::Tensor &at_tensor = storage_shape_tensor.tensor_;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    aclDataType acl_data_type = ConvertToAclDataType(at_tensor.scalar_type());
    if (!at_tensor.is_contiguous() || !IsOpInputBaseFormat(at_tensor) || Is4BitDtype(acl_data_type)) {
        return ConvertType(at_tensor);
    }
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(at_tensor.scalar_type())) + " has not been supported")
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());
    std::vector<int64_t> storageDims(at_tensor.sizes().begin(), at_tensor.sizes().end());
    aclFormat format = ACL_FORMAT_ND;
    switch (at_tensor.sizes().size()) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }
    return aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                           at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                           const_cast<void *>(at_tensor.storage().data()));
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = ConvertToAclDataType(scalar_data_type);
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        default:
            acl_scalar = nullptr;
            break;
    }
    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

inline aclFloatArray *ConvertType(const std::vector<float> &value)
{
    static const auto aclCreateFloatArray = GET_OP_API_FUNC(aclCreateFloatArray);
    if (aclCreateFloatArray == nullptr) {
        return nullptr;
    }
    return aclCreateFloatArray(value.data(), value.size());
}

template <std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    if (at_tensor_list.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }
    return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }
    return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return ConvertToAclDataType(scalarType);
}

inline const char *ConvertType(const string &str)
{
    return str.c_str();
}

aclTensor *ConvertType(const TensorWrapper &tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &at_tensor = tensor_r.tensor_;

    if (!at_tensor.defined()) {
        return nullptr;
    }

    aclDataType acl_data_type = tensor_r.dtype;
    std::vector<int64_t> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    const bool isBaseFormat = IsOpInputBaseFormat(at_tensor);
    if (!isBaseFormat) {
        format = static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims = at_npu::native::get_npu_storage_sizes(at_tensor);
        }
    } else {
        switch (dimNum) {
            case 3:
                format = ACL_FORMAT_NCL;
                break;
            case 4:
                format = ACL_FORMAT_NCHW;
                break;
            case 5:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (acl_data_type != ACL_STRING && Is4BitDtype(acl_data_type)) {
        CollectB4ShapeInfo(at_tensor, wrapperStride, wrapperShape);
        storageDims.back() *= FP4_IN_INT8;
        if (!isBaseFormat) {
            auto realFormat = FORMAT_FAKE_TO_REAL.find(format);
            TORCH_CHECK(realFormat != FORMAT_FAKE_TO_REAL.end(), "not support convert ", format, ".");
            format = realFormat->second;
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        ConvertToAclStorageOffset(at_tensor, acl_data_type), format, storageDims.data(),
                        storageDims.size(), const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper)
{
    if (tensor_list_wrapper.tensor_list_.size() == 0) {
        return nullptr;
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclFloatArray *p)
{
    static const auto aclDestroyFloatArray = GET_OP_API_FUNC(aclDestroyFloatArray);
    if (aclDestroyFloatArray == nullptr) {
        return;
    }

    aclDestroyFloatArray(p);
}

inline void Release(aclBoolArray *p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

inline const c10::optional<at::Tensor> get_valid_tensor(const c10::optional<at::Tensor> &tensor_opt, at::Device device)
{
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void AddParamToBuf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void AddParamToBuf(const at::Tensor &);
void AddParamToBuf(const at::Scalar &);
void AddParamToBuf(const at::IntArrayRef &);
void AddParamToBuf(const at::ArrayRef<bool> &);
void AddParamToBuf(const at::TensorList &);
void AddParamToBuf(const c10::optional<at::Tensor> &);
void AddParamToBuf(const c10::optional<at::IntArrayRef> &);
void AddParamToBuf(const c10::optional<at::Scalar> &);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string &);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T &arg, Args &...args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

uint64_t CalcHashId();
using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = void (*)(void *, bool);
using ReleaseHugeMem = void (*)(void *, bool);

/**
 * check arg is at::Tensor ?
 */
template <typename T>
struct is_at_tensor : std::false_type {};

template <>
struct is_at_tensor<at::Tensor> : std::true_type {};

/**
 * check arg is at::TensorList ?
 */
template <typename T>
struct is_at_tensor_list : std::false_type {};

template <>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

template <typename T>
struct is_tensor_wrapper : std::false_type {};

template <>
struct is_tensor_wrapper<TensorWrapper> : std::true_type {};

template <>
struct is_tensor_wrapper<StorageShapeTensor> : std::true_type {};

/**
 * find first at::Tensor
 */
template <std::size_t I = 0, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{}

template <std::size_t I = 0, typename... Ts>
    typename std::enable_if < I<sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    } else if constexpr (is_tensor_wrapper<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t).tensor_;
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

/**
 * get the device
 */
template <typename... Ts>
auto DecodeDevice(Ts &...args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

// 弱符号签名须与 torch_npu >= 2.9.0.post3 一致，返回类型不参与名字修饰，修改前请核对对应版本头文件：
//   uint32_t c10_npu::GetDeterministicLevel();                                    (NPUFunctions.h)
//   void at_npu::native::ApplyDeterministicLevel(int64_t, bool, bool);            (op_plugin 内部导出符号)
namespace at_npu {
namespace native {
__attribute__((weak)) void ApplyDeterministicLevel(int64_t level, bool deterministicAlgorithmsStatus, bool isOpapi);
} // namespace native
} // namespace at_npu

namespace c10_npu {
__attribute__((weak)) uint32_t GetDeterministicLevel();
} // namespace c10_npu

inline void ApplyDeterministicConfig()
{
    constexpr bool kUseOpApiPath = true; // isOpapi=true: 走 opapi 路径下发，aclnn_common.h 全场景均为 opapi
    auto deterministicAlgorithmsStatus = at::globalContext().deterministicAlgorithms();
    auto applyLevelFn = at_npu::native::ApplyDeterministicLevel;
    auto getLevelFn = c10_npu::GetDeterministicLevel;
    if (applyLevelFn != nullptr && getLevelFn != nullptr) {
        int64_t level = static_cast<int64_t>(getLevelFn());
        applyLevelFn(level, deterministicAlgorithmsStatus, kUseOpApiPath);
        return;
    }
    static bool fallbackWarned = false;
    if (!fallbackWarned) {
        fallbackWarned = true;
        if (applyLevelFn == nullptr && getLevelFn == nullptr) {
            ASCEND_LOGW("ApplyDeterministicLevel and GetDeterministicLevel are both missing in torch_npu, "
                        "fallback to ACL_OPT_DETERMINISTIC binary switch, deterministic_level will be ignored.");
        } else if (applyLevelFn == nullptr) {
            ASCEND_LOGW("ApplyDeterministicLevel is missing in torch_npu, "
                        "fallback to ACL_OPT_DETERMINISTIC binary switch, deterministic_level will be ignored.");
        } else {
            ASCEND_LOGW("GetDeterministicLevel is missing in torch_npu, "
                        "fallback to ACL_OPT_DETERMINISTIC binary switch, deterministic_level will be ignored.");
        }
    }
    aclError ret = aclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0);
    if (ret != ACL_SUCCESS) {
        ASCEND_LOGW("aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC) failed, ret=%d, "
                    "deterministic config may be incorrect.",
                    static_cast<int>(ret));
    }
}

#define ACLNN_CMD(aclnn_api, ...) \
    do { \
        auto device = DecodeDevice(__VA_ARGS__); \
        const c10::OptionalDeviceGuard device_guard(device); \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize"); \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api); \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal"); \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal"); \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem"); \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ", \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(), ", or ", \
                    GetTransformerOpApiLibName(), "not found."); \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false); \
        if (c10_npu::check_enqueue_need_use(acl_stream)) { \
            aclrtUseStreamResInCurrentThread(acl_stream); \
        } \
        uint64_t workspace_size = 0; \
        uint64_t *workspace_size_addr = &workspace_size; \
        aclOpExecutor *executor = nullptr; \
        aclOpExecutor **executor_addr = &executor; \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr); \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr); \
        if (initMemFunc) { \
            initMemFunc(nullptr, false); \
        } \
        ApplyDeterministicConfig(); \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr); \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr); \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params); \
        if (workspace_status != 0) { \
            ReleaseConvertTypes(converted_params); \
            TORCH_CHECK(false, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
        } \
        at::Tensor workspace_tensor; \
        void *workspace_addr = nullptr; \
        if (workspace_size != 0) { \
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type()); \
            workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte)); \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
        } \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int { \
            if (c10_npu::check_dequeue_need_use(acl_stream)) { \
                aclrtUseStreamResInCurrentThread(acl_stream); \
            } \
            typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream); \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr); \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream); \
            ReleaseConvertTypes(converted_params); \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr); \
            if (releaseMemFunc) { \
                releaseMemFunc(nullptr, false); \
            } \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
            return api_ret; \
        }; \
        at_npu::native::OpCommand cmd; \
        cmd.Name(#aclnn_api); \
        cmd.SetCustomHandler(acl_call); \
        cmd.Run(); \
        if (unInitMemFunc) { \
            unInitMemFunc(nullptr, false); \
        } \
    } while (false)

#endif // CANN_OPS_TRANSFORMER_ACLNN_COMMON_H
