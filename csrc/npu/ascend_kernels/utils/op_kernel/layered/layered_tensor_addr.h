/**
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description:
 * Shared layered addressing primitive for cam_async _layered operators.
 *
 * Resolves the current layer's data pointer from an all-layer TensorList
 * descriptor. The descriptor layout matches AscendC ListTensorDesc /
 * GetTensorAddr: the first uint64 holds the byte offset from the descriptor
 * head to the data-pointer array; element i of that array is tensor i's
 * GM data address. Index semantics here is the LAYER number (one list
 * element per layer, each element carrying all experts of that layer).
 *
 * Create: 2026-09-18
 * Note:
 * History: 2026-09-18
 */
#ifndef CAM_UTILS_OP_KERNEL_LAYERED_TENSOR_ADDR_H
#define CAM_UTILS_OP_KERNEL_LAYERED_TENSOR_ADDR_H

template <typename T>
__aicore__ inline __gm__ T* GetLayerTensorAddr(int64_t layerIndex, GM_ADDR allTensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(allTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;

    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + layerIndex));
}

#endif // CAM_UTILS_OP_KERNEL_LAYERED_TENSOR_ADDR_H
