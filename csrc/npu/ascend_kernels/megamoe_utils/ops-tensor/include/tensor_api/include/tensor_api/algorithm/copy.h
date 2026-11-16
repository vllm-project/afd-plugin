/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ALGORITHM_COPY_H
#define INCLUDE_TENSOR_API_ALGORITHM_COPY_H

#include "tensor_api/tensor/layout_interface.h"
#include "tensor_api/tensor/pointer.h"
#include "tensor_api/atom/cube/copy_atom.h"
#include "impl/tensor_api/algorithm/copy_impl.h"

namespace asc {
namespace te {

/**
 * @brief Perform a copy operation with a preconstructed copy_atom.
 * @param atom : Copy atom object that determines the copy behavior.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 */
template <typename Atom, typename DstTensor, typename SrcTensor>
__aicore__ inline void copy(const copy_atom<Atom>& atom, const DstTensor& dst, const SrcTensor& src);

/**
 * @brief Perform a copy operation with a preconstructed copy_atom and a quantization parameter.
 * @param atom : Copy atom object that determines the copy behavior.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param quant : Quantization parameter, which can be a scalar or a Tensor API tensor.
 */
template <typename Atom, typename DstTensor, typename SrcTensor, typename Quant,
          Std::enable_if_t<is_valid_quant_v<Quant>, int> Enable>
__aicore__ inline void copy(const copy_atom<Atom>& atom, const DstTensor& dst, const SrcTensor& src,
                            const Quant& quant);

/**
 * @brief Perform a copy operation using the default atom inferred from the tensor memory locations.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 */
template <typename DstTensor, typename SrcTensor,
          Std::enable_if_t<is_attr_tensor_v<DstTensor> && is_attr_tensor_v<SrcTensor>, int> Enable>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src);

/**
 * @brief Perform a quantized copy using the default atom inferred from the tensor memory locations.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param quant : Quantization parameter, which can be a scalar or a Tensor API tensor.
 */
template <typename DstTensor, typename SrcTensor, typename QuantParam,
    Std::enable_if_t<is_attr_tensor_v<DstTensor> && is_attr_tensor_v<SrcTensor> &&
        is_valid_quant_v<QuantParam>, int> Enable>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src, const QuantParam& quant);

/**
 * @brief Perform a copy operation using source/destination coordinates and a requested shape.
 * @param atom : Copy atom object that determines the copy behavior.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param dst_coord : Start coordinate in the destination tensor, or zero_coord.
 * @param src_coord : Start coordinate in the source tensor, or zero_coord.
 * @param copy_shape : Shape of the region to copy.
 */
template <typename Atom, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
    typename CopyShape, Std::enable_if_t<is_valid_coord_v<DstCoord> && is_valid_coord_v<SrcCoord> &&
        is_valid_shape_v<CopyShape>, int> Enable>
__aicore__ inline void copy(const copy_atom<Atom>& atom, const DstTensor& dst, const SrcTensor& src,
    const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape);

/**
 * @brief Perform a quantized copy operation using source/destination coordinates and a requested shape.
 * @param atom : Copy atom object that determines the copy behavior.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param quant : Quantization parameter, which can be a scalar or a Tensor API tensor.
 * @param dst_coord : Start coordinate in the destination tensor, or zero_coord.
 * @param src_coord : Start coordinate in the source tensor, or zero_coord.
 * @param copy_shape : Shape of the region to copy.
 */
template <typename Atom, typename DstTensor, typename SrcTensor, typename Quant, typename DstCoord,
    typename SrcCoord, typename CopyShape,
    Std::enable_if_t<is_valid_quant_v<Quant> && is_valid_coord_v<DstCoord> &&
        is_valid_coord_v<SrcCoord> && is_valid_shape_v<CopyShape>, int> Enable>
__aicore__ inline void copy(const copy_atom<Atom>& atom, const DstTensor& dst, const SrcTensor& src,
    const Quant& quant, const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape);

/**
 * @brief Perform a region copy using the default atom inferred from the tensor memory locations.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param dst_coord : Start coordinate in the destination tensor, or zero_coord.
 * @param src_coord : Start coordinate in the source tensor, or zero_coord.
 * @param copy_shape : Shape of the region to copy.
 */
template <typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord, typename CopyShape,
    Std::enable_if_t<is_attr_tensor_v<DstTensor> && is_attr_tensor_v<SrcTensor> &&
        is_valid_coord_v<DstCoord> && is_valid_coord_v<SrcCoord> && is_valid_shape_v<CopyShape>, int> Enable>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
    const SrcCoord& src_coord, const CopyShape& copy_shape);

/**
 * @brief Perform a quantized region copy using the default atom inferred from the tensor memory locations.
 * @param dst : Destination tensor.
 * @param src : Source tensor.
 * @param quant : Quantization parameter, which can be a scalar or a Tensor API tensor.
 * @param dst_coord : Start coordinate in the destination tensor, or zero_coord.
 * @param src_coord : Start coordinate in the source tensor, or zero_coord.
 * @param copy_shape : Shape of the region to copy.
 */
template <typename DstTensor, typename SrcTensor, typename QuantParam, typename DstCoord, typename SrcCoord,
    typename CopyShape, Std::enable_if_t<is_attr_tensor_v<DstTensor> && is_attr_tensor_v<SrcTensor> &&
        is_valid_quant_v<QuantParam> && is_valid_coord_v<DstCoord> && is_valid_coord_v<SrcCoord> &&
        is_valid_shape_v<CopyShape>, int> Enable>
__aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src, const QuantParam& quant,
    const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape);

/**
 * @brief Construct a copy_atom from the copy operation object.
 * @param operation : Copy operation object.
 */
template <typename CopyOperation>
__aicore__ inline constexpr auto make_copy(const CopyOperation& operation);

/**
 * @brief Construct a copy_atom from the copy operation object and trait object.
 * @param operation : Copy operation object.
 * @param trait : Copy trait object.
 */
template <typename CopyOperation, typename CopyTrait>
__aicore__ inline constexpr auto make_copy(const CopyOperation& operation, const CopyTrait& trait);

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_ALGORITHM_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
