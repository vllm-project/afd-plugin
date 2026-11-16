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
#warning                                                                                                               \
    "impl/tensor_api/tensor/pointer_adaptor_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file pointer_adaptor_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename Iterator, typename DerivedType>
struct iter_adaptor {
    using iterator = Iterator;
    using reference = typename iter_ref<iterator>::type;   // T&
    using element_type = typename iter_ele<iterator>::type; // rm_ref
    using value_type = typename iter_val<iterator>::type;   // rm_cvf

    __aicore__ inline constexpr iter_adaptor(iterator ptr = {}) : ptr(ptr) {}

    __aicore__ inline constexpr reference operator*() const
    {
        return *ptr;
    }

    template <typename Index>
    __aicore__ inline constexpr reference operator[](const Index& i) const
    {
        auto ic = index_correct(i);
        return ptr[ic];
    }

    template <typename Index>
    __aicore__ inline constexpr DerivedType operator+(const Index& i) const
    {
        auto ic = index_correct(i);
        return {ptr + ic};
    }

    __aicore__ inline constexpr DerivedType operator&(uint64_t mask) const
    {
        return {reinterpret_cast<iterator>(reinterpret_cast<uint64_t>(ptr) & mask)};
    }

    __aicore__ inline constexpr DerivedType operator|(uint64_t mask) const
    {
        return {reinterpret_cast<iterator>(reinterpret_cast<uint64_t>(ptr) | mask)};
    }

    __aicore__ inline constexpr iterator get() const
    {
        return ptr;
    }

    __aicore__ inline constexpr iterator Get() const
    {
        return get();
    }

    __aicore__ inline constexpr friend bool operator==(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr == y.ptr;
    }

    __aicore__ inline constexpr friend bool operator!=(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr != y.ptr;
    }

    __aicore__ inline constexpr friend bool operator<(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr < y.ptr;
    }

    __aicore__ inline constexpr friend bool operator<=(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr <= y.ptr;
    }

    __aicore__ inline constexpr friend bool operator>(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr > y.ptr;
    }

    __aicore__ inline constexpr friend bool operator>=(const DerivedType& x, const DerivedType& y)
    {
        return x.ptr >= y.ptr;
    }

private:
    template <typename Index>
    __aicore__ inline constexpr Index index_correct(const Index& i) const
    {
        if constexpr (is_b4_type<value_type>) {
            return i >> 1;
        } else {
            return i;
        }
    }

    iterator ptr;
};

} // namespace te
} // namespace asc



#endif // IMPL_TENSOR_API_TENSOR_POINTER_ADAPTOR_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
