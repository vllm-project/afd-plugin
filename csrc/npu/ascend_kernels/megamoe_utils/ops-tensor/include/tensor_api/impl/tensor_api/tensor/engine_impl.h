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
    "impl/tensor_api/tensor/engine_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file engine_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H
#define IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename Iterator>
struct view_engine {
    using iterator = Iterator;
    using reference = typename iter_ref<iterator>::type;    // T&
    using element_type = typename iter_ele<iterator>::type; // rm_ref
    using value_type = typename iter_val<iterator>::type;   // rm_cvf
    __aicore__ inline constexpr iterator const& begin() const
    {
        return storage;
    }

    __aicore__ inline constexpr iterator& begin()
    {
        return storage;
    }

    __aicore__ inline constexpr view_engine(iterator storage = {}) : storage(storage) {}
    __aicore__ inline constexpr view_engine(iterator storage, cache_mode mode) : storage(storage), mode(mode) {}
    __aicore__ inline constexpr view_engine(iterator storage, AscendC::Te::CacheMode mode) :
        storage(storage), mode(normalize_cache_mode(mode))
    {}

    __aicore__ inline constexpr uint8_t get_cache_mode() const
    {
        return static_cast<uint8_t>(mode);
    }

    __aicore__ inline constexpr void set_cache_mode(cache_mode mode)
    {
        this->mode = mode;
    }

    template <typename Index>
    __aicore__ inline constexpr view_engine operator+(const Index& i) const
    {
        auto iter = begin() + i;
        return {iter, mode};
    }

    __aicore__ inline constexpr iterator const& Begin() const
    {
        return begin();
    }

    __aicore__ inline constexpr iterator& Begin()
    {
        return begin();
    }

    __aicore__ inline constexpr uint8_t GetCacheMode() const
    {
        return get_cache_mode();
    }

    __aicore__ inline constexpr void SetCacheMode(AscendC::Te::CacheMode cache_mode)
    {
        set_cache_mode(normalize_cache_mode(cache_mode));
    }

private:
    iterator storage;
    cache_mode mode = cache_mode::normal;
};

template <typename Iterator>
struct const_view_engine {
    using iterator = Iterator;
    using reference = typename iter_ref<iterator>::type;    // T&
    using element_type = typename iter_ele<iterator>::type; // rm_ref
    using value_type = typename iter_val<iterator>::type;   // rm_cvf

    __aicore__ inline constexpr iterator const& begin() const
    {
        return storage;
    }

    __aicore__ inline constexpr iterator const& Begin() const
    {
        return begin();
    }
    __aicore__ inline constexpr const_view_engine(iterator storage = {}) : storage(storage) {}

private:
    iterator storage;
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_ENGINE_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
