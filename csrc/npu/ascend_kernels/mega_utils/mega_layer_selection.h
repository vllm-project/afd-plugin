/* Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2. */
#pragma once
#include <cstdint>

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#define CAM_MEGA_LAYER_INLINE __aicore__ inline
#else
#define CAM_MEGA_LAYER_INLINE inline
#endif

namespace AfdAsyncLayer {
constexpr uint32_t TASK_BYTES = 64U;
constexpr uint32_t MAX_LAYERS = 1U << 24U;
constexpr uint32_t MAX_TENSORS = 1U << 16U;

CAM_MEGA_LAYER_INLINE uint32_t LayerCount(uint32_t tensors, uint32_t localExperts, bool perExpert)
{
    if (tensors == 0U || tensors > MAX_TENSORS || localExperts == 0U ||
        (perExpert && tensors % localExperts != 0U)) {
        return 0U;
    }
    return perExpert ? tensors / localExperts : tensors;
}

CAM_MEGA_LAYER_INLINE uint64_t TaskAreaBytes(uint32_t worldSize)
{
    return (static_cast<uint64_t>(worldSize) * TASK_BYTES + 511U) / 512U * 512U;
}

CAM_MEGA_LAYER_INLINE uint32_t LayerTag(uint32_t sequence, uint32_t layer)
{
    return (((sequence & 0x7FU) | 0x80U) << 24U) | layer;
}

CAM_MEGA_LAYER_INLINE bool Matches(uint32_t tag, uint32_t fullSequence, uint32_t expected)
{
    return fullSequence == expected && (tag >> 24U) == ((expected & 0x7FU) | 0x80U);
}

CAM_MEGA_LAYER_INLINE uint32_t DecodeLayer(uint32_t tag)
{
    return tag & 0x00FFFFFFU;
}

struct WeightIndex {
    uint32_t tensor;
    uint32_t expert;
};

// Per-expert: one 2D tensor per (layer,expert). Stacked: one 3D tensor per layer.
CAM_MEGA_LAYER_INLINE WeightIndex SelectWeight(uint32_t layer, uint32_t expert,
                                                  uint32_t localExperts, bool perExpert)
{
    return perExpert ? WeightIndex{layer * localExperts + expert, 0U} : WeightIndex{layer, expert};
}
} // namespace AfdAsyncLayer

#undef CAM_MEGA_LAYER_INLINE
