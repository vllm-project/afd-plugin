// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_COMBINE_TILE_PLAN_H
#define AFD_CAM_ASYNC_COMBINE_TILE_PLAN_H

#include <cstdint>

#include "cam_async_combine_prefetch.h"

#ifndef CAM_ASYNC_COMBINE_TILE_INLINE
#define CAM_ASYNC_COMBINE_TILE_INLINE inline
#define CAM_ASYNC_COMBINE_TILE_DEFAULT_INLINE
#endif

namespace Cam::AsyncCombineTiles {
constexpr uint64_t TILE_ALIGNMENT_BYTES = 32;
constexpr uint32_t TILE_INPUT_SLOTS = 2;
constexpr uint32_t TILE_FLOAT_ROWS = 3;
constexpr uint64_t MAX_TILE_VALUE = ~uint64_t{0};
static_assert(TILE_INPUT_SLOTS == AsyncCombine::PREFETCH_SLOT_COUNT,
    "tile plan and event leases must own the same input slots");

struct TileScratchPlan {
    bool enabled{false};
    uint32_t tile_elements{0};
    uint64_t float_row_bytes{0};
    uint64_t input_row_bytes{0};
    uint64_t address_bytes{0};
    uint64_t scale_bytes{0};
    uint64_t prefix_bytes{0};
    uint64_t address_offset{0};
    uint64_t scale_offset{0};
    uint64_t prefix_offset{0};
    uint64_t row_offset{0};
    uint64_t product_offset{0};
    uint64_t sum_offset{0};
    uint64_t input_offset[TILE_INPUT_SLOTS]{};
    uint64_t total_bytes{0};
};

CAM_ASYNC_COMBINE_TILE_INLINE bool aligned_product(uint64_t count, uint64_t bytes, uint64_t &result)
{
    if (bytes != 0 && count > (MAX_TILE_VALUE - (TILE_ALIGNMENT_BYTES - 1)) / bytes) return false;
    const uint64_t value = count * bytes;
    result = (value + TILE_ALIGNMENT_BYTES - 1) / TILE_ALIGNMENT_BYTES * TILE_ALIGNMENT_BYTES;
    return true;
}
CAM_ASYNC_COMBINE_TILE_INLINE bool append_bytes(uint64_t bytes, uint64_t &cursor, uint64_t &offset)
{
    if (bytes > MAX_TILE_VALUE - cursor) return false;
    offset = cursor;
    cursor += bytes;
    return true;
}

// available_bytes begins after the unchanged expert metadata. The original
// mandatory full-H layout must still fit; this path does not relax host limits.
// K has no new fixed cap: checked descriptor storage either fits or selects the
// existing path. Prefer this path only if it expands the legacy prefetch tile.
CAM_ASYNC_COMBINE_TILE_INLINE TileScratchPlan plan_tile_scratch(uint64_t available_bytes,
    uint32_t hidden_elements, uint64_t input_element_bytes, uint32_t topk,
    uint32_t owned_tokens, uint64_t prefix_bytes)
{
    if (hidden_elements == 0 || topk == 0 || owned_tokens == 0 || input_element_bytes == 0
        || input_element_bytes > TILE_ALIGNMENT_BYTES
        || TILE_ALIGNMENT_BYTES % input_element_bytes != 0
        || prefix_bytes % TILE_ALIGNMENT_BYTES != 0) return {};
    uint64_t full_float, full_input;
    if (!aligned_product(hidden_elements, sizeof(float), full_float)
        || !aligned_product(hidden_elements, input_element_bytes, full_input)
        || full_float > (MAX_TILE_VALUE - prefix_bytes) / TILE_FLOAT_ROWS) return {};
    uint64_t baseline = prefix_bytes + full_float * TILE_FLOAT_ROWS;
    if (full_input > MAX_TILE_VALUE - baseline) return {};
    baseline += full_input;
    if (baseline > available_bytes) return {};
    TileScratchPlan plan;
    plan.prefix_bytes = prefix_bytes;
    if (!aligned_product(topk, sizeof(uint64_t), plan.address_bytes)
        || !aligned_product(topk, sizeof(float), plan.scale_bytes)) return {};
    uint64_t cursor = 0;
    if (!append_bytes(plan.address_bytes, cursor, plan.address_offset)
        || !append_bytes(plan.scale_bytes, cursor, plan.scale_offset)
        || !append_bytes(prefix_bytes, cursor, plan.prefix_offset)
        || cursor > available_bytes) return {};
    const uint64_t bytes_per_element = TILE_FLOAT_ROWS * sizeof(float)
        + TILE_INPUT_SLOTS * input_element_bytes;
    uint64_t element_alignment = TILE_ALIGNMENT_BYTES / input_element_bytes;
    if (element_alignment < TILE_ALIGNMENT_BYTES / sizeof(float))
        element_alignment = TILE_ALIGNMENT_BYTES / sizeof(float);
    const uint64_t capacity = (available_bytes - cursor) / bytes_per_element
        / element_alignment * element_alignment;
    if (capacity == 0 || capacity >= hidden_elements) return {};
    plan.tile_elements = static_cast<uint32_t>(capacity);
    if (!aligned_product(plan.tile_elements, sizeof(float), plan.float_row_bytes)
        || !aligned_product(plan.tile_elements, input_element_bytes, plan.input_row_bytes)
        || plan.input_row_bytes < AsyncCombine::MIN_PREFETCH_TILE_BYTES) return {};
    const auto legacy = AsyncCombine::PlanCombinePrefetch(available_bytes - baseline,
        hidden_elements, input_element_bytes, static_cast<uint64_t>(owned_tokens) * topk);
    if (legacy.enabled && plan.tile_elements <= legacy.tileElements) return {};
    if (!append_bytes(plan.float_row_bytes, cursor, plan.row_offset)
        || !append_bytes(plan.float_row_bytes, cursor, plan.product_offset)
        || !append_bytes(plan.float_row_bytes, cursor, plan.sum_offset)) return {};
    for (uint32_t slot = 0; slot < TILE_INPUT_SLOTS; ++slot)
        if (!append_bytes(plan.input_row_bytes, cursor, plan.input_offset[slot])) return {};
    if (cursor > available_bytes) return {};
    plan.total_bytes = cursor;
    plan.enabled = true;
    return plan;
}

struct HiddenKTile {
    uint32_t logical_index{0};
    uint32_t hidden_offset{0};
    uint32_t hidden_elements{0};
};
class HiddenKCursor {
public:
    CAM_ASYNC_COMBINE_TILE_INLINE void init(uint32_t hidden, uint32_t tile, uint32_t topk)
    {
        hidden_ = hidden;
        tile_ = tile;
        topk_ = topk;
        offset_ = 0;
        logical_ = 0;
    }
    CAM_ASYNC_COMBINE_TILE_INLINE bool has_next() const
    {
        return offset_ < hidden_ && tile_ != 0 && topk_ != 0;
    }
    CAM_ASYNC_COMBINE_TILE_INLINE HiddenKTile current() const
    {
        const uint32_t remaining = hidden_ - offset_;
        return {logical_, offset_, remaining < tile_ ? remaining : tile_};
    }
    CAM_ASYNC_COMBINE_TILE_INLINE void advance()
    {
        ++logical_;
        if (logical_ == topk_) {
            offset_ += current().hidden_elements;
            logical_ = 0;
        }
    }
private:
    uint32_t hidden_{0};
    uint32_t tile_{0};
    uint32_t topk_{0};
    uint32_t offset_{0};
    uint32_t logical_{0};
};

// The caller supplies bounded compact-v2 row/prefix values and a registered
// window; arithmetic uses 64-bit addresses, matching the existing wire contract.
// Each descriptor is frozen once, before any hidden tile issues a source read.
template <typename RoutingBackend>
CAM_ASYNC_COMBINE_TILE_INLINE void capture_token_sources(uint32_t token_id,
    uint32_t topk, uint32_t hidden_elements, uint64_t input_element_bytes,
    uint64_t source_window, uint64_t wire_header_bytes, RoutingBackend &backend)
{
    const uint64_t row_bytes = static_cast<uint64_t>(hidden_elements) * input_element_bytes;
    for (uint32_t logical = 0; logical < topk; ++logical) {
        const uint32_t contribution = token_id * topk + logical;
        const uint32_t expert = backend.route_expert(contribution);
        const uint32_t occurrence = backend.take_occurrence(expert);
        const uint64_t row_index = static_cast<uint64_t>(backend.rank_prefix(expert))
            + backend.expert_prefix(expert) + occurrence;
        const uint64_t address = source_window + wire_header_bytes + row_bytes * row_index;
        backend.store_descriptor(logical, address, backend.weight(contribution));
    }
}

// Descriptor storage may be overwritten for the next token only after closing
// every V/output -> MTE2 credit back to Scalar. Input packets capture scalar
// addresses/weights at issue time; this boundary also drains any delayed reads.
// Event IDs stay leased across tokens; only the caller's final Drain releases them.
template <typename Pipeline, typename Backend>
CAM_ASYNC_COMBINE_TILE_INLINE void retire_token_descriptor_use(Pipeline &pipeline, Backend &backend)
{
    for (uint32_t slot = 0; slot < TILE_INPUT_SLOTS; ++slot) pipeline.WaitForReuse(slot);
    backend.FenceToScalar();
}
} // namespace Cam::AsyncCombineTiles
#ifdef CAM_ASYNC_COMBINE_TILE_DEFAULT_INLINE
#undef CAM_ASYNC_COMBINE_TILE_DEFAULT_INLINE
#undef CAM_ASYNC_COMBINE_TILE_INLINE
#endif
#endif // AFD_CAM_ASYNC_COMBINE_TILE_PLAN_H
