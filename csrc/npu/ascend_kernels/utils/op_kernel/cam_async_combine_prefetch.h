// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_COMBINE_PREFETCH_H
#define AFD_CAM_ASYNC_COMBINE_PREFETCH_H

#include <cstdint>

#include "kernel_operator.h"

namespace Cam::AsyncCombine {
constexpr uint32_t PREFETCH_SLOT_COUNT = 2;
constexpr uint64_t PREFETCH_ALIGNMENT_BYTES = 32;
// Structural guard against many tiny transfers, not a measured A5 crossover.
constexpr uint64_t MIN_PREFETCH_TILE_BYTES = 2048;
constexpr uint64_t MAX_PREFETCH_VALUE = ~uint64_t{0};

struct CombinePrefetchPlan {
    uint32_t tileElements{0};
    uint64_t secondSlotBytes{0};
    bool enabled{false};
};

// The original input row and three full-hidden FP32 scratch rows remain owned by
// CR. Only this second input slot is carved from spare UB. A small spare region
// selects hidden tiles instead of raising the original minimum UB requirement.
// Tiles below the structural transfer threshold retain the original serial path;
// a final hidden tail may be shorter than that threshold.
__aicore__ inline CombinePrefetchPlan PlanCombinePrefetch(uint64_t spareBytes,
    uint32_t hiddenElements, uint64_t inputElementBytes, uint64_t ownedContributions)
{
    if (hiddenElements == 0 || inputElementBytes == 0 || ownedContributions == 0
        || inputElementBytes > PREFETCH_ALIGNMENT_BYTES
        || PREFETCH_ALIGNMENT_BYTES % inputElementBytes != 0
        || hiddenElements > MAX_PREFETCH_VALUE / inputElementBytes) return {};
    const uint64_t rowBytes = static_cast<uint64_t>(hiddenElements) * inputElementBytes;
    if (rowBytes > MAX_PREFETCH_VALUE - (PREFETCH_ALIGNMENT_BYTES - 1)) return {};
    const uint64_t alignedRowBytes = (rowBytes + PREFETCH_ALIGNMENT_BYTES - 1)
        / PREFETCH_ALIGNMENT_BYTES * PREFETCH_ALIGNMENT_BYTES;
    const uint64_t alignedSpareBytes = spareBytes / PREFETCH_ALIGNMENT_BYTES * PREFETCH_ALIGNMENT_BYTES;
    const uint64_t secondSlotBytes = alignedSpareBytes < alignedRowBytes
        ? alignedSpareBytes : alignedRowBytes;
    if (secondSlotBytes == 0) return {};
    const uint64_t capacityElements = secondSlotBytes / inputElementBytes;
    const uint32_t tileElements = capacityElements < hiddenElements
        ? static_cast<uint32_t>(capacityElements) : hiddenElements;
    const bool enabled = secondSlotBytes >= MIN_PREFETCH_TILE_BYTES
        && (ownedContributions >= PREFETCH_SLOT_COUNT || tileElements < hiddenElements);
    return {tileElements, secondSlotBytes, enabled};
}

struct CombineTile {
    uint32_t contributionIndex{0};
    uint32_t hiddenOffset{0};
    uint32_t hiddenElements{0};
};

// Issue order is contribution-major then hidden-major. Different hidden elements
// are independent; each element still observes increasing logical top-k order.
// Metadata occurrence counters advance once per contribution, not once per tile.
class CombineTileCursor {
public:
    __aicore__ inline void Init(uint32_t contributionBegin, uint32_t contributionEnd,
        uint32_t hiddenElements, uint32_t tileElements)
    {
        contribution_ = contributionBegin;
        end_ = contributionEnd;
        hidden_ = hiddenElements;
        tile_ = tileElements;
        offset_ = 0;
    }
    __aicore__ inline bool HasNext() const
    {
        return contribution_ < end_ && hidden_ != 0 && tile_ != 0;
    }
    __aicore__ inline CombineTile Current() const
    {
        const uint32_t remaining = hidden_ - offset_;
        return {contribution_, offset_, remaining < tile_ ? remaining : tile_};
    }
    __aicore__ inline void Advance()
    {
        offset_ += Current().hiddenElements;
        if (offset_ == hidden_) {
            offset_ = 0;
            ++contribution_;
        }
    }

private:
    uint32_t contribution_{0};
    uint32_t end_{0};
    uint32_t hidden_{0};
    uint32_t tile_{0};
    uint32_t offset_{0};
};

class CannPrefetchBackend {
public:
    using EventId = AscendC::TEventID;
    __aicore__ inline void Init(AscendC::TPipe *pipe) { pipe_ = pipe; }
    template <AscendC::HardEvent event>
    __aicore__ inline EventId Allocate() { return pipe_->AllocEventID<event>(); }
    template <AscendC::HardEvent event>
    __aicore__ inline void Release(EventId id) { pipe_->ReleaseEventID<event>(id); }
    template <AscendC::HardEvent event>
    __aicore__ inline void Signal(EventId id) { AscendC::SetFlag<event>(static_cast<int32_t>(id)); }
    template <AscendC::HardEvent event>
    __aicore__ inline void Wait(EventId id) { AscendC::WaitFlag<event>(static_cast<int32_t>(id)); }
    __aicore__ inline void FenceToScalar()
    {
        const auto id = pipe_->FetchEventID<AscendC::HardEvent::MTE2_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
    }
private:
    AscendC::TPipe *pipe_{nullptr};
};

// MTE2 owns a slot until MTE2_V ready; V returns credit only after its last read.
// At final top-k the caller also waits MTE3_V for the cast/output using that slot
// before MarkSourceConsumed. Backend injection exercises these exact leases in
// CPU tests; it does not establish transport visibility or GM flag semantics.
template <typename Backend>
class PipelineEvents {
public:
    __aicore__ inline PipelineEvents() = default;
    __aicore__ inline explicit PipelineEvents(Backend backend) : backend_(backend) {}
    __aicore__ inline void Init(AscendC::TPipe *pipe)
    {
        backend_.Init(pipe);
        for (uint32_t slot = 0; slot < PREFETCH_SLOT_COUNT; ++slot) {
            ready_[slot] = backend_.template Allocate<AscendC::HardEvent::MTE2_V>();
            free_[slot] = backend_.template Allocate<AscendC::HardEvent::V_MTE2>();
            busy_[slot] = false;
        }
        initialized_ = true;
    }
    __aicore__ inline void WaitForReuse(uint32_t slot)
    {
        if (busy_[slot]) {
            backend_.template Wait<AscendC::HardEvent::V_MTE2>(free_[slot]);
            busy_[slot] = false;
        }
    }
    __aicore__ inline void MarkLoaded(uint32_t slot)
    {
        backend_.template Signal<AscendC::HardEvent::MTE2_V>(ready_[slot]);
    }
    __aicore__ inline void WaitLoaded(uint32_t slot)
    {
        backend_.template Wait<AscendC::HardEvent::MTE2_V>(ready_[slot]);
    }
    __aicore__ inline void MarkSourceConsumed(uint32_t slot)
    {
        backend_.template Signal<AscendC::HardEvent::V_MTE2>(free_[slot]);
        busy_[slot] = true;
    }
    __aicore__ inline void Drain()
    {
        if (!initialized_) return;
        for (uint32_t slot = 0; slot < PREFETCH_SLOT_COUNT; ++slot) WaitForReuse(slot);
        // Close V -> MTE2 -> S before releasing IDs or resetting remote flags.
        backend_.FenceToScalar();
        for (uint32_t slot = 0; slot < PREFETCH_SLOT_COUNT; ++slot) {
            backend_.template Release<AscendC::HardEvent::MTE2_V>(ready_[slot]);
            backend_.template Release<AscendC::HardEvent::V_MTE2>(free_[slot]);
        }
        initialized_ = false;
    }
private:
    Backend backend_;
    typename Backend::EventId ready_[PREFETCH_SLOT_COUNT]{};
    typename Backend::EventId free_[PREFETCH_SLOT_COUNT]{};
    bool busy_[PREFETCH_SLOT_COUNT]{};
    bool initialized_{false};
};
using TwoSlotPrefetch = PipelineEvents<CannPrefetchBackend>;
} // namespace Cam::AsyncCombine
#endif // AFD_CAM_ASYNC_COMBINE_PREFETCH_H
