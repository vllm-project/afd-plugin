// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_MTE_PIPELINE_H
#define AFD_CAM_ASYNC_MTE_PIPELINE_H

#include <cstdint>

#include "kernel_operator.h"

namespace Cam::AsyncMte {
constexpr uint32_t BUFFER_COUNT = 2;

struct CombineSendBufferPlan {
    uint64_t rowsPerSlot;
    uint64_t bytesPerSlot;
    bool doubleBuffered;
};

__aicore__ inline CombineSendBufferPlan PlanCombineSendBuffers(
    uint64_t reusableBytes, uint64_t rowBytes, bool allowDoubleBuffer)
{
    if (rowBytes == 0) {
        return {0, 0, false};
    }
    const uint64_t fullRows = reusableBytes / rowBytes;
    const bool useTwoSlots = allowDoubleBuffer && fullRows >= BUFFER_COUNT;
    const uint64_t rowsPerSlot = useTwoSlots ? fullRows / BUFFER_COUNT : fullRows;
    return {rowsPerSlot, rowsPerSlot * rowBytes, useTwoSlots};
}

// DR already owns its first row. Its existing address slice has priority over
// the optional second row, so the original one-row UB boundary stays valid.
__aicore__ inline bool CanUseReceiveSecondSlot(uint64_t reusableBytes,
    uint64_t addressSliceBytes, uint64_t wireRowBytes, uint64_t rowCount)
{
    return rowCount >= BUFFER_COUNT && wireRowBytes != 0
        && addressSliceBytes <= reusableBytes
        && wireRowBytes <= reusableBytes - addressSliceBytes;
}

class CannEventBackend {
public:
    using EventId = AscendC::TEventID;

    __aicore__ inline void Init(AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
    }

    template <AscendC::HardEvent event>
    __aicore__ inline EventId Allocate()
    {
        return pipe_->AllocEventID<event>();
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void Release(EventId id)
    {
        pipe_->ReleaseEventID<event>(id);
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void Signal(EventId id)
    {
        AscendC::SetFlag<event>(static_cast<int32_t>(id));
    }

    template <AscendC::HardEvent event>
    __aicore__ inline void Wait(EventId id)
    {
        AscendC::WaitFlag<event>(static_cast<int32_t>(id));
    }

    __aicore__ inline void FenceToScalar()
    {
        const auto id = pipe_->FetchEventID<AscendC::HardEvent::MTE2_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
    }

private:
    AscendC::TPipe *pipe_{nullptr};
};

// The backend seam lets CPU tests execute this same event/slot lifecycle with
// asynchronous FIFO pipelines. It does not change production event ownership.
template <typename Backend>
class PipelineEvents {
public:
    __aicore__ inline PipelineEvents() = default;
    __aicore__ inline explicit PipelineEvents(Backend backend) : backend_(backend) {}

    __aicore__ inline void Init(AscendC::TPipe *pipe)
    {
        backend_.Init(pipe);
        for (uint32_t slot = 0; slot < BUFFER_COUNT; ++slot) {
            ready_[slot] = backend_.template Allocate<AscendC::HardEvent::MTE2_MTE3>();
            free_[slot] = backend_.template Allocate<AscendC::HardEvent::MTE3_MTE2>();
            busy_[slot] = false;
        }
        initialized_ = true;
    }

    __aicore__ inline void WaitForReuse(uint32_t slot)
    {
        if (busy_[slot]) {
            backend_.template Wait<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
            busy_[slot] = false;
        }
    }

    __aicore__ inline void MarkLoaded(uint32_t slot)
    {
        backend_.template Signal<AscendC::HardEvent::MTE2_MTE3>(ready_[slot]);
    }

    __aicore__ inline void WaitLoaded(uint32_t slot)
    {
        backend_.template Wait<AscendC::HardEvent::MTE2_MTE3>(ready_[slot]);
    }

    // All MTE3 reads of this slot, including a quantized row's scale copy,
    // must be submitted before returning the slot credit to MTE2.
    __aicore__ inline void MarkWritten(uint32_t slot)
    {
        backend_.template Signal<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
        busy_[slot] = true;
    }

    __aicore__ inline void Drain()
    {
        if (!initialized_) {
            return;
        }
        for (uint32_t slot = 0; slot < BUFFER_COUNT; ++slot) {
            WaitForReuse(slot);
        }
        // The free waits run on MTE2. Close MTE3 -> MTE2 -> S before releasing
        // IDs, reusing UB for completion metadata, or publishing flags.
        backend_.FenceToScalar();
        for (uint32_t slot = 0; slot < BUFFER_COUNT; ++slot) {
            backend_.template Release<AscendC::HardEvent::MTE2_MTE3>(ready_[slot]);
            backend_.template Release<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
        }
        initialized_ = false;
    }

private:
    Backend backend_;
    typename Backend::EventId ready_[BUFFER_COUNT]{};
    typename Backend::EventId free_[BUFFER_COUNT]{};
    bool busy_[BUFFER_COUNT]{};
    bool initialized_{false};
};

using TwoSlotPipeline = PipelineEvents<CannEventBackend>;
} // namespace Cam::AsyncMte

#endif // AFD_CAM_ASYNC_MTE_PIPELINE_H
