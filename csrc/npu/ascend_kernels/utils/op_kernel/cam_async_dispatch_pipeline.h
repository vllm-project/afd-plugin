// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_DISPATCH_PIPELINE_H
#define AFD_CAM_ASYNC_DISPATCH_PIPELINE_H

#include <cstdint>

#include "kernel_operator.h"

namespace Cam::AsyncDispatch {
constexpr uint32_t TOKEN_SLOT_COUNT = 2;
constexpr uint64_t SLOT_ALIGNMENT_BYTES = 32;
constexpr uint64_t MAX_BUFFER_VALUE = ~uint64_t{0};

struct DispatchBufferPlan {
    uint64_t secondSlotBytes{0};
    bool enabled{false};
};

// DS already aliases its original input and wire-output views. Quantization
// first reads that input into its existing FP32 scratch, then overwrites the
// same row with wire bytes and the original four-byte scale. The optional slot
// repeats that aliasing layout and requires only one extra input-row extent.
// Shared FP32 scratch is not duplicated. The spare budget already excludes the
// original mandatory arena and, for SIMT keys, the original reserved resources.
__aicore__ inline DispatchBufferPlan plan_dispatch_buffers(uint64_t spareBytes,
    uint32_t inputElements, uint64_t inputElementBytes, uint32_t wireElements,
    uint64_t wireElementBytes, uint64_t ownedTokens)
{
    if (ownedTokens < TOKEN_SLOT_COUNT || inputElements == 0 || wireElements == 0
        || inputElementBytes == 0 || wireElementBytes == 0
        || inputElementBytes > SLOT_ALIGNMENT_BYTES || wireElementBytes > SLOT_ALIGNMENT_BYTES
        || SLOT_ALIGNMENT_BYTES % inputElementBytes != 0
        || SLOT_ALIGNMENT_BYTES % wireElementBytes != 0
        || inputElements > MAX_BUFFER_VALUE / inputElementBytes
        || wireElements > MAX_BUFFER_VALUE / wireElementBytes) return {};
    const uint64_t inputBytes = static_cast<uint64_t>(inputElements) * inputElementBytes;
    const uint64_t wireBytes = static_cast<uint64_t>(wireElements) * wireElementBytes;
    if (inputBytes > MAX_BUFFER_VALUE - (SLOT_ALIGNMENT_BYTES - 1)) return {};
    const uint64_t slotBytes = (inputBytes + SLOT_ALIGNMENT_BYTES - 1)
        / SLOT_ALIGNMENT_BYTES * SLOT_ALIGNMENT_BYTES;
    // The mandatory first row cannot be enlarged by this optional feature.
    if (wireBytes > slotBytes || slotBytes > spareBytes) return {};
    return {slotBytes, true};
}

class CannDispatchBackend {
public:
    using EventId = AscendC::TEventID;
    __aicore__ inline void init(AscendC::TPipe *pipe) { pipe_ = pipe; }
    template <AscendC::HardEvent Event>
    __aicore__ inline EventId allocate() { return pipe_->AllocEventID<Event>(); }
    template <AscendC::HardEvent Event>
    __aicore__ inline void release(EventId id) { pipe_->ReleaseEventID<Event>(id); }
    template <AscendC::HardEvent Event>
    __aicore__ inline void signal(EventId id) { AscendC::SetFlag<Event>(static_cast<int32_t>(id)); }
    template <AscendC::HardEvent Event>
    __aicore__ inline void wait(EventId id) { AscendC::WaitFlag<Event>(static_cast<int32_t>(id)); }
    __aicore__ inline void fence_to_scalar()
    {
        const auto id = pipe_->FetchEventID<AscendC::HardEvent::MTE2_S>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<int32_t>(id));
    }
private:
    AscendC::TPipe *pipe_{nullptr};
};

// One complete row slot owns both input and its in-place wire representation.
// q1 consumes input on V; q0 consumes it directly on MTE3. QuantProcess retains
// its original V FIFO arithmetic and V/S/MTE3 scale handoffs. The MTE3->MTE2
// free signal follows ALL peer reads of the token, including its scale bytes.
// It does not retire tokenOffsetBatchTensor metadata: that separate scalar-owned
// staging still needs the original S_MTE3 / MTE3_S flush fences.
template <typename Backend, bool DynamicQuant>
class PipelineEvents {
public:
    static constexpr AscendC::HardEvent INPUT_READY_EVENT = DynamicQuant
        ? AscendC::HardEvent::MTE2_V : AscendC::HardEvent::MTE2_MTE3;

    __aicore__ inline PipelineEvents() = default;
    __aicore__ inline explicit PipelineEvents(Backend backend) : backend_(backend) {}
    __aicore__ inline void init(AscendC::TPipe *pipe)
    {
        backend_.init(pipe);
        for (uint32_t slot = 0; slot < TOKEN_SLOT_COUNT; ++slot) {
            ready_[slot] = backend_.template allocate<INPUT_READY_EVENT>();
            free_[slot] = backend_.template allocate<AscendC::HardEvent::MTE3_MTE2>();
            if constexpr (DynamicQuant) {
                vectorReady_[slot] = backend_.template allocate<AscendC::HardEvent::V_MTE3>();
                scaleReady_[slot] = backend_.template allocate<AscendC::HardEvent::S_MTE3>();
            }
            busy_[slot] = false;
        }
        initialized_ = true;
    }
    __aicore__ inline void wait_for_reuse(uint32_t slot)
    {
        if (busy_[slot]) {
            backend_.template wait<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
            busy_[slot] = false;
        }
    }
    __aicore__ inline void mark_loaded(uint32_t slot)
    {
        backend_.template signal<INPUT_READY_EVENT>(ready_[slot]);
    }
    __aicore__ inline void wait_loaded(uint32_t slot)
    {
        backend_.template wait<INPUT_READY_EVENT>(ready_[slot]);
    }
    __aicore__ inline void mark_token_sent(uint32_t slot)
    {
        backend_.template signal<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
        busy_[slot] = true;
    }
    // Original QuantProcess tail handoffs with slot-exclusive IDs. Borrowed
    // SyncFunc IDs cannot be reused by another token before MTE3 consumes them.
    __aicore__ inline void mark_quantized(uint32_t slot)
    {
        if constexpr (DynamicQuant) {
            backend_.template signal<AscendC::HardEvent::V_MTE3>(vectorReady_[slot]);
            backend_.template wait<AscendC::HardEvent::V_MTE3>(vectorReady_[slot]);
        }
    }
    __aicore__ inline void mark_scale_ready(uint32_t slot)
    {
        if constexpr (DynamicQuant) {
            backend_.template signal<AscendC::HardEvent::S_MTE3>(scaleReady_[slot]);
            backend_.template wait<AscendC::HardEvent::S_MTE3>(scaleReady_[slot]);
        }
    }
    __aicore__ inline void drain()
    {
        if (!initialized_) return;
        for (uint32_t slot = 0; slot < TOKEN_SLOT_COUNT; ++slot) wait_for_reuse(slot);
        // Both free waits execute on MTE2. Close MTE3 -> MTE2 -> S before
        // releasing IDs, final metadata flushes or the original SetFlags.
        backend_.fence_to_scalar();
        for (uint32_t slot = 0; slot < TOKEN_SLOT_COUNT; ++slot) {
            backend_.template release<INPUT_READY_EVENT>(ready_[slot]);
            backend_.template release<AscendC::HardEvent::MTE3_MTE2>(free_[slot]);
            if constexpr (DynamicQuant) {
                backend_.template release<AscendC::HardEvent::V_MTE3>(vectorReady_[slot]);
                backend_.template release<AscendC::HardEvent::S_MTE3>(scaleReady_[slot]);
            }
        }
        initialized_ = false;
    }
private:
    Backend backend_;
    typename Backend::EventId ready_[TOKEN_SLOT_COUNT]{};
    typename Backend::EventId free_[TOKEN_SLOT_COUNT]{};
    typename Backend::EventId vectorReady_[TOKEN_SLOT_COUNT]{};
    typename Backend::EventId scaleReady_[TOKEN_SLOT_COUNT]{};
    bool busy_[TOKEN_SLOT_COUNT]{};
    bool initialized_{false};
};

template <bool DynamicQuant>
using TwoSlotPipeline = PipelineEvents<CannDispatchBackend, DynamicQuant>;
} // namespace Cam::AsyncDispatch
#endif // AFD_CAM_ASYNC_DISPATCH_PIPELINE_H
