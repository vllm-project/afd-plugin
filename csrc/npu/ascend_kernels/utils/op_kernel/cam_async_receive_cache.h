// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_RECEIVE_CACHE_H
#define AFD_CAM_ASYNC_RECEIVE_CACHE_H

#include <cstdint>

#ifndef CAM_ASYNC_RECEIVE_CACHE_INLINE
#define CAM_ASYNC_RECEIVE_CACHE_INLINE inline
#define CAM_ASYNC_RECEIVE_CACHE_DEFAULT_INLINE
#endif

namespace Cam::AsyncReceiveCache {
constexpr uint32_t CACHE_WAYS = 2;

// Called after the original expert-prefix walk locates the AIV slice. A slice
// starting exactly at that expert's end conservatively remains cache eligible.
// Single selected experts and slices inside one initial expert keep P0 slots.
CAM_ASYNC_RECEIVE_CACHE_INLINE bool CanUseCache(uint32_t startExpert,
    uint32_t endExpert, uint32_t tokenEnd, uint32_t initialExpertEnd)
{
    return startExpert != endExpert && tokenEnd > initialExpertEnd;
}

struct RowKey {
    uint64_t sourceAddress;
    uint32_t wireIndex;
};

struct RowSelection {
    uint32_t slot;
    bool hit;
};

// Metadata only: the existing two leased UB rows retain complete wire bytes,
// including the dynamic quantization scale. This cache is local to one routed
// receive call; no entry survives a TP/source/chunk change or source-window reuse.
// A remembered key identifies a real load whose MTE2-ready and MTE3-free events
// have been scheduled. It does not claim that the scalar has observed completion.
// Every selected slot, including a hit, must first acquire its prior MTE3-free
// credit and perform the MTE2->MTE3 handoff before another output reads that row.
class TwoRowCache {
public:
    CAM_ASYNC_RECEIVE_CACHE_INLINE TwoRowCache() = default;

    CAM_ASYNC_RECEIVE_CACHE_INLINE RowSelection Select(const RowKey &key) const
    {
        for (uint32_t slot = 0; slot < CACHE_WAYS; ++slot) {
            if (valid_[slot] && keys_[slot].sourceAddress == key.sourceAddress
                && keys_[slot].wireIndex == key.wireIndex) {
                return {slot, true};
            }
        }
        for (uint32_t slot = 0; slot < CACHE_WAYS; ++slot) {
            if (!valid_[slot]) {
                return {slot, false};
            }
        }
        return {(mostRecentSlot_ + 1) % CACHE_WAYS, false};
    }

    // slot must come from Select. Call only after a miss has queued a real full
    // wire-row load, both cases have queued their readiness wait and all output
    // reads (payload and scale), and the slot's MTE3-free credit has been signaled.
    // Remembering after MarkWritten ensures even a pending hit has the ownership
    // dependency that proves the earlier real load completed before it is read.
    CAM_ASYNC_RECEIVE_CACHE_INLINE void Remember(uint32_t slot, const RowKey &key)
    {
        keys_[slot] = key;
        valid_[slot] = true;
        mostRecentSlot_ = slot;
    }

private:
    RowKey keys_[CACHE_WAYS]{};
    bool valid_[CACHE_WAYS]{};
    uint32_t mostRecentSlot_{0};
};
} // namespace Cam::AsyncReceiveCache

#ifdef CAM_ASYNC_RECEIVE_CACHE_DEFAULT_INLINE
#undef CAM_ASYNC_RECEIVE_CACHE_DEFAULT_INLINE
#undef CAM_ASYNC_RECEIVE_CACHE_INLINE
#endif
#endif // AFD_CAM_ASYNC_RECEIVE_CACHE_H
