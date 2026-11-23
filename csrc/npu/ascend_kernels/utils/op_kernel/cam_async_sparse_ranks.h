// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_SPARSE_RANKS_H
#define AFD_CAM_ASYNC_SPARSE_RANKS_H

#include <cstdint>

#ifndef CAM_ASYNC_SPARSE_RANKS_INLINE
#define CAM_ASYNC_SPARSE_RANKS_INLINE inline
#define CAM_ASYNC_SPARSE_RANKS_DEFAULT_INLINE
#endif

namespace Cam::AsyncSparseRanks {
constexpr uint32_t MASK_RANK_COUNT = 64;
constexpr uint32_t A5_MAX_WORLD_RANKS = 64;

// Matches the existing A5 host admission: positive A/F populations, A+F=world
// and world<=64. In particular production F<=63; a portable 64-bit mask test
// does not admit an additional A5 shape. A3 does not use this path.
CAM_ASYNC_SPARSE_RANKS_INLINE bool CanUseSparseRanks(uint32_t worldRanks,
    uint32_t attentionRanks, uint32_t moeRanks)
{
    return worldRanks != 0 && worldRanks <= A5_MAX_WORLD_RANKS
        && attentionRanks != 0 && attentionRanks < worldRanks
        && moeRanks != 0 && moeRanks == worldRanks - attentionRanks;
}

enum class Visit : uint32_t { First, Repeated, InvalidRank };

// Scalar metadata only. The rank list reuses DS's existing uint8[F] flag arena;
// one first visit appends one owner, so capacity is bounded by min(topk,F).
// BeginToken resets two scalar fields without reading or clearing stale list
// entries. Payload and every assignment keep the token's current wire index;
// only AdvanceOffsets at token end advances each visited owner once.
class TokenRankTracker {
public:
    CAM_ASYNC_SPARSE_RANKS_INLINE explicit TokenRankTracker(uint32_t rankCount)
        : rankCount_(rankCount) {}

    CAM_ASYNC_SPARSE_RANKS_INLINE void BeginToken()
    {
        seen_ = 0;
        visitedCount_ = 0;
    }

    template <typename Backend, typename RankList>
    CAM_ASYNC_SPARSE_RANKS_INLINE Visit RecordRank(RankList &visited, uint32_t rank)
    {
        if (rankCount_ == 0 || rankCount_ > MASK_RANK_COUNT || rank >= rankCount_) {
            return Visit::InvalidRank;
        }
        const uint64_t bit = uint64_t{1} << rank;
        if ((seen_ & bit) != 0) {
            return Visit::Repeated;
        }
        Backend::WriteRank(visited, visitedCount_, rank);
        ++visitedCount_;
        seen_ |= bit;
        return Visit::First;
    }

    // Call once, after all assignments of the current token have been placed.
    // Backend supplies direct list reads and the existing uint16 wire-counter
    // increment. No stale list tail is read, and expert counters are untouched.
    template <typename Backend, typename RankList, typename Offsets>
    CAM_ASYNC_SPARSE_RANKS_INLINE void AdvanceOffsets(RankList &visited, Offsets &offsets) const
    {
        for (uint32_t index = 0; index < visitedCount_; ++index) {
            Backend::IncrementOffset(offsets, Backend::ReadRank(visited, index));
        }
    }

    CAM_ASYNC_SPARSE_RANKS_INLINE uint32_t visitedCount() const { return visitedCount_; }

private:
    const uint32_t rankCount_;
    uint64_t seen_{0};
    uint32_t visitedCount_{0};
};
} // namespace Cam::AsyncSparseRanks

#ifdef CAM_ASYNC_SPARSE_RANKS_DEFAULT_INLINE
#undef CAM_ASYNC_SPARSE_RANKS_DEFAULT_INLINE
#undef CAM_ASYNC_SPARSE_RANKS_INLINE
#endif
#endif // AFD_CAM_ASYNC_SPARSE_RANKS_H
