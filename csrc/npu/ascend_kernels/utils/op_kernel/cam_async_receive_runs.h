// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_ASYNC_RECEIVE_RUNS_H
#define AFD_CAM_ASYNC_RECEIVE_RUNS_H

#include <cstdint>

#ifndef CAM_ASYNC_RECEIVE_RUNS_INLINE
#define CAM_ASYNC_RECEIVE_RUNS_INLINE inline
#define CAM_ASYNC_RECEIVE_RUNS_DEFAULT_INLINE
#endif

namespace Cam::AsyncReceiveRuns {
constexpr uint64_t RUN_ALIGNMENT_BYTES = 32;
constexpr uint32_t RUN_TILE_COUNT = 2;
constexpr uint32_t MIN_TILE_ROWS = 2;
constexpr uint32_t MAX_TILE_ROWS = 128;
constexpr uint64_t MAX_TILE_BYTES = 64 * 1024;
constexpr uint64_t QUANT_WIRE_PADDING_BYTES = 32;

struct ReceiveRunPlan {
    uint32_t rowsPerTile{0};
    uint64_t bytesPerTile{0};
    bool enabled{false};
};

// Two NEW contiguous tile arenas follow the retained address slice. The old
// mandatory row and optional P0 row are not assumed adjacent. A failed plan
// retains the original cache/P0 path and its original minimum UB requirement.
CAM_ASYNC_RECEIVE_RUNS_INLINE ReceiveRunPlan plan_receive_runs(uint64_t reusableBytes,
    uint64_t addressSliceBytes, uint64_t wireRowBytes, uint32_t ownedRows)
{
    if (addressSliceBytes > reusableBytes || addressSliceBytes % RUN_ALIGNMENT_BYTES != 0 || wireRowBytes == 0
        || wireRowBytes % RUN_ALIGNMENT_BYTES != 0 || wireRowBytes > MAX_TILE_BYTES
        || ownedRows < MIN_TILE_ROWS) return {};
    const uint64_t spare = reusableBytes - addressSliceBytes;
    const uint64_t budgetRows = spare / RUN_TILE_COUNT / wireRowBytes;
    const uint64_t sizeRows = MAX_TILE_BYTES / wireRowBytes;
    uint64_t rows = budgetRows < sizeRows ? budgetRows : sizeRows;
    if (rows > MAX_TILE_ROWS) rows = MAX_TILE_ROWS;
    if (rows > ownedRows) rows = ownedRows;
    if (rows < MIN_TILE_ROWS) return {};
    return {static_cast<uint32_t>(rows), rows * wireRowBytes, true};
}

struct PayloadCopyPlan {
    uint16_t blockCount{0};
    uint32_t blockBytes{0};
    uint32_t sourceStrideBlocks{0};
};

// CANN 9.0 dav_c310 CopyUbufToGmAlignV2 uses 32B source stride units for
// DataCopyPad, while blockLen and GM destination stride use bytes. The quant
// payload is aligned, so one source block skips its retained 32B padding.
// Scale extraction remains separate, exact 4B reads from the same tile lease.
CAM_ASYNC_RECEIVE_RUNS_INLINE PayloadCopyPlan plan_payload_copy(uint64_t payloadRowBytes,
    uint64_t wireRowBytes, uint32_t rows)
{
    if (rows == 0 || rows > MAX_TILE_ROWS || payloadRowBytes == 0
        || payloadRowBytes % RUN_ALIGNMENT_BYTES != 0 || wireRowBytes > MAX_TILE_BYTES
        || wireRowBytes < payloadRowBytes || rows > MAX_TILE_BYTES / wireRowBytes) return {};
    if (wireRowBytes == payloadRowBytes) {
        return {1, static_cast<uint32_t>(rows * payloadRowBytes), 0};
    }
    if (wireRowBytes - payloadRowBytes != QUANT_WIRE_PADDING_BYTES) return {};
    return {static_cast<uint16_t>(rows), static_cast<uint32_t>(payloadRowBytes),
        static_cast<uint32_t>(QUANT_WIRE_PADDING_BYTES / RUN_ALIGNMENT_BYTES)};
}

struct ReceiveRun {
    uint32_t expert{0};
    uint32_t wireIndex{0};
    uint32_t outputRow{0};
    uint32_t rows{0};
};

// The backend reads the retained, immutable UB metadata/address snapshot:
// read_count(expert), read_total(expert), read_prior(expert), read_wire(index).
// outputRow follows the original expert-major/TP-major placement. Runs never
// cross an expert boundary or this source call's AIV slice. Every address entry
// is visited in its original order, including singleton/random rows.
template <typename Backend>
class ReceiveRunCursor {
public:
    CAM_ASYNC_RECEIVE_RUNS_INLINE explicit ReceiveRunCursor(Backend backend) : backend_(backend) {}

    CAM_ASYNC_RECEIVE_RUNS_INLINE void init(uint32_t tokenBegin, uint32_t tokenEnd,
        uint32_t startExpert, uint32_t endExpert, uint32_t rowsPerTile)
    {
        begin_ = tokenBegin;
        end_ = tokenEnd;
        position_ = tokenBegin;
        expert_ = startExpert;
        lastExpert_ = endExpert;
        expertBegin_ = 0;
        expertEnd_ = startExpert <= endExpert ? backend_.read_count(startExpert) : 0;
        outputBase_ = 0;
        limit_ = rowsPerTile;
    }

    CAM_ASYNC_RECEIVE_RUNS_INLINE bool next(ReceiveRun &run)
    {
        run = {};
        if (position_ >= end_ || limit_ == 0 || expert_ > lastExpert_) return false;
        while (position_ >= expertEnd_) {
            outputBase_ += backend_.read_total(expert_);
            expertBegin_ = expertEnd_;
            if (expert_ == lastExpert_) return false;
            ++expert_;
            expertEnd_ += backend_.read_count(expert_);
        }
        const uint32_t wireIndex = backend_.read_wire(position_ - begin_);
        uint32_t rows = 1;
        while (rows < limit_ && rows < end_ - position_ && rows < expertEnd_ - position_) {
            // Use promoted arithmetic: uint16 wire index 65535 must never wrap
            // to zero and be mistaken for the next row of a contiguous run.
            if (backend_.read_wire(position_ - begin_ + rows)
                != static_cast<uint64_t>(wireIndex) + rows) break;
            ++rows;
        }
        run = {expert_, wireIndex,
            outputBase_ + backend_.read_prior(expert_) + position_ - expertBegin_, rows};
        position_ += rows;
        return true;
    }

private:
    Backend backend_;
    uint32_t begin_{0};
    uint32_t end_{0};
    uint32_t position_{0};
    uint32_t expert_{0};
    uint32_t lastExpert_{0};
    uint32_t expertBegin_{0};
    uint32_t expertEnd_{0};
    uint32_t outputBase_{0};
    uint32_t limit_{0};
};

template <typename Backend>
CAM_ASYNC_RECEIVE_RUNS_INLINE bool has_contiguous_run(Backend backend,
    uint32_t tokenBegin, uint32_t tokenEnd, uint32_t startExpert, uint32_t endExpert,
    uint32_t rowsPerTile)
{
    ReceiveRunCursor<Backend> cursor(backend);
    cursor.init(tokenBegin, tokenEnd, startExpert, endExpert, rowsPerTile);
    ReceiveRun run;
    while (cursor.next(run)) {
        if (run.rows >= MIN_TILE_ROWS) return true;
    }
    return false;
}
} // namespace Cam::AsyncReceiveRuns

#ifdef CAM_ASYNC_RECEIVE_RUNS_DEFAULT_INLINE
#undef CAM_ASYNC_RECEIVE_RUNS_DEFAULT_INLINE
#undef CAM_ASYNC_RECEIVE_RUNS_INLINE
#endif
#endif // AFD_CAM_ASYNC_RECEIVE_RUNS_H
