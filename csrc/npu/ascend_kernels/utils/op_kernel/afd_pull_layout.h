// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_PULL_LAYOUT_H
#define AFD_PULL_LAYOUT_H

#include <cstdint>

#ifdef __CCE_AICORE__
#define AFD_PULL_DEVICE __aicore__
#else
#define AFD_PULL_DEVICE
#endif

namespace Cam { namespace Pull {
constexpr uint32_t VERSION = 1;
constexpr uint32_t ALIGNMENT = 64;
constexpr uint32_t MAX_AIVS = 48;
constexpr uint32_t ROW_PADDING = 512;
constexpr uint64_t MAX_SOURCE_ASSIGNMENTS = 65535;
constexpr uint64_t MAX_CAPACITY = 262144;
constexpr int64_t MAGIC = 0x41464401;
constexpr int64_t INIT_MAGIC = 0x41464402;
constexpr uint32_t WORDS_PER_LINE = ALIGNMENT / sizeof(int64_t);
enum State : uint32_t {
    SESSION = 0, SOURCE_CAPACITY = 8, TP_SIZE = 9, MAP_SIZE = 16, FAILURE = 24,
    SOURCE = 32, ACTIVE_SEQUENCE = 40, ROUND_BEGIN = 48, ROUND_END = 56,
    IDLE_BUDGET = 64, ACTIVE_LAYER = 72, CURSOR = 80, NEXT_SOURCE = 88,
    COMPLETED = 96, STOP = 104, LAST_SEQUENCE = 112
};
AFD_PULL_DEVICE constexpr uint64_t LayerMapOffset(uint64_t sources) {
    return LAST_SEQUENCE + sources * WORDS_PER_LINE;
}
AFD_PULL_DEVICE constexpr uint64_t LaneErrorsOffset(uint64_t sources, uint64_t mapSize) {
    return LayerMapOffset(sources) + (mapSize + WORDS_PER_LINE - 1) / WORDS_PER_LINE * WORDS_PER_LINE;
}
AFD_PULL_DEVICE constexpr uint64_t StateWords(uint64_t sources, uint64_t mapSize) {
    return LaneErrorsOffset(sources, mapSize) + MAX_AIVS * WORDS_PER_LINE;
}

// One control line has one writer; payload/counts precede the commit line.
enum Header : uint32_t { EPOCH, SEQUENCE, LAYER, TOKENS, ROWS, COMMIT, ERROR, HEADER_LINES };

struct Shape {
    uint64_t sourceCapacity, hidden, topK, ffnRanks, attnRanks, experts, tp;
};

AFD_PULL_DEVICE constexpr uint64_t Align(uint64_t n) {
    return (n + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

struct Layout {
    uint64_t stats, flags, returned, header, errors, counts, payload, rowStride, bytes;
};

// The old Combine reads 32B-aligned statistics and flags at these exact offsets.
AFD_PULL_DEVICE constexpr uint64_t Align32(uint64_t n) { return (n + 31) / 32 * 32; }

AFD_PULL_DEVICE inline bool MakeLayout(const Shape &s, Layout &out) {
    if (s.sourceCapacity == 0 || s.hidden == 0 || s.hidden > 8192 || s.hidden % 256 ||
        s.topK == 0 || s.ffnRanks == 0 || s.attnRanks == 0 || s.experts == 0 ||
        s.ffnRanks > 384 || s.attnRanks > 384 || s.experts > 256 ||
        s.ffnRanks + s.attnRanks > 384 || s.tp == 0 || s.attnRanks % s.tp ||
        s.topK > s.ffnRanks * s.experts ||
        s.sourceCapacity > MAX_SOURCE_ASSIGNMENTS / s.topK ||
        s.sourceCapacity > MAX_CAPACITY / s.tp) return false;
    out.stats = 0;
    out.flags = Align32(4 * (s.ffnRanks + s.ffnRanks * s.experts) * MAX_AIVS);
    out.returned = out.flags + Align32(4 * s.ffnRanks);
    out.header = Align(out.returned + 2 * s.sourceCapacity * s.topK * s.hidden);
    out.errors = out.header + HEADER_LINES * ALIGNMENT;
    out.counts = Align(out.errors + s.ffnRanks * ALIGNMENT);
    out.payload = Align(out.counts + 8 * s.ffnRanks * s.experts);
    out.rowStride = s.hidden + ROW_PADDING;
    out.bytes = out.payload + s.sourceCapacity * s.topK * out.rowStride;
    return true;
}

AFD_PULL_DEVICE constexpr uint64_t MetadataWords(uint64_t tp, uint64_t experts) {
    return 7 + tp + tp * experts;
}
AFD_PULL_DEVICE constexpr uint64_t TransportUbBytes(const Shape &s) {
    return ALIGNMENT + Align(8 * MetadataWords(s.tp, s.experts)) +
        Align(8 * s.ffnRanks * s.experts) + Align(8 * s.experts) +
        Align(2 * s.hidden + ROW_PADDING);
}

// Publish retains these buffers alongside Transport for the entire phase.
AFD_PULL_DEVICE inline uint64_t PublisherStatsBytes(const Shape &s, uint64_t cores) {
    const auto stats = Align(4 * (s.ffnRanks + s.ffnRanks * s.experts) * cores);
    // Reuse this buffer for the per-lane error lines before loading token offsets.
    return stats > cores * ALIGNMENT ? stats : cores * ALIGNMENT;
}
AFD_PULL_DEVICE inline uint64_t PublisherUbBytes(const Shape &s, uint64_t cores) {
    const auto e = s.ffnRanks * s.experts;
    return PublisherStatsBytes(s, cores) + 2 * Align(4 * e) +
        Align(4 * s.topK) + Align(11 * s.hidden + 128);
}

// Retain the former admission bound while accounting for Mega's per-token routes.
AFD_PULL_DEVICE inline uint64_t CombineUbBytes(const Shape &s, uint64_t cores) {
    const auto e = s.ffnRanks * s.experts;
    const auto base = 65536 + Align32(2 * e);
    const auto preprocess = base + Align32(4 * (s.ffnRanks + e) * cores);
    const auto retained = base + Align32(4 * e) + Align32(2 * e) +
        3 * Align32(4 * s.hidden) + Align32(2 * s.hidden) + Align32(4 * s.ffnRanks);
    const auto legacy = preprocess > retained ? preprocess : retained;
    const auto mega = 2 * Align(4 * e) + 2 * Align(4 * s.topK) + Align(14 * s.hidden);
    return legacy > mega ? legacy : mega;
}

struct Slice {
    uint64_t count, skip, local;
};

// Empty intersections carry no usable source offset.
AFD_PULL_DEVICE inline Slice Intersect(uint64_t block, uint64_t count,
                                         uint64_t begin, uint64_t capacity) {
    const uint64_t lo = block > begin ? block : begin;
    const uint64_t end = begin + capacity;
    const uint64_t hi = block + count < end ? block + count : end;
    return hi > lo ? Slice{hi - lo, lo - block, lo - begin} : Slice{0, 0, 0};
}

AFD_PULL_DEVICE constexpr uint64_t Rounds(uint64_t rows, uint64_t capacity) {
    return rows == 0 ? 1 : 1 + (rows - 1) / capacity;
}

AFD_PULL_DEVICE constexpr uint64_t ReturnRow(uint64_t rankPrefix, uint64_t expertPrefix,
                                            uint64_t skip, uint64_t row) {
    return rankPrefix + expertPrefix + skip + row;
}
} } // namespace Cam::Pull

#undef AFD_PULL_DEVICE
#endif
