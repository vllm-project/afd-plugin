// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_FFN_FAAS_TILING_H
#define AFD_FFN_FAAS_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#if defined(__CCE_AICORE__)
#define AFD_FAAS_INLINE __aicore__ inline
#else
#define AFD_FAAS_INLINE inline
#endif

namespace AfdFaas {
constexpr uint32_t ABI_VERSION = 1;
constexpr uint64_t LINE_BYTES = 64;
constexpr uint32_t STATUS_FIELDS = 8;
constexpr uint32_t CONFIG_FIELDS = 13;
constexpr uint32_t TILE_M = 16;
enum ScratchField : uint32_t {
    WORK_CODE, WORK_ROWS, WEIGHT_BASE, WORK_SOURCE, WORK_SEQUENCE, WORK_LAYER
};
constexpr int64_t MAX_GENERATION = 2147483646;
enum ConfigField : uint32_t {
    HIDDEN, INTERMEDIATE, EXPERTS, TOPK, ATTENTION_RANKS, FFN_RANKS,
    ATTENTION_TP, SLOTS, CAPACITY, WORLD_RANK, MODEL_LAYERS, LOADED_LAYERS,
    POLL_BUDGET
};
enum Mode : uint32_t { INITIALIZE, PUBLISH, EXECUTE, COMBINE, END_SOURCE, ABORT_SESSION, PEEK_RESULTS, SERVE, EXCHANGE, SCOPED_EXCHANGE, PREPARE, FINISH };
constexpr uint32_t PUBLISHED_STATE = 0;
constexpr uint32_t CONSUMED_STATE = 1;
enum Result : int64_t { OK = 0, IDLE = 1, BACKPRESSURE = 2, DRAINED = 3,
                        INVALID = -1, TIMED_OUT = -2, PEER_FAILED = -3 };
struct FaaSInfo {
    int64_t config[CONFIG_FIELDS];
    uint32_t mode;
    uint32_t blocks;
    uint32_t inputRows;
    // Internal tiling metadata uses the former alignment padding. This does
    // not change window layout or the FP32 probability contract for callers.
    uint32_t probabilityBf16;
    uint64_t windowBytes;
};
static_assert(sizeof(FaaSInfo) == 128);
struct AfdFfnFaasTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    FaaSInfo info;
};

// Private, executor-owned scratch, reused only after the active submission
// retires. Route numbers include TP rank and capacity, so unequal TP row
// counts do not change another member's offsets. No wire-layout change.
struct ScratchLayout {
    uint32_t counts, prefix, cursors, tpCounts, routes, paddedRows;
    uint64_t packed, gateUp, activation, down, bytes;
    AFD_FAAS_INLINE ScratchLayout(const FaaSInfo &info)
    {
        const auto *c = info.config;
        const uint32_t localExperts = c[EXPERTS] / c[FFN_RANKS];
        paddedRows = c[ATTENTION_TP] * c[CAPACITY] * c[TOPK] + (TILE_M - 1) * localExperts;
        counts = LINE_BYTES / sizeof(int32_t);
        prefix = counts + localExperts;
        cursors = prefix + localExperts + 1;
        tpCounts = cursors + localExperts;
        routes = tpCounts + c[ATTENTION_TP];
        packed = ((routes + paddedRows) * sizeof(int32_t) + LINE_BYTES - 1) / LINE_BYTES * LINE_BYTES;
        gateUp = packed + paddedRows * c[HIDDEN] * sizeof(uint16_t);
        activation = gateUp + paddedRows * 2 * c[INTERMEDIATE] * sizeof(uint16_t);
        down = activation + paddedRows * c[INTERMEDIATE] * sizeof(uint16_t);
        bytes = down + paddedRows * c[HIDDEN] * sizeof(uint16_t);
    }
};

// Each ready/credit word has a separate cache line. Metadata and payload are
// completed by MTE3 before a producer publishes ready; consumers DMA ready
// through MTE2 rather than caching a scalar load from a remote window.
struct Layout {
    uint64_t controlBytes;
    uint64_t inputStride;
    uint64_t resultStride;
    uint64_t idsOffset;
    uint64_t inputBytes;
    uint64_t resultBytes;
    AFD_FAAS_INLINE Layout(const FaaSInfo &info)
    {
        const auto *c = info.config;
        controlBytes = (1 + c[ATTENTION_RANKS] + c[FFN_RANKS] + c[ATTENTION_RANKS]) * LINE_BYTES;
        idsOffset = 3 * LINE_BYTES + c[CAPACITY] * c[HIDDEN] * sizeof(uint16_t);
        inputStride = Align(idsOffset + c[CAPACITY] * c[TOPK] * sizeof(int32_t));
        resultStride = Align(2 * LINE_BYTES + c[CAPACITY] * c[TOPK] * c[HIDDEN] * sizeof(uint16_t));
        inputBytes = controlBytes + c[ATTENTION_RANKS] * c[SLOTS] * inputStride;
        resultBytes = controlBytes + c[FFN_RANKS] * c[SLOTS] * resultStride;
    }
    AFD_FAAS_INLINE static uint64_t Align(uint64_t n)
    {
        return (n + LINE_BYTES - 1) / LINE_BYTES * LINE_BYTES;
    }
};
}  // namespace AfdFaas
#undef AFD_FAAS_INLINE
#endif
