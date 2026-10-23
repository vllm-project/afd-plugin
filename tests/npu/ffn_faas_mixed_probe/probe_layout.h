// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_FAAS_MIXED_PROBE_LAYOUT_H
#define AFD_FAAS_MIXED_PROBE_LAYOUT_H
#include <cstdint>

namespace FaaSProbe {
constexpr uint32_t H = 2048;
constexpr uint32_t I = 1408;
constexpr uint32_t TOP_K = 6;
constexpr uint32_t MAX_TOKENS = 16;
constexpr uint32_t MAX_ROUTES = MAX_TOKENS * TOP_K;
constexpr uint32_t MAX_EXPERTS = 32;
constexpr uint32_t TILE_M = 16;
constexpr uint32_t TILE_N = 128;
constexpr uint32_t PADDED_ROWS = MAX_ROUTES + (TILE_M - 1) * MAX_EXPERTS;
constexpr uint32_t LINE_WORDS = 16;
constexpr uint32_t CODE = 0;
constexpr uint32_t ROWS = 1;
constexpr uint32_t EXPERTS = 2;
constexpr uint32_t TOKENS = 3;
constexpr uint32_t COUNTS = LINE_WORDS;
constexpr uint32_t PREFIX = COUNTS + MAX_EXPERTS;
constexpr uint32_t CURSORS = PREFIX + MAX_EXPERTS + 1;
constexpr uint32_t ROUTES = CURSORS + MAX_EXPERTS;
constexpr uint64_t META_BYTES = ((ROUTES + PADDED_ROWS + LINE_WORDS - 1) / LINE_WORDS) * 64ULL;
constexpr uint64_t PACKED = META_BYTES;
constexpr uint64_t GATE_UP = PACKED + PADDED_ROWS * H * 2ULL;
constexpr uint64_t ACTIVATION = GATE_UP + PADDED_ROWS * 2ULL * I * 2ULL;
constexpr uint64_t DOWN = ACTIVATION + PADDED_ROWS * I * 2ULL;
constexpr uint64_t SCRATCH_BYTES = DOWN + PADDED_ROWS * H * 2ULL;
constexpr int32_t OK = 0;
constexpr int32_t IDLE = 1;
constexpr int32_t INVALID = -1;
constexpr uint16_t COMMAND = 8;
constexpr uint16_t GATE_DONE = 9;
constexpr uint16_t DOWN_READY = 10;
constexpr uint16_t DOWN_DONE = 11;
}  // namespace FaaSProbe
#endif
