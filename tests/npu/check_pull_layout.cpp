// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>
#include "utils/op_kernel/afd_pull_layout.h"
#include "utils/op_kernel/afd_native_mega_workspace.h"

int main() {
    using namespace Cam::Pull;
    Layout layout{};
    assert(MakeLayout({2048, 4096, 8, 8, 8, 32, 4}, layout));
    assert(layout.payload >= layout.returned + 128ULL * 1024 * 1024);
    assert(layout.bytes - layout.payload == 72ULL * 1024 * 1024);
    assert(!MakeLayout({8192, 4096, 8, 8, 8, 32, 4}, layout));
    assert(!MakeLayout({2048, 4096, 8, 8, 7, 32, 4}, layout));
    assert(!MakeLayout({UINT64_MAX, 4096, 8, 8, 8, 32, 4}, layout));
    assert(TransportUbBytes({8, 256, 2, 383, 1, 256, 1}) > 192 * 1024);
    assert(PublisherUbBytes({8, 256, 2, 383, 1, 256, 1}, 48) > 192 * 1024);
    const Shape largePublisher{8, 7168, 8, 4, 1, 128, 1};
    assert(PublisherUbBytes(largePublisher, 48) + TransportUbBytes(largePublisher) > 192 * 1024);
    assert(CombineUbBytes({8, 256, 2, 4, 1, 256, 1}, 48) > 192 * 1024);
    assert(CombineUbBytes({128, 4096, 6, 4, 4, 64, 4}, 48) <= 192 * 1024);
    const Cam::NativeMegaWorkspace native(512, 4096, 2048, 128, 4);
    assert(native.a2 - native.a1 >= 512ULL * 4096);
    assert(native.c1 - native.a2 >= 512ULL * 2048);
    assert(native.c2 - native.c1 >= 8ULL * 512 * 2048);
    assert(native.bytes - native.c2 >= 4ULL * 512 * 4096);
    assert(native.layer - native.tileCounts >= 8ULL * 128 * (512 / Cam::NATIVE_TILE_ROWS));
    assert(native.a1 - native.x >= 512ULL * 4096);
    std::mt19937 random(20261027);
    for (uint32_t trial = 0; trial < 1000; ++trial) {
        const uint32_t sources = 1 + random() % 4, experts = 1 + random() % 8;
        const uint64_t capacity = 1 + random() % 23;
        std::vector<uint64_t> counts(sources * experts), prefixes(counts.size());
        uint64_t rows = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            for (uint32_t s = 0; s < sources; ++s) {
                const auto i = s * experts + e;
                prefixes[i] = rows;
                counts[i] = random() % 29;
                rows += counts[i];
            }
        }
        std::vector<unsigned> visits(rows);
        for (uint64_t q = 0; q < Rounds(rows, capacity); ++q) {
            uint64_t roundRows = 0;
            for (uint32_t s = 0; s < sources; ++s) {
                uint64_t expertPrefix = 0;
                for (uint32_t e = 0; e < experts; ++e) {
                    const auto i = s * experts + e;
                    const auto slice = Intersect(prefixes[i], counts[i], q * capacity, capacity);
                    for (uint64_t r = 0; r < slice.count; ++r) {
                        ++visits[prefixes[i] + slice.skip + r];
                        assert(slice.local + r < capacity);
                        assert(ReturnRow(0, expertPrefix, slice.skip, r) <
                               expertPrefix + counts[i]);
                    }
                    expertPrefix += counts[i];
                    roundRows += slice.count;
                }
            }
            assert(roundRows == std::min(capacity, rows - q * capacity));
        }
        assert(std::all_of(visits.begin(), visits.end(), [](unsigned n) { return n == 1; }));
    }
}
