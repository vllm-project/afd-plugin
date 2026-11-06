// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_PULL_PROTOCOL_H
#define AFD_PULL_PROTOCOL_H

#include "kernel_operator.h"
#include "cam_async_comm_group.h"
#include "afd_pull_layout.h"

namespace Cam::Pull {
template <AscendC::HardEvent event> __aicore__ inline void Fence() {
    const auto id = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

// Uses the SDK's complete context; ordinary tensor addresses are never peer addresses.
__aicore__ inline GM_ADDR Window(uint32_t rank, uint32_t self) {
    auto ctx = reinterpret_cast<__gm__ HcclOpResParam *>(AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
    if (rank == self) return reinterpret_cast<GM_ADDR>(ctx->localWindowsIn);
    return reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
        ctx->remoteRes[rank].nextDevicePtr)->windowsIn);
}

class Transport {
public:
    __aicore__ Transport(GM_ADDR state, Shape shape, uint32_t rank, uint32_t cores,
                        AscendC::TPipe &pipe, uint32_t id = AscendC::GetBlockIdx())
        : state_(state), shape_(shape), rank_(rank), cores_(cores), id_(id), pipe_(pipe) {
        pipe_.Reset();
        pipe_.InitBuffer(control_, ALIGNMENT);
        MakeLayout(shape_, layout_);
        const auto ctx = reinterpret_cast<__gm__ HcclOpResParam *>(
            AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>());
        if (ctx->winSize < layout_.bytes || ctx->rankSize != shape_.attnRanks + shape_.ffnRanks ||
            ctx->localUsrRankId != rank_) {
            if (id_ == 0) WriteState(FAILURE, -7);
        }
        if (!ReadState(FAILURE) &&
            (ReadState(SOURCE_CAPACITY) != int64_t(shape_.sourceCapacity) ||
             ReadState(TP_SIZE) != int64_t(shape_.tp))) {
            if (id_ == 0) Fail(0, -6);
        }
        const uint64_t metaBytes = Align(8 * MetadataWords(shape_.tp, shape_.experts));
        pipe_.InitBuffer(metadata_, metaBytes);
        pipe_.InitBuffer(counts_, Align(8 * shape_.ffnRanks * shape_.experts));
        pipe_.InitBuffer(groups_, Align(8 * shape_.experts));
        pipe_.InitBuffer(row_, Align(2 * shape_.hidden + ROW_PADDING));
    }

    __aicore__ int64_t Read(GM_ADDR address) {
        AscendC::GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
        auto value = control_.Get<int64_t>();
        AscendC::DataCopyPad(value, gm, {1U, 8U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE2_S>();
        const int64_t result = value.GetValue(0);
        Fence<AscendC::HardEvent::S_MTE2>();
        return result;
    }

    __aicore__ void Write(GM_ADDR address, int64_t value) {
        auto local = control_.Get<int64_t>();
        local.SetValue(0, value);
        AscendC::GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
        Fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(gm, local, {1U, 8U, 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ int64_t ReadState(uint32_t field) { return Read(state_ + field * 8ULL); }
    __aicore__ void WriteState(uint32_t field, int64_t value) { Write(state_ + field * 8ULL, value); }

    __aicore__ void Initialize() {
        if (rank_ >= shape_.attnRanks) return;
        AscendC::SyncAll<true>();
        if (ReadState(FAILURE)) return;
        auto zero = row_.Get<uint8_t>();
        AscendC::Duplicate(row_.Get<uint32_t>(), uint32_t(0), ALIGNMENT / sizeof(uint32_t));
        Fence<AscendC::HardEvent::V_MTE3>();
        // Keep the payload untouched; only protocol state requires initialization.
        for (uint64_t offset = id_ * ALIGNMENT;
             offset < layout_.counts; offset += cores_ * ALIGNMENT) {
            if (offset >= layout_.returned && offset < layout_.header) continue;
            AscendC::GlobalTensor<uint8_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(Peer(rank_) + offset));
            AscendC::DataCopy(dst, zero, ALIGNMENT);
        }
        Fence<AscendC::HardEvent::MTE3_S>();
        AscendC::SyncAll<true>();
        if (id_ == 0)
            Write(Peer(rank_) + layout_.header + EPOCH * ALIGNMENT, ReadState(SESSION));
    }

    __aicore__ void Fail(uint32_t source, int64_t error) {
        const auto first = ReadState(FAILURE);
        if (!first) WriteState(FAILURE, error);
        else error = first;
        if (rank_ >= shape_.attnRanks) {
            for (uint32_t s = 0; s < shape_.tp; ++s)
                Write(Peer(source + s) + layout_.errors +
                      (rank_ - shape_.attnRanks) * ALIGNMENT, error);
        }
    }

    __aicore__ bool Select() {
        if (ReadState(FAILURE)) return false;
        if (ReadState(SOURCE) >= 0) return true;
        if (ReadState(STOP)) return false;
        const auto sources = shape_.attnRanks / shape_.tp;
        const auto first = uint32_t(ReadState(NEXT_SOURCE));
        const auto budget = ReadState(IDLE_BUDGET);
        const auto session = ReadState(SESSION);
        uint64_t scans = 0;
        while (!ReadState(STOP)) {
            for (uint32_t n = 0; n < sources; ++n) {
                const uint32_t dp = (first + n) % sources;
                const uint32_t source = dp * shape_.tp;
                const auto expected = ReadState(LAST_SEQUENCE + dp * WORDS_PER_LINE) + 1;
                bool ready = true;
                int64_t layer = -1;
                for (uint32_t s = 0; s < shape_.tp; ++s) {
                    const auto header = Peer(source + s) + layout_.header;
                    const auto commit = Read(header + COMMIT * ALIGNMENT);
                    if (commit < expected) { ready = false; break; }
                    AscendC::GlobalTensor<int64_t> gm;
                    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(header));
                    auto values = row_.Get<int64_t>();
                    AscendC::DataCopy(values, gm, HEADER_LINES * WORDS_PER_LINE);
                    Fence<AscendC::HardEvent::MTE2_S>();
                    if (commit != expected || values.GetValue(EPOCH * WORDS_PER_LINE) != session) {
                        Fail(source, -2); return false;
                    }
                    const auto currentLayer = values.GetValue(LAYER * WORDS_PER_LINE);
                    if (values.GetValue(ERROR * WORDS_PER_LINE) ||
                        values.GetValue(SEQUENCE * WORDS_PER_LINE) != commit ||
                        values.GetValue(COMMIT * WORDS_PER_LINE) != commit ||
                        (layer >= 0 && layer != currentLayer)) {
                        Fail(source, -3); return false;
                    }
                    layer = currentLayer;
                    if (Read(header + COMMIT * ALIGNMENT) != commit) { ready = false; break; }
                }
                if (!ready) continue;
                const auto mapSize = ReadState(MAP_SIZE);
                if (layer < 0 || layer >= mapSize ||
                    Read(state_ + 8 * (LayerMapOffset(sources) + layer)) < 0) {
                    Fail(source, -4); return false;
                }
                WriteState(SOURCE, source);
                WriteState(ACTIVE_SEQUENCE, expected);
                WriteState(ACTIVE_LAYER, layer);
                WriteState(CURSOR, 0);
                return true;
            }
            if (budget > 0 && ++scans >= uint64_t(budget)) return false;
        }
        return false;
    }

    // Load immutable original counts; slice counts are separate from return coordinates.
    __aicore__ uint64_t LoadMetadata() {
        auto meta = metadata_.Get<int64_t>();
        const auto source = uint32_t(ReadState(SOURCE));
        const uint32_t ffn = rank_ - shape_.attnRanks;
        uint64_t total = 0;
        for (uint32_t s = 0; s < shape_.tp; ++s) {
            AscendC::GlobalTensor<int64_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(Peer(source + s) + layout_.counts));
            auto counts = counts_.Get<int64_t>();
            AscendC::DataCopyPad(counts, gm,
                {1U, uint32_t(8 * shape_.ffnRanks * shape_.experts), 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
            Fence<AscendC::HardEvent::MTE2_S>();
            uint64_t prefix = 0, sourceTotal = 0;
            for (uint32_t i = 0; i < shape_.ffnRanks * shape_.experts; ++i) {
                const auto count = counts.GetValue(i);
                if (count < 0 || uint64_t(count) > shape_.sourceCapacity * shape_.topK) {
                    if (id_ == 0) Fail(source, -5);
                    return 0;
                }
                if (i < ffn * shape_.experts) prefix += count;
                sourceTotal += count;
                if (i / shape_.experts == ffn) {
                    const uint32_t index = 5 + shape_.tp + s * shape_.experts + i % shape_.experts;
                    meta.SetValue(index, count);
                    total += count;
                }
            }
            if (sourceTotal > shape_.sourceCapacity * shape_.topK ||
                uint64_t(Read(Peer(source + s) + layout_.header + ROWS * ALIGNMENT)) != sourceTotal) {
                if (id_ == 0) Fail(source, -5);
                return 0;
            }
            meta.SetValue(5 + s, prefix);
            Fence<AscendC::HardEvent::S_MTE2>();
        }
        meta.SetValue(0, total);
        meta.SetValue(1, source);
        meta.SetValue(2, ReadState(ACTIVE_LAYER));
        meta.SetValue(MetadataWords(shape_.tp, shape_.experts) - 2, ReadState(SESSION));
        meta.SetValue(MetadataWords(shape_.tp, shape_.experts) - 1, ReadState(ACTIVE_SEQUENCE));
        return total;
    }

    // Mega's fused pipeline owns payload DMA; this prepares only control metadata.
    __aicore__ void PrepareRound(GM_ADDR batch, GM_ADDR groupCounts, uint64_t capacity) {
        const bool continuing = ReadState(SOURCE) >= 0;
        Select();
        auto meta = metadata_.Get<int64_t>();
        auto groups = groups_.Get<int64_t>();
        if (ReadState(FAILURE) || ReadState(SOURCE) < 0) {
            ClearMetadata();
            SaveMetadata(batch, groupCounts);
            return;
        }
        // A retains the original counts until the last round has returned its output.
        if (continuing) LoadSnapshot(batch);
        else LoadMetadata();
        const uint64_t total = meta.GetValue(0);
        const auto source = ReadState(SOURCE);
        const auto sequence = ReadState(ACTIVE_SEQUENCE);
        if (!ReadState(FAILURE)) {
            for (uint32_t s = 0; s < shape_.tp; ++s) {
                if (Read(Peer(source + s) + layout_.header + COMMIT * ALIGNMENT) != sequence) {
                    Fail(source, -9); break;
                }
            }
        }
        if (ReadState(FAILURE)) {
            ClearMetadata();
            SaveMetadata(batch, groupCounts);
            return;
        }
        const uint64_t begin = ReadState(CURSOR);
        if (begin > total) {
            Fail(source, -9);
            ClearMetadata();
            SaveMetadata(batch, groupCounts);
            return;
        }
        const uint64_t end = begin + capacity < total ? begin + capacity : total;
        meta.SetValue(3, begin);
        meta.SetValue(4, end);
        if (id_ == 0) {
            WriteState(ROUND_BEGIN, begin);
            WriteState(ROUND_END, end);
        }
        uint64_t block = 0;
        for (uint32_t e = 0; e < shape_.experts; ++e) {
            uint64_t group = 0;
            for (uint32_t s = 0; s < shape_.tp; ++s) {
                const uint64_t count = meta.GetValue(5 + shape_.tp + s * shape_.experts + e);
                const auto slice = Intersect(block, count, begin, capacity);
                group += slice.count;
                block += count;
            }
            groups.SetValue(e, group);
        }
        SaveMetadata(batch, groupCounts);
        Fence<AscendC::HardEvent::MTE3_S>();
    }

    // Called only by the control owner after the mixed-core output drain.
    __aicore__ bool ValidateReturn(GM_ADDR batch, uint64_t inputRows) {
        if (ReadState(FAILURE) || ReadState(SOURCE) < 0) return false;
        LoadSnapshot(batch);
        auto meta = metadata_.Get<int64_t>();
        const auto source = ReadState(SOURCE);
        if (source < 0 || uint64_t(source) + shape_.tp > shape_.attnRanks ||
            source % shape_.tp || meta.GetValue(1) != source ||
            meta.GetValue(2) != ReadState(ACTIVE_LAYER) ||
            meta.GetValue(MetadataWords(shape_.tp, shape_.experts) - 2) != ReadState(SESSION) ||
            meta.GetValue(MetadataWords(shape_.tp, shape_.experts) - 1) != ReadState(ACTIVE_SEQUENCE) ||
            meta.GetValue(3) != ReadState(CURSOR) ||
            meta.GetValue(3) != ReadState(ROUND_BEGIN) ||
            meta.GetValue(4) != ReadState(ROUND_END) ||
            meta.GetValue(3) < 0 || meta.GetValue(4) < meta.GetValue(3) ||
            meta.GetValue(4) > meta.GetValue(0) ||
            uint64_t(meta.GetValue(4) - meta.GetValue(3)) > inputRows) {
            Fail(source >= 0 && uint64_t(source) + shape_.tp <= shape_.attnRanks ? source : 0, -9);
        }
        if (ReadState(FAILURE)) return false;
        return true;
    }

    __aicore__ void CompleteRound(GM_ADDR batch, uint64_t inputRows) {
        if (!ValidateReturn(batch, inputRows)) return;
        auto meta = metadata_.Get<int64_t>();
        const uint64_t end = meta.GetValue(4);
        const uint32_t source = meta.GetValue(1);
        if (end < uint64_t(meta.GetValue(0))) { WriteState(CURSOR, end); return; }
        const auto sequence = ReadState(ACTIVE_SEQUENCE);
        WriteState(LAST_SEQUENCE + source / shape_.tp * WORDS_PER_LINE, sequence);
        WriteState(SOURCE, -1);
        WriteState(CURSOR, 0);
        WriteState(NEXT_SOURCE, (source / shape_.tp + 1) % (shape_.attnRanks / shape_.tp));
        WriteState(COMPLETED, ReadState(COMPLETED) + 1);
        auto flag = control_.Get<int32_t>();
        for (uint32_t s = 0; s < shape_.tp; ++s) {
            flag.SetValue(0, meta.GetValue(5 + s) + 1);
            AscendC::GlobalTensor<int32_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Peer(source + s) + layout_.flags) +
                                rank_ - shape_.attnRanks);
            Fence<AscendC::HardEvent::S_MTE3>();
            AscendC::DataCopyPad(dst, flag, {1U, 4U, 0U, 0U, 0U});
            Fence<AscendC::HardEvent::MTE3_S>();
        }
    }

private:
    __aicore__ GM_ADDR Peer(uint32_t rank) { return Window(rank, rank_); }
    __aicore__ void LoadSnapshot(GM_ADDR batch) {
        AscendC::GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(batch));
        AscendC::DataCopyPad(metadata_.Get<int64_t>(), gm,
            {1U, uint32_t(8 * MetadataWords(shape_.tp, shape_.experts)), 0U, 0U, 0U},
            {false, 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE2_S>();
    }
    __aicore__ void ClearMetadata() {
        auto meta = metadata_.Get<int64_t>();
        auto groups = groups_.Get<int64_t>();
        for (uint32_t i = 0; i < MetadataWords(shape_.tp, shape_.experts); ++i) meta.SetValue(i, 0);
        for (uint32_t e = 0; e < shape_.experts; ++e) groups.SetValue(e, 0);
        const auto offset = LayerMapOffset(shape_.attnRanks / shape_.tp);
        for (int64_t layer = 0; layer < ReadState(MAP_SIZE); ++layer) {
            if (Read(state_ + 8 * (offset + layer)) >= 0) { meta.SetValue(2, layer); break; }
        }
    }
    __aicore__ void SaveMetadata(GM_ADDR batch, GM_ADDR counts) {
        if (id_ != 0) return;
        AscendC::GlobalTensor<int64_t> dstBatch, dstCounts;
        dstBatch.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(batch));
        dstCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(counts));
        Fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(dstBatch, metadata_.Get<int64_t>(),
            {1U, uint32_t(8 * MetadataWords(shape_.tp, shape_.experts)), 0U, 0U, 0U});
        AscendC::DataCopyPad(dstCounts, groups_.Get<int64_t>(),
            {1U, uint32_t(8 * shape_.experts), 0U, 0U, 0U});
        Fence<AscendC::HardEvent::MTE3_S>();
    }
    GM_ADDR state_;
    Shape shape_;
    Layout layout_{};
    uint32_t rank_, cores_, id_;
    AscendC::TPipe &pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> control_, metadata_, counts_, groups_, row_;
};
} // namespace Cam::Pull
#endif
