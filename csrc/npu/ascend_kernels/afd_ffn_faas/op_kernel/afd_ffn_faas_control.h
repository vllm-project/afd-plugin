// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_FFN_FAAS_CONTROL_H
#define AFD_FFN_FAAS_CONTROL_H
#include "kernel_operator.h"
#include "afd_ffn_faas_tiling.h"
#include "cam_async_comm_group.h"

using namespace AscendC;


namespace AfdFaas {
constexpr uint32_t EXPERTS_PER_LINE = LINE_BYTES / sizeof(int32_t);

__aicore__ inline void FlushScratch(const GlobalTensor<int32_t> &meta)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(meta);
}

template <HardEvent Event>
__aicore__ inline void Fence()
{
    const auto id = GetTPipePtr()->FetchEventID(Event);
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// The control entry owns readiness, source/layer selection, packing and
// result publication. The executor inserts model computation between phases.
class FaaSKernel {
public:
    __aicore__ inline FaaSKernel(const FaaSInfo &info, TPipe &pipe)
        : info_(info), layout_(info), pipe_(pipe)
    {
        c_ = info_.config;
        block_ = GetBlockIdx();
        context_ = reinterpret_cast<__gm__ HcclOpResParam *>(GetHcclContext<HCCL_GROUP_ID_0>());
        const uint32_t width = c_[HIDDEN] > c_[INTERMEDIATE] ? c_[HIDDEN] : c_[INTERMEDIATE];
        pipe_.InitBuffer(metaBuffer_, LINE_BYTES);
        pipe_.InitBuffer(copyBuffer_, width * sizeof(float));
        pipe_.InitBuffer(xBuffer_, width * sizeof(float));
        pipe_.InitBuffer(sumBuffer_, Layout::Align(c_[TOPK] * sizeof(float)));
        pipe_.InitBuffer(outputBuffer_, c_[HIDDEN] * sizeof(float));
        meta_ = metaBuffer_.Get<int64_t>();
        copy_ = copyBuffer_.Get<bfloat16_t>();
        x_ = xBuffer_.Get<float>();
        sum_ = sumBuffer_.Get<float>();
        output_ = outputBuffer_.Get<float>();
    }

    __aicore__ inline void Bind(GM_ADDR x, GM_ADDR ids, GM_ADDR probabilities,
                               GM_ADDR layers,
                               GM_ADDR request, GM_ADDR state, GM_ADDR scratch, GM_ADDR y, GM_ADDR status)
    {
        input_ = x;
        ids_ = ids;
        probabilities_ = probabilities;
        layers_ = layers;
        request_ = request;
        state_ = state;
        scratch_ = scratch;
        y_ = y;
        status_ = status;
    }

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR ids, GM_ADDR probabilities,
                              GM_ADDR layers,
                              GM_ADDR request, GM_ADDR state, GM_ADDR scratch, GM_ADDR y, GM_ADDR status)
    {
        Bind(x, ids, probabilities, layers, request, state, scratch, y, status);
        if (info_.mode == INITIALIZE) {
            Initialize();
        } else if (info_.mode == PUBLISH) {
            Publish();
        } else if (info_.mode == EXCHANGE || info_.mode == SCOPED_EXCHANGE) {
            const bool published = Publish();
            // Publish's final ready/state/status writes belong to block zero.
            // All AIVs must observe their completion before deciding whether
            // to combine. BACKPRESSURE means nothing was published;
            // TIMED_OUT can only come from the subsequent combine phase.
            SyncAll<true>();
            if (published) {
                Combine(false);
            }
            if (info_.mode == SCOPED_EXCHANGE) {
                // Scoped consumers can be queued before host status validation.
                // Preserve the failure, but never expose uninitialized storage.
                SyncAll<true>();
                ReadLine(status_);
                if (meta_.GetValue(0) != OK) {
                    Duplicate(copy_, static_cast<bfloat16_t>(0.0f), c_[HIDDEN]);
                    Fence<HardEvent::V_MTE3>();
                    GlobalTensor<bfloat16_t> result;
                    result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y_));
                    for (uint32_t row = block_; row < info_.inputRows; row += info_.blocks) {
                        DataCopy(result[row * c_[HIDDEN]], copy_, c_[HIDDEN]);
                    }
                    Fence<HardEvent::MTE3_S>();
                }
                SyncAll<true>();
            }

        } else if (info_.mode == PREPARE) {
            Prepare();
        } else if (info_.mode == FINISH) {
            Finalize();

        } else if (info_.mode == COMBINE || info_.mode == PEEK_RESULTS) {
            Combine(info_.mode == PEEK_RESULTS);
        } else if (info_.mode == END_SOURCE || info_.mode == ABORT_SESSION) {
            Control();
        }
    }

private:
    __aicore__ inline GM_ADDR Window(uint32_t rank)
    {
        if (rank == c_[WORLD_RANK]) {
            return reinterpret_cast<GM_ADDR>(context_->localWindowsIn);
        }
        return reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ HcclRankRelationResV2 *>(
            context_->remoteRes[rank].nextDevicePtr)->windowsIn);
    }
    __aicore__ inline GM_ADDR Input(uint32_t rank, uint32_t attention, uint32_t slot)
    {
        return Window(rank) + layout_.controlBytes + (attention * c_[SLOTS] + slot) * layout_.inputStride;
    }
    __aicore__ inline GM_ADDR Output(uint32_t rank, uint32_t ffn, uint32_t slot)
    {
        return Window(rank) + layout_.controlBytes + (ffn * c_[SLOTS] + slot) * layout_.resultStride;
    }
    __aicore__ inline void ReadLine(GM_ADDR address)
    {
        GlobalTensor<int64_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
        Fence<HardEvent::S_MTE2>();
        DataCopy(meta_, source, LINE_BYTES / sizeof(int64_t));
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline int64_t ReadWord(GM_ADDR address)
    {
        GlobalTensor<int64_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
        Fence<HardEvent::S_MTE2>();
        DataCopyPad(meta_, source, {1U, sizeof(int64_t), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        return meta_.GetValue(0);
    }
    __aicore__ inline void WriteLine(GM_ADDR address)
    {
        GlobalTensor<int64_t> destination;
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address));
        Fence<HardEvent::S_MTE3>();
        DataCopy(destination, meta_, LINE_BYTES / sizeof(int64_t));
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void WriteWord(GM_ADDR address, int64_t value)
    {
        for (uint32_t i = 0; i < LINE_BYTES / sizeof(int64_t); ++i) {
            meta_.SetValue(i, 0);
        }
        meta_.SetValue(0, value);
        WriteLine(address);
    }
    __aicore__ inline int32_t ReadSignal(GM_ADDR address)
    {
        GlobalTensor<int32_t> signal;
        signal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
        auto value = metaBuffer_.Get<int32_t>();
        Fence<HardEvent::S_MTE2>();
        DataCopyPad(value, signal, {1U, sizeof(int32_t), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        return value.GetValue(0);
    }
    __aicore__ inline void WriteSignal(GM_ADDR address, int32_t generation)
    {
        GlobalTensor<int32_t> signal;
        signal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
        auto value = metaBuffer_.Get<int32_t>();
        value.SetValue(0, generation);
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(signal, value, {1U, sizeof(int32_t), 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Status(int64_t code, int64_t source = -1,
                                 int64_t seq = 0, int64_t layer = -1, int64_t count = 0)
    {
        for (uint32_t i = 0; i < STATUS_FIELDS; ++i) {
            meta_.SetValue(i, 0);
        }
        meta_.SetValue(0, code);
        meta_.SetValue(1, source);
        meta_.SetValue(2, seq);
        meta_.SetValue(3, layer);
        meta_.SetValue(4, count);
        WriteLine(status_);
    }
    __aicore__ inline void StoreState(uint32_t index, int64_t value)
    {
        GlobalTensor<int64_t> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(state_));
        meta_.SetValue(0, value);
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(state[index], meta_, {1U, sizeof(int64_t), 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline bool Failed()
    {
        for (uint32_t rank = 0; rank < c_[ATTENTION_RANKS] + c_[FFN_RANKS]; ++rank) {
            if (ReadSignal(Window(c_[WORLD_RANK]) + (rank + 1) * LINE_BYTES) != 0) {
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void Initialize()
    {
        const uint64_t bytes = c_[WORLD_RANK] < c_[ATTENTION_RANKS] ? layout_.resultBytes : layout_.inputBytes;
        const uint32_t chunk = c_[HIDDEN] * sizeof(bfloat16_t);
        Duplicate(copy_, static_cast<bfloat16_t>(0.0f), c_[HIDDEN]);
        Fence<HardEvent::V_MTE3>();
        GlobalTensor<bfloat16_t> window;
        window.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(Window(c_[WORLD_RANK])));
        for (uint64_t offset = block_ * chunk; offset < bytes; offset += info_.blocks * chunk) {
            const uint32_t n = offset + chunk > bytes ? bytes - offset : chunk;
            DataCopyPad(window[offset / sizeof(bfloat16_t)], copy_, {1U, n, 0U, 0U, 0U});
        }
        Fence<HardEvent::MTE3_S>();
        SyncAll<true>();
        if (block_ == 0) {
            const uint32_t sources = c_[ATTENTION_RANKS] / c_[ATTENTION_TP];
            // State is private to the executor, not the HCCL window.
            for (uint32_t i = 0; i <= sources; ++i) {
                GlobalTensor<int64_t> state;
                state.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(state_));
                meta_.SetValue(0, 0);
                Fence<HardEvent::S_MTE3>();
                DataCopyPad(state[i], meta_, {1U, sizeof(int64_t), 0U, 0U, 0U});
                Fence<HardEvent::MTE3_S>();
            }
            Status(OK);
        }
    }
    __aicore__ inline void Request(int64_t &seq, int64_t &layer, int64_t &count)
    {
        seq = ReadWord(request_);
        layer = ReadWord(request_ + sizeof(int64_t));
        count = ReadWord(request_ + 2 * sizeof(int64_t));
    }
    __aicore__ inline void CopyRow(GM_ADDR destination, GM_ADDR source, uint32_t bytes)
    {
        GlobalTensor<bfloat16_t> from;
        GlobalTensor<bfloat16_t> to;
        from.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(source));
        to.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(destination));
        Fence<HardEvent::S_MTE2>();
        DataCopyPad(copy_, from, {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_MTE3>();
        DataCopyPad(to, copy_, {1U, bytes, 0U, 0U, 0U});
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline bool Publish()
    {
        int64_t seq, layer, count;
        Request(seq, layer, count);
        const uint32_t slot = (seq - 1) % c_[SLOTS];
        if (block_ == 0) {
            int64_t code = seq > 0 && layer >= 0 && layer < c_[MODEL_LAYERS] &&
                           count >= 0 && count <= c_[CAPACITY] && count <= info_.inputRows ? OK : INVALID;
            if (seq > MAX_GENERATION) {
                code = INVALID;
            }
            // A owns independent published/consumed counters. Only a fresh
            // sequence can publish; a backpressure return leaves both intact.
            // This check precedes every payload/metadata/ready write.
            if (seq != ReadWord(state_ + PUBLISHED_STATE * sizeof(int64_t)) + 1) {
                code = INVALID;
            }
            if (code == OK) {
                auto indices = metaBuffer_.Get<int32_t>();
                for (uint32_t base = 0; base < count * c_[TOPK]; base += EXPERTS_PER_LINE) {
                    const uint32_t remaining = count * c_[TOPK] - base;
                    const uint32_t n = remaining < EXPERTS_PER_LINE ? remaining : EXPERTS_PER_LINE;
                    ReadExperts(ids_ + base * sizeof(int32_t), n);
                    for (uint32_t j = 0; j < n; ++j) {
                        const auto expert = indices.GetValue(j);
                        if (expert < -1 || expert >= c_[EXPERTS]) {
                            code = INVALID;
                        }
                    }
                }
            }
            if (code == OK) {
                for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
                    const auto credit = ReadSignal(Input(c_[ATTENTION_RANKS] + f, c_[WORLD_RANK], slot) + 2 * LINE_BYTES);
                    if (seq > c_[SLOTS] && credit != seq - c_[SLOTS]) {
                        code = BACKPRESSURE;
                    }
                }
                if (Failed()) {
                    code = PEER_FAILED;
                }
            }
            Status(code);
        }
        SyncAll<true>();
        ReadLine(status_);
        if (meta_.GetValue(0) != OK) {
            return false;
        }
        for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
            const auto destination = Input(c_[ATTENTION_RANKS] + f, c_[WORLD_RANK], slot);
            for (uint32_t row = block_; row < count; row += info_.blocks) {
                CopyRow(destination + 3 * LINE_BYTES + row * c_[HIDDEN] * sizeof(bfloat16_t),
                        input_ + row * c_[HIDDEN] * sizeof(bfloat16_t), c_[HIDDEN] * sizeof(bfloat16_t));
                CopyRow(destination + layout_.idsOffset + row * c_[TOPK] * sizeof(int32_t),
                        ids_ + row * c_[TOPK] * sizeof(int32_t), c_[TOPK] * sizeof(int32_t));
            }
        }
        SyncAll<true>();
        if (block_ == 0) {
            for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
                const auto destination = Input(c_[ATTENTION_RANKS] + f, c_[WORLD_RANK], slot);
                WriteWord(destination + LINE_BYTES, layer);
                meta_.SetValue(1, count);
                meta_.SetValue(2, ABI_VERSION);
                WriteLine(destination + LINE_BYTES);
                WriteSignal(destination, seq);
            }
            StoreState(PUBLISHED_STATE, seq);
            Status(OK, c_[WORLD_RANK] / c_[ATTENTION_TP], seq, layer, count);
        }
        return true;
    }
    __aicore__ inline int32_t Expert(GM_ADDR address)
    {
        GlobalTensor<int32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
        auto integers = metaBuffer_.Get<int32_t>();
        Fence<HardEvent::S_MTE2>();
        DataCopyPad(integers, source, {1U, sizeof(int32_t), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
        return integers.GetValue(0);
    }
    __aicore__ inline void ReadExperts(GM_ADDR address, uint32_t count)
    {
        GlobalTensor<int32_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address));
        auto indices = metaBuffer_.Get<int32_t>();
        Fence<HardEvent::S_MTE2>();
        DataCopyPad(indices, source, {1U, static_cast<uint32_t>(count * sizeof(int32_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Select()
    {
        const uint32_t sources = c_[ATTENTION_RANKS] / c_[ATTENTION_TP];
        const uint32_t cursor = ReadWord(state_) % sources;
        for (uint32_t attempt = 0; attempt < c_[POLL_BUDGET]; ++attempt) {
            if (Failed()) {
                Status(PEER_FAILED);
                return;
            }
            bool drained = true;
            for (uint32_t distance = 0; distance < sources; ++distance) {
                const uint32_t source = (cursor + distance) % sources;
                const int64_t seq = ReadWord(state_ + (source + 1) * sizeof(int64_t)) + 1;
                const uint32_t slot = (seq - 1) % c_[SLOTS];
                bool ready = true;
                int64_t layer = -1;
                int64_t total = 0;
                for (uint32_t tp = 0; tp < c_[ATTENTION_TP]; ++tp) {
                    const uint32_t a = source * c_[ATTENTION_TP] + tp;
                    const auto end = ReadSignal(Window(c_[WORLD_RANK]) +
                        (1 + c_[ATTENTION_RANKS] + c_[FFN_RANKS] + a) * LINE_BYTES);
                    drained = drained && end == seq;
                    const auto input = Input(c_[WORLD_RANK], a, slot);
                    if (ReadSignal(input) != seq) {
                        ready = false;
                        continue;
                    }
                    ReadLine(input + LINE_BYTES);
                    const auto candidate = meta_.GetValue(0);
                    const auto count = meta_.GetValue(1);
                    if ((layer >= 0 && layer != candidate) || candidate < 0 || candidate >= c_[MODEL_LAYERS] ||
                        count < 0 || count > c_[CAPACITY] || meta_.GetValue(2) != ABI_VERSION) {
                        Status(INVALID);
                        return;
                    }
                    layer = candidate;
                    total += count;
                }
                if (ready) {
                    const int32_t compact = Expert(layers_ + layer * sizeof(int32_t));
                    if (compact < 0 || compact >= c_[LOADED_LAYERS]) {
                        Status(INVALID);
                    } else {
                        Status(OK, source, seq, layer, total);
                    }
                    return;
                }
            }
            if (drained) {
                Status(DRAINED);
                return;
            }
        }
        Status(IDLE);
    }
    __aicore__ inline void Group(GlobalTensor<int32_t> &meta, const ScratchLayout &scratch)
    {
        ReadLine(status_);
        meta.SetValue(WORK_CODE, meta_.GetValue(0));
        meta.SetValue(WORK_ROWS, 0);
        if (meta_.GetValue(0) != OK) {
            FlushScratch(meta);
            return;
        }
        const uint32_t source = meta_.GetValue(1);
        const int64_t sequence = meta_.GetValue(2);
        const int64_t modelLayer = meta_.GetValue(3);
        const uint32_t slot = (sequence - 1) % c_[SLOTS];
        const int32_t compact = Expert(layers_ + modelLayer * sizeof(int32_t));
        const uint32_t experts = c_[EXPERTS] / c_[FFN_RANKS];
        const uint32_t first = (c_[WORLD_RANK] - c_[ATTENTION_RANKS]) * experts;
        meta.SetValue(WEIGHT_BASE, compact * experts);
        meta.SetValue(WORK_SOURCE, source);
        meta.SetValue(WORK_SEQUENCE, sequence);
        meta.SetValue(WORK_LAYER, modelLayer);
        for (uint32_t e = 0; e < experts; ++e) {
            meta.SetValue(scratch.counts + e, 0);
        }
        for (uint32_t tp = 0; tp < c_[ATTENTION_TP]; ++tp) {
            const auto input = Input(c_[WORLD_RANK], source * c_[ATTENTION_TP] + tp, slot);
            ReadLine(input + LINE_BYTES);
            const uint32_t count = meta_.GetValue(1);
            meta.SetValue(scratch.tpCounts + tp, count);
            auto indices = metaBuffer_.Get<int32_t>();
            for (uint32_t base = 0; base < count * c_[TOPK]; base += EXPERTS_PER_LINE) {
                const uint32_t remaining = count * c_[TOPK] - base;
                const uint32_t n = remaining < EXPERTS_PER_LINE ? remaining : EXPERTS_PER_LINE;
                ReadExperts(input + layout_.idsOffset + base * sizeof(int32_t), n);
                for (uint32_t j = 0; j < n; ++j) {
                    const int32_t e = indices.GetValue(j) - first;
                    if (e >= 0 && e < experts) {
                        meta.SetValue(scratch.counts + e, meta.GetValue(scratch.counts + e) + 1);
                    }
                }
            }
        }
        uint32_t prefix = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            meta.SetValue(scratch.prefix + e, prefix);
            meta.SetValue(scratch.cursors + e, prefix);
            prefix += ((meta.GetValue(scratch.counts + e) + TILE_M - 1) / TILE_M) * TILE_M;
        }
        meta.SetValue(scratch.prefix + experts, prefix);
        meta.SetValue(WORK_ROWS, prefix);
        for (uint32_t row = 0; row < prefix; ++row) {
            meta.SetValue(scratch.routes + row, -1);
        }
        for (uint32_t tp = 0; tp < c_[ATTENTION_TP]; ++tp) {
            const auto input = Input(c_[WORLD_RANK], source * c_[ATTENTION_TP] + tp, slot);
            const uint32_t count = meta.GetValue(scratch.tpCounts + tp) * c_[TOPK];
            auto indices = metaBuffer_.Get<int32_t>();
            for (uint32_t base = 0; base < count; base += EXPERTS_PER_LINE) {
                const uint32_t remaining = count - base;
                const uint32_t n = remaining < EXPERTS_PER_LINE ? remaining : EXPERTS_PER_LINE;
                ReadExperts(input + layout_.idsOffset + base * sizeof(int32_t), n);
                for (uint32_t j = 0; j < n; ++j) {
                    const int32_t e = indices.GetValue(j) - first;
                    if (e >= 0 && e < experts) {
                        const uint32_t cursor = meta.GetValue(scratch.cursors + e);
                        meta.SetValue(scratch.routes + cursor, tp * c_[CAPACITY] * c_[TOPK] + base + j);
                        meta.SetValue(scratch.cursors + e, cursor + 1);
                    }
                }
            }
        }
        FlushScratch(meta);
    }
    __aicore__ inline void Pack(const GlobalTensor<int32_t> &meta, const ScratchLayout &scratch,
                               uint32_t source, uint32_t slot)
    {
        GlobalTensor<bfloat16_t> packed;
        packed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scratch_ + scratch.packed));
        for (uint32_t row = block_; row < meta.GetValue(WORK_ROWS); row += info_.blocks) {
            const int32_t route = meta.GetValue(scratch.routes + row);
            if (route >= 0) {
                const uint32_t tp = route / (c_[CAPACITY] * c_[TOPK]);
                const uint32_t token = (route % (c_[CAPACITY] * c_[TOPK])) / c_[TOPK];
                CopyRow(scratch_ + scratch.packed + row * c_[HIDDEN] * sizeof(bfloat16_t),
                        Input(c_[WORLD_RANK], source * c_[ATTENTION_TP] + tp, slot) +
                        3 * LINE_BYTES + token * c_[HIDDEN] * sizeof(bfloat16_t),
                        c_[HIDDEN] * sizeof(bfloat16_t));
            } else {
                Duplicate(copy_, static_cast<bfloat16_t>(0), c_[HIDDEN]);
                Fence<HardEvent::V_MTE3>();
                DataCopy(packed[row * c_[HIDDEN]], copy_, c_[HIDDEN]);
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_S>();
            }
        }
    }
    __aicore__ inline void Scatter(const GlobalTensor<int32_t> &meta, const ScratchLayout &scratch,
                                  uint32_t source, uint32_t slot)
    {
        const uint32_t f = c_[WORLD_RANK] - c_[ATTENTION_RANKS];
        for (uint32_t row = block_; row < meta.GetValue(WORK_ROWS); row += info_.blocks) {
            const int32_t route = meta.GetValue(scratch.routes + row);
            if (route >= 0) {
                const uint32_t tp = route / (c_[CAPACITY] * c_[TOPK]);
                const uint32_t localRoute = route % (c_[CAPACITY] * c_[TOPK]);
                CopyRow(Output(source * c_[ATTENTION_TP] + tp, f, slot) + 2 * LINE_BYTES +
                        localRoute * c_[HIDDEN] * sizeof(bfloat16_t),
                        scratch_ + scratch.down + row * c_[HIDDEN] * sizeof(bfloat16_t),
                        c_[HIDDEN] * sizeof(bfloat16_t));
            }
        }
    }
public:
    // These two phases also delimit the quantized backend's one device entry.
    // Scratch owns the selected transaction until Finalize publishes ready.
    __aicore__ inline int64_t Prepare()
    {
        const ScratchLayout scratch(info_);
        GlobalTensor<int32_t> meta;
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratch_));
        if (block_ == 0) {
            Select();
            Group(meta, scratch);
        }
        SyncAll<true>();
        FlushScratch(meta);
        const bool work = meta.GetValue(WORK_CODE) == OK;
        if (work) {
            const uint32_t source = meta.GetValue(WORK_SOURCE);
            const int64_t seq = meta.GetValue(WORK_SEQUENCE);
            const uint32_t slot = (seq - 1) % c_[SLOTS];
            Pack(meta, scratch, source, slot);
        }
        SyncAll<true>();
        if (work && info_.mode == PREPARE && block_ == 0) {
            Status(OK, meta.GetValue(WORK_SOURCE), meta.GetValue(WORK_SEQUENCE),
                   meta.GetValue(WORK_LAYER), meta.GetValue(WORK_ROWS));
        }
        return meta.GetValue(WORK_CODE);
    }

    __aicore__ inline int64_t Finalize()
    {
        const ScratchLayout scratch(info_);
        GlobalTensor<int32_t> meta;
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratch_));
        FlushScratch(meta);
        const int64_t code = meta.GetValue(WORK_CODE);
        if (code != OK) {
            if (info_.mode == FINISH && block_ == 0) {
                Status(code);
            }
            return code;
        }
        const uint32_t source = meta.GetValue(WORK_SOURCE);
        const int64_t seq = meta.GetValue(WORK_SEQUENCE);
        const int64_t layer = meta.GetValue(WORK_LAYER);
        const uint32_t slot = (seq - 1) % c_[SLOTS];
        Scatter(meta, scratch, source, slot);
        SyncAll<true>();
        if (block_ == 0) {
            const uint32_t f = c_[WORLD_RANK] - c_[ATTENTION_RANKS];
            for (uint32_t tp = 0; tp < c_[ATTENTION_TP]; ++tp) {
                const uint32_t a = source * c_[ATTENTION_TP] + tp;
                WriteWord(Output(a, f, slot) + LINE_BYTES, layer);
                meta_.SetValue(1, meta.GetValue(scratch.tpCounts + tp));
                meta_.SetValue(2, ABI_VERSION);
                WriteLine(Output(a, f, slot) + LINE_BYTES);
                WriteSignal(Output(a, f, slot), seq);
            }
            StoreState(source + 1, seq);
            StoreState(0, (source + 1) % (c_[ATTENTION_RANKS] / c_[ATTENTION_TP]));
            if (info_.mode == FINISH) {
                uint32_t total = 0;
                for (uint32_t tp = 0; tp < c_[ATTENTION_TP]; ++tp) {
                    total += meta.GetValue(scratch.tpCounts + tp);
                }
                Status(OK, source, seq, layer, total);
            }
        }
        return OK;
    }

    __aicore__ inline void Combine(bool peek)
    {
        int64_t seq, layer, count;
        Request(seq, layer, count);
        const uint32_t slot = (seq - 1) % c_[SLOTS];
        if (block_ == 0) {
            int64_t code = TIMED_OUT;
            if (seq < 1 || seq > MAX_GENERATION || count < 0 || count > info_.inputRows ||
                count > c_[CAPACITY] || layer < 0 || layer >= c_[MODEL_LAYERS]) {
                code = INVALID;
            }
            if (seq != ReadWord(state_ + CONSUMED_STATE * sizeof(int64_t)) + 1 ||
                seq > ReadWord(state_ + PUBLISHED_STATE * sizeof(int64_t))) {
                code = INVALID;
            }
            for (uint32_t attempt = 0; code != INVALID && attempt < c_[POLL_BUDGET]; ++attempt) {
                if (Failed()) {
                    code = PEER_FAILED;
                    break;
                }
                bool ready = true;
                for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
                    const auto output = Output(c_[WORLD_RANK], f, slot);
                    if (ReadSignal(output) != seq) {
                        ready = false;
                        break;
                    }
                    ReadLine(output + LINE_BYTES);
                    if (meta_.GetValue(0) != layer || meta_.GetValue(1) != count || meta_.GetValue(2) != ABI_VERSION) {
                        Status(INVALID);
                        ready = false;
                        code = INVALID;
                        break;
                    }
                }
                if (ready) {
                    code = OK;
                    break;
                }
                if (code == INVALID) {
                    break;
                }
            }
            Status(code, c_[WORLD_RANK] / c_[ATTENTION_TP], seq, layer, count);
        }
        SyncAll<true>();
        ReadLine(status_);
        if (meta_.GetValue(0) != OK) {
            return;
        }
        if (peek) {
            // Functional diagnostics only: read unweighted F results while A
            // still owns the slot. Production COMBINE does not take this path.
            for (uint32_t route = block_; route < count * c_[TOPK]; route += info_.blocks) {
                const int32_t expert = Expert(ids_ + route * sizeof(int32_t));
                const auto destination = y_ + route * c_[HIDDEN] * sizeof(bfloat16_t);
                if (expert >= 0 && expert < c_[EXPERTS]) {
                    const uint32_t f = expert / (c_[EXPERTS] / c_[FFN_RANKS]);
                    CopyRow(destination, Output(c_[WORLD_RANK], f, slot) + 2 * LINE_BYTES +
                            route * c_[HIDDEN] * sizeof(bfloat16_t), c_[HIDDEN] * sizeof(bfloat16_t));
                } else {
                    Duplicate(copy_, static_cast<bfloat16_t>(0.0f), c_[HIDDEN]);
                    Fence<HardEvent::V_MTE3>();
                    GlobalTensor<bfloat16_t> result;
                    result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(destination));
                    DataCopy(result, copy_, c_[HIDDEN]);
                    Fence<HardEvent::MTE3_S>();
                }
            }
            SyncAll<true>();
            return;
        }
        for (uint32_t token = block_; token < count; token += info_.blocks) {
            Duplicate(output_, 0.0f, c_[HIDDEN]);
            if (info_.probabilityBf16) {
                // The caller already performed the model's BF16 rounding.
                // Expand one token's probabilities once, without an extra A
                // tensor operation or changing generic FP32 combine weights.
                GlobalTensor<bfloat16_t> probabilities;
                probabilities.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(probabilities_));
                Fence<HardEvent::S_MTE2>();
                DataCopyPad(copy_, probabilities[token * c_[TOPK]],
                            {1U, static_cast<uint32_t>(c_[TOPK] * sizeof(bfloat16_t)), 0U, 0U, 0U},
                            {false, 0U, 0U, 0U});
                Fence<HardEvent::MTE2_V>();
                Cast(sum_, copy_, RoundMode::CAST_NONE, c_[TOPK]);
                Fence<HardEvent::V_S>();
            }
            for (uint32_t k = 0; k < c_[TOPK]; ++k) {
                const uint32_t route = token * c_[TOPK] + k;
                const int32_t expert = Expert(ids_ + route * sizeof(int32_t));
                if (expert < 0 || expert >= c_[EXPERTS]) {
                    continue;
                }
                const uint32_t f = expert / (c_[EXPERTS] / c_[FFN_RANKS]);
                float probability;
                if (info_.probabilityBf16) {
                    probability = sum_.GetValue(k);
                } else {
                    GlobalTensor<float> probabilities;
                    probabilities.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(probabilities_));
                    auto scalar = sumBuffer_.Get<float>();
                    Fence<HardEvent::S_MTE2>();
                    DataCopyPad(scalar, probabilities[route], {1U, sizeof(float), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                    Fence<HardEvent::MTE2_S>();
                    probability = scalar.GetValue(0);
                }
                GlobalTensor<bfloat16_t> remote;
                remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(Output(c_[WORLD_RANK], f, slot) +
                    2 * LINE_BYTES + route * c_[HIDDEN] * sizeof(bfloat16_t)));
                Fence<HardEvent::S_MTE2>();
                DataCopy(copy_, remote, c_[HIDDEN]);
                Fence<HardEvent::MTE2_V>();
                Cast(x_, copy_, RoundMode::CAST_NONE, c_[HIDDEN]);
                PipeBarrier<PIPE_V>();
                Muls(x_, x_, probability, c_[HIDDEN]);
                PipeBarrier<PIPE_V>();
                Add(output_, output_, x_, c_[HIDDEN]);
                PipeBarrier<PIPE_V>();
            }
            Cast(copy_, output_, RoundMode::CAST_RINT, c_[HIDDEN]);
            Fence<HardEvent::V_MTE3>();
            GlobalTensor<bfloat16_t> result;
            result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y_));
            DataCopy(result[token * c_[HIDDEN]], copy_, c_[HIDDEN]);
            Fence<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        if (block_ == 0) {
            for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
                WriteSignal(Input(c_[ATTENTION_RANKS] + f, c_[WORLD_RANK], slot) + 2 * LINE_BYTES, seq);
            }
            StoreState(CONSUMED_STATE, seq);
        }
    }
    __aicore__ inline void Control()
    {
        if (block_ != 0) {
            return;
        }
        if (info_.mode == ABORT_SESSION) {
            for (uint32_t peer = 0; peer < c_[ATTENTION_RANKS] + c_[FFN_RANKS]; ++peer) {
                WriteSignal(Window(peer) + (c_[WORLD_RANK] + 1) * LINE_BYTES, 1);
            }
        } else {
            const int64_t last = ReadWord(request_);
            if (last < 0 || last > MAX_GENERATION ||
                last != ReadWord(state_ + PUBLISHED_STATE * sizeof(int64_t)) ||
                last != ReadWord(state_ + CONSUMED_STATE * sizeof(int64_t))) {
                Status(INVALID);
                return;
            }
            for (uint32_t f = 0; f < c_[FFN_RANKS]; ++f) {
                WriteSignal(Window(c_[ATTENTION_RANKS] + f) +
                    (1 + c_[ATTENTION_RANKS] + c_[FFN_RANKS] + c_[WORLD_RANK]) * LINE_BYTES, last + 1);
            }
            // A source is terminal after EOS, including EOS(0). Negative
            // published encodes the boundary without another state/ABI field.
            // No later nonnegative generation can pass Publish's next check.
            StoreState(PUBLISHED_STATE, -(last + 1));
        }
        Status(OK);
    }

    const FaaSInfo &info_;
    Layout layout_;
    TPipe &pipe_;
    const int64_t *c_;
    uint32_t block_;
    __gm__ HcclOpResParam *context_;
    GM_ADDR input_;
    GM_ADDR ids_;
    GM_ADDR probabilities_;
    GM_ADDR layers_;
    GM_ADDR request_;
    GM_ADDR state_;
    GM_ADDR scratch_;
    GM_ADDR y_;
    GM_ADDR status_;
    TBuf<> metaBuffer_, copyBuffer_, xBuffer_, sumBuffer_;
    TBuf<> outputBuffer_;
    LocalTensor<int64_t> meta_;
    LocalTensor<bfloat16_t> copy_;
    LocalTensor<float> x_, sum_, output_;
};
}  // namespace AfdFaas

#endif  // AFD_FFN_FAAS_CONTROL_H
