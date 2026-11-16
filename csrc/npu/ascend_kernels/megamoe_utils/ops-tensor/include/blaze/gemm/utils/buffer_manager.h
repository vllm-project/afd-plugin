/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file buffer_manager.h
 * \brief 全流水线 Buffer 与 Event ID 统一管理器，覆盖 L1(A/B/Bias)、L0A、L0B、BT、L0C。
 *
 * === 核心设计 ===
 *
 * BufferSlot 是原子绑定单元：持有一个 buffer 的地址偏移 + 同步 Event ID，
 * 可通过 .Lock(pipe) 直接派生 RAII 锁，不再需要单独传递 buffer 索引。
 *
 * === Event ID 命名空间布局 ===
 *
 *   [0, MaxL1ASlots) → L1 A 数据 buffer
 *   [MaxL1ASlots, MaxL1ASlots+MaxL1BSlots) → L1 B 数据 buffer
 *   [MaxL1ASlots+MaxL1BSlots, +2) → BT buffer
 *   [...+2, +MaxL0Slots) → L0A/L0B ping-pong
 *   [...+MaxL0Slots, +MaxL0Slots) → L0C ping-pong
 *
 *   注意：L1 Bias 复用 A 的事件 ID（Bias 与 A 在同一 pipeline 阶段加载），
 *   无独立 MTE1 event — 其 MTE2 写入受 A 数据 event 保护，
 *   MTE1 读取侧通过 BT event 同步。
 *
 * === 使用示例 ===
 *
 *   // 场景1: A/B 独立 event（适合不同 stage 数）
 *   BufferManager<4, 2, 2> bufMgr;
 *   bufMgr.InitAL1(0, offsetA0, 0);
 *   bufMgr.InitBL1(0, offsetB0, 4);
 *   bufMgr.InitBias(0, offsetBias0, 0);
 *
 *   // 场景2: A/B 共享 event（适合相同 stage 数）
 *   BufferManager<4, 4, 2> bufMgr;
 *   bufMgr.InitAL1(0, offsetA0, 0);
 *   bufMgr.InitBL1(0, offsetB0, 0);      // 同 bufferId
 *   bufMgr.InitBias(0, offsetBias0, 0);  // 同 bufferId
 *   bufMgr.InitBT(btOneSize);
 *   bufMgr.InitL0();
 *   bufMgr.InitL0C();
 *
 *   auto& aSlot  = bufMgr.GetL1ASlot(aBufIdx);
 *   auto& bSlot  = bufMgr.GetL1BSlot(bBufIdx);
 *   auto& btSlot = bufMgr.GetBTSlot(btBufIdx);
 *   auto& l0Slot = bufMgr.GetL0Slot(l0Idx);
 *
 *   { auto lk = aSlot.LockMte2();
 *     make_tensor(make_mem_ptr<L1, AType>(aSlot.Addr()), ...); }
 *   { auto lk = btSlot.LockMte1();
 *     make_tensor(make_mem_ptr<BIAS, float>(btSlot.Addr()), ...); }
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {

/* =========================================================================
 * BufferIdLayout — 全流水线 Event ID 命名空间常量（编译期）
 * ========================================================================= */
template <uint32_t MaxL1ASlots = 4, uint32_t MaxL1BSlots = 4, uint32_t MaxL0Slots = 2>
struct BufferIdLayout {
    static constexpr uint32_t L1A_DATA_BASE = 0;
    static constexpr uint32_t L1A_DATA_MAX = L1A_DATA_BASE + MaxL1ASlots;

    static constexpr uint32_t L1B_DATA_BASE = L1A_DATA_MAX;
    static constexpr uint32_t L1B_DATA_MAX = L1B_DATA_BASE + MaxL1BSlots;

    static constexpr uint32_t BT_BASE = L1B_DATA_MAX;
    static constexpr uint32_t BT_MAX = BT_BASE + MaxL0Slots;

    static constexpr uint32_t L0_BASE = BT_MAX;
    static constexpr uint32_t L0_MAX = L0_BASE + MaxL0Slots;

    static constexpr uint32_t L0C_BASE = L0_MAX;
    static constexpr uint32_t L0C_MAX = L0C_BASE + MaxL0Slots;

    static constexpr uint32_t SCALE_BASE = L0C_MAX;
    static constexpr uint32_t SCALE_MAX = SCALE_BASE + 1;

    static constexpr uint8_t L1ADataBufferId(uint32_t idx) { return L1A_DATA_BASE + idx; }
    static constexpr uint8_t L1BDataBufferId(uint32_t idx) { return L1B_DATA_BASE + idx; }
    static constexpr uint8_t BTBufferId(uint32_t idx) { return BT_BASE + idx; }
    static constexpr uint8_t L0BufferId(uint32_t idx) { return L0_BASE + idx; }
    static constexpr uint8_t L0CBufferId(uint32_t idx) { return L0C_BASE + idx; }
    static constexpr uint8_t ScaleBufferId() { return SCALE_BASE; }
};

/* =========================================================================
 * ScopedSyncLock — RAII 锁守卫，构造 asc_lock / 析构 asc_unlock
 * ========================================================================= */
template <pipe_t Pipe>
class ScopedSyncLock {
public:
    __aicore__ inline ScopedSyncLock(uint8_t bufferId) : bufferId_(bufferId)
    {
        if ASCEND_IS_AIV {
            if constexpr (Pipe == pipe_t::PIPE_M || Pipe == pipe_t::PIPE_FIX || Pipe == pipe_t::PIPE_MTE1) {
                return;
            }
        }
        asc_lock(Pipe, bufferId_);
    }

    __aicore__ inline ~ScopedSyncLock()
    {
        if ASCEND_IS_AIV {
            if constexpr (Pipe == pipe_t::PIPE_M || Pipe == pipe_t::PIPE_FIX || Pipe == pipe_t::PIPE_MTE1) {
                return;
            }
        }
        asc_unlock(Pipe, bufferId_);
    }

    ScopedSyncLock(const ScopedSyncLock&) = delete;
    ScopedSyncLock& operator=(const ScopedSyncLock&) = delete;

private:
    uint8_t bufferId_;
};

/* =========================================================================
 * BufferSlot — 原子绑定单元：地址偏移 + Event ID + 自锁能力
 *
 * 每个 slot 在 Init 时绑定唯一的 event ID，之后通过 slot 引用即可：
 *   - slot.Addr()   → 获取 buffer 字节偏移
 *   - slot.Lock(p)  → 对该 buffer 加指定 pipe 锁（返回 ScopedSyncLock）
 *
 * 调用侧只需持有 slot 引用，不再需要单独传递 buffer 索引或 event ID。
 * ========================================================================= */
struct BufferSlot {
    uint64_t byteOffset = 0;
    uint8_t bufferId = 0;

    __aicore__ inline uint64_t Addr() const { return byteOffset; }
    __aicore__ inline uint8_t Id() const { return bufferId; }

    template <pipe_t Pipe>
    __aicore__ inline auto Lock() const
    {
        return ScopedSyncLock<Pipe>(bufferId);
    }

    __aicore__ inline auto LockMte2() const { return Lock<pipe_t::PIPE_MTE2>(); }
    __aicore__ inline auto LockMte1() const { return Lock<pipe_t::PIPE_MTE1>(); }
    __aicore__ inline auto LockMte3() const { return Lock<pipe_t::PIPE_MTE3>(); }
    __aicore__ inline auto LockM() const { return Lock<pipe_t::PIPE_M>(); }
    __aicore__ inline auto LockV() const { return Lock<pipe_t::PIPE_V>(); }
    __aicore__ inline auto LockFix() const { return Lock<pipe_t::PIPE_FIX>(); }
};

/* =========================================================================
 * BufferManager — 全流水线 Buffer 管理器
 *
 * 模板参数:
 *   MaxL1ASlots  L1 A 最大槽位数（4 = QUADRUPLE_BUFFER_COUNT）
 *   MaxL1BSlots  L1 B 最大槽位数（2 或 4）
 *   MaxL0Slots   L0 ping-pong 槽位数（2 = DOUBLE_BUFFER_COUNT）
 *
 * L1 初始化由调用方通过 InitAL1/InitBL1/InitBias 逐槽位指定 byteOffset 和 bufferId，
 * 自行决定 A/B 共享或独立 event：
 *   - 共享: InitAL1(i, offA, i) / InitBL1(i, offB, i)    // 同 bufferId
 *   - 独立: InitAL1(i, offA, i) / InitBL1(i, offB, 4+i)  // 不同 bufferId
 * ========================================================================= */
template <uint32_t MaxL1ASlots = 4, uint32_t MaxL1BSlots = 4, uint32_t MaxL0Slots = 2>
class BufferManager {
    using Layout = BufferIdLayout<MaxL1ASlots, MaxL1BSlots, MaxL0Slots>;

public:
    BufferManager() = default;

    // =====================================================================
    // L1 初始化（由调用方自行管理 offset 和 bufferId）
    // =====================================================================
    __aicore__ inline void InitAL1(uint32_t idx, uint64_t byteOffset, uint8_t bufferId)
    {
        aL1Slots_[idx] = {byteOffset, bufferId};
    }
    __aicore__ inline void InitBL1(uint32_t idx, uint64_t byteOffset, uint8_t bufferId)
    {
        bL1Slots_[idx] = {byteOffset, bufferId};
    }
    __aicore__ inline void InitBias(uint32_t idx, uint64_t byteOffset, uint8_t bufferId)
    {
        biasL1Slots_[idx] = {byteOffset, bufferId};
    }

    __aicore__ inline void InitScaleL1(uint64_t byteOffset) { scaleL1Slot_ = {byteOffset, Layout::ScaleBufferId()}; }

    __aicore__ inline void InitBT(uint64_t btOneSize)
    {
        for (uint32_t i = 0; i < MaxL0Slots; ++i) {
            btSlots_[i] = {btOneSize * i, Layout::BTBufferId(i)};
        }
    }
    __aicore__ inline void InitL0()
    {
        for (uint32_t i = 0; i < MaxL0Slots; ++i) {
            l0Slots_[i] = {(AscendC::TOTAL_L0A_SIZE / MaxL0Slots) * i, Layout::L0BufferId(i)};
        }
    }
    __aicore__ inline void InitL0C()
    {
        for (uint32_t i = 0; i < MaxL0Slots; ++i) {
            l0cSlots_[i] = {(AscendC::TOTAL_L0C_SIZE / MaxL0Slots) * i, Layout::L0CBufferId(i)};
        }
    }

    // =====================================================================
    // Slot 引用访问
    // =====================================================================
    __aicore__ inline const BufferSlot& GetL1ASlot(uint32_t idx) const { return aL1Slots_[idx]; }
    __aicore__ inline const BufferSlot& GetL1BSlot(uint32_t idx) const { return bL1Slots_[idx]; }
    __aicore__ inline const BufferSlot& GetL1BiasSlot(uint32_t idx) const { return biasL1Slots_[idx]; }
    __aicore__ inline const BufferSlot& GetBTSlot(uint32_t idx) const { return btSlots_[idx]; }
    __aicore__ inline const BufferSlot& GetL0Slot(uint32_t idx) const { return l0Slots_[idx]; }
    __aicore__ inline const BufferSlot& GetL0CSlot(uint32_t idx) const { return l0cSlots_[idx]; }
    __aicore__ inline const BufferSlot& GetScaleL1Slot() const { return scaleL1Slot_; }

private:
    BufferSlot aL1Slots_[MaxL1ASlots];
    BufferSlot bL1Slots_[MaxL1BSlots];
    BufferSlot biasL1Slots_[MaxL1ASlots > MaxL1BSlots ? MaxL1ASlots : MaxL1BSlots];
    BufferSlot btSlots_[MaxL0Slots];
    BufferSlot l0Slots_[MaxL0Slots];
    BufferSlot l0cSlots_[MaxL0Slots];
    BufferSlot scaleL1Slot_;
};

/* =========================================================================
 * 独立锁函数
 * ========================================================================= */
__aicore__ inline void LockMte2(uint8_t bufferId) { asc_lock(pipe_t::PIPE_MTE2, bufferId); }
__aicore__ inline void UnLockMte2(uint8_t bufferId) { asc_unlock(pipe_t::PIPE_MTE2, bufferId); }
__aicore__ inline void LockMte1(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_lock(pipe_t::PIPE_MTE1, bufferId);
    }
}
__aicore__ inline void UnLockMte1(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_unlock(pipe_t::PIPE_MTE1, bufferId);
    }
}
__aicore__ inline void LockM(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_lock(pipe_t::PIPE_M, bufferId);
    }
}
__aicore__ inline void UnLockM(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_unlock(pipe_t::PIPE_M, bufferId);
    }
}
__aicore__ inline void LockFix(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_lock(pipe_t::PIPE_FIX, bufferId);
    }
}
__aicore__ inline void UnLockFix(uint8_t bufferId)
{
    if ASCEND_IS_NOT_AIV {
        asc_unlock(pipe_t::PIPE_FIX, bufferId);
    }
}
} // namespace Gemm
} // namespace Blaze
