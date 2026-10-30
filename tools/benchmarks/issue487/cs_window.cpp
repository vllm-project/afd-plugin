// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// TEST ONLY. Compiled in an isolated copy of AfdAsyncCombineSend's run package.
// Reuse its MC2 tiling/ACLNN ABI to initialize/read a target's local window.
// Senders must load the unmodified native package, in separate processes.

#include "kernel_operator.h"
#include "afd_async_combine_send_tiling.h"
#include "cam_async_comm_group.h"

using namespace AscendC;
using namespace Cam;

namespace {
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_AIV = 48;
constexpr uint32_t COPY_ELEMENTS = 16384;
constexpr uint32_t ACTION_CLEAR = 0;
constexpr uint32_t ACTION_READ = 1;
constexpr uint32_t ACTION_INIT = 2;

__aicore__ inline uint64_t Align(uint64_t value)
{
    return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

template <HardEvent event>
__aicore__ inline void Fence()
{
    int32_t id = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}
}  // namespace

// Same signature as the native kernel; only target-side setup/readback uses it.
// batchInfo[0] = rows, batchInfo[1] = clear/read/init. commArgs is a writable
// float16 buffer holding 32 state bytes followed by BF16 payload bytes.
extern "C" __global__ __aicore__ void afd_async_combine_send(
    GM_ADDR expandX, GM_ADDR commArgs, GM_ADDR batchInfo,
    GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AfdAsyncCombineSendTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", AfdAsyncCombineSendTilingData);
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncCombineSendTilingData, tiling, tilingGM);
    TPipe pipe;
    TBuf<> buffer;
    pipe.InitBuffer(buffer, COPY_ELEMENTS * sizeof(uint16_t));
    auto metadata = buffer.Get<int64_t>();
    GlobalTensor<int64_t> info;
    info.SetGlobalBuffer((__gm__ int64_t *)batchInfo);
    DataCopy(metadata, info, ALIGN_BYTES / sizeof(int64_t));
    Fence<HardEvent::MTE2_S>();
    uint64_t rows = metadata.GetValue(0);
    uint32_t action = metadata.GetValue(1);
    Fence<HardEvent::S_MTE2>();

    auto context = (__gm__ HcclOpResParam *)GetHcclContext<HCCL_GROUP_ID_0>();
    const auto &layout = tiling.moeDistributeCombineInfo;
    // The original Python binding requires a sender rank; callers pass 2.
    // The real target identity comes from HCCL, never from that placeholder.
    if (context->localUsrRankId >= layout.attnRankNum || action == ACTION_INIT) {
        return;
    }
    if (action != ACTION_CLEAR && action != ACTION_READ) {
        return;
    }
    uint64_t dispatchOffset = Align(sizeof(uint32_t) * layout.moeRankNum *
                                   (1 + layout.routeExpertNumPerMoe) * MAX_AIV);
    uint64_t bytes = Align(sizeof(uint32_t) * layout.moeRankNum) +
                     rows * layout.hiddenSize * sizeof(uint16_t);
    if (dispatchOffset + bytes > context->winSize) {
        return;
    }
    GlobalTensor<uint16_t> window, output;
    window.SetGlobalBuffer((__gm__ uint16_t *)(context->localWindowsIn + dispatchOffset));
    output.SetGlobalBuffer((__gm__ uint16_t *)commArgs);
    auto payload = buffer.Get<uint16_t>();
    if (action == ACTION_CLEAR) {
        Duplicate(payload, static_cast<uint16_t>(0), COPY_ELEMENTS);
        Fence<HardEvent::V_MTE3>();
    }
    uint64_t elements = bytes / sizeof(uint16_t);
    for (uint64_t offset = GetBlockIdx() * COPY_ELEMENTS; offset < elements;
         offset += layout.aivNum * COPY_ELEMENTS) {
        uint32_t count = elements - offset < COPY_ELEMENTS ? elements - offset : COPY_ELEMENTS;
        if (action == ACTION_READ) {
            DataCopy(payload, window[offset], count);
            Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_MTE2>();
            Fence<HardEvent::S_MTE3>();
            DataCopy(output[offset], payload, count);
        } else {
            DataCopy(window[offset], payload, count);
        }
        Fence<HardEvent::MTE3_S>();
    }
}
