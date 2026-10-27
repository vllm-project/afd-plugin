#pragma once

// Split MegaMoE for the AFD M2N topology, derived from
// `sm100_fp8_fp4_mega_moe.cuh`.
//
// The symmetric mega kernel is split into two role kernels over one union
// world of `kNumAGRanks + kNumEGRanks` ranks (AG ranks first):
//   - AG kernel (`sm100_mega_moe_m2n_ag_impl`): send + recv only.  Pushes
//     source token-topk indices and per-expert counts to EG owner ranks,
//     then waits for the EG combine write-back and reduces top-k locally.
//     No GEMM, no TMEM, no cluster.
//   - EG kernel (`sm100_mega_moe_m2n_eg_impl`): recv + experts + send.
//     Pulls FP8 tokens/SF/weights from AG ranks via NVLink TMA, runs the
//     FP8xFP4 L1/L2 grouped GEMM pipeline, and pushes BF16 expert outputs
//     into the source AG rank combine buffers.  Sources no tokens and does
//     no local combine reduction.
//
// Both role kernels execute a compatible NVLink barrier sequence so the
// union-wide phase counter stays aligned.  The per-expert receive-count
// high-32 convention is changed: each AG rank's SM0 pushes
// `(1 << 32) | count` once, so EG waits for `high32 == kNumAGRanks`
// (decoupled from SM counts).

#include <cstdint>
#include <type_traits>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_m2n.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// NVLink barrier tags shared by both role kernels (must match in order)
namespace m2n_detail {
// Per layer the protocol is: [deferred tag-3 wait from the previous layer] ->
// count push (release per source rank; no pre-pull barrier) -> tag-2 full
// barrier (combine readiness) -> tag-3 arrive (cleanup signal).
constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;
constexpr uint32_t kDispatchGridSyncIndex = 0;
constexpr uint32_t kEpilogueGridSyncIndex = 1;

// Acquire fence for the volatile per-expert count spins: orders the
// subsequent remote src-idx / token reads after the observed releases.
CUTLASS_DEVICE void fence_acquire_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

// Fire-and-forget bulk L2 prefetch with an evict-last policy.  Weights
// prefetched during the recv-count wait window must survive the other
// microbatch lane's streaming GEMM, which otherwise turns the whole L2
// over several times before this lane's work phase starts.
CUTLASS_DEVICE void prefetch_l2_evict_last(const void* ptr, const uint32_t& num_bytes) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                 :: "l"(ptr), "r"(num_bytes), "l"(policy) : "memory");
}

enum class EGTimingSlot: uint32_t {
    LaneStart = 0,
    Tag3Wait = 1,
    SchedFetchBarrierDispatch = 2,
    RecvCountWaitDispatch = 3,
    PullLoopDispatch = 4,
    DispatchEpilogueSync = 5,
    CleanupArrive = 6,
    SchedFetchBarrierTmaA = 7,
    L1ArrivalWait = 8,
    L2ArrivalWait = 9,
    L2RemoteWriteback = 10,
    Tag2Barrier = 11,
    LaneEnd = 12,
    TmaALoop = 13,
    TmaBLoop = 14,
    MmaLoop = 15,
    EpilogueLoop = 16,
    TotalRecvTokens = 17,
    MaxExpertTokens = 18,
    TotalMBlocks = 19,
    RecvCountReadyTime = 20,
    PullDoneTime = 21,
    TmaAStartTime = 22,
    TmaAEndTime = 23,
    TmaBStartTime = 24,
    TmaBEndTime = 25,
    MmaStartTime = 26,
    MmaEndTime = 27,
    EpiStartTime = 28,
    EpiEndTime = 29,
    Tag2EndTime = 30,
    NumSlots = 31,
};

enum class AGTimingSlot: uint32_t {
    KernelStart = 0,
    Tag3Wait = 1,
    Quant = 2,
    Gate = 3,
    DispatchCount = 4,
    DispatchOffsets = 5,
    DispatchIndexPush = 6,
    DispatchGridSync = 7,
    DispatchCountPush = 8,
    CombineBarrierWait = 9,
    CombineReduce = 10,
    CleanupArrive = 11,
    KernelEnd = 12,
    NumSlots = 16,
    Tag3WaitEndTime = 16,
    QuantEndTime = 17,
    GateEndTime = 18,
    DispatchCountEndTime = 19,
    DispatchOffsetsEndTime = 20,
    DispatchIndexPushEndTime = 21,
    DispatchGridSyncEndTime = 22,
    DispatchCountPushEndTime = 23,
    CombineBarrierStartTime = 24,
    CombineBarrierEndTime = 25,
    CleanupArriveEndTime = 26,
    CombineReduceStartTime = 27,
    CombineReduceEndTime = 28,
    NumSlotsWithTimestamps = 29,
};

CUTLASS_DEVICE unsigned long long* ag_timing_ptr(
    unsigned long long* debug_timings,
    const AGTimingSlot& slot) {
    if (debug_timings == nullptr)
        return nullptr;
    return debug_timings + static_cast<uint32_t>(slot);
}

CUTLASS_DEVICE void ag_timing_add(
    unsigned long long* debug_timings,
    const AGTimingSlot& slot,
    const unsigned long long& value) {
    auto ptr = ag_timing_ptr(debug_timings, slot);
    if (ptr != nullptr)
        atomicAdd(ptr, value);
}

CUTLASS_DEVICE void ag_timing_store(
    unsigned long long* debug_timings,
    const AGTimingSlot& slot,
    const unsigned long long& value) {
    auto ptr = ag_timing_ptr(debug_timings, slot);
    if (ptr != nullptr)
        *ptr = value;
}

template <uint32_t kNumLanes>
CUTLASS_DEVICE unsigned long long* eg_timing_ptr(
    unsigned long long* debug_timings,
    const uint32_t& layer_i,
    const uint32_t& lane_i,
    const EGTimingSlot& slot) {
    if (debug_timings == nullptr)
        return nullptr;
    return debug_timings + (layer_i * kNumLanes + lane_i) *
        static_cast<uint32_t>(EGTimingSlot::NumSlots) + static_cast<uint32_t>(slot);
}

template <uint32_t kNumLanes>
CUTLASS_DEVICE void eg_timing_add(
    unsigned long long* debug_timings,
    const uint32_t& layer_i,
    const uint32_t& lane_i,
    const EGTimingSlot& slot,
    const unsigned long long& value) {
    auto ptr = eg_timing_ptr<kNumLanes>(debug_timings, layer_i, lane_i, slot);
    if (ptr != nullptr)
        atomicAdd(ptr, value);
}

template <uint32_t kNumLanes>
CUTLASS_DEVICE void eg_timing_store(
    unsigned long long* debug_timings,
    const uint32_t& layer_i,
    const uint32_t& lane_i,
    const EGTimingSlot& slot,
    const unsigned long long& value) {
    auto ptr = eg_timing_ptr<kNumLanes>(debug_timings, layer_i, lane_i, slot);
    if (ptr != nullptr)
        *ptr = value;
}
}  // namespace m2n_detail

// ---------------------------------------------------------------------------
// AG kernel: dispatch send + combine recv only
// ---------------------------------------------------------------------------
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kInputTopk, uint32_t kNumRoutedExperts,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumDispatchThreads, uint32_t kNumCombineThreads,
    uint32_t kNumSMs,
    uint32_t kNumAGRanks, uint32_t kNumEGRanks,
    bool kTopkIdxIs64,
    bool kInKernelGate,
    // 0 = fused dispatch+combine (one kernel holds its SMs through the EG
    // round trip); 1 = dispatch-only (quant + pushes, exits immediately so
    // the SMs go back to attention); 2 = combine-only (combine-ready wait +
    // top-k reduce + local cleanup), launched later in the graph.
    uint32_t kAGPhase = 0,
    // The per-32 FP8 quant + topk-weight staging were already produced by an
    // upstream kernel (fused into the router gate, which reads the hidden
    // states anyway); skip phase 0 so this kernel's active window shrinks to
    // just the metadata pushes.  Safe w.r.t. the EG cleanup-done wait: the
    // upstream quant of layer L happens after layer L-1's combine output was
    // consumed; the deferred cleanup wait proves every EG rank finished
    // pulling layer L-1's x/sf/weights.
    bool kExternalQuant = false,
    bool kDirectCombine = false,
    bool kEnableDebugTiming = false,
    uint32_t kNumRanks = kNumAGRanks + kNumEGRanks,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumCombineWarps = kNumCombineThreads / 32,
    uint32_t kNumThreads = kNumDispatchThreads + kNumCombineThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerEGRank = kNumExperts / kNumEGRanks
>
__global__ __launch_bounds__(kNumThreads, 1) void
sm100_mega_moe_m2n_ag_impl(void* y,
                           const nv_bfloat16* hidden_bf16,
                           const void* topk_ids_raw,
                           const float* topk_weights_in,
                           const nv_bfloat16* gate_weight,
                           const uint32_t renormalize,
                           const uint32_t num_tokens,
                           const float shared_topk_weight,
                           unsigned long long* debug_timings,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumCombineThreads % 128 == 0, "Invalid number of combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumEGRanks == 0, "Invalid number of experts or EG ranks");
    DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
    DG_STATIC_ASSERT(kInputTopk > 0 and kInputTopk <= kNumTopk, "Invalid input topk");
    DG_STATIC_ASSERT(kInputTopk == kNumTopk or kInputTopk + 1 == kNumTopk,
                     "Only one synthetic shared-expert slot is supported");
    DG_STATIC_ASSERT(kNumRoutedExperts <= kNumExperts, "Invalid routed expert count");
    DG_STATIC_ASSERT(kNumRoutedExperts % kNumEGRanks == 0, "Routed experts must divide EG ranks");
    DG_STATIC_ASSERT(kInputTopk == kNumTopk or kNumRoutedExperts < kNumExperts,
                     "Synthetic shared slot requires protocol-only experts");
    DG_STATIC_ASSERT(kInputTopk == kNumTopk or not kInKernelGate,
                     "Synthetic shared slot is not supported with in-kernel gate");

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const bool do_timing_dispatch_leader =
        kEnableDebugTiming and sm_idx == 0 and warp_idx == 0 and lane_idx == 0;
    const bool do_timing_combine_leader =
        kEnableDebugTiming and sm_idx == 0 and warp_idx == kNumDispatchWarps and lane_idx == 0;
    constexpr uint32_t kNumRoutedExpertsPerEGRank = kNumRoutedExperts / kNumEGRanks;
    if (do_timing_dispatch_leader)
        m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::KernelStart, clock64());

    // Workspaces (must match the EG kernel byte-for-byte)
    const auto workspace = layout::WorkspaceM2N(
        sym_buffer.get_base_ptr(), kNumAGRanks, kNumEGRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk);

    // Buffer layout chain (identical on AG/EG; only the offsets matter here)
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // Shared memory: combine chunk slots first (reused as nothing else after
    // dispatch), expert counters next, barriers last.
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kNumChunkSlots = 3;
    constexpr uint32_t kCombineSmemBudget = 160 * 1024;
    constexpr uint32_t kNumChunks =
        (kNumChunkSlots * kNumCombineWarps * kNumHiddenBytes <= kCombineSmemBudget) ? 1 : 2;
    constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
    constexpr uint32_t SMEM_CHUNK_SIZE = math::constexpr_align(
        kNumChunkSlots * kNumCombineWarps * kNumChunkBytes, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE = math::constexpr_align<uint32_t>(
        kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);

    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_CHUNK_SIZE);
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_CHUNK_SIZE + SMEM_EXPERT_COUNT_SIZE);
    auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });

    // Initialization
    if (warp_idx == 0) {
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumCombineWarps * 2; i += 32)
            combine_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    using topk_idx_t = std::conditional_t<kTopkIdxIs64, int64_t, int32_t>;
    // Route ids come either from the caller (host gate + topk) or from the
    // in-kernel gate phase, which writes them into the symmetric buffer.
    const topk_idx_t* route_ids = kInKernelGate
        ? input_topk_idx_buffer.get_base_ptr<topk_idx_t>()
        : static_cast<const topk_idx_t*>(topk_ids_raw);

    using namespace m2n_detail;

    // ---- deferred cleanup-barrier wait -----------------------------------
    // The previous layer's tag-3 barrier is split: its signals were sent at
    // the end of the previous kernel (arrive); the wait is consumed HERE so
    // the inter-layer attention window hides the cross-rank latency.  It must
    // complete before phase 0 overwrites x/sf/weights that the EG ranks may
    // still be pulling, and before any remote workspace write.
    if constexpr (kAGPhase != 2) {
        unsigned long long timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        comm::nvlink_barrier_wait<kNumRanks, kNumSMs, kNumThreads,
                                  kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { __syncthreads(); }
        );
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::Tag3Wait, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::Tag3WaitEndTime, timing_t1);
        }
    }

    // ---- fused per-32 FP8 quant + route metadata staging ----------------
    // hidden [num_tokens, kHidden] bf16 -> x (e4m3) + x_sf (UE8M0 bytes,
    // ceil) in the LOCAL symmetric buffer, and the top-k weights staged next
    // to them: the EG kernel pulls all three regions remotely, so they must
    // live in the symmetric buffer before the tag-1 barrier (the grid sync
    // ahead of the count push orders these writes for every SM).
    if constexpr (kAGPhase != 2 and not kExternalQuant) {
        unsigned long long timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        constexpr uint32_t kNumGroups = kHidden / 32;
        constexpr uint32_t kNumWarps = kNumThreads / 32;
        const auto x_base = input_token_buffer.get_base_ptr<uint8_t>();
        const auto sf_base = input_sf_buffer.get_base_ptr<uint8_t>();
        const auto w_base = input_topk_weights_buffer.get_base_ptr<float>();
        for (uint32_t t = sm_idx * kNumWarps + warp_idx; t < num_tokens; t += kNumSMs * kNumWarps) {
            const auto src = hidden_bf16 + t * kHidden;
            const auto x_row = x_base + t * kHidden;
            const auto sf_row = sf_base + t * kNumGroups;
            #pragma unroll
            for (uint32_t g = lane_idx; g < kNumGroups; g += 32) {
                uint4 raw[4];
                const auto src_vec = reinterpret_cast<const uint4*>(src + g * 32);
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++ j)
                    raw[j] = __ldg(src_vec + j);
                float vals[32];
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++ j) {
                    const auto h = reinterpret_cast<const nv_bfloat16*>(raw + j);
                    #pragma unroll
                    for (uint32_t k = 0; k < 8; ++ k)
                        vals[j * 8 + k] = __bfloat162float(h[k]);
                }
                float amax = 1e-4f;
                #pragma unroll
                for (uint32_t k = 0; k < 32; ++ k)
                    amax = fmaxf(amax, fabsf(vals[k]));
                // ceil-to-UE8M0 of (amax / 448): bump the exponent when any
                // mantissa bit is set, then scale by the exact power of two.
                const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
                const uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
                sf_row[g] = static_cast<uint8_t>(e);
                const float inv_scale = __uint_as_float((254u - e) << 23);
                uint32_t packed[8];
                #pragma unroll
                for (uint32_t k = 0; k < 16; ++ k) {
                    reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[k] = __nv_cvt_float2_to_fp8x2(
                        make_float2(vals[2 * k] * inv_scale, vals[2 * k + 1] * inv_scale),
                        __NV_SATFINITE, __NV_E4M3);
                }
                const auto dst_vec = reinterpret_cast<uint4*>(x_row + g * 32);
                dst_vec[0] = *reinterpret_cast<uint4*>(packed);
                dst_vec[1] = *reinterpret_cast<uint4*>(packed + 4);
            }
            if constexpr (!kInKernelGate) {
                if (lane_idx < kInputTopk) {
                    w_base[t * kNumTopk + lane_idx] =
                        __ldg(topk_weights_in + t * kInputTopk + lane_idx);
                } else if constexpr (kInputTopk < kNumTopk) {
                    if (lane_idx == kInputTopk)
                        w_base[t * kNumTopk + lane_idx] = shared_topk_weight;
                }
            }
        }
        __syncthreads();
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::Quant, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::QuantEndTime, timing_t1);
        }
    }

    // ---- fused router gate + top-k (decode path) -------------------------
    // One warp per token: bf16 GEMV against the [kNumExperts, kHidden] gate
    // matrix (each lane owns kNumExperts/32 experts), then softmax + top-k
    // (+ optional renormalize) entirely in registers.  Ids (int64) and fp32
    // weights land in the symmetric-buffer input regions, where the dispatch
    // and combine phases read them; the grid sync below makes the writes
    // visible across SMs before the route counting starts.
    if constexpr (kInKernelGate and kAGPhase != 2) {
        unsigned long long timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        DG_STATIC_ASSERT(kNumExperts % 32 == 0 and kNumExperts / 32 <= 8, "Too many experts for the in-kernel gate");
        DG_STATIC_ASSERT(kTopkIdxIs64, "In-kernel gate writes int64 route ids");
        constexpr uint32_t kNumExpertsPerLane = kNumExperts / 32;
        constexpr uint32_t kNumWarps = kNumThreads / 32;
        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumWarps;
        const auto idx_base = input_topk_idx_buffer.get_base_ptr<int64_t>();
        const auto w_base = input_topk_weights_buffer.get_base_ptr<float>();
        // Logit scratch reuses the head of the combine buffer: the EG ranks
        // only write combine rows after this rank pushes its counts, which
        // happens after the gate phase, and the previous layer's rows were
        // consumed before the deferred barrier wait above.
        const auto logits_scratch = reinterpret_cast<float*>(combine_token_buffer.get_base_ptr());

        // Phase A: one warp per (token, 4-expert group): the h row loads are
        // shared and the 4 independent accumulator chains keep the memory
        // pipeline full (the gate matrix stays L2-resident across warps).
        constexpr uint32_t kGateExpertsPerWarp = 4;
        const uint32_t num_groups = num_tokens * (kNumExperts / kGateExpertsPerWarp);
        for (uint32_t d = sm_idx * kNumWarps + warp_idx; d < num_groups; d += kNumGlobalWarps) {
            const uint32_t t = d / (kNumExperts / kGateExpertsPerWarp);
            const uint32_t e0 = (d % (kNumExperts / kGateExpertsPerWarp)) * kGateExpertsPerWarp;
            const auto hrow = hidden_bf16 + t * kHidden;
            float acc[kGateExpertsPerWarp] = {};
            // Constant trip count so the unroll actually happens and the L2
            // latencies of consecutive iterations overlap
            DG_STATIC_ASSERT(kHidden % 256 == 0, "Invalid hidden for the gate GEMV");
            #pragma unroll 8
            for (uint32_t kk = 0; kk < kHidden / 256; ++ kk) {
                const uint32_t k = kk * 256 + lane_idx * 8;
                const uint4 hraw = __ldg(reinterpret_cast<const uint4*>(hrow + k));
                const auto hv2 = reinterpret_cast<const nv_bfloat162*>(&hraw);
                float2 hf[4];
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++ j)
                    hf[j] = __bfloat1622float2(hv2[j]);
                uint4 wraw[kGateExpertsPerWarp];
                #pragma unroll
                for (uint32_t i = 0; i < kGateExpertsPerWarp; ++ i)
                    wraw[i] = __ldg(reinterpret_cast<const uint4*>(gate_weight + (e0 + i) * kHidden + k));
                #pragma unroll
                for (uint32_t i = 0; i < kGateExpertsPerWarp; ++ i) {
                    const auto wv2 = reinterpret_cast<const nv_bfloat162*>(wraw + i);
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++ j) {
                        const float2 wf = __bfloat1622float2(wv2[j]);
                        acc[i] = fmaf(hf[j].x, wf.x, acc[i]);
                        acc[i] = fmaf(hf[j].y, wf.y, acc[i]);
                    }
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < kGateExpertsPerWarp; ++ i) {
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1)
                    acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
                if (lane_idx == 0)
                    logits_scratch[t * kNumExperts + e0 + i] = acc[i];
            }
        }
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx, [=]() { __syncthreads(); });

        // Phase B: one warp per token, top-k over its kNumExperts logits
        for (uint32_t t = sm_idx * kNumWarps + warp_idx; t < num_tokens; t += kNumGlobalWarps) {
            float logit[kNumExpertsPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i)
                logit[i] = logits_scratch[t * kNumExperts + i * 32 + lane_idx];

            // Stability max + full-softmax denominator (for renormalize=0)
            float m_all = -CUDART_INF_F;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i)
                m_all = fmaxf(m_all, logit[i]);
            #pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                m_all = fmaxf(m_all, __shfl_xor_sync(0xffffffff, m_all, off));
            float denom_all = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i)
                denom_all += expf(logit[i] - m_all);
            #pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                denom_all += __shfl_xor_sync(0xffffffff, denom_all, off);

            // Iterative warp arg-max top-k; lane j keeps selection j
            uint32_t used = 0;
            float my_logit = -CUDART_INF_F;
            int my_expert = -1;
            float top0 = -CUDART_INF_F;
            #pragma unroll
            for (uint32_t j = 0; j < kNumTopk; ++ j) {
                float best = -CUDART_INF_F;
                int best_i = -1;
                #pragma unroll
                for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
                    if (!((used >> i) & 1) and logit[i] > best)
                        best = logit[i], best_i = static_cast<int>(i);
                }
                float v = best;
                uint32_t src = lane_idx;
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffff, v, off);
                    const uint32_t os = __shfl_xor_sync(0xffffffff, src, off);
                    if (ov > v or (ov == v and os < src))
                        v = ov, src = os;
                }
                const int wbi = __shfl_sync(0xffffffff, best_i, src);
                if (lane_idx == src)
                    used |= 1u << wbi;
                if (j == 0)
                    top0 = v;
                if (lane_idx == j) {
                    my_logit = v;
                    my_expert = wbi * 32 + static_cast<int>(src);
                }
            }

            // Weights: renormalize -> within the selected set; otherwise the
            // full softmax probability of each selected expert
            float w = lane_idx < kNumTopk ? expf(my_logit - (renormalize ? top0 : m_all)) : 0.0f;
            float w_sum = w;
            #pragma unroll
            for (uint32_t off = 16; off > 0; off >>= 1)
                w_sum += __shfl_xor_sync(0xffffffff, w_sum, off);
            if (lane_idx < kNumTopk) {
                idx_base[t * kNumTopk + lane_idx] = static_cast<int64_t>(my_expert);
                w_base[t * kNumTopk + lane_idx] = renormalize ? w / w_sum : w / denom_all;
            }
        }
        // Cross-SM visibility for the route ids before dispatch counting
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx, [=]() { __syncthreads(); });
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::Gate, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::GateEndTime, timing_t1);
        }
    }

    // Intra-SM named barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;

    using namespace m2n_detail;

    if (warp_idx < kNumDispatchWarps) {
        // ---- dispatch warps: route + push indices + counts -------------------
        if constexpr (kAGPhase != 2) {
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                const uint32_t token_idx = i + (lane_idx / kNumTopk);
                const uint32_t topk_slot = lane_idx % kNumTopk;
                if (token_idx < num_tokens and lane_idx < kNumActivateLanes) {
                    if (topk_slot < kInputTopk) {
                        const int real_expert_idx = static_cast<int>(
                            __ldg(route_ids + token_idx * kInputTopk + topk_slot));
                        if constexpr (kNumRoutedExperts == kNumExperts) {
                            expert_idx = real_expert_idx;
                        } else {
                            if (real_expert_idx >= 0) {
                                const uint32_t routed_owner =
                                    static_cast<uint32_t>(real_expert_idx) / kNumRoutedExpertsPerEGRank;
                                const uint32_t routed_local =
                                    static_cast<uint32_t>(real_expert_idx) % kNumRoutedExpertsPerEGRank;
                                expert_idx = static_cast<int>(
                                    routed_owner * kNumExpertsPerEGRank + routed_local);
                            }
                        }
                    } else if constexpr (kInputTopk < kNumTopk) {
                        if (topk_slot == kInputTopk) {
                            const uint32_t shared_owner = sym_buffer.rank_idx % kNumEGRanks;
                            expert_idx = static_cast<int>(
                                shared_owner * kNumExpertsPerEGRank + kNumRoutedExpertsPerEGRank);
                        }
                    }
                    if (expert_idx >= 0)
                        process(token_idx * kNumTopk + topk_slot, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        unsigned long long timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::DispatchCount, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::DispatchCountEndTime, timing_t1);
        }

        // Get SM offsets in the global send counters
        timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::DispatchOffsets, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::DispatchOffsetsEndTime, timing_t1);
        }

        // Write source indices into the owner EG rank's slots
        timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = kNumAGRanks + expert_idx / kNumExpertsPerEGRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerEGRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::DispatchIndexPush, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::DispatchIndexPushEndTime, timing_t1);
        }

        // Grid sync
        timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::DispatchGridSync, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::DispatchGridSyncEndTime, timing_t1);
        }

        // Write expert counts to the owner EG ranks.  The high 32 bits are
        // re-packed to 1 per source rank, so EG waits for `kNumAGRanks`.
        timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = kNumAGRanks + i / kNumExpertsPerEGRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerEGRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                // AFD plugin change (FastAFD 3c71619 uses a relaxed atom here):
                // release orders the per-source count above before the ready
                // bits that the EG side spins on and then acquires.
                ptx::atomic_add_rel_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    (1ull << 32) | (expert_status & 0xffffffff));
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::DispatchCountPush, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::DispatchCountPushEndTime, timing_t1);
        }

        // No barrier before EG pulls: the per-rank release on the count push
        // plus the EG side's per-expert count spin (acquire) order the data;
        // EG pulls from each source as soon as its counts arrive.
        }  // kAGPhase != 2 (route + pushes)

        // Dispatch-only kernels exit here: the SMs go back to the attention
        // stream while the EG round trip is in flight; the combine kernel
        // (phase 2) picks up at the tag-2 wait.
        if constexpr (kAGPhase != 1) {
        // Joint sync A with combine warps
        ptx::sync_unaligned(kNumDispatchThreads + kNumCombineThreads, kDispatchWithEpilogueBarrierIdx);

        // No pull phase on AG ranks.

        // Joint sync B (combine warps reach this after the combine-ready wait)
        ptx::sync_unaligned(kNumDispatchThreads + kNumCombineThreads, kDispatchWithEpilogueBarrierIdx);

        // Clean own send counters for the next call (overlaps combine reduce)
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        }

        // Cleanup barrier: signal-only (arrive).  The matching wait is
        // deferred into the next kernel launch, where the inter-layer gap
        // has already hidden the cross-rank latency.
        unsigned long long timing_t0 = do_timing_dispatch_leader ? clock64() : 0;
        comm::nvlink_barrier_arrive<kNumRanks, kNumSMs, kNumDispatchThreads,
                                    kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true
        );
        if (do_timing_dispatch_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::CleanupArrive, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::CleanupArriveEndTime, timing_t1);
        }
        }  // kAGPhase != 1
    } else if constexpr (kAGPhase != 1) {
        // ---- combine warps: wait expert outputs + reduce top-k ---------------
        const auto combine_warp_idx = warp_idx - kNumDispatchWarps;
        const auto combine_thread_idx = combine_warp_idx * 32 + lane_idx;

        // Joint sync A (dispatch finished the tag-1 barrier)
        ptx::sync_unaligned(kNumDispatchThreads + kNumCombineThreads, kDispatchWithEpilogueBarrierIdx);

        // Wait until every EG rank has pushed all combine rows
        unsigned long long timing_t0 = do_timing_combine_leader ? clock64() : 0;
        if (do_timing_combine_leader)
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::CombineBarrierStartTime, timing_t0);
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumCombineThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, combine_thread_idx,
            [&]() { ptx::sync_aligned(kNumCombineThreads, kEpilogueFullBarrierIdx); }
        );
        if (do_timing_combine_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::CombineBarrierWait, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::CombineBarrierEndTime, timing_t1);
        }

        // Joint sync B with dispatch warps (they clean the workspace next)
        ptx::sync_unaligned(kNumDispatchThreads + kNumCombineThreads, kDispatchWithEpilogueBarrierIdx);

        // Combine: reduce top-k results and write back (same as the symmetric kernel)
        timing_t0 = do_timing_combine_leader ? clock64() : 0;
        if (do_timing_combine_leader)
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::CombineReduceStartTime, timing_t0);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 uint4");
        DG_STATIC_ASSERT(kHidden / kNumChunks <= 32 * 128, "Too many combine registers");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (combine_warp_idx + i * kNumCombineWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (combine_warp_idx + kNumCombineWarps * 2) * kNumChunkBytes);
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + combine_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumCombineWarps + combine_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumCombineWarps) {
            int stored_topk_slot_idx = -1;
            if (lane_idx < kInputTopk) {
                stored_topk_slot_idx =
                    static_cast<int>(__ldg(route_ids + token_idx * kInputTopk + lane_idx));
            } else if constexpr (kInputTopk < kNumTopk) {
                if (lane_idx == kInputTopk)
                    stored_topk_slot_idx = static_cast<int>(kNumRoutedExperts);
            }
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                if constexpr (kDirectCombine) {
                    uint32_t mask = total_mask;
                    while (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        const auto src_ptr = reinterpret_cast<const uint4*>(math::advance_ptr<uint8_t>(
                            combine_token_buffer.get_rank_buffer(slot_idx)
                                                .get_data_buffer(token_idx).get_base_ptr(),
                            chunk_byte_offset));
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            const auto uint4_values = __ldg(src_ptr + j * 32 + lane_idx);
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                        }
                    }
                } else {
                    uint32_t mask = total_mask;
                    const auto move_mask_and_load = [&](const uint32_t& i) {
                        if (mask) {
                            const uint32_t slot_idx = __ffs(mask) - 1;
                            mask ^= 1 << slot_idx;
                            if (cute::elect_one_sync()) {
                                const auto src_ptr = math::advance_ptr<uint8_t>(
                                    combine_token_buffer.get_rank_buffer(slot_idx)
                                                        .get_data_buffer(token_idx).get_base_ptr(),
                                    chunk_byte_offset);
                                ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                                ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                            }
                            __syncwarp();
                            return true;
                        }
                        return false;
                    };

                    bool do_reduce = move_mask_and_load(load_stage_idx);
                    while (do_reduce) {
                        do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                        combine_load_barriers[load_stage_idx]->wait(combine_phase);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                            const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                            const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                            #pragma unroll
                            for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                        }
                        combine_phase ^= load_stage_idx;
                        load_stage_idx ^= 1;
                    }
                }

                if constexpr (kDirectCombine) {
                    const auto dst_ptr = reinterpret_cast<uint4*>(math::advance_ptr<uint8_t>(
                        y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset));
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);
                        dst_ptr[j * 32 + lane_idx] = casted;
                    }
                } else {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        uint4 casted;
                        auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                        if (j == 0) {
                            ptx::tma_store_wait<0>();
                            __syncwarp();
                        }
                        ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                       casted.x, casted.y, casted.z, casted.w);
                    }
                    __syncwarp();

                    if (cute::elect_one_sync()) {
                        cute::tma_store_fence();
                        ptx::tma_store_1d(
                            math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                            combine_store_buffer, kNumChunkBytes);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }
        }
        if (do_timing_combine_leader) {
            const auto timing_t1 = clock64();
            m2n_detail::ag_timing_add(debug_timings, m2n_detail::AGTimingSlot::CombineReduce, timing_t1 - timing_t0);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::CombineReduceEndTime, timing_t1);
            m2n_detail::ag_timing_store(debug_timings, m2n_detail::AGTimingSlot::KernelEnd, timing_t1);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

// ---------------------------------------------------------------------------
// EG kernel: dispatch recv + expert FP8xFP4 MLP + combine send
// ---------------------------------------------------------------------------
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t STORE_BLOCK_M,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumDispatchPullStages,
    uint32_t kNumSMs,
    uint32_t kNumAGRanks, uint32_t kNumEGRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kUseFP8Weights,
    uint32_t kNumLanes,
    bool kEnableDebugTiming = false,
    uint32_t kNumRanks = kNumAGRanks + kNumEGRanks,
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K = kHidden,
    uint32_t L2_SHAPE_N = kHidden,
    uint32_t L2_SHAPE_K = kIntermediateHidden,
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumEGRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_mega_moe_m2n_eg_impl(int* cumulative_local_expert_recv_stats,
                                   unsigned long long* debug_timings,
                                   const uint32_t num_prefetch_bytes,
                                   const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer0,
                                   const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer1,
                                   const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer2,
                                   const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer3,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts0,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf0,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output0,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts0,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf0,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts1,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf1,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output1,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts1,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf1,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts2,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf2,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output2,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts2,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf2,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts3,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf3,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output3,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts3,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf3,
                                   // Per-layer weight descriptors in GLOBAL memory, [num_layers][4]:
                                   // (l1_w, l1_w_sf, l2_w, l2_w_sf); consumed via tensormap fences.
                                   const cute::TmaDescriptor* __restrict__ weight_descs,
                                   const void* const* __restrict__ l1_weight_ptrs,
                                   const uint32_t num_layers) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumEGRanks == 0, "Invalid number of experts or EG ranks");
    DG_STATIC_ASSERT(1 <= kNumLanes and kNumLanes <= 4, "M2N EG supports one to four microbatch lanes");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts0);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf0);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output0);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts0);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf0);
        if constexpr (kNumLanes >= 2) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts1);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf1);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output1);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts1);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf1);
        }
        if constexpr (kNumLanes >= 3) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts2);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf2);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output2);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts2);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf2);
        }
        if constexpr (kNumLanes >= 4) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts3);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf3);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output3);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts3);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf3);
        }
    }

    // Token and buffer layouts
    constexpr auto fp8_token_layout = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // Data types: FP8 activations; weights are FP4 (e2m1) or FP8 (e4m3).
    // Both are 1 byte per element in shared memory, so the B tile sizes,
    // swizzles and UMMA descriptors are identical; only the TMA gmem format
    // and the block-scaled MMA operand encoding differ.
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = std::conditional_t<kUseFP8Weights,
                                         cutlass::float_e4m3_t,
                                         cutlass::detail::float_e2m1_unpacksmem_t>;

    // MMA configs (always swap A/B, 2-CTA MMA, K-major)
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t kNumDispatchBarrierSlots = kNumDispatchWarps * kNumDispatchPullStages;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchBarrierSlots, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, kNumDispatchPullStages,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchBarrierSlots + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchBarrierSlots + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchBarrierSlots + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchBarrierSlots + kNumStages * 2 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem       = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumDispatchBarrierSlots + kNumStages * 2 + kNumEpilogueStages * 2);

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchBarrierSlots; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Cross-lane persistent phase counters (the mbarrier/TMEM phases carry
    // across the lane loop; resetting them would deadlock the pipelines)
    uint32_t pull_mbarrier_phase[kNumDispatchPullStages] = {};
    uint32_t mma_iter_idx = 0, epi_iter_idx = 0;

    // ---- layer loop: ONE launch serves the whole step ---------------------
    // Per-launch costs (smem barrier init, cluster syncs, TMEM alloc,
    // launch ramp) amortize across all layers; the per-layer weight
    // descriptors live in global memory behind tensormap fences.
    for (uint32_t layer_i = 0; layer_i < num_layers; ++ layer_i) {
    const cute::TmaDescriptor* tensor_map_l1_weights = weight_descs + layer_i * 4 + 0;
    const cute::TmaDescriptor* tensor_map_l1_weights_sf = weight_descs + layer_i * 4 + 1;
    const cute::TmaDescriptor* tensor_map_l2_weights = weight_descs + layer_i * 4 + 2;
    const cute::TmaDescriptor* tensor_map_l2_weights_sf = weight_descs + layer_i * 4 + 3;

    // ---- microbatch-lane loop: one full-width pass per lane ----------------
    for (uint32_t lane_i = 0; lane_i < kNumLanes; ++ lane_i) {
    const auto& sym_buffer = lane_i == 0 ? sym_buffer0 :
                             (lane_i == 1 ? sym_buffer1 :
                             (lane_i == 2 ? sym_buffer2 : sym_buffer3));
    const auto& tensor_map_l1_acts = lane_i == 0 ? tensor_map_l1_acts0 :
                                     (lane_i == 1 ? tensor_map_l1_acts1 :
                                     (lane_i == 2 ? tensor_map_l1_acts2 : tensor_map_l1_acts3));
    const auto& tensor_map_l1_acts_sf = lane_i == 0 ? tensor_map_l1_acts_sf0 :
                                        (lane_i == 1 ? tensor_map_l1_acts_sf1 :
                                        (lane_i == 2 ? tensor_map_l1_acts_sf2 : tensor_map_l1_acts_sf3));
    const auto& tensor_map_l1_output = lane_i == 0 ? tensor_map_l1_output0 :
                                       (lane_i == 1 ? tensor_map_l1_output1 :
                                       (lane_i == 2 ? tensor_map_l1_output2 : tensor_map_l1_output3));
    const auto& tensor_map_l2_acts = lane_i == 0 ? tensor_map_l2_acts0 :
                                     (lane_i == 1 ? tensor_map_l2_acts1 :
                                     (lane_i == 2 ? tensor_map_l2_acts2 : tensor_map_l2_acts3));
    const auto& tensor_map_l2_acts_sf = lane_i == 0 ? tensor_map_l2_acts_sf0 :
                                        (lane_i == 1 ? tensor_map_l2_acts_sf1 :
                                        (lane_i == 2 ? tensor_map_l2_acts_sf2 : tensor_map_l2_acts_sf3));
    // Workspaces (must match the AG kernel byte-for-byte)
    const auto workspace = layout::WorkspaceM2N(
        sym_buffer.get_base_ptr(), kNumAGRanks, kNumEGRanks, kNumExperts,
        kNumMaxTokensPerRank, kNumTopk);

    // Registered inputs (live on AG ranks; EG pulls through `sym_buffer.map`)
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine outputs (live on AG ranks; EG pushes through `sym_buffer.map`)
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // Task scheduler: experts of this EG rank; sources are the AG ranks
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumAGRanks,
        /* kRecvCountHighTarget = */ kNumAGRanks,
        layout::WorkspaceM2N>(workspace);

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;
    // Gates the free-running GEMM warps' (TMA-A/TMA-B/MMA) per-iteration
    // scheduler fetch on the dispatch warps' per-iteration release point.
    // Without it they can read the PREVIOUS layer's expert counts, since the
    // count spin target is already satisfied before cleanup zeroes the sums.
    constexpr uint32_t kSchedFetchBarrierIdx = 8;
    constexpr uint32_t kNumSchedFetchThreads = kNumDispatchThreads + 3 * 32;

    using namespace m2n_detail;
    const bool do_timing_leader = kEnableDebugTiming and sm_idx == 0 and lane_idx == 0;
    const bool do_timing_dispatch_leader = do_timing_leader and warp_idx == 0;
    if (do_timing_dispatch_leader)
        eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::LaneStart, clock64());

    // Adjust registers
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Prefetch the head of the L1 weights into L2 while the recv counts
        // are still in flight: the weight stream is the largest serial
        // component of the work phase, and the wait window has idle HBM
        // bandwidth.  No hazard (static read-only data) and no completion
        // dependency, so it goes before the deferred barrier wait.
        if (lane_i == 0 and warp_idx == 0 and num_prefetch_bytes > 0) {
            const void* l1_weights_prefetch_l = l1_weight_ptrs[layer_i];
            constexpr uint32_t kPrefetchChunk = 128 * 1024;
            const uint32_t slice = (num_prefetch_bytes / kNumSMs) & ~15u;
            const auto slice_base = static_cast<const uint8_t*>(l1_weights_prefetch_l) +
                                    static_cast<uint64_t>(sm_idx) * slice;
            for (uint32_t off = lane_idx * kPrefetchChunk; off < slice; off += 32 * kPrefetchChunk)
                prefetch_l2_evict_last(slice_base + off, cute::min(kPrefetchChunk, (slice - off) & ~15u));
            __syncwarp();
        }

        // Deferred tag-3 wait from the previous layer (signals were sent at
        // the previous kernel's end; this consumes the phase nearly for
        // free).  Pulls themselves are gated by the per-expert recv-count
        // spins, so no pre-pull barrier is needed.
        unsigned long long t0 = do_timing_dispatch_leader ? clock64() : 0;
        comm::nvlink_barrier_wait<kNumRanks, kNumSMs, kNumDispatchThreads,
                                  kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::Tag3Wait, clock64() - t0);
        // Release the free-running GEMM warps (TMA-A/TMA-B/MMA) for this
        // iteration: the deferred wait above globally orders the previous
        // iteration's workspace cleanup, so their count fetches can no
        // longer observe stale (pre-cleanup) values.
        t0 = do_timing_dispatch_leader ? clock64() : 0;
        ptx::sync_unaligned(kNumSchedFetchThreads, kSchedFetchBarrierIdx);
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::SchedFetchBarrierDispatch, clock64() - t0);

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from AG ranks into the local L1 buffer.
        // Each dispatch warp owns two pull stages so the next remote TMA load
        // can run while the current token's SF/weight/local-store work drains.

        // Cache expert token counts in registers (same pattern as scheduler);
        // the spin is volatile, so add an acquire fence before reading the
        // remotely-written src indices / token data it gates.
        t0 = do_timing_dispatch_leader ? clock64() : 0;
        scheduler.fetch_expert_recv_count();
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::RecvCountWaitDispatch, clock64() - t0);
        if (do_timing_dispatch_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::RecvCountReadyTime, clock64());
        if constexpr (kEnableDebugTiming) {
            constexpr uint32_t kNumExpertsPerTimingLane =
                math::constexpr_ceil_div(kNumExpertsPerRank, 32u);
            uint32_t local_total_tokens = 0;
            uint32_t local_max_tokens = 0;
            uint32_t local_total_m_blocks = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kNumExpertsPerTimingLane; ++ i) {
                const uint32_t expert_idx = i * 32 + lane_idx;
                if (expert_idx < kNumExpertsPerRank) {
                    const uint32_t count = scheduler.stored_num_tokens_per_expert[i];
                    local_total_tokens += count;
                    local_total_m_blocks += math::ceil_div(count, BLOCK_M);
                    local_max_tokens = local_max_tokens > count ? local_max_tokens : count;
                }
            }
            const uint32_t total_tokens = __reduce_add_sync(0xffffffff, local_total_tokens);
            const uint32_t total_m_blocks = __reduce_add_sync(0xffffffff, local_total_m_blocks);
            uint32_t max_tokens = local_max_tokens;
            #pragma unroll
            for (uint32_t offset = 16; offset > 0; offset >>= 1) {
                const uint32_t other = __shfl_down_sync(0xffffffff, max_tokens, offset);
                max_tokens = max_tokens > other ? max_tokens : other;
            }
            if (do_timing_dispatch_leader) {
                eg_timing_store<kNumLanes>(
                    debug_timings, layer_i, lane_i,
                    EGTimingSlot::TotalRecvTokens,
                    static_cast<unsigned long long>(total_tokens));
                eg_timing_store<kNumLanes>(
                    debug_timings, layer_i, lane_i,
                    EGTimingSlot::MaxExpertTokens,
                    static_cast<unsigned long long>(max_tokens));
                eg_timing_store<kNumLanes>(
                    debug_timings, layer_i, lane_i,
                    EGTimingSlot::TotalMBlocks,
                    static_cast<unsigned long long>(total_m_blocks));
            }
        }
        fence_acquire_sys();

        uint32_t expert_pool_block_offset = 0;

        // Per-source-rank counts for current expert (re-loaded when expert changes)
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumAGRanks, 32u);
        int current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        t0 = do_timing_dispatch_leader ? clock64() : 0;

        struct PullTokenRecord {
            bool valid;
            uint32_t rank_idx;
            uint32_t src_token_topk_idx;
            uint32_t src_token_idx;
            uint32_t src_topk_idx;
            uint32_t token_idx_in_expert;
            uint32_t pool_token_idx;
            uint32_t sf_pool_token_idx;
            uint32_t pool_block_idx;
        };

        auto prepare_pull_token = [&](const uint32_t& token_idx, PullTokenRecord& record) {
            record.valid = false;

            // Advance expert until within the range
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    return;

                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }

            // Load per-source-rank counts when expert changes
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumAGRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin source rank selection via iterative min-peeling
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by the AG dispatch via NVLink)
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            record.valid = true;
            record.rank_idx = current_rank_in_expert_idx;
            record.src_token_topk_idx = src_token_topk_idx;
            record.src_token_idx = src_token_idx;
            record.src_topk_idx = src_topk_idx;
            record.token_idx_in_expert = token_idx_in_expert;
            record.pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            record.sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_expert);
            record.pool_block_idx = expert_pool_block_offset + token_idx_in_expert / BLOCK_M;
        };

        auto issue_pull_token = [&](const PullTokenRecord& record, const uint32_t& pull_stage_idx) {
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(pull_stage_idx);
            const auto pull_mbarrier = dispatch_barriers[warp_idx * kNumDispatchPullStages + pull_stage_idx];

            if (record.valid and cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(record.src_token_idx).get_base_ptr(),
                                   record.rank_idx),
                    pull_mbarrier, kHidden);
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
            }
            __syncwarp();
        };

        auto finish_pull_token = [&](const PullTokenRecord& record, const uint32_t& pull_stage_idx) {
            const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(pull_stage_idx);
            const auto pull_mbarrier = dispatch_barriers[warp_idx * kNumDispatchPullStages + pull_stage_idx];

            // Load and store SF (overlaps with TMA token load)
            constexpr uint32_t kNumSFUint32 = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(record.src_token_idx).get_base_ptr<uint32_t>(),
                record.rank_idx);
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + record.sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and token data
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + record.src_token_topk_idx,
                    record.rank_idx);
                *l1_topk_weights_buffer.get_data_buffer(record.pool_token_idx).get_base_ptr<float>() = weight;

                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase[pull_stage_idx]);

                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(record.pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                // Write source metadata for combine write-back (AG union rank)
                *workspace.get_token_src_metadata_ptr(record.pool_token_idx) =
                    {record.rank_idx, record.src_token_idx, record.src_topk_idx};

                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(record.pool_block_idx), 1);
            }
            __syncwarp();
        };

        PullTokenRecord current_record = {};
        PullTokenRecord next_record = {};
        uint32_t current_pull_stage = 0;
        uint32_t next_token_idx = sm_idx * kNumDispatchWarps + warp_idx;
        prepare_pull_token(next_token_idx, current_record);
        next_token_idx += kNumGlobalWarps;
        issue_pull_token(current_record, current_pull_stage);

        if constexpr (kNumDispatchPullStages == 1) {
            while (current_record.valid) {
                finish_pull_token(current_record, current_pull_stage);
                prepare_pull_token(next_token_idx, next_record);
                next_token_idx += kNumGlobalWarps;
                issue_pull_token(next_record, current_pull_stage);
                current_record = next_record;
            }
        } else {
            while (current_record.valid) {
                const uint32_t next_pull_stage =
                    current_pull_stage + 1 == kNumDispatchPullStages ? 0 : current_pull_stage + 1;
                prepare_pull_token(next_token_idx, next_record);
                next_token_idx += kNumGlobalWarps;
                issue_pull_token(next_record, next_pull_stage);
                finish_pull_token(current_record, current_pull_stage);
                current_record = next_record;
                current_pull_stage = next_pull_stage;
            }
        }
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::PullLoopDispatch, clock64() - t0);
        if (do_timing_dispatch_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::PullDoneTime, clock64());

        // Clean workspace for the next usage, and also do cumulative stats
        t0 = do_timing_dispatch_leader ? clock64() : 0;
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::DispatchEpilogueSync, clock64() - t0);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx != 0) {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-source-rank token count
                for (uint32_t j = thread_idx; j < kNumAGRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 arrival stuffs
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }

        // Cleanup barrier: signal-only (arrive).  The matching wait is
        // deferred into the next kernel launch, where the inter-layer gap
        // has already hidden the cross-rank latency.
        t0 = do_timing_dispatch_leader ? clock64() : 0;
        comm::nvlink_barrier_arrive<kNumRanks, kNumSMs, kNumDispatchThreads,
                                    kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true
        );
        if (do_timing_dispatch_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::CleanupArrive, clock64() - t0);
    } else if (warp_idx == kNumDispatchWarps) {
        // Wait until dispatch warps have reached the scheduler-fetch point.
        const bool do_timing_tmaa_leader = do_timing_leader and warp_idx == kNumDispatchWarps;
        unsigned long long t0 = do_timing_tmaa_leader ? clock64() : 0;
        ptx::sync_unaligned(kNumSchedFetchThreads, kSchedFetchBarrierIdx);
        if (do_timing_tmaa_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::SchedFetchBarrierTmaA, clock64() - t0);

        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        unsigned long long loop_t0 = do_timing_tmaa_leader ? clock64() : 0;
        if (do_timing_tmaa_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaAStartTime, loop_t0);
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait the entire token arrival for linear 1
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                unsigned long long wait_t0 = (kEnableDebugTiming and lane_idx == 0) ? clock64() : 0;
                while (ptx::ld_acq(ptr) != expected);
                if (kEnableDebugTiming and lane_idx == 0)
                    eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::L1ArrivalWait, clock64() - wait_t0);
            } else {
                DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;
                unsigned long long wait_t0 = (kEnableDebugTiming and lane_idx == 0) ? clock64() : 0;
                while (ptx::ld_acq_gpu(ptr) != expected);
                if (kEnableDebugTiming and lane_idx == 0)
                    eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::L2ArrivalWait, clock64() - wait_t0);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx;

                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * 2 + SMEM_SFA_SIZE_PER_STAGE * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
        if (do_timing_tmaa_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaALoop, clock64() - loop_t0);
        if (do_timing_tmaa_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaAEndTime, clock64());
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // Wait until dispatch warps have reached the scheduler-fetch point.
        ptx::sync_unaligned(kNumSchedFetchThreads, kSchedFetchBarrierIdx);
        const bool do_timing_tmab_leader = do_timing_leader and warp_idx == kNumDispatchWarps + 1;

        // The weight descriptors live in gmem: acquire the tensormap proxy
        // before the first TMA issue of this layer (cheap, per-thread)
        if (cute::elect_one_sync()) {
            cute::tma_descriptor_fence_acquire(tensor_map_l1_weights);
            cute::tma_descriptor_fence_acquire(tensor_map_l1_weights_sf);
            cute::tma_descriptor_fence_acquire(tensor_map_l2_weights);
            cute::tma_descriptor_fence_acquire(tensor_map_l2_weights_sf);
        }
        __syncwarp();
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        unsigned long long loop_t0 = do_timing_tmab_leader ? clock64() : 0;
        if (do_timing_tmab_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaBStartTime, loop_t0);
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? tensor_map_l2_weights : tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? tensor_map_l2_weights_sf : tensor_map_l1_weights_sf;

            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;
            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;

                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        // Each CTA issues the 2SM weight load; the transaction
                        // counts GMEM bytes, so packed FP4 credits half the
                        // unpacked smem bytes per issue while FP8 credits full.
                        constexpr uint32_t kBArrivalBytes =
                            kUseFP8Weights ? SMEM_B_SIZE_PER_STAGE * 2 : SMEM_B_SIZE_PER_STAGE;
                        full_barriers[stage_idx]->arrive_and_expect_tx(kBArrivalBytes + SMEM_SFB_SIZE_PER_STAGE * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
        if (do_timing_tmab_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaBLoop, clock64() - loop_t0);
        if (do_timing_tmab_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::TmaBEndTime, clock64());
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // Wait until dispatch warps have reached the scheduler-fetch point.
        ptx::sync_unaligned(kNumSchedFetchThreads, kSchedFetchBarrierIdx);
        const bool do_timing_mma_leader = do_timing_leader and warp_idx == kNumDispatchWarps + 2 and is_leader_cta;

        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        unsigned long long loop_t0 = do_timing_mma_leader ? clock64() : 0;
        if (do_timing_mma_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::MmaStartTime, loop_t0);
        if (is_leader_cta) {
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            auto& current_iter_idx = mma_iter_idx;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        // UTCCP copy SFA and SFB to TMEM
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }

                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
        if (do_timing_mma_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::MmaLoop, clock64() - loop_t0);
        if (do_timing_mma_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::MmaEndTime, clock64());
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Adjust registers
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // GEMM epilogue warps
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // Ensure the epilogue barrier cannot run with the pull barrier
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Persistently schedule over blocks
        const bool do_timing_epi_leader = do_timing_leader and epilogue_warp_idx == 0;
        unsigned long long loop_t0 = do_timing_epi_leader ? clock64() : 0;
        if (do_timing_epi_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::EpiStartTime, loop_t0);
        auto& current_iter_idx = epi_iter_idx;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // Wait UMMA arrival
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (block_phase == sched::BlockPhase::Linear1) {
                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                //   (values[0], values[2]), (values[1], values[3]),
                //   (values[4], values[6]), (values[5], values[7])
                float stored_cached_weight = 0;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    float2 swiglu_values[kNumAtomsPerStore * 2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .get_base_ptr<float>();
                        }

                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // Load from TMEM
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // Apply SwiGLU: silu(gate) * up
                        // Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
                        }

                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());
                        if (lane_idx < 4)
                            smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const float2 wp_amax =
                            smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        uint32_t row = lane_idx;
                        uint32_t col = warp_idx_in_wg;
                        const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                     + i * ATOM_M * L1_OUT_BLOCK_N
                                                                     + row * L1_OUT_BLOCK_N
                                                                     + (col ^ (row / 2)) * kNumBankGroupBytes;
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                        if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M);
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
            } else {
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;
                unsigned long long writeback_t0 =
                    (kEnableDebugTiming and lane_idx == 0) ? clock64() : 0;

                // L2 BF16 epilogue: write GEMM output to the source AG rank's
                // combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (kEnableDebugTiming and lane_idx == 0)
                    eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::L2RemoteWriteback, clock64() - writeback_t0);
            }
        });
        if (do_timing_epi_leader)
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::EpilogueLoop, clock64() - loop_t0);
        if (do_timing_epi_leader)
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::EpiEndTime, clock64());

        // Deallocate tensor memory (only at the very END: last layer, last lane)
        if (layer_i == num_layers - 1 and lane_i == kNumLanes - 1 and epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // Signal the AG ranks that all combine rows have been pushed
        unsigned long long tag2_t0 = (do_timing_leader and epilogue_warp_idx == 0) ? clock64() : 0;
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        if (do_timing_leader and epilogue_warp_idx == 0) {
            eg_timing_add<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::Tag2Barrier, clock64() - tag2_t0);
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::Tag2EndTime, clock64());
            eg_timing_store<kNumLanes>(debug_timings, layer_i, lane_i, EGTimingSlot::LaneEnd, clock64());
        }

        // Barrier with dispatch warps, so that they can clean the workspace.
        // No local combine reduction on EG ranks.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
    }
    }  // lane loop
    }  // layer loop
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
