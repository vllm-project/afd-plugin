// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// No Torch dependency: pointers and streams are owned by the Python window.
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
constexpr int FIELD_COUNT = 5;  // Header, indices, weights, routed and shared rows.
struct Field {
    size_t offset;
    size_t bytes;
};
struct Payload {
    Field fields[FIELD_COUNT];
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void publish(char* dest, const char* src, Payload payload,
                        uint64_t* signal, const int32_t* sequence,
                        uint64_t value, int peer) {
    // Read on the GPU on every graph replay. Sign extension agrees with Torch's
    // int32 -> int64 copy in the mapped-peer path, including counter wraparound.
    if (sequence) value = static_cast<uint64_t>(static_cast<int64_t>(*sequence));
    // Counts come from tensor shapes, so graph replay needs no routing D2H.
    // Fixed field offsets keep the receive layout unchanged while avoiding
    // unused capacity bytes for short batches. Blocking puts plus fence order
    // every payload field before the header's completion signal.
    for (int i = 1; i < FIELD_COUNT; ++i) {
        const auto field = payload.fields[i];
        if (field.bytes) {
            nvshmem_putmem(dest + field.offset, src + field.offset, field.bytes, peer);
        }
    }
    nvshmem_fence();
    const auto header = payload.fields[0];
    nvshmem_putmem_signal(dest + header.offset, src + header.offset, header.bytes,
                         signal, value, NVSHMEM_SIGNAL_SET, peer);
    // Complete before this stream repacks its symmetric outbox or frees it.
    nvshmem_quiet();
}

__global__ void wait_signal(uint64_t* signal, uint64_t value) {
    nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, value);
}

__global__ void fetch_signals(uint64_t* signals, uint64_t* observations, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && nvshmem_uint64_test(signals + i, NVSHMEM_CMP_NE, observations[i])) {
        // test supplies NVSHMEM's payload visibility guarantee. A producer
        // cannot change this slot again until the consumer has replied.
        observations[i] = nvshmem_signal_fetch(signals + i);
    }
}
}  // namespace

PYBIND11_MODULE(_C_gpu_async, m) {
    m.def("initialize", [](uintptr_t attr) {
        // Full initialization registers this module's linked NVSHMEM device
        // state. Calling hostlib_init_attr alone is insufficient for kernels.
        return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID,
                                 reinterpret_cast<nvshmemx_init_attr_t*>(attr));
    });
    m.def("finalize", [] { nvshmem_finalize(); });
    m.def("put_signal", [](uintptr_t dest, uintptr_t src,
                           const std::array<std::array<size_t, 2>, FIELD_COUNT>& fields,
                           uintptr_t signal, uintptr_t sequence, uint64_t value,
                           int peer, uintptr_t stream) {
        Payload payload;
        for (int i = 0; i < FIELD_COUNT; ++i) {
            payload.fields[i] = {fields[i][0], fields[i][1]};
        }
        publish<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<char*>(dest), reinterpret_cast<const char*>(src), payload,
            reinterpret_cast<uint64_t*>(signal), reinterpret_cast<const int32_t*>(sequence),
            value, peer);
        check(cudaGetLastError());
    });
    m.def("wait_signal", [](uintptr_t signal, uint64_t value, uintptr_t stream) {
        wait_signal<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<uint64_t*>(signal), value);
        check(cudaGetLastError());
    });
    m.def("fetch_signals", [](uintptr_t signals, uintptr_t observations, int count,
                              uintptr_t stream) {
        constexpr int THREADS = 128;
        fetch_signals<<<(count + THREADS - 1) / THREADS, THREADS, 0,
                        reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<uint64_t*>(signals),
            reinterpret_cast<uint64_t*>(observations), count);
        check(cudaGetLastError());
    });
}
