# Async GPU transport extension

`_C_gpu_async` supplies NVSHMEM device initialization, a graph-safe
put-with-signal kernel, and signal wait/test operations. It has no Torch build
dependency. CPU and NPU installations do not build or import it by default.

Build on a host whose CUDA toolkit and C++ runtime match the deployment nodes:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
export AFD_BUILD_ASCEND_OPS=0 AFD_BUILD_GPU_ASYNC_OPS=1
export AFD_CUDA_ARCHITECTURES=90 MAX_JOBS=4
uv pip install --no-build-isolation -e .
```

The build environment needs setuptools, setuptools-scm, wheel, pybind11,
CMake >= 3.24, CUDA 13 and the runtime's `nvidia-nvshmem-cu13==3.4.5` package.
Install build dependencies first when using `--no-build-isolation`; alternatively
set `NVSHMEM_HOME` to the matching NVSHMEM installation for an isolated build.
When using an explicit `NVSHMEM_HOME`, set it at build time and on each worker.
It must contain `include/` and `lib/`, including both the host and device
libraries. Otherwise both loaders use the importable `nvidia.nvshmem` package.
CUDA architectures may be a semicolon-separated CMake list.
The same compatible extension and NVSHMEM libraries must exist on every PE.
NVSHMEM 3.4.5 requires equal PE counts on every node; use balanced placements
such as 2A on one node and 2F on the other, or 1A1F on each node.
The runtime validates node counts and maps logical A-first ranks to
node-contiguous NVSHMEM PEs before initialization.

The receive window is allocated symmetrically across all ranks; when any rank
has unmapped remote peers, a second collective outbox allocation reserves one
slot per remote peer and ring (omitted entirely when all peers are locally
mapped).
Packing performs dtype conversion before the byte transfer. Blocking puts send
only populated field prefixes at their fixed offsets; their lengths come from
tensor shapes without a routing readback. A device `fence` orders those puts
before the header's `putmem_signal`, and `quiet` completes the transfer before
its stream may reuse the outbox. The target is the receive window's symmetric
address plus the slot offset,
with the destination PE passed separately. The kernel reads the int32 dispatch
counter on the GPU on every graph replay and publishes an aligned uint64 signal.
Mapped peers retain direct Torch writes. Replies retain separate routed/shared
fields and FP32 accumulation.

Protocol v5 uses 64-bit signals; v4 deployments cannot communicate with it.
Bootstrap checks protocol, extension availability and geometry across ranks.
Graphs and streams must drain before collective heap release/finalization.
Worker failure requires restarting the entire AFD group; PE hot replacement
is unsupported.

Set `NVSHMEM_REMOTE_TRANSPORT=ibrc` for the initial RoCE/RDMA qualification.
`NVSHMEM_SYMMETRIC_SIZE` must cover receive slots **and** outboxes; use the
connector's logged allocation size to select it. NIC selection and any IBGDA
configuration remain deployment-specific.

Both native kernels that call `nvshmem_uint64_test` (reply wait and arrival
poll) use `nvshmemx_collective_launch`. In the pinned release this API checks
local GPU occupancy, launches on the calling PE's cooperative stream and
orders that stream against the caller's stream with events. It requires no
matching launch from an idle Attention PE. The successful test supplies
NVSHMEM's target-side consistency operation before the receiver reads the
payload. See NVIDIA's [launch contract](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html),
[3.4.5 launch implementation](https://github.com/NVIDIA/nvshmem/blob/v3.4.5-0/src/device/launch/collective_launch.cpp)
and [3.4.5 device test](https://github.com/NVIDIA/nvshmem/blob/v3.4.5-0/src/include/device/nvshmem_defines.h).

Reply waits use the configured `afd_process_group_timeout_s` on the GPU, so
CUDA Graph replay has the same deadline as eager execution. The mapped
fallback uses a system-scope acquire load. A missing reply traps the CUDA
context and must restart the whole
AFD world. Rebuild the Python package and native bridge together: `wait_signal`
now requires a timeout argument. Native RMA operations and collective teardown
still depend on the transport and process supervisor; this is not in-process
fault recovery.
