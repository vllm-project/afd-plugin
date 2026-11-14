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
`NVSHMEM_HOME` must contain `include/` and `lib/`, including both the host and
device libraries. CUDA architectures may be a semicolon-separated CMake list.
The same compatible extension and NVSHMEM libraries must exist on every PE.
NVSHMEM 3.4.5 requires equal PE counts on every node; use balanced placements
such as 2A on one node and 2F on the other, or 1A1F on each node.
The runtime validates node counts and maps logical A-first ranks to
node-contiguous NVSHMEM PEs before initialization.

The receive window and per-peer/stage outboxes are allocated symmetrically.
Packing performs dtype conversion before the byte transfer. One blocking
`putmem_signal` plus `quiet` completes each transfer before its stream may reuse
the outbox; the target is the local symmetric address plus the slot offset,
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
configuration remain deployment-specific. See the
[GPU tests](../../tests/gpu/README.md) and
[validation record](../../docs/design/async_gpu_cross_node_research.md).
