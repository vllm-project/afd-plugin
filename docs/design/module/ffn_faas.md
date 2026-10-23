---
title: Asynchronous FFN FaaS
kind: module
status: draft
owners:
  - "@jiaran-king"
primary_code_paths:
  - "afd_plugin/ffn_faas/**"
  - "csrc/npu/ascend_kernels/afd_ffn_faas/**"
  - "csrc/npu/pybind/torch_binding_ffn_faas.cpp"
validation_paths:
  - "tests/unit/ffn_faas/**"
  - "tests/e2e/models/deepseek_v4_flash/ffn_faas_inference.py"
  - "tests/e2e/models/deepseek_v2_lite/ffn_faas_inference.py"
  - "tests/npu/ffn_faas_native.py"
  - "tests/npu/ffn_faas_service.py"
  - "tests/npu/ffn_faas_service_driver.py"
  - "tests/npu/ffn_faas_mixed_probe/**"
last_reviewed: 2026-10-23
---

# Asynchronous FFN FaaS

This module owns the standalone FFN lifecycle and Ascend execution resources.
Two checkpoint contracts coexist: DeepSeek-V2 BF16 with the fused device
megakernel, and DeepSeek-V4 Flash ModelSlim per-channel W4A8 with native host
compute operators. Both use the same session transport, service and ownership
boundaries; eager Attention is required for both.

`FFNService` exposes CPU-only health, readiness, status and stop endpoints.
`FFNModelRunner` selects the model and executor from the checkpoint type, then
coordinates Model `load`/`native_view` and Executor `initialize`, `submit`,
`retire`, `close`. F constructs no EngineCore, Worker, ModelRunner, request
scheduler, KV cache or full CausalLM.

## V4 W4A8 contract

CLI selects `V4FFNModel` and `NativeW4A8Executor` for the V4 checkpoint.
`V4FFNModel` owns routed parameters for the 43 main layers. Its parameter-only
factory container reuses native routed-expert construction, checkpoint name
mapping, EP sharding and packed/NZ conversion, without a native MoERunner or
its communication dispatchers. Necessary F-local loading groups end after
packing; the separate A/F session group owns transport. Each borrowed view
contains its model-layer ID, packed INT32 INT4 W13/W2, encoded INT64 scales,
FP32 compensation and static clamp/output semantics.

`NativeW4A8Executor` borrows those views and implements one transaction between
`PREPARE` and `FINISH`. The service and runner do not expand GMM arguments or
quantization details. Completed `PREPARE` returns only
`(result, model_layer, packed_rows)` to select a borrowed layer; scratch,
routes, source and generation remain owned until `FINISH`, and no second work
can be selected while computation is pending. Native dynamic quantization, GMM1
with SwiGLU/clamp/INT8 quantization, and native GMM2 run on the session stream.
The executor converts padded expert prefixes into counts for both native GMMs.

V4 gate has only an upper clamp of 10; up is clamped to [-10, 10] before SiLU.
The native fused operation reconstructs MSD intermediates in FP32; it does not
round full gate/up outputs to BF16 first. A owns shared experts and already
scaled FP32 probabilities. Native dynamic input quantization is not claimed
bit-identical to the old CAM dispatch scale formula.

## BF16 fused contract

CLI selects `FFNModel` and the base `NPUExecutor` for the unquantized V2
checkpoint. `FFNModel` plans only this F rank's routed experts from the
safetensors index, validates every required key and shard before allocating,
then streams BF16 gate/up/down tensors into borrowed `NativeWeightView` storage.
It allocates no EngineCore or native MoERunner state. Historical V2 device
evidence and the prepared comparison are recorded in
[validation](../../npu/FFN_FAAS_VALIDATION.md) and the
[benchmark plan](../../npu/FFN_FAAS_BENCHMARK.md).

The device-fused `EXECUTE` entry receives from an HCCL window, selects the
layer on device, groups routes by local expert, runs both Cube projections with
FP32 accumulation and BF16 outputs, evaluates SwiGLU on AIV and publishes the
result. `SERVE` repeats that bounded transaction up to `MAX_WORK_PER_SERVE`
times and returns one aggregate receipt; `completed_work` validates the count
before the runner advances the service counter. Scratch and borrowed weights
outlive the active completion and are reused without duplicating the expert
set. Only valid routes are projected and scattered; zero-work transactions
still retire normally so peer readiness and draining stay uniform.

The production mixed-core entry performs device selection, expert grouping,
Cube projections, AIV SwiGLU and remote result publication. Its executor owns
one reusable scratch buffer and one active F completion. Eager execution can
queue four serial exchanges inside a model scope, with status validation before
return and worker exit on failure; ordinary Session calls remain synchronous.
The C++ submitter retains every scoped input/output before enqueue and keeps a
pending owner if a partially submitted work cannot return to Python. F
initializes its host thread affinity through the existing Ascend CPU allocation
planner; CPU-only session metadata does not import that runtime planner.

## Shared lifecycle

`NPUExecutor` owns communication windows, stream, scratch, request storage,
completion and peer lifecycle; `NativeW4A8Executor` only replaces the compute
between `PREPARE` and `FINISH`, while the base class owns the fused
`EXECUTE`/`SERVE` submissions. Paired startup warmup uses two zero-input
transactions: one repeats expert 0 with zero local work on other F ranks; the
next distributes routes across F. Zero routed rows skip computation; real zero
input remains a valid row.

After the output copy completes, `FINISH` (V4) or the fused `EXECUTE`/`SERVE`
publication (V2) writes result metadata/ready and advances the F source cursor.
A returns slot credit only after it consumes every F result. PREPARE and host
layer selection return no credit. Unknown enqueue/wait failures retain owners
and abort the session; cleanup releases Model only after Executor proves peer
quiescence, otherwise the owning process exits with resources retained.
Scoped A submissions preserve producer/consumer events, storage lifetime,
bounded queueing and validation before model return.

See [service usage](../../npu/FFN_FAAS_SERVICE.md) and the
[control ABI](../../npu/FFN_FAAS_ABI.md). CPU contracts, actual weight loading,
native numerical checks, full-model requests and cleanup are separate
verification stages; one does not establish the others. The standalone source
under `tests/npu/ffn_faas_mixed_probe/` checks the mixed 1:2 core geometry and
finite Cube/AIV handoffs separately; it is not registered in the executor.
V4 fusion speedup remains subsequent work, not a gate for either backend.
