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
last_reviewed: 2026-10-23
---

# Asynchronous FFN FaaS

This module owns the standalone FFN lifecycle and Ascend execution resources.
Its initial model is DeepSeek-V4 Flash ModelSlim per-channel W4A8 on A3,
with native compute operators and eager Attention. Fusion is subsequent work.

`FFNService` exposes CPU-only health, readiness, status and stop endpoints.
`FFNModelRunner` coordinates Model `load`/`native_view` and Executor
`initialize`, `submit(EXECUTE)`, `retire`, `close`. F constructs no EngineCore,
Worker, ModelRunner, request scheduler, KV cache or full CausalLM.

`V4FFNModel` owns routed parameters for the 43 main layers. Its parameter-only
factory container reuses native routed-expert construction, checkpoint name
mapping, EP sharding and packed/NZ conversion, without a native MoERunner or
its communication dispatchers. Necessary F-local loading groups end after
packing; the separate A/F session group owns transport. Each borrowed view
contains its model-layer ID, packed INT32 INT4 W13/W2, encoded INT64 scales,
FP32 compensation and static clamp/output semantics.

`NPUExecutor` owns communication windows, stream, scratch, request storage,
completion and peer lifecycle. `NativeW4A8Executor` borrows model views and
implements one transaction between `PREPARE` and `FINISH`. The service and
runner do not expand GMM arguments or quantization details; a subsequent fused
executor can implement the same submission/completion boundary.

The native backend explicitly uses the host. Completed `PREPARE` returns only
`(result, model_layer, packed_rows)` to select a borrowed layer. Scratch,
routes, source and generation remain owned until `FINISH`; no second work can
be selected while computation is pending. Native dynamic quantization, GMM1
with SwiGLU/clamp/INT8 quantization, and native GMM2 run on the session stream.
The executor converts padded expert prefixes into counts for both native GMMs.
Paired startup warmup uses two zero-input transactions: one repeats expert 0
with zero local work on other F ranks; the next distributes routes across F.
Zero routed rows skip computation; real zero input remains a valid row.

V4 gate has an upper clamp of 10; up is clamped to [-10, 10] before SiLU.
The native fused operation reconstructs MSD intermediates in FP32; it does not
round full gate/up outputs to BF16 first. A owns shared experts and already
scaled FP32 probabilities. Native dynamic input quantization is not claimed
bit-identical to the old CAM dispatch scale formula.

`FINISH` publishes result ready after the output copy and advances F's source
cursor. A returns input credit only after consuming every F result. Unknown
enqueue/wait failures retain owners; Model is released only after Executor
proves peer quiescence, otherwise the owning process exits with resources
retained. A scoped submissions preserve producer/consumer events, storage
lifetime, bounded queueing and validation before model return.

See [service usage](../../npu/FFN_FAAS_SERVICE.md) and the
[control ABI](../../npu/FFN_FAAS_ABI.md). CPU contracts, actual weight loading,
native numerical checks, full-model requests and cleanup are separate
verification stages; one does not establish the others. Historical V2 fusion
experiments are outside this framework change.
