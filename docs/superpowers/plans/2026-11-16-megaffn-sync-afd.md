# MegaFFN synchronous AFD implementation plan

> Execute inline with executing-plans and test-driven-development.

**Goal:** Opt-in F-side MegaFFN normal-mode execution with A-side routing and existing a2e/e2a.
**Architecture:** Retain role-local vLLM weight loading and wire protocol. After loading, convert BF16 expert matrices into MXFP8/NZ once. A per-layer adapter owns at most two fixed-capacity graphs and returns an owned, completed output to E2A.
**Tech Stack:** Python, PyTorch NPU, blockrt, CANN; local Dhq_AFD branches in both repositories.
**Spec:** User-approved single-layer V4 integration in this chat; no standalone design file.

## Global constraints
- Preserve default native backend, existing modifications, build/output links, host drivers and other users' processes.
- Support only V4, A-side gate, CAMP2p, one A/one F, TP/DP/PP/CP=1, eager, BF16 unquantized source weights. No DBO/speculation.
- Current MegaFFN SwiGLU lacks V4 activation clipping. Reject nonzero swiglu_limit; smoke model explicitly disables clipping. Do not silently change model semantics.
- A supplies already-scaled routing weights; F never gates, adds shared experts or scales routing again.
- Actual weights are quantized from the loaded model, never silently replaced by uninitialized synthetic tensors.
- Normal mode only. Async dispatch/combine and persistent mode are out of scope.

## Tasks
1. [x] Mega runtime: test exact input/weight contracts and MXFP8 packing; implement injected weights and checked fixed-shape IO, owned synchronous output.
2. [x] AFD: test backend selection/rejection and native fallback; implement config, runtime validation and lazy per-layer adapter.
3. [x] Device validation: compare small deterministic FFN against a reference; then run two-card HTTP smoke with a separate model/config/log directory. Record failures honestly.
4. [x] Review full diff; record evidence and remaining limitations. Keep local branches, no push.

## Review focus
- Token-count changes must not broadcast decode rows into a prefill graph.
- Dynamic top-k values and int32 wire IDs must reach the graph correctly.
- Retained output must not be overwritten before E2A consumes it.
- Model loading must finish before weight conversion, with the correct logical matrix orientation.
- SDK and device mapping must not be changed by sourcing the standalone mega environment.

## Progress / decisions
- Branches created from AFD 9393311 and mega 5ccda55. Existing build/output preserved.
- Ruling: work in user-requested checkouts on Dhq_AFD, no extra worktree; preserves existing deployment paths.
- Ruling: use unclipped V4 smoke config and reject clipping; production V4 clamp support remains separate work.
- Pre-flight: runner accepts [M,H] BF16, [M,K] FP32 weights, INT32/INT64 IDs; captured kernel buffers use checked INT32 IDs and loaded [E,K,N] BF16 weights. Normal-mode output must be synchronized and cloned.

## Outcome

- Implementation, focused CPU tests, actual NPU reference, two-card HTTP smoke and independent review completed.
- AFD: 204 passed, 1 skipped. Mega: 8 passed. NPU reference: M=128/8/1, two varied replays each, all passed at rtol=0.12/atol=0.002.
- E2E: physical A=2/F=3, one-layer dummy-weight V4, two HTTP 200 responses, 8 generated tokens each, finite logprobs and explicit MegaFFN execution logs. Repeated after removal of diagnostic tensor dump.
- Fixed runtime device ordinal mapping and BF16/FP32 finalize tiling. Built libmega and ascendmk_rt in existing private deployment.
- Replaced graph eviction with capacity 1/max_tokens graphs and zero-weight padding; native graph allocations live until process exit.
- Namespaced custom AICPU package in private CANN 9.2 SDK; no host driver, firmware or DeviceOS security changes.
- Remaining scope and exact launch/evidence paths: [A5 synchronous MegaFFN guide](../../npu/mega_ffn_sync_a5.md).
- Keep both Dhq_AFD branches local, no merge or push. Preserve pre-existing build/output symlinks.
