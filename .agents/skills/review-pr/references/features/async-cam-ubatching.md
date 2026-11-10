# Async CAM, Layered FFN, and Ubatching

Load with the primary module contract for changes to the async CAM data path:
`CAMAsyncAFDConnector` and `AFDAsyncTransferState`, AFD-managed native
ubatching, the NPU async FFN runner and its two-stream mode, DBO yield
interplay, the layered GMM executor, and the DSV3.2/DSV4 async CAM forward
modules under `afd_plugin/model_executor/models/npu/`.

Designs: `docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`,
`docs/design/npu/async_ffn_runtime.md`, `docs/design/npu/cam_async_routed_ops.md`,
`docs/design/module/connector_contracts.md`,
`docs/design/module/ffn_runtime.md`.

## Feature checks

- Respect the layer's status. `CAMAsyncAFDConnector` itself is a supported
  product path for DeepSeek-V4 W4A8/W8A8; the *experimental* parts are FFN
  two-stream (`ffn_two_stream`, default off) and any DP layout beyond the
  documented ones. Experimental flags must not become defaults or silently
  required paths; the synchronous `CAMP2pAFDConnector` path keeps working, and
  DSpark must not acquire DBO or two-stream.
- Layered GMM is the only Async CAM FFN compute path. Incompatible weights
  fail startup rather than selecting another FFN path — a PR that adds a
  fallback contradicts `ffn_runtime.md`. `dynamicQuant=1` is required;
  `w13_scale_bias`/`w2_scale_bias` are required for W4A8; W8A8 uses INT8 NZ
  tensors with a single shared zero bias; `swiglu_limit` is applied in both
  eager and FULL while W8A8 keeps its unclipped computation.
- Keep `AsyncCAMLayeredExecutor` and `LayeredFFNWeights` pure about device
  side effects where the design says so: planning (`async_cam_layout.py`) is
  host-side and testable on CPU, device execution belongs to the executor.
- Prove state-machine safety: `AFDAsyncTransferState`, async FFN work items,
  ubatch contexts, and two-stream slots handle abort, timeout, and shutdown at
  every pending state; a dropped work item or a lost completion signal hangs a
  rank. FULL retires a slot only after its send is observed complete; eager
  slot reuse requires the same guarantee.
- Keep hidden-state handoff correct at ubatch boundaries: residual streams,
  layernorm state, and graph/ubatch metadata keys must be consistent between
  attention and FFN sides and across capture.
- Both roles select `VLLM_USE_V2_MODEL_RUNNER=1`; Attention stays eager under
  Async CAM (`--enforce-eager`) and may enable native DBO with exactly two
  microbatches, while FFN never joins Attention's DBO thread schedule. Native
  DBO on FFN is rejected.
- Require evidence proportional to risk: control-path or scheduling changes
  here carry perf implications and have historically shipped as `[PERF][NPU]`
  changes — demand comparable base/head NPU measurements or an explicit
  no-impact rationale; unit tests alone do not prove device behavior.
- Cleanup is part of the contract: a pending CAM receive cannot be interrupted
  cleanly, so the E2E harness uses a marker-based SIGKILL sweep with a 120 s
  survivor window. Changes that touch receiver teardown must keep that path
  working; a "fix" that makes teardown look clean by suppressing survivors is
  a regression.

Name the NPU evidence (who ran what, on which environment per
`tests/npu/README.md`) for any behavioral claim; CPU tests cover planning and
state transitions only.
