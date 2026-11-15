# Engine Orchestration

Primary design: `docs/design/module/attention_runtime.md` and
`docs/design/module/ffn_runtime.md` (role execution flow) in the reviewed
head.

Use for cross-role request flow semantics: async-DP engine behavior from the
patched `EngineCore` / `EngineCoreProc` / `DPEngineCoreProc` busy loops,
`DPAsyncMPClient.add_request_async`, AFD request routing between attention and
FFN, the DBO yield custom op (`v1/worker/dbo.py` →
`maybe_apply_dbo_yield`), native microbatch metadata integration
(`v1/worker/attention_metadata.py`), and the `AFDBalancedRoutingStrategy` routing
simulator. Does not own the patch mechanics
([compat-patches.md](compat-patches.md)) or worker internals
([workers-runners.md](workers-runners.md)).

## Contract checks

- Keep one owner for attention→FFN routing; models and workers must not
  independently select a route or a peer.
- Keep the async-DP patches externally transparent: non-AFD requests observe
  upstream scheduling, output order, and finish semantics unchanged, on both
  `EngineCoreProc` and `DPEngineCoreProc` paths.
- Preserve request identity and ordering across the two roles; correlate every
  connector result, ubatch work item, and DBO yield to the originating
  request, and make terminal state monotonic (no forwarding after finish,
  abort, or failure).
- Define shutdown ordering between the worlds: FFN shutdown must not hang
  attention and vice versa; cancellation propagates to in-flight transfers and
  queued ubatch items. DeepSeek-V2-Lite has known DBO shutdown failures — check
  whether a PR claims to fix that honestly, with evidence, rather than
  suppressing the symptom.
- Native DBO is a vLLM capability that AFD reuses: exactly two microbatches,
  enabled on Attention only. Async CAM FFN consumes work items outside
  Attention's DBO thread schedule. GPU DBO accepts Attention DP1 or DP > 1;
  NPU graph DBO with plain MLA requires `FULL_DECODE_ONLY`. A change that adds
  a second scheduling path or relaxes the two-ubatch bound is a design change.
- Keep the balanced routing strategy benchmark-only and explicitly registered
  (gated by `AFD_FORCE_BALANCED_TOPK_IDS`); it must never become the default
  routing behavior silently.

Test engine-loop patch behavior (AFD on/off), ubatch state transitions, and
shutdown/cancellation in `tests/unit/compat/patches/`
(`test_engine_core.py`, `test_async_dp_engine.py`,
`test_async_dp_forward_context.py`, `test_gpu_mrv2_ubatching.py`),
`tests/unit/v1/worker/test_dbo.py`,
`tests/unit/model_executor/test_async_cam_dbo.py`, and
`tests/unit/model_executor/test_routing_simulator.py`; name the DBO E2E
scenario that exercises the real path.
