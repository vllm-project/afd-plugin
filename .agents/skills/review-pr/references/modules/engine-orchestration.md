# Engine Orchestration

Primary design: `docs/design/module/attention_runtime.md` and
`docs/design/module/ffn_runtime.md` (role execution flow) in the reviewed
head.

Use for cross-role request flow semantics: async-DP engine behavior from the
patched `EngineCore`/`DPEngineCoreProc` busy loops, AFD request routing
between attention and FFN, the DBO yield custom op, ubatching orchestration,
and the `AFDBalancedRoutingStrategy` routing simulator. Does not own the patch
mechanics ([compat-patches.md](compat-patches.md)) or worker internals
([workers-runners.md](workers-runners.md)).

## Contract checks

- Keep one owner for attention→FFN routing; models and workers must not
  independently select a route or a peer.
- Keep the async-DP patches externally transparent: non-AFD requests observe
  upstream vLLM 0.26.0 scheduling, output order, and finish semantics
  unchanged, on both `EngineCoreProc` and `DPEngineCoreProc` paths.
- Preserve request identity and ordering across the two roles; correlate every
  connector result, ubatch work item, and DBO yield to the originating
  request, and make terminal state monotonic (no forwarding after finish,
  abort, or failure).
- Define shutdown ordering between the worlds: FFN shutdown must not hang
  attention and vice versa; cancellation propagates to in-flight transfers and
  queued ubatch items.
- Keep the balanced routing strategy benchmark-only and explicitly registered;
  it must never become the default routing behavior silently.

Test engine-loop patch behavior (AFD on/off), ubatch state transitions, and
shutdown/cancellation in `tests/unit/compat/patches/` and
`tests/unit/model_executor/`; name the DBO E2E scenario that exercises the
real path.
