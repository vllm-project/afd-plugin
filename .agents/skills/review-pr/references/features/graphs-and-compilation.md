# Graphs and Compilation

Load with the primary module contract for changes to graph capture or
compilation paths: FULL_DECODE_ONLY CUDA graph (`cuda_graph.py`, GPU),
ACL graph via the `mla_graph` patches (NPU), `_AFDCaptureEventTracker`, graph
metadata preparation, and graph/eager fallback.

Designs: `docs/design/module/attention_runtime.md`,
`docs/design/module/execution_platforms.md`.

## Feature checks

- Keep capture/replay parity: every input to a captured graph (metadata,
  buffers, connector state, ubatch keys) must be identical between capture
  and replay; flag any capture-time-only branch that replay can also reach.
- Respect decode-only scoping: CUDA graph paths must not trigger for prefill
  or mixed batches; the FULL_DECODE_ONLY gate stays enforced where the
  connector contract declares it.
- Keep ACL graph patches pinned: `mla_graph` patches
  `vllm_ascend/attention/mla_v1.py` at commit `80d8c194f` — apply the patch
  contract (markers, upstream source, signature parity) and re-verify the copy
  against that ref.
- Preserve the eager escape hatch: every graph path has a tested eager
  fallback, and fallback selection is explicit config or documented platform
  behavior, not an accident of ordering.
- State resource implications: graph size, memory pools, and capture-count
  changes belong in the PR description and the relevant user guide; a capture
  change that shifts memory is a perf claim needing evidence.

Test graph gating and metadata parity CPU-side where possible
(`tests/unit/v1/worker/`); device behavior claims need the named GPU scenario
(`afd-graph-*`, `afd-graph-dbo-*`) or NPU run.
