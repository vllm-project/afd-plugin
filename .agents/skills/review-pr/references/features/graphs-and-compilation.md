# Graphs and Compilation

Load with the primary module contract for changes to graph capture or
compilation paths: `FULL_DECODE_ONLY` / `FULL` CUDA graph
(`v1/worker/cuda_graph.py`, GPU), ACL graph via the `mla_graph` module and
patches (`v1/worker/npu/mla_graph.py`, `compat/patches/npu/mla_graph.py`, NPU),
`_AFDCaptureEventTracker`, graph metadata preparation, and graph/eager
fallback.

Designs: `docs/design/module/attention_runtime.md`,
`docs/design/module/execution_platforms.md`.

## Feature checks

- Keep capture/replay parity: every input to a captured graph (metadata,
  buffers, connector state, ubatch keys, async CAM graph payload) must be
  identical between capture and replay; flag any capture-time-only branch that
  replay can also reach. `_AFDCaptureEventTracker` asserts that capture input
  preparation ran exactly as expected — do not accept weakening or deleting
  those assertions without a replacement guarantee.
- Respect mode scoping. GPU uses `FULL_DECODE_ONLY` where the connector
  contract says so; Ascend CAM uses `FULL`, rejects piecewise and decode-only
  modes, and requires plain MLA graph + DBO to use `FULL_DECODE_ONLY`. A change
  that lets an unsupported mode through to capture is a P1.
- Keep ACL graph patches pinned: `compat/patches/npu/mla_graph.py` patches
  `vllm_ascend/attention/mla_v1.py` at the pinned vLLM-Ascend ref — apply the
  patch contract (markers, upstream source, signature parity) and re-verify the
  copy against that ref.
- Preserve the eager escape hatch: every graph path has a tested eager
  fallback, and fallback selection is explicit config or documented platform
  behavior, not an accident of ordering. Async CAM explicitly has *no*
  automatic eager fallback: unsupported FULL configurations and mismatched
  transport parameters fail startup.
- Graph-owned resources are bounded: async CAM FULL owns four private graph
  pools and four graphs; FFN two-stream FULL owns its own pools per slot. State
  resource implications (graph size, memory pools, capture counts) in the PR
  description and the relevant user guide; a capture change that shifts memory
  is a perf claim needing evidence.
- Async CAM graph startup is a host-only handshake
  (`afd_plugin/async_cam_startup.py`, `FFNGraphPlan`): the requested
  eager/full mode, executor error, and signature must agree between roles
  before HCCL creation, with `GRAPH_STARTUP_TIMEOUT_SECONDS` bounding the wait.
  A change to those keys is a cross-role protocol change.

Test graph gating and metadata parity CPU-side where possible
(`tests/unit/v1/worker/test_cuda_graph.py`, `test_npu_mla_graph.py`,
`test_async_ffn_graph.py`, `tests/unit/connectors/test_async_cam_startup.py`);
device behavior claims need the named GPU scenario (`afd-v2-graph-*`,
`afd-v2-graph-dbo-*`) or the NPU run recorded per `tests/npu/README.md`.
