# Async CAM and Ubatching

Load with the primary module contract for changes to the experimental async
CAM data path: `CAMAsyncAFDConnector`, AFD-managed two-stage MoE ubatching,
the NPU ubatch wrappers/metadata, DBO yield interplay, and the DSV3.2/DSV4
async CAM forward modules under `afd_plugin/model_executor/models/npu/`.

Designs: `docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`,
`docs/design/module/connector_contracts.md`; recent work concentrates here
(DSV4 CAM FFN control path, `v0.26.0_camasync_dsv4`).

## Feature checks

- Respect experimental status: async CAM must not become a default or silently
  required path; the opt-in remains explicit in config/recipes, and the sync
  CAM P2P path keeps working.
- Preserve MoE semantics across the two-stage split: gate computation
  placement, shared-expert handling, and topk routing inputs must match the
  single-stage semantics the model defines; the execution planner
  (`model_executor/npu/async_cam_ubatching.py`) is pure planning — no device
  side effects.
- Prove state-machine safety: `AFDAsyncTransferState`, async FFN work items,
  and ubatch contexts handle abort, timeout, and shutdown at every pending
  state; a dropped work item or a lost completion signal hangs a rank.
- Keep hidden-state handoff correct at ubatch boundaries: residual streams,
  layernorm state, and metadata keys (`AscendUbatchMetadata`, graph metadata)
  must be consistent between attention and FFN sides and across graph capture.
- Require evidence proportional to risk: control-path or scheduling changes
  here carry perf implications (the DSV4 CAM FFN work was a `[PERF][NPU]`
  change) — demand comparable base/head NPU measurements or an explicit
  no-impact rationale; unit tests alone do not prove device behavior.

Name the NPU evidence (who ran what, on which environment per
`docs/npu/TESTING.md`) for any behavioral claim; CPU tests cover planning and
state transitions only.
