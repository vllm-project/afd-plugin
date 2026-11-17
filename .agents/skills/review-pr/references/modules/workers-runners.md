# Workers and Model Runners

Primary design: `docs/design/module/attention_runtime.md` and
`docs/design/module/ffn_runtime.md` in the reviewed head.

Use for `afd_plugin/v1/worker/` on both platforms: `AFDAttentionWorker` /
`AFDFFNWorker` and `AFDNPUAttentionWorker` / `AFDNPUFFNWorker`, the
Attention ModelRunnerV2 variants (`AFDAttentionModelRunnerV2`,
`AFDNPUAttentionModelRunnerV2`), the FFN runners (`GPUFFNModelRunner`,
`GPUFFNRoutedExpertsModelRunner`, `AFDNPUFFNModelRunner`,
`AsyncNPUFFNModelRunner`), attention/FFN metadata builders, `cuda_graph.py`,
`dbo.py`, `attention_metadata.py`, and the NPU `ubatch_runner.py`,
`mla_graph.py`, `async_ffn_two_stream.py`, and `dspark_speculator.py`. Does
not own platform selection ([execution-platforms.md](execution-platforms.md)),
connector transport, or model wrappers ([model-integration.md](model-integration.md)).

## Contract checks

- Preserve the role split: FFN-side `execute_model()` is connector-driven and
  fails fast on scheduler-driven calls; attention-side runners never construct
  FFN components and vice versa.
- There is no V1 runner and no fallback. Both roles require
  `VLLM_USE_V2_MODEL_RUNNER=1`; Attention runs native ModelRunnerV2 (GPU) or
  the NPU V2 runner, while FFN keeps a connector-driven runner that is *not*
  a V2 runner. `validate_gpu_model_runner_v2_config` and
  `validate_npu_model_runner_v2_config` enforce the flag; do not accept a
  change that turns the requirement into a warning or a silent default.
- Keep the three FFN execution paths distinct and selected by contract, not by
  convenience:
  | Path | Selected by | Runner |
  | --- | --- | --- |
  | Control-plane-driven | `connector.control_plane is not None` | `GPUFFNModelRunner` / `AFDNPUFFNModelRunner` |
  | Routed experts | `issubclass(connector_cls, AFDRoutedExpertsConnector)` | `GPUFFNRoutedExpertsModelRunner` |
  | Async CAM (no control plane) | `CAMAsyncAFDConnector` on NPU | `AsyncNPUFFNModelRunner` |
  A change that collapses two of these, or routes by connector *name*, is a
  finding.
- Enforce the patch contract on every patched worker/runner method — the
  Attention runners are the heaviest patch sites on both platforms: marked AFD
  differences, upstream-copy fidelity against the pinned ref, identical
  signatures, `Patch reason:`/`Patch functionality:` comments, and a row in
  `tests/unit/compat/patches/test_signatures.py`.
- Keep graph capture sound: metadata must be prepared identically for capture
  and replay; capture-only state must not leak into eager paths; DBO yield and
  ubatch wrappers behave the same under graph and eager.
  `_AFDCaptureEventTracker` exists to detect capture input-preparation drift —
  a change that silences those assertions is worse than the drift.
- Keep the lazy export maps CPU-safe: `afd_plugin/v1/worker/__init__.py` and
  `afd_plugin/v1/worker/npu/__init__.py` resolve dotted paths through
  `__getattr__`; new runtime classes must be added there rather than imported
  eagerly, or plugin import on a CPU host breaks.
- Update the GPU and NPU twins together or prove platform neutrality; the NPU
  runners (largest patch surface) and GPU runners drift independently, and the
  NPU sync/async FFN runners diverge by design.

Test role contracts, metadata building, and gating in
`tests/unit/v1/worker/` and its `npu/` subtree using the CPU-safe runtime-guard
convention (see
[test-quality-evaluation.md](../checks/test-quality-evaluation.md)); name the
GPU/NPU gap for anything not run.
