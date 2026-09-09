# NPU Runtime Compat

Primary design: `docs/design/module/compatibility_and_patches.md` (NPU
sections) in the reviewed head; platform init ordering lives in
`docs/design/module/execution_platforms.md`.

Use for `afd_plugin/compat/npu/`: deferred patch application
(`runtime.py::apply_afd_ascend_patches_if_needed`), runtime config, forward
context, feature validation, Ascend ops loading (`ops.py`), and the NPU
profiler. Does not own the vLLM-Ascend patch files under
`compat/patches/npu/` ([compat-patches.md](compat-patches.md)) or the NPU
workers ([workers-runners.md](workers-runners.md)).

## Contract checks

- Apply NPU patches only after vLLM-Ascend completes platform initialization —
  during AFD config construction and worker startup, never at import time.
- Keep `feature_validation.py` fail-fast: unsupported NPU feature combinations
  must be rejected before any device work, with the unsupported pair named.
- Verify Ascend op namespace presence (CAM/AFD torch ops) once, cache the
  result, and fail with an actionable message; do not re-probe per call or
  silently continue when ops are missing.
- Guard every `torch_npu`/`vllm_ascend` import so module import stays
  CPU-safe; runtime checks happen inside guarded functions.
- Keep NPU behavior observable: a deferred patch or runtime tweak that changes
  execution must be loggable and attributable when troubleshooting per
  `docs/npu/TROUBLESHOOTING.md`.

Test in `tests/unit/compat/npu/`: application ordering, idempotence, feature
rejections, and op-presence failure paths — all CPU-safe via guarded imports.
