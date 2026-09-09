# Execution Platforms

Primary design: `docs/design/module/execution_platforms.md` in the reviewed
head.

Use for NPU/GPU selection and platform mechanics: `validation.py`
(`assert_compatible_afd_stack`,
`afd_worker_qualname_for_platform_default`), `setup.py` Ascend-op build
gating, and `csrc/` (a2e/e2a ACLNN operators, `aclnn_torch_adapter`,
`torch_extension` → `afd_plugin._C_ascend`). Does not own connector behavior
or the deferred NPU patch application ([npu-compat.md](npu-compat.md)).

## Contract checks

- Derive worker selection from live platform detection (vLLM/vLLM-Ascend
  default worker classes), never hardcoded device strings; reject unsupported
  platforms (Ascend 310P/xlite) explicitly with an actionable error.
- Keep the build contract: compile Ascend ops only in a detected Ascend
  environment or with `AFD_BUILD_ASCEND_OPS=1`; honor
  `AFD_SKIP_ACLNN_BUILD=1`; document `SOC_VERSION` defaults and any new build
  knob in the install docs.
- Pair `csrc/` kernel changes (a2e/e2a hosts and kernels) with the torch
  adapter/extension updates they require, and state accuracy and performance
  impact; kernel behavior changes need NPU evidence, never simulation.
- Keep version pins coherent: vLLM 0.26.0 (`TARGET_VLLM_VERSION`) and
  vLLM-Ascend `80d8c194f` move together across the compat gate, docker base
  image, docs, and design-page `upstream_refs`, or the PR states why not.
- Keep GPU path parity visible: `csrc/gpu/` is reserved — a PR adding GPU
  native ops must state where the GPU counterpart lives.

Test platform selection and build gating in `tests/unit/v1/worker/` (runtime
classpath tests) and `tests/unit/package/`; kernel changes additionally name
the NPU run that produced their evidence.
