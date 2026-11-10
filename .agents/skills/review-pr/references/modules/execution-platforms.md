# Execution Platforms

Primary design: `docs/design/module/execution_platforms.md` in the reviewed
head.

Use for NPU/GPU selection and platform mechanics: `validation.py`
(`assert_compatible_afd_stack`, `afd_worker_qualname_for_platform_default`,
`validate_gpu_model_runner_v2_config`,
`validate_npu_model_runner_v2_config`), `setup.py` Ascend-op build gating, and
`csrc/npu/` (`ascend_kernels/` a2e/e2a and `grouped_matmul_layered` ACLNN
operators under the `npu_op_*` build system, `pybind/` → `afd_plugin._C_ascend`,
`build_aclnn.sh` installing into `afd_plugin/_cann_ops_custom`, and the drivers
in `scripts/`). Does not own connector behavior or the deferred NPU patch
application ([npu-compat.md](npu-compat.md)).

## Contract checks

- Derive worker selection from live platform detection (vLLM/vLLM-Ascend
  default worker classes), never hardcoded device strings; reject unsupported
  platforms (Ascend 310P/xlite) explicitly with an actionable error.
- Keep the build contract: compile Ascend ops only in a detected Ascend
  environment or with `AFD_BUILD_ASCEND_OPS=1` (unset/empty means auto-detect;
  an invalid value must raise, not default); honor `AFD_SKIP_ACLNN_BUILD=1`;
  document `SOC_VERSION` defaults and any new build knob in the install docs.
- Pair `csrc/` kernel changes (a2e/e2a and grouped-matmul-layered hosts and
  kernels) with the torch adapter/extension updates they require, and state
  accuracy and performance impact; kernel behavior changes need NPU evidence,
  never simulation. Layered GMM selection changes belong to the
  `upgrade-layered-gmm-ops` workflow.
- Keep version pins coherent: `TARGET_VLLM_VERSION` in `compat/vllm.py`, the
  `vllm` extra in `pyproject.toml`, `VLLM_BASE_TAG` and its version assertion
  in `docker/Dockerfile.ci`, installation/support references, and design-page `upstream_refs` move
  together, or the PR states why not. Read the actual pin; do not trust the
  value quoted in a review note or in this skill.
- Keep GPU path parity visible: `csrc/gpu/` contains only a README reserving
  the directory — the plugin builds no GPU native extension. A PR adding GPU
  native ops must state where the GPU counterpart lives and why it is not
  plugin-owned today.
- Vendored `.run`/`.whl` artifacts under `afd_plugin/connectors/npu/bin/` and
  the generated `_cann_ops_custom` tree are packaging, not source; a change
  there needs version/CANN justification (see
  [connectors.md](connectors.md#vendored-ascend-artifacts)).

Test platform selection and build gating in `tests/unit/v1/worker/`
(`test_runtime_classpaths.py`), `tests/unit/config/test_validation.py`, and
`tests/unit/package/` (`test_package.py`, `test_ascend_build_files.py`,
`test_ascend_op_host_build.py`, `test_cam_async_build.py`); kernel changes
additionally name the NPU run that produced their evidence.
