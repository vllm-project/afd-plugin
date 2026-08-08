---
title: Compatibility and patches
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/compat/__init__.py"
  - "afd_plugin/compat/vllm.py"
  - "afd_plugin/compat/npu/__init__.py"
  - "afd_plugin/compat/npu/feature_validation.py"
  - "afd_plugin/compat/npu/runtime.py"
  - "afd_plugin/compat/npu/runtime_config.py"
  - "afd_plugin/compat/patches/**/*.py"
related_code_paths:
  - "afd_plugin/__init__.py"
  - "afd_plugin/v1/worker/**"
  - "AGENTS.md"
depends_on:
  - "plugin_boundary.md"
validation_paths:
  - "tests/unit/compat/**"
  - "tests/unit/package/test_package.py"
  - "tests/unit/v1/worker/test_runtime_classpaths.py"
  - "tests/unit/v1/worker/test_npu_runtime.py"
upstream_refs:
  - "vLLM vllm.v1.engine.core.EngineCore and EngineCoreProc"
  - "vLLM vllm.v1.engine.core_client.DPAsyncMPClient"
  - "vLLM vllm.forward_context.set_forward_context"
  - "vLLM vllm.engine.arg_utils.EngineArgs and vllm.config.VllmConfig"
  - "vLLM vllm.v1.worker.gpu_model_runner.GPUModelRunner"
  - "vLLM-Ascend platform and fused-MoE symbols referenced by NPU patches"
verified_platform_refs:
  - "CPU/runtime compatibility tests in tests/unit/compat"
  - "Ascend patch evidence in NPU unit and E2E paths"
related_issues:
  - "#86"
  - "#129"
last_reviewed: 2026-08-08
---

# Compatibility and patches

## Purpose and boundary

This document owns supported vLLM compatibility, compatibility adapters,
monkey-patch boundaries, upstream deltas, validation, and patch removal or
upstream plans. Backend mechanisms that do not alter upstream behavior belong
to [execution platforms](execution_platforms.md).

## Ownership and dependency direction

Compatibility code consumes the common plugin boundary and adapts pinned
upstream behavior for AFD roles. Role modules may rely on those adaptations;
patch modules must not become a general home for AFD-owned functionality.

## Supported upstream boundary

The package extra pins vLLM `0.26.0`, and
[`compat/vllm.py`](../../../afd_plugin/compat/vllm.py) enforces the same target.
Direct strict calls raise for a missing or different vLLM; plugin registration
calls the check with `strict=False`, so it warns and continues. This warning
policy does not make another vLLM release supported.

The NPU v0.26 refresh is based on vLLM-Ascend commit
[`80d8c194f`](https://github.com/vllm-project/vllm-ascend/commit/80d8c194f7584b17fe08065ea99a130916f6b0e7).
The repository does not declare a vLLM-Ascend package dependency in
`pyproject.toml`, so this source commit and the recorded NPU validation are the
compatibility evidence rather than a released package or container tag.

## Implementation evidence

| Area | Source | Focused validation |
| --- | --- | --- |
| Version policy | [`compat/vllm.py`](../../../afd_plugin/compat/vllm.py) | [`test_package.py`](../../../tests/unit/package/test_package.py) |
| Core vLLM patches | [`compat/patches/`](../../../afd_plugin/compat/patches) | [`tests/unit/compat/patches/`](../../../tests/unit/compat/patches) |
| NPU adapters | [`compat/npu/`](../../../afd_plugin/compat/npu) | [`test_runtime.py`](../../../tests/unit/compat/test_runtime.py), [`test_ascend_ops.py`](../../../tests/unit/compat/test_ascend_ops.py) |
| NPU patch paths | [`compat/patches/npu/`](../../../afd_plugin/compat/patches/npu) | [`test_force_load_balance.py`](../../../tests/unit/compat/patches/test_force_load_balance.py), [`test_npu_runtime.py`](../../../tests/unit/v1/worker/test_npu_runtime.py) |
| CUDA DBO Attention metadata workaround | [`attention_model_runner.py`](../../../afd_plugin/v1/worker/attention_model_runner.py) | restoration and error-path coverage in [`test_attention_model_runner.py`](../../../tests/unit/v1/worker/test_attention_model_runner.py) |

## Patch application lifecycle

The vLLM plugin entry point applies compatibility in this order:

1. perform the non-strict vLLM version check;
2. import `async_dp_engine`, `async_dp_forward_context`, `config_validation`,
   and `engine_core` in one best-effort block;
3. register the plugin-owned DBO yield operator;
4. call the idempotent Ascend runtime facade, which installs the NPU platform
   config wrapper when vLLM-Ascend is importable;
5. import the force-load-balance patch only when vLLM-Ascend is discoverable;
6. register model mappings, which is required for registration to complete.

Python module import provides process-level one-time execution in the ordinary
path. Some patches also preserve originals or set explicit sentinels, but this
is not consistent across the inventory. There is no transaction or rollback:
because core patch imports share one `try`, a failure can leave earlier patches
installed and later ones skipped. Registration logs these failures at debug
level and continues to model registration.

## Monkey-patch inventory

Every row describes current pinned behavior. `Guard` is the patch's own guard,
not the package dependency policy.

| Patch and upstream symbols | AFD delta and non-AFD path | Application, guard, and idempotence | Validation | Removal or upstream plan |
| --- | --- | --- | --- | --- |
| [`async_dp_engine.py`](../../../afd_plugin/compat/patches/async_dp_engine.py): `EngineCoreProc.run_engine_core`, `vllm.v1.engine.utils.launch_core_engines` and its imported client alias, `DPAsyncMPClient.add_request_async` | Async-DP Attention selects regular `EngineCoreProc`, keeps coordinator stats but disables DP wave coordination, and skips `FIRST_REQ`; other configs run the copied upstream branches. | Imported by `register_afd`; accepts target-version prefix, development versions, or missing version metadata. Direct assignments; reload/idempotence behavior has a focused test. | [`test_async_dp_engine.py`](../../../tests/unit/compat/patches/test_async_dp_engine.py) covers process selection, non-AFD MoE DP, coordinator mode, wakeup, and reload. | Remove when vLLM exposes role-selectable async-DP engine scheduling, wave policy, and wakeup hooks. |
| [`async_dp_forward_context.py`](../../../afd_plugin/compat/patches/async_dp_forward_context.py): `vllm.forward_context.set_forward_context` plus already-imported worker aliases | Skips native `DPMetadata` construction/coordination only for AFD async-DP; otherwise uses the copied upstream flow. | Imported by `register_afd`; same target/dev/unknown guard. Rebinds known already-imported aliases so callers do not retain the old function. | [`test_async_dp_forward_context.py`](../../../tests/unit/compat/patches/test_async_dp_forward_context.py) covers async skip and non-async coordination. | Remove when vLLM supports a per-engine-role opt-out from native MoE DP metadata coordination. |
| [`config_validation.py`](../../../afd_plugin/compat/patches/config_validation.py): `EngineArgs.create_engine_config`, `VllmConfig.__post_init__` | For AFD-owned ubatching with a non-DeepEP backend, temporarily presents `deepep_low_latency` during upstream validation and restores the configured backend. After upstream platform normalization, maps an initial `worker_cls="auto"` to the role-specific CUDA or standard Ascend AFD worker. | Imported by `register_afd`; accepts the target version, development versions, or missing version metadata. Saves originals on upstream modules under AFD-specific attributes before installing wrappers. Explicit worker paths and non-AFD configs are not remapped. | [`test_config_validation.py`](../../../tests/unit/compat/patches/test_config_validation.py) covers backend relaxation, four role/platform mappings, explicit and non-AFD preservation, repeated validation, unsupported platforms, and dev versions. | Remove the backend branch when upstream validation distinguishes plugin-owned ubatching; remove worker mapping when vLLM offers plugin-owned role-aware worker selection. |
| [`engine_core.py`](../../../afd_plugin/compat/patches/engine_core.py): `EngineCore.__init__`, `_initialize_kv_caches`, `shutdown`; `EngineCoreProc.run_busy_loop`; `DPEngineCoreProc.run_busy_loop` | AFD FFN becomes a connector daemon: construct executor, skip scheduler/KV setup, return an empty KV-shaped result on late paths, start/monitor/stop the FFN worker loop, and use FFN-safe shutdown. Non-FFN branches copy pinned upstream behavior. | Imported by `register_afd`; **no patch-local version guard and no saved-original sentinel**. Direct class assignment means the package pin and review discipline are the compatibility guard. | [`test_engine_core.py`](../../../tests/unit/compat/patches/test_engine_core.py) covers FFN initialization, non-FFN behavior, and daemon start/stop; role runtime tests cover error propagation. | Remove when vLLM offers a headless connector-daemon engine lifecycle or an executor mode that does not require scheduler/KV ownership. |
| [`npu/ascend_platform.py`](../../../afd_plugin/compat/patches/npu/ascend_platform.py): `NPUPlatform.check_and_update_config` | Snapshots AFD DBO state, runs upstream normalization, and restores configured `enable_dbo`, `ubatch_size`, and `all2all_backend` in `finally`; non-AFD behavior is unchanged. | Called through `apply_afd_ascend_patches_if_needed`; no version guard. Saves the original on the class and uses a class sentinel. The runtime facade caches success only after the wrapper is installed, so an early missing vLLM-Ascend import remains retryable. | [`test_runtime.py`](../../../tests/unit/compat/test_runtime.py) and [`test_npu_runtime.py`](../../../tests/unit/v1/worker/test_npu_runtime.py). | Remove when vLLM-Ascend recognizes plugin-owned DBO workers or no longer clears these fields. |
| [`npu/force_load_balance.py`](../../../afd_plugin/compat/patches/npu/force_load_balance.py): `AscendW8A8DynamicFusedMoEMethod.__init__`, `AscendW8A8DynamicFusedMoEMethod.apply` | Captures AFD profiling configuration as method-owned state and replaces routed expert IDs with a deterministic balanced buffer only when the method-owned switch is enabled; normal model-selected routing remains unchanged. This switch changes outputs and is not a correctness feature. | Imported only when vLLM-Ascend is discoverable; **no patch-local version guard or explicit reload sentinel**. Functions copy the current upstream bodies with marked AFD deltas. | [`test_force_load_balance.py`](../../../tests/unit/compat/patches/test_force_load_balance.py) covers buffer bounds, determinism, growth, override, and pass-through. | Upstream a deterministic expert-routing profiling hook in vLLM-Ascend, then delete both copied functions. |

## Non-patch compatibility adapters

These modules adapt upstream behavior without replacing a global symbol:

| Adapter | Current purpose |
| --- | --- |
| [`compat/npu/runtime_config.py`](../../../afd_plugin/compat/npu/runtime_config.py) | Mirrors vLLM-Ascend's non-SP all-to-all backend rewrite for custom AFD workers and reports the active NPU ubatch count. |
| [`compat/npu/feature_validation.py`](../../../afd_plugin/compat/npu/feature_validation.py) | Parses connector-owned typed extra information through the factory and fails before execution for unsupported NPU connector, quantization, graph, DBO, gate, or async MoE combinations. |
| [`compat/npu/forward_context.py`](../../../afd_plugin/compat/npu/forward_context.py) | Enters the pinned Ascend forward context for connector-driven FFN compute and installs AFD metadata in `additional_kwargs`. |
| [`compat/npu/ops.py`](../../../afd_plugin/compat/npu/ops.py) | Discovers plugin-owned CAMP2P operators and external CAM operators with explicit missing-runtime errors; operator build/runtime ownership is documented in [execution platforms](execution_platforms.md). |
| [`v1/worker/attention_model_runner.py`](../../../afd_plugin/v1/worker/attention_model_runner.py) | Disables vLLM 0.26's block-table-only Attention metadata cache only while multiple DBO ubatches are built, then restores every affected builder in `finally`. Remove this workaround when [vLLM #48659](https://github.com/vllm-project/vllm/pull/48659), or an equivalent ubatch-aware cache key, reaches the pinned release. |

Two scoped mutations live with their semantic owners rather than this patch
directory: model dummy runs temporarily wrap `create_forward_context` as
described in [model integration](model_integration.md), and connector process
groups call private PyTorch/vLLM helpers as described in
[connector contracts](connector_contracts.md). Both must be included in an
upstream upgrade audit.

## Patch review and refresh procedure

For every new patch or upstream upgrade, reviewers must:

1. identify the exact upstream file, symbol, version/tag, and signature;
2. confirm that the patch targets the owning upstream layer rather than hiding
   AFD-owned behavior in a convenient global hook;
3. copy the pinned upstream function when feasible and mark only AFD deltas
   with `# ### PATCH START` / `# ### PATCH END`;
4. document any original-function delegation exception immediately above the
   function;
5. test both the selected AFD branch and the non-AFD/upstream branch, including
   initialization, failure, shutdown, and idempotence where relevant;
6. assess hot-path, graph/compile, memory, and distributed-lifecycle impact;
7. record an upstream contribution or concrete removal condition.

When the pinned upstream version changes, copied bodies must be refreshed from
that version before reapplying marked AFD deltas. Passing unit tests against a
different signature is not sufficient compatibility evidence.

## Failure visibility and resource ownership

- Patch modules own symbol replacement only; the patched engine or connector
  continues to own its runtime resources.
- Patch import errors are currently best effort at plugin startup and may be
  visible only in debug logs. Runtime paths must therefore fail explicitly if
  a required adaptation is absent rather than silently serving incorrect
  results.
- Copied non-AFD branches are intended to preserve pinned upstream behavior;
  they still require regression tests because global replacement affects every
  process that loads the plugin.
- A patch that saves an original callable must never overwrite that saved
  original with its own wrapper on reload.
- Patch removal requires deleting registration/application code and focused
  tests only after the replacement upstream behavior is part of the supported
  pinned stack.

## Candidate invariants

The following RFC candidates are non-normative while this document is draft:

- `PATCH-INV-001`: new or modified patches follow `AGENTS.md`, including
  upstream context, marked AFD deltas, validation, and a removal or upstream
  plan.
- `PATCH-INV-002`: a global patch preserves the pinned non-AFD path and has a
  focused regression test for it.
- `PATCH-INV-003`: unsupported upstream versions are not treated as supported
  merely because best-effort registration continued.
- `PATCH-INV-004`: patch initialization is idempotent or explicitly guarded,
  and partial application is observable before the affected runtime executes.

## Upstream relationship and validation requirements

Patch signatures and copied code must match their pinned upstream symbols.
Every modification requires focused tests for AFD and non-AFD behavior plus
the architectural review required by `AGENTS.md`. Version guard changes also
require package compatibility tests. Ascend patch refreshes require an exact
vLLM-Ascend source reference and the NPU unit/E2E evidence for the affected
path.

## Limitations and open issues

The inventory covers current production patch files, but the document remains
**draft** because:

- `engine_core` and force-load-balance lack patch-local version/idempotence
  guards;
- best-effort grouped imports can produce a partially applied patch set;
- the exact vLLM-Ascend source commit is documented but not pinned as package metadata;
- owners have not approved the candidate invariants as normative contracts.

Runtime refactor decisions in
[#86](https://github.com/JiusiServe/afd-plugin/issues/86) may remove or relocate
patches. The documentation RFC remains tracked by
[#129](https://github.com/JiusiServe/afd-plugin/issues/129).
