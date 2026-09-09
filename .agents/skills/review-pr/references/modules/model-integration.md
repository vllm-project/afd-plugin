# Model Integration

Primary design: `docs/design/module/model_integration.md` in the reviewed
head.

Use for `afd_plugin/model_executor/`: the registration maps in
`afd_plugin/__init__.py` (`_DEEPSEEK_MODEL_REGISTRATIONS`,
`_QWEN_MODEL_REGISTRATIONS`, `_QWEN3_5_MODEL_REGISTRATIONS`), role-split model
wrappers (`RemoteFFNProxy`, `AFDAttentionFusedMoE`, `GateOnlyRemoteMoE`, the
`AFDDeepseekV2/V3/V4` and `AFDQwen3Moe`/`AFDQwen3_5` families), checkpoint
weight-role filtering, and the NPU-specific forward/gate modules under
`models/npu/`. Does not own runner execution, connector transport, or the
routing simulator's registration mechanics.

## Contract checks

- Keep registrations explicit in the `register_afd()` maps; select NPU vs GPU
  variants (for example `DeepseekV4ForCausalLM` branching on `torch_npu`) at
  registration time, never deep inside forward paths.
- Preserve the role construction contract: each role builds only
  role-required components, and weight filtering via
  `_checkpoint_weight_roles` must not silently drop weights that role later
  reads (a missing gate/expert weight is a runtime failure on device).
- Keep remote proxies faithful: `RemoteFFNProxy` and friends preserve upstream
  signatures, hidden-state layouts, and auxiliary outputs (residuals, gate
  values) across the role boundary.
- Access upstream model config attributes directly per AGENTS.md; a
  `getattr`/`hasattr` guard around vLLM model fields hides the exact
  incompatibility the pinned-version contract exists to surface.
- Mark copied upstream model code with the patch contract (markers, upstream
  source pointers); the qwen3_5 and deepseek_v4 files are among the heaviest
  patch surfaces.
- Register new model families against the
  [model-addition-checklist.md](../checks/model-addition-checklist.md): both
  role wrappers, registration, recipe, E2E suite, and docs.

Test wrappers and weight filtering in `tests/unit/model_executor/`
(CPU-safe, checkpoint-driven); name the E2E model suite covering the family.
