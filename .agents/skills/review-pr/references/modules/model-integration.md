# Model Integration

Primary design: `docs/design/module/model_integration.md` in the reviewed
head.

Use for `afd_plugin/model_executor/`: the registration maps in
`afd_plugin/__init__.py` (`_DEEPSEEK_MODEL_REGISTRATIONS`,
`_QWEN_MODEL_REGISTRATIONS`, `_QWEN3_5_MODEL_REGISTRATIONS`), role-split model
wrappers (`RemoteFFNProxy`, `AFDAttentionFusedMoE`, `GateOnlyRemoteMoE`, the
`AFDDeepseekV2/V3/V4` and `AFDQwen3Moe`/`AFDQwen3_5` families, plus
`AFDGlmMoeDsaForCausalLM` and `AFDDSparkDeepseekV4ForCausalLM`), checkpoint
weight-role filtering, the NPU-specific forward/gate modules under
`models/npu/`, and the layered executor under `model_executor/npu/`. Does not
own runner execution, connector transport, or the routing simulator's
registration mechanics.

## Contract checks

- Keep registrations explicit in the `register_afd()` maps; select NPU vs GPU
  variants (for example `DeepseekV4ForCausalLM` branching on `torch_npu`) at
  registration time, never deep inside forward paths. The maps are frozen with
  `MappingProxyType` — a change that mutates them at runtime is a finding.
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
  source pointers); `qwen3_5.py`, `deepseek_v4.py`, and
  `models/npu/deepseek_v4.py` are among the heaviest patch surfaces.
- Family-specific constraints are contracts, not suggestions: Qwen3.5/3.6 GDN
  DBO is unsupported; DeepEP M2N covers Qwen3 MoE (all-MoE layers, no shared
  expert); Async CAM covers DeepSeek-V4 W4A8/W8A8 with static EP and
  Attention-local SP, and no longer supports DeepSeek-V2/V2-Lite. A PR that
  widens a family's supported matrix without E2E and design-page evidence is a
  finding.
- Register new model families against the
  [model-addition-checklist.md](../checks/model-addition-checklist.md): both
  role wrappers, registration, recipe, E2E suite, and docs.

Test wrappers and weight filtering in `tests/unit/model_executor/`
(`models/test_*_construction.py`, `models/test_*_weight_policy.py`,
`models/test_*_proxy.py`, `models/npu/`, and `test_async_cam_*.py`); name the
E2E model suite covering the family.
