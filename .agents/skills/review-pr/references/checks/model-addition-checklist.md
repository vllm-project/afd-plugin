# Model Addition Checklist

Use when a PR adds a model family or architecture to AFD support: new
registration entries, role-split wrappers, NPU/GPU variants, or a new
`models/npu/` forward module. The author-side workflow is the `adapt-model`
skill; this checklist is the reviewer's evidence bar.

## Required pieces

- **Both role wrappers.** Attention-side and FFN-side classes for the family
  (or a stated reason one is absent, e.g. dense FFN handled by remote proxy
  only). Each role constructs only its role-required components.
- **Registration.** Entries in the `register_afd()` registration maps with the
  correct upstream architecture names; NPU/GPU variant selection
  (`torch_npu` branch) resolved at registration, not in forward code. The maps
  are frozen (`MappingProxyType`) and the AFD alias is only installed when
  `is_afd_active` is true — non-AFD workers must keep the checkpoint
  architecture and resolve to the native vLLM class.
- **Weight-role filtering.** `_checkpoint_weight_roles` coverage so no role
  loads weights it never reads and none drops weights it later needs; verify
  against the checkpoint's full weight list, including gate/expert and
  embedding/LM-head placement.
- **Upstream fidelity.** Copied model code follows the patch contract: marked
  AFD differences, `# Upstream source:` pointers, identical signatures.
- **Config compatibility.** Direct attribute access to upstream config;
  documented assumptions about quantization, MLA/attention backend, and MoE
  settings the family requires.
- **Connector/execution fit.** State which connectors and topologies the family
  actually supports, and make the unsupported combinations fail at startup
  rather than silently degrade. Family restrictions are real: Qwen3.5/3.6 GDN
  DBO is unsupported; DeepEP M2N requires an all-MoE family with no shared
  expert; Async CAM is DeepSeek-V4 W4A8/W8A8 only.

## Evidence required in the PR

- A unit suite under `tests/unit/model_executor/` (CPU-safe, checkpoint- or
  fixture-driven) covering role construction, proxy fidelity, and weight
  filtering — the existing naming is `test_<family>_construction.py`,
  `test_<family>_proxy.py`, `test_<family>_weight_policy.py`.
- An E2E suite or scenario under `tests/e2e/models/<family>/` with the suite's
  gate scenarios and GSM8K accuracy evidence, respecting the four-device bound
  (`E2E-INV-002`) and the no-skip rule (`E2E-INV-004`).
- A recipe under `recipe/` for at least one working topology, with launch
  scripts consistent with the user guides.
- Docs: the relevant `docs/design/module/` page (model integration) and, for
  user-facing support, a user guide or guide section; the design-page
  registration table must gain the new row.
- For NPU support: manual validation evidence per `tests/npu/README.md`;
  there is no NPU CI.

A family missing any required piece is a P1 finding; state which piece and
the smallest way to complete it.
