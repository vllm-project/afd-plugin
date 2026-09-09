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
  (`torch_npu` branch) resolved at registration, not in forward code.
- **Weight-role filtering.** `_checkpoint_weight_roles` coverage so no role
  loads weights it never reads and none drops weights it later needs; verify
  against the checkpoint's full weight list, including gate/expert and
  embedding/LM-head placement.
- **Upstream fidelity.** Copied model code follows the patch contract: marked
  AFD differences, `# Upstream source:` pointers, identical signatures.
- **Config compatibility.** Direct attribute access to upstream config;
  documented assumptions about quantization, MLA/attention backend, and
  MoE settings the family requires.

## Evidence required in the PR

- A unit suite under `tests/unit/model_executor/` (CPU-safe, checkpoint- or
  fixture-driven) covering role construction and weight filtering.
- An E2E suite or scenario under `tests/e2e/models/` with the four gate
  scenarios (or the suite's stated gate set) and GSM8K-7 accuracy evidence.
- A recipe under `recipe/` for at least one working topology, with launch
  scripts consistent with the user guides.
- Docs: the relevant `docs/design/module/` page mention (model integration)
  and, for user-facing support, a user guide or guide section.
- For NPU support: manual validation evidence per `docs/npu/TESTING.md`;
  there is no NPU CI.

A family missing any required piece is a P1 finding; state which piece and
the smallest way to complete it.
