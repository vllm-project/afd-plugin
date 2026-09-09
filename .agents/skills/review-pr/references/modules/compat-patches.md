# Compatibility Patches

Primary design: `docs/design/module/compatibility_and_patches.md` in the
reviewed head. Authoring rules: `AGENTS.md` ("Patching Requirement").

Use for monkey patches under `afd_plugin/compat/patches/` (generic
engine/GPU: `async_dp_engine.py`, `async_dp_forward_context.py`,
`config_validation.py`, `engine_core.py`) and the version gate in
`compat/vllm.py`. Does not own the deferred NPU runtime layer
([npu-compat.md](npu-compat.md)) or plugin-owned classes.

## Contract checks

- Prefer plugin-owned classes, inheritance, or upstream contribution over a
  new monkey patch; a new patch needs an architectural-review rationale:
  correct upstream target, minimal scope, understood performance impact, and a
  long-term upstream-or-removal plan.
- Copy the upstream function wholesale from the pinned ref (vLLM 0.26.0;
  vLLM-Ascend commit `80d8c194f`) and mark only AFD-specific differences with
  `# ### PATCH START:` / `# ### PATCH END:`; keep the marker text short and
  specific.
- Keep a patched function's signature and return type identical to upstream;
  any added parameter must be documented in the comments immediately above the
  patch function.
- Require `Patch reason:` / `Patch functionality:` comments immediately above
  every patch function, and keep `# Upstream source:` pointers accurate.
- Treat `_original_*` delegation as the exception, not the default non-AFD
  path; require the exception to be called out in the patch comments.
- Keep patches idempotent and version-aware (guarded by
  `TARGET_VLLM_VERSION`); re-applying or importing twice must not corrupt
  state.
- Do not flag upstream style inside patch copies — ruff ignores E501/N806/N807
  there so diffs stay comparable to upstream; do flag upstream drift (the copy
  no longer matches the pinned ref).
- Patched behavior must be transparent when AFD is inactive: non-AFD requests
  execute the copied upstream logic unchanged.

Test every patch in `tests/unit/compat/patches/` with CPU-safe tests covering
idempotence, version guarding, and both AFD-on and AFD-off behavior.
