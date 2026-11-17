# Compatibility Patches

Primary design: `docs/design/module/compatibility_and_patches.md` in the
reviewed head. Authoring rules: `AGENTS.md` ("Patching Requirement").

Use for monkey patches under `afd_plugin/compat/patches/` (generic
engine/GPU: `async_dp_engine.py`, `async_dp_forward_context.py`,
`config_validation.py`, `engine_core.py`, `gpu_mrv2_ubatching.py`,
`gpu_deepseek_v4_dbo.py`), the vLLM-Ascend patches under
`compat/patches/npu/` (`ascend_config.py`, `ascend_platform.py`, `mla_graph.py`,
`mrv2_ubatching.py`), the version gate in `compat/vllm.py`, and the patched
methods in `afd_plugin/v1/worker/**` (`attention_worker.py`, `ffn_worker.py`,
`ffn_model_runner.py`, `attention_model_runner_v2.py`, and the NPU twins).
Does not own the deferred NPU runtime layer ([npu-compat.md](npu-compat.md)) or
plugin-owned classes.

## Contract checks

- Prefer plugin-owned classes, inheritance, or upstream contribution over a
  new monkey patch; a new patch needs an architectural-review rationale:
  correct upstream target, minimal scope, understood performance impact, and a
  long-term upstream-or-removal plan.
- Copy the upstream function wholesale from the pinned ref and mark only
  AFD-specific differences with `# ### PATCH START:` / `# ### PATCH END:`;
  keep the marker text short and specific.
- Keep a patched function's signature and return type identical to upstream;
  any added parameter must be documented in the comments immediately above the
  patch function.
- Require, immediately above every patch function: `# Upstream source:`
  (module, symbol, and the pinned ref — a bare `vLLM 0.32.0` label is weaker
  than the commit but acceptable when it names the file and symbol),
  `Patch reason:`, and `Patch functionality:`. `# Removal plan:` and
  `# Expansion exception:` are expected where the patch delegates to
  `_original_*` or expands a large upstream function; their absence there is a
  finding.
- Treat `_original_*` delegation as the exception, not the default non-AFD
  path; require the exception to be called out in the patch comments.
- Keep patches idempotent and version-aware (guarded by
  `TARGET_VLLM_VERSION` / `is_target_vllm_compatible`); re-applying or
  importing twice must not corrupt state.
- Do not flag upstream style inside patch copies — ruff ignores E501/N806/N807
  under `afd_plugin/compat/patches/**` so diffs stay comparable to upstream; do
  flag upstream drift (the copy no longer matches the pinned ref).
- Patched behavior must be transparent when AFD is inactive: non-AFD requests
  execute the copied upstream logic unchanged.

## Signature alignment

`tests/unit/compat/patches/test_signatures.py` is the single parametrized table
that owns upstream signature checks for every scoped runtime patch, including
the ones that live in `afd_plugin/v1/worker/**` (capture inputs, fullgraph
replay, workspace slot, ubatch prepare, CAMP2P dispatch) and the NPU
`vllm_ascend` installers. It reads the installed upstream definition from
source rather than comparing a patch to itself. A newly patched function with
no table row, or a row whose target symbol no longer exists upstream, is a P1
finding: the next pin move will fail or, worse, silently stop being checked.

Test every patch in `tests/unit/compat/patches/` (and
`tests/unit/compat/patches/npu/`) with CPU-safe tests covering idempotence,
version guarding, and both AFD-on and AFD-off behavior.
