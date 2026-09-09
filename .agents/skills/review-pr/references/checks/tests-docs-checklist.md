# Tests, Docs, and PR Evidence Checklist

Use to review CI coverage, docs impact, and the PR's own evidence — after the
technical review, before delivery.

## CI coverage

- GitHub Actions runs pre-commit only (ruff, format, mypy/ty, typos,
  markdownlint, actionlint, shellcheck, SPDX headers, Buildkite YAML checks,
  DCO). Treat lint findings as gate status, not review findings.
- Buildkite: `test-ready` (L2 premerge) runs unit tests (`l4_1`) plus the
  DeepSeek-V2-Lite E2E gate (`l4_4`); `test-merge` (L3) repeats post-merge;
  `test-weekly` (main or `weekly-test` label) adds Qwen3 MoE, Qwen3.6, and
  DSV2-Lite 2A1F DBO on `h100_4`.
- A change touching an area no pipeline covers (NPU paths, non-gate model
  suites, `csrc/` kernels) needs author-provided evidence; name the gap.

## PR template and checklist

The template requires Purpose / Issue / Scope / Implementation Notes / Test
Plan / Test Result / Docs Impact, plus the Essential PR Checklist. Verify the
high-signal items rather than restating them:

- vLLM 0.26.0 compatibility considered; no vLLM source-checkout changes.
- Plugin-owned classes or dotted paths preferred over monkey patches; any
  shim isolated, idempotent, version-guarded, documented, tested.
- Imports remain CPU-safe; CUDA-heavy work delayed or GPU-gated.
- Validation evidence included, including skipped GPU tests and NPU manual
  evidence where applicable.
- DCO `Signed-off-by` present (the pre-commit hook auto-adds it; a missing
  sign-off blocks the DCO gate).

## Docs impact

- Behavior or contract changes update the matching `docs/design/module/`
  page in the same PR (see
  [design-contracts.md](../process/design-contracts.md)).
- User-facing changes (config fields, connector usage, launch flows) update
  the relevant user guide (`docs/gpu/`, `docs/npu/`) or state why not.
- Topology/launch changes update `recipe/` scripts consistently; E2E gate
  structure changes update `docs/design/module/e2e_testing.md`.
- New source files carry SPDX headers; user guides keep the documented env
  vars and defaults in sync with `AFDConfig`.

Report gaps as one consolidated P2 finding listing the missing pieces, not as
per-item nits; escalate to P1 only when the missing evidence hides a blocking
claim.
