# Tests, Docs, and PR Evidence Checklist

Use to review CI coverage, docs impact, and the PR's own evidence — after the
technical review, before delivery.

## CI coverage

- GitHub Actions runs pre-commit only (check-yaml, merge-conflict, end-of-file,
  mixed-line-ending, trailing-whitespace, ruff check/format, typos,
  actionlint, markdownlint, mypy on changed files for 3.11 with 3.12/3.13
  manual-only, shellcheck, SPDX headers, Buildkite YAML validation, DCO). Treat
  lint findings as gate status, not review findings.
- Pre-commit only inspects files changed in the push/PR (`--from-ref/--to-ref`);
  a violation in an untouched file is not this PR's finding.
- Buildkite image: `docker/Dockerfile.ci` builds on
  `vllm/vllm-openai:${VLLM_BASE_TAG}` (currently `v0.32.0`) and asserts the
  installed vLLM version equals the pin. A PR that changes the pin without
  changing that assertion leaves CI unable to catch drift.
- `test-ready` (L2) and `test-merge` (L3) run on `l4_1`: the CPU unit
  selection (`pytest -m "not gpu and not vllm_runtime" tests/unit` plus
  `tests/npu/test_async_cam_harness.py`, `test_async_cam_performance.py`,
  `test_async_cam_golden.py`, `test_gmm_layered_performance.py`) and
  `pytest tests/gpu/test_attention_routing.py`. The DeepSeek-V2-Lite
  seven-case GPU E2E gate runs on `l4_4`.
- `test-weekly` (main + `WEEKLY=1`, or PR label `weekly-test`) adds Qwen3 MoE
  (four cases), Qwen3.6 MoE (three cases), and DeepSeek-V2-Lite
  `afd-graph-dbo-2a1f` on `h100_4`.
- `skip_ci.py` suppresses the whole default pipeline for docs-only
  (`docs/**`, any `*.md`, `mkdocs.yml`) or skip-mark-only diffs; label-only CI
  YAML edits gate per platform and level. Say what a green run does and does
  not cover.
- A change touching an area no pipeline covers (NPU paths, `csrc/` kernels,
  non-gate model suites, `tests/gpu/` device tests) needs author-provided
  evidence; name the gap.

## PR template and checklist

The template requires Purpose / Issue / Scope / Implementation Notes / Test
Plan / Test Result / Docs Impact, plus the Essential PR Checklist. Verify the
high-signal items rather than restating them:

- vLLM and, for NPU changes, the pinned vLLM-Ascend revision compatibility
  considered; no vLLM source-checkout changes.
- Plugin-owned classes or dotted paths preferred over monkey patches; any shim
  isolated, idempotent, version-guarded, documented, and tested.
- Imports remain CPU-safe; CUDA/NPU-heavy work delayed or device-gated.
- Validation evidence included, including skipped GPU tests and NPU manual
  evidence, with exact upstream revisions and backend-scoped results where
  native builds or ABIs matter.
- DCO `Signed-off-by` present (the pre-commit hook auto-adds it; a missing
  sign-off blocks the DCO gate).

## Docs impact

- Behavior or contract changes update the matching `docs/design/module/`
  page in the same PR (see
  [design-contracts.md](../process/design-contracts.md)).
- User-facing changes (config fields, connector usage, launch flows) update
  the relevant user guide (`docs/gpu/`, `docs/npu/`) or state why not.
- Topology/launch changes update `recipe/` scripts and their READMEs
  consistently; E2E gate structure changes update
  `docs/design/module/e2e_testing.md`.
- `.agents/skills/**` are executable instructions, not prose: a change to
  commands, case lists, pins, or workflows that leaves a SKILL.md stating the
  old behavior is a P2 finding. Markdownlint excludes `.agents/`, so nothing
  else catches it.
- New source files carry SPDX headers; user guides keep the documented env
  vars and defaults in sync with `AFDConfig`.

Report gaps as one consolidated P2 finding listing the missing pieces, not as
per-item nits; escalate to P1 only when the missing evidence hides a blocking
claim.
