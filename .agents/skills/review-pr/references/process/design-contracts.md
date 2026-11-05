# Resolve Design Contracts

Use this reference for every production review, after the diff census and
before routing. The branch-local design pages under `docs/design/module/` are
the ownership and status source of truth for the frozen head.

## Resolution order

1. Open `docs/design/module/index.md` in the reviewed checkout. It declares
   the reading order, the dependency direction, and a "Production path routing"
   table that assigns every shipped path, native build path, and packaging
   file to exactly one primary document. Tests, tools, recipes, generated
   files, user guides, and design docs are outside that inventory; the E2E
   harness is owned separately by `e2e_testing.md`.
2. Match the changed production paths against each design page's
   `primary_code_paths` (and `related_code_paths`) frontmatter to identify the
   owning design document; a path may appear in `related_code_paths` of several
   pages while having one primary owner. A new shipped file that appears in no
   page is an ownership gap — that is a finding, not a reason to route by
   guesswork.
3. Read the matched page's `status`, `owners`, `depends_on`,
   `validation_paths`, `upstream_refs`, and `last_reviewed`. Today only
   `plugin_boundary.md` and `e2e_testing.md` are `normative`; every other page
   is `draft`. `status: draft` means its boundaries and invariants are
   candidates, not enforced contracts — treat them as questions, and raise a
   blocking finding only when current code, tests, or AGENTS.md enforce them.
   A `normative` page's `MUST`/`MUST NOT` invariants (`ENTRY-INV-*`,
   `CFG-INV-*`, `E2E-INV-*`) are blockers.
4. Prefer live code and tests over stale prose when they disagree; report the
   drift against the design page as a finding when the PR touches that area.
5. Honor the declared dependency direction: role modules (attention/FFN/model
   integration) depend on shared boundaries (connector contracts, plugin
   boundary, execution platforms, compatibility). A PR that adds a dependency
   from a shared boundary back into a role implementation contradicts the
   documented architecture; question it.
6. Compare `upstream_refs` and `last_reviewed` against the pin in
   `afd_plugin/compat/vllm.py` and `docs/design/module/index.md`. A page whose
   `upstream_refs` still name the previous pair while the code moved is drift
   worth reporting when the PR touches that area — but do not report a page as
   stale merely because it records historical evidence; those rows are
   explicitly historical.
7. Never import a newer design page into an older branch's review; release
   branches (for example `release/v0.19.1rc1`, `update/v0.28.0`) are judged
   against their own head's pages.

## When a PR must update the design pages

A PR changes a documented contract, and must update the matching
`docs/design/module/` page in the same PR, when it:

- adds, moves, or repurposes a path listed in `primary_code_paths`;
- changes connector, worker, platform, or compatibility behavior that a design
  page describes;
- changes the E2E gate structure or pass criteria (`e2e_testing.md`);
- moves the pinned vLLM or vLLM-Ascend refs recorded in `upstream_refs`, or
  changes the `TARGET_VLLM_VERSION` gate in `afd_plugin/compat/vllm.py`;
- adds, retires, or renames an E2E case, or changes a gate's sample count,
  device count, or required set (`e2e_testing.md`);
- changes behavior that an `.agents/skills/**` SKILL.md describes as a command,
  case list, or pin.

Doc-only updates to a design page route back to the production contract the
page claims; verify the claim against code before accepting the page edit as
the review's whole surface.

## Ownership

CODEOWNERS is no longer a single global rule. The global catch-all maps to
`@hsliuustc0106 @jiangkuaixue123`, with path-specific owners on top:
`/afd_plugin/connectors/gpu/`, `/csrc/gpu/`, `/recipe/gpu/`, `/tests/gpu/`,
`/docs/gpu/`, and `/.buildkite/cuda/` add `@ronenkat`; `/csrc/npu/` adds
`@francisleely`; `/tests/e2e/`, `/.agents/skills/run-e2e/`, and
`/.agents/skills/deploy-afd-k8s/` add `@ronenkat`. Design pages name their own
`owners`; `e2e_testing.md` is owned by `@yujuancao07` alone. When a design page
and CODEOWNERS disagree about who owns a contract, report the disagreement
rather than choosing one silently.
