# Resolve Design Contracts

Use this reference for every production review, after the diff census and
before routing. The branch-local design pages under `docs/design/module/` are
the ownership and status source of truth for the frozen head.

## Resolution order

1. Open `docs/design/module/index.md` in the reviewed checkout. It declares
   the reading order and the dependency direction between module designs.
2. Match the changed production paths against each design page's
   `primary_code_paths` (and `related_code_paths`) frontmatter to identify the
   owning design document; a path may appear in `related_code_paths` of several
   pages while having one primary owner.
3. Read the matched page's `status`, `depends_on`, `validation_paths`, and
   `last_reviewed` fields. `status: draft` means its boundaries and invariants
   are candidates, not enforced contracts — treat them as questions, and raise
   a blocking finding only when current code, tests, or AGENTS.md enforce them.
4. Prefer live code and tests over stale prose when they disagree; report the
   drift against the design page as a finding when the PR touches that area.
5. Honor the declared dependency direction: role modules (attention/FFN/model
   integration) depend on shared boundaries (connector contracts, plugin
   boundary, execution platforms, compatibility). A PR that adds a dependency
   from a shared boundary back into a role implementation contradicts the
   documented architecture; question it.
6. Never import a newer design page into an older branch's review; release
   branches (for example `release/v0.19.1rc1`, `update/v0.28.0`) are judged
   against their own head's pages.

## When a PR must update the design pages

A PR changes a documented contract, and must update the matching
`docs/design/module/` page in the same PR, when it:

- adds, moves, or repurposes a path listed in `primary_code_paths`;
- changes connector, worker, platform, or compatibility behavior that a design
  page describes;
- changes the E2E gate structure or pass criteria (`e2e_testing.md`);
- moves the pinned vLLM or vLLM-Ascend refs recorded in `upstream_refs`.

Doc-only updates to a design page route back to the production contract the
page claims; verify the claim against code before accepting the page edit as
the review's whole surface.
