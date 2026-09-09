# Review Execution and Delivery

Use this reference for every review. It defines snapshot collection, gates,
validation records, delivery permissions, tone, and re-review.

## Contents

- [Freeze the review surface](#freeze-the-review-surface)
- [Report status before analysis](#report-status-before-analysis)
- [Apply review gates](#apply-review-gates)
- [Run bounded validation](#run-bounded-validation)
- [Deliver maintainer-style findings](#deliver-maintainer-style-findings)
- [Re-review safely](#re-review-safely)

## Freeze the review surface

For a GitHub PR, read the base/head SHA before and after fetching metadata and
the diff:

```bash
gh api "repos/vllm-project/afd-plugin/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
REVIEW_FIELDS="number,url,title,body,isDraft,baseRefName,headRefName,mergeable,mergeStateStatus,statusCheckRollup,files"
gh pr view <PR> --repo vllm-project/afd-plugin --json "${REVIEW_FIELDS}"
gh pr diff <PR> --repo vllm-project/afd-plugin
gh api "repos/vllm-project/afd-plugin/pulls/<PR>" \
  --jq '{base_sha: .base.sha, head_sha: .head.sha}'
```

Discard the snapshot if either SHA changed. Do not mix comments or validation
from different heads.

After the SHAs stabilize, fetch and materialize a trusted `head_sha` in an
isolated detached worktree. Run every source read, `rg` search, import, and test
from that snapshot, not the caller's checkout. A detached worktree provides
snapshot isolation, not a security boundary.

Treat a fork head as untrusted unless the user and environment policy explicitly
establish trust. Never run its imports, tests, builds, hooks, package setup, or
repo-configurable linters/pre-commit plugins on the reviewer host. Execute them
only in a disposable sandbox with no credentials or inherited secrets, minimal
read-only host mounts, disabled or explicitly allowlisted network, resource/time
limits, and destruction after the run. If that boundary is unavailable, limit
the review to the remote diff, SHA-addressed reads such as
`git show <head_sha>:<path>`, and existing CI evidence; report all executable
validation as a gap.

Before the first source read, record a pristine snapshot fingerprint outside
the reviewed worktree. Include `HEAD`, the NUL-delimited index entries and blob
IDs, and a NUL-safe manifest of every worktree entry except Git metadata,
including tracked, untracked, and ignored paths with file type, mode, symlink
target, and content hash. Before each validation group and delivery, recompute
and byte-compare the fingerprint as well as asserting `HEAD == head_sha`. If a
tool changes or creates any entry, discard affected evidence and recreate the
snapshot before another group; do not rely on `HEAD` alone or clean an unknown
worktree in place. If an exact snapshot cannot be created, use SHA-addressed
reads and report filesystem-dependent validation as a gap.

For a local branch/worktree, determine the target ref from the user, current PR,
or configured upstream; never infer it from the branch name. Resolve the target
once and include all task-owned worktree state:

```bash
git status --porcelain=v2 -z
git rev-parse HEAD
git rev-parse <target-ref>
git merge-base <target-base-sha> HEAD
git diff --stat <comparison-commit>
git diff --name-status <comparison-commit>
git diff --binary <comparison-commit>
git diff --cached --binary <comparison-commit>
git diff --binary
git ls-files --others --exclude-standard -z
```

Freeze the exact bytes of every in-scope untracked file from the NUL-delimited
list, plus a NUL-safe path, file-type, mode, and content-hash manifest. Also
fingerprint `HEAD`, the target and merge base, porcelain status, and both binary
index and worktree patches. Recording only untracked names is insufficient.
Keep these snapshots in the evidence packet and do not mutate the reviewed
checkout during a read-only review.

For local changes without PR metadata, mark the title/body as synthetic. Base
them on the user request and commit messages without inventing validation claims.

If no PR, branch, or usable worktree is identifiable, ask for a PR URL/number or
explicit base and head rather than guessing.

## Report status before analysis

Within 60 seconds of starting, and before source searches or tests, send this
host update:

```text
Pinned head: <SHA>
Base/comparison: <ref and SHA>
CI: <pre-commit / Buildkite test-ready / test-merge / weekly: pass/fail/pending/not applicable>
Mergeability: <state/not applicable>
Preliminary findings: <brief finding or none yet>
```

Mark early findings as preliminary and continue. This is not a GitHub comment.

## Apply review gates

- Report draft/WIP state. Continue a local review when explicitly requested,
  but do not publish review events without separate authorization.
- Record DCO (`Signed-off-by`), pre-commit, required CI, and mergeability.
  Pending or unknown gates do not block source review.
- Treat a failed gate as its own evidence. Do not restate pre-commit's
  formatting/lint output as a new code-review finding.
- Open CI logs only when the first failing step overlaps the frozen diff or
  blocks the verdict. Start with the first error, not the last cascade.
- Treat the PR body as navigation. Commands, benchmark tables, and claimed tests
  become evidence only after their provenance and relevance are checked.

## Run bounded validation

Every command runs against the frozen SHA plus the recorded fingerprint, or a
recreated pristine snapshot; otherwise its output is not evidence. Record:

```text
repo, head SHA, snapshot fingerprint, command, result, Python/platform, dependency fingerprint
```

Order validation from cheapest to most expensive and stop when the changed
paths have supported findings or explicit no-issue conclusions:

1. Import preflight on a trusted head: `python -c "import afd_plugin"` and the
   changed modules import without torch/torch_npu present.
2. Narrowest unit tests for the changed area, CPU-safe by default:
   `pytest tests/unit/<area> -m "not gpu and not vllm_runtime"`.
3. Targeted GPU-gated tests only when a GPU and a trusted head exist.
4. E2E gate scenarios only through the `run-e2e` skill, only when the review
   genuinely needs active evidence and devices are free; otherwise request
   author evidence.
5. NPU paths have no in-repo CI. Require author-provided evidence per
   `docs/npu/TESTING.md`, or a manually executed run the user explicitly asks
   for. Never simulate device evidence.

## Deliver maintainer-style findings

Findings only, ordered by severity, each in this format:

```text
[P1] Short imperative title — path/to/file.py:<line>
<Trigger or call path>. <Current behavior and impact>. <Smallest fix direction>.
```

Severity tiers are defined in
[review-routing.md](review-routing.md#calibrate-findings). Calibrate to roughly
1–5 short comments; treat that as calibration, not a quota. Zero findings is a
valid result — say so and name material validation gaps instead.

Posting to GitHub is read-only by default. Do not submit review events, add
labels, or comment unless the user explicitly authorizes it for this review or
standing session.

## Re-review safely

1. Re-freeze the new head and diff the old head against it.
2. Re-verify only findings whose lines changed; carry the rest forward.
3. Re-run validation groups whose inputs changed.
4. Re-check gate status and mergeability.
5. Reset the comment budget: prior accepted findings do not license new volume.
6. Deliver as a delta; do not repaste the earlier review.
