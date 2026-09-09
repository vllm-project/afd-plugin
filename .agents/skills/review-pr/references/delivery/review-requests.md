# Reviewer Requests

Use only when the user asks to identify, suggest, request, or ping reviewers.
Identifying or suggesting reviewers is read-only; requesting reviewers or
posting `@mention` comments changes external state and requires explicit user
authorization for this PR or session.

## Owners

`CODEOWNERS` is a single global rule: `* @hsliuustc0106 @jiangkuaixue123`.
The design pages under `docs/design/module/` declare the same owners. There
are no path-specific owners, so route by contract expertise instead:

| Changed contract | Ask for |
| --- | --- |
| Patches, version pins, platform selection | Owner with upstream vLLM/vLLM-Ascend upgrade context |
| Connectors, topology, async CAM | Owner with the connector/NPU runtime context |
| Model families, E2E gates | Owner with the model-suite context |
| CI / Buildkite pipelines | Owner with CI pool context |

## Procedure

1. From the frozen head, map the diff census to the primary module contract
   and feature overlays.
2. Propose one to three focused reviewers with an explicit rationale naming
   the contract ("patch contract + async CAM ubatching"), never a bare
   "PTAL".
3. Check existing reviews and requests first; do not duplicate an owner
   already engaged on the PR threads.
4. When authorized to post, recheck the head SHA, then post at most one
   consolidated comment with the request; never a per-file ping.
5. Prefer the GitHub reviewer-request UI (`gh pr edit --add-reviewer`) over a
   comment when the goal is only to add reviewers.
