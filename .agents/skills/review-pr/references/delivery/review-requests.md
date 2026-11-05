# Reviewer Requests

Use only when the user asks to identify, suggest, request, or ping reviewers.
Identifying or suggesting reviewers is read-only; requesting reviewers or
posting `@mention` comments changes external state and requires explicit user
authorization for this PR or session.

## Owners

`.github/CODEOWNERS` has a global catch-all
(`* @hsliuustc0106 @jiangkuaixue123`) plus path-specific entries:

| Path | Adds |
| --- | --- |
| `/afd_plugin/connectors/gpu/`, `/csrc/gpu/`, `/recipe/gpu/`, `/tests/gpu/`, `/docs/gpu/`, `/.buildkite/cuda/` | `@ronenkat` |
| `/csrc/npu/` | `@francisleely` |
| `/tests/e2e/`, `/.agents/skills/run-e2e/`, `/.agents/skills/deploy-afd-k8s/` | `@ronenkat` |

Design pages declare their own `owners` under `docs/design/module/`; most list
`@hsliuustc0106` and `@jiangkuaixue123`, while `e2e_testing.md` is owned by
`@yujuancao07` alone. Prefer the design page's owners for the *contract*, and
CODEOWNERS for the *paths*; when they disagree, say which one you are
following.

## Procedure

1. From the frozen head, map the diff census to the primary module contract
   and feature overlays.
2. Propose one to three focused reviewers with an explicit rationale naming
   the contract ("patch contract + async CAM layered FFN", "E2E gate
   selection"), never a bare "PTAL".
3. Check existing reviews and requests first; do not duplicate an owner
   already engaged on the PR threads.
4. When authorized to post, recheck the head SHA, then post at most one
   consolidated comment with the request; never a per-file ping.
5. Prefer the GitHub reviewer-request UI (`gh pr edit --add-reviewer`) over a
   comment when the goal is only to add reviewers.
