# Maintainer Delivery Style

Use when findings are ready for delivery. These rules adapt the repo's own
review culture — AGENTS.md's "strict review required" patching bar and the
direct engineering tone of its merged history — into comment craft.

## Recurring review behavior

- Start from the changed contract, not style: pre-commit already owns ruff,
  format, typos, SPDX, markdownlint, shellcheck, and DCO; never restate them.
- Protect the boundaries that make the plugin maintainable: patch fidelity
  against pinned refs, CPU-safe imports, role split, platform twins, the
  no-V1/ModelRunnerV2-only contract, AFD-off transparency, and E2E gate
  integrity. These are the findings this repo's maintainers need most.
- Challenge scope when a PR smuggles in an unexplained abstraction, a new
  global, a second mechanism for an existing one, or a permissive fallback
  where the design says "fail at startup"; ask for the smallest version.
- Demand evidence exactly where it is load-bearing: perf claims, NPU behavior,
  E2E gate impact — and nowhere else.
- Address the owner who actually holds the contract. CODEOWNERS is no longer a
  single global rule: GPU paths add `@ronenkat`, `/csrc/npu/` adds
  `@francisleely`, E2E paths add `@ronenkat`, and `e2e_testing.md` is owned by
  `@yujuancao07`. Use the design page's `owners` plus the path-matched
  CODEOWNERS entry, not a generic "PTAL".
- Defer cleanup to follow-ups; do not block on debt the PR did not create.
- Known open failures are context, not ammunition: cite DeepSeek-V2-Lite DBO
  shutdown or NPU graceful-exit limitations only to explain why the PR's
  evidence differs from a clean pass.

## Write the finding

Lead with the issue; no preamble, no praise sandwich. One sentence when the
fix is obvious. Prefer root-cause comments ("this breaks patch comparability
at the next pin move") over restating the diff. Hedge only when evidence is
genuinely incomplete, and say what would complete it.

Format (severity tiers from
[review-routing.md](../process/review-routing.md#calibrate-findings)):

```text
[P1] Short imperative title — path/to/file.py:<line>
<Trigger or call path>. <Current behavior and impact>. <Smallest fix direction>.
```

Patch-contract findings name the exact missing element (marker, `Patch reason:`
/ `Patch functionality:` comment, signature parity, upstream source pointer,
pinned ref) because that is what the next upgrade will trip over. Gate findings
name the invariant (`E2E-INV-004`) and the case ID.

Calibrate to roughly 1–5 short comments; treat that as calibration, not a
quota. Zero comments is a valid outcome — say so and name material validation
gaps.

## Anti-patterns

- Generic praise, "Nit:" prefixes, decorative severity labels.
- Audit-template artifacts: rule IDs as decoration, checklist restatements,
  comment-count narration.
- Style findings pre-commit owns, or AGENTS.md wording quotes with no concrete
  failure.
- Batching many small findings that share one root cause; merge them.
- Reporting a stale pin as a finding when the head deliberately moved it in an
  upgrade PR that also moved the gate, image, docs, and design pages.
