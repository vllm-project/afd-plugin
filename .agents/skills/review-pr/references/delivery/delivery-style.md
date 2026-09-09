# Maintainer Delivery Style

Use when findings are ready for delivery. These rules adapt the repo's own
review culture — AGENTS.md's "strict review required" patching bar and the
direct engineering tone of its merged history — into comment craft.

## Recurring review behavior

- Start from the changed contract, not style: pre-commit already owns ruff,
  format, SPDX, markdownlint, and DCO; never restate them.
- Protect the boundaries that make the plugin maintainable: patch fidelity
  against pinned refs, CPU-safe imports, role split, platform twins, AFD-off
  transparency. These are the findings this repo's maintainers need most.
- Challenge scope when a PR smuggles in an unexplained abstraction, a new
  global, or a second mechanism for an existing one; ask for the smallest
  version.
- Demand evidence exactly where it is load-bearing: perf claims, NPU
  behavior, E2E gate impact — and nowhere else.
- Address named owners (`@hsliuustc0106`, `@jiangkuaixue123`) for
  architectural questions, not a generic "PTAL".
- Defer cleanup to follow-ups; do not block on debt the PR did not create.

## Write the finding

Lead with the issue; no preamble, no praise sandwich. One sentence when the
fix is obvious. Prefer root-cause comments ("this breaks patch comparability
at the next upgrade") over restating the diff. Hedge only when evidence is
genuinely incomplete, and say what would complete it.

Format (severity tiers from
[review-routing.md](../process/review-routing.md#calibrate-findings)):

```text
[P1] Short imperative title — path/to/file.py:<line>
<Trigger or call path>. <Current behavior and impact>. <Smallest fix direction>.
```

Patch-contract findings name the exact missing element (marker, signature
parity, upstream source pointer, pinned ref) because that is what the next
upgrade will trip over.

Calibrate to roughly 1–5 short comments; treat that as calibration, not a
quota. Zero comments is a valid outcome — say so and name material
validation gaps.

## Anti-patterns

- Generic praise, "Nit:" prefixes, decorative severity labels.
- Audit-template artifacts: rule IDs, checklist restatements, comment-count
  narration.
- Style findings pre-commit owns, or AGENTS.md wording quotes with no
  concrete failure.
- Batching many small findings that share one root cause; merge them.
