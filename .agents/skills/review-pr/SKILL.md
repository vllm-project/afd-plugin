---
name: review-pr
description: Review pull requests and local branches for vllm-project/afd-plugin with a frozen snapshot, module-design ownership, feature overlays, patch-contract enforcement, targeted validation, and concise evidence-backed findings. Use for default, detailed, or repeat reviews; checking correctness, NPU/GPU compatibility, vLLM 0.26.0 patch discipline, connectors, distributed topology, workers and model runners, tests, benchmarks, or model-family additions; and identifying or explicitly requesting code-owner reviewers. Use run-e2e instead to execute E2E suites, and AGENTS.md for authoring rules.
---

# Review AFD Plugin Pull Requests

Review like a maintainer: direct, selective, and focused on issues that CI does
not prove. Prefer a few high-confidence findings over exhaustive commentary.
Zero findings is a valid result.

## Quality contract

Make every finding:

- **Correct:** prove a reachable failure, not a suspicion.
- **Prioritized:** lead with merge blockers and high-impact defects.
- **Actionable:** identify the smallest safe fix direction.
- **Evidence-based:** cite code, tests, docs, CI, or measurements.
- **Concise:** avoid review templates and repeated summaries.
- **Calibrated:** match severity to user and maintainer impact.

Do not report unrelated backlog, style already enforced by pre-commit (ruff,
mypy, SPDX headers, markdownlint, DCO), or a missing test that would not
protect changed behavior.

## Select the input and depth

Use `vllm-project/afd-plugin` as the base repository. Accept its forks and local
checkouts; use another skill for unrelated repositories.

| Input | Review surface |
| --- | --- |
| PR number or URL | Frozen PR metadata, full diff, and relevant threads. |
| Local branch/worktree | Frozen target-base SHA through committed, staged, unstaged, and in-scope untracked changes. |
| Pre-filled context | Reuse supplied metadata; fetch only missing facts and the full diff. |

Default to maintainer brevity. A detailed or audit request expands coverage and
lists `path:line` findings, but keeps the same confidence and severity bar.

## Reference guide

Load references in review order: process, one primary module contract, matching
feature overlays, evidence checks, then delivery. Every file is linked directly
below; do not load unrelated module references.

Each module contract links to a design page under `docs/design/module/`. For
branch-specific behavior, inspect the matching design page in the reviewed
checkout first; the frozen head's design pages are authoritative for that
review. If docs and live code disagree, verify the code/tests and report the
drift.

### Review process

| Reference | Read when |
| --- | --- |
| [review-execution.md](references/process/review-execution.md) | Every review; freeze inputs, inspect safely, and deliver against the same snapshot. |
| [general-checks.md](references/process/general-checks.md) | Every review; apply repository-wide correctness, patch-contract, and evidence rules. |
| [design-contracts.md](references/process/design-contracts.md) | Every production review; resolve branch-local module design status. |
| [review-routing.md](references/process/review-routing.md) | After the diff census; select one primary module and conditional overlays. |

### Primary module contract

| Reference | Read when |
| --- | --- |
| [plugin-config.md](references/modules/plugin-config.md) | Plugin registration, `AFDConfig` schema, parse/validation, extra-config, or env flags change. |
| [compat-patches.md](references/modules/compat-patches.md) | A monkey patch is added or changed under `afd_plugin/compat/patches/`, or the vLLM/vLLM-Ascend version gate moves. |
| [npu-compat.md](references/modules/npu-compat.md) | `afd_plugin/compat/npu/` runtime init, deferred NPU patch application, feature validation, or Ascend ops loading changes. |
| [connectors.md](references/modules/connectors.md) | An attention↔FFN connector backend, its control plane, or transfer metadata changes. |
| [distributed-topology.md](references/modules/distributed-topology.md) | AFD process groups, rank mapping, or topology validation changes. |
| [engine-orchestration.md](references/modules/engine-orchestration.md) | Async-DP engine behavior, cross-role request routing, DBO yield, or ubatching orchestration changes. |
| [workers-runners.md](references/modules/workers-runners.md) | Role workers, model runners (V1/V2), graph capture, or attention/FFN metadata change. |
| [execution-platforms.md](references/modules/execution-platforms.md) | NPU/GPU selection, platform checks, Ascend op build, or `csrc/` kernels change. |
| [model-integration.md](references/modules/model-integration.md) | Model registration, role-split wrappers, weight filtering, or a model family changes. |

### Feature-design overlays

| Reference | Read when |
| --- | --- |
| [disaggregation-topologies.md](references/features/disaggregation-topologies.md) | 2A2F/2A1F, colocation vs disaggregation placement, or recipe launch scripts change. |
| [async-cam-ubatching.md](references/features/async-cam-ubatching.md) | The async CAM connector, two-stage MoE ubatching, or DSV3.2/DSV4 async CAM forward paths change. |
| [graphs-and-compilation.md](references/features/graphs-and-compilation.md) | CUDA graph, ACL graph, capture metadata, or graph/eager fallback behavior changes. |

### Evidence and quality checks

| Reference | Read when |
| --- | --- |
| [model-addition-checklist.md](references/checks/model-addition-checklist.md) | A model family, role wrapper, registration, or model config is added. |
| [perf-verification.md](references/checks/perf-verification.md) | The PR makes a latency, throughput, memory, or accuracy claim. |
| [test-quality-evaluation.md](references/checks/test-quality-evaluation.md) | Tests change, are absent for risky code, or may not exercise production behavior. |
| [tests-docs-checklist.md](references/checks/tests-docs-checklist.md) | Coverage, CI markers, docs impact, PR evidence, or the PR checklist need review. |
| [verification.md](references/checks/verification.md) | Hardware, a server, or a runnable affected path is available for active verification. |

### Delivery and reviewer coordination

| Reference | Read when |
| --- | --- |
| [delivery-style.md](references/delivery/delivery-style.md) | Findings are ready for concise maintainer-style delivery. |
| [review-requests.md](references/delivery/review-requests.md) | The user asks to identify, suggest, request, or ping code-owner reviewers. |

## Workflow

### 1. Freeze and report the snapshot

Pin the base and head before reading source or running validation. Within 60 seconds,
report the pinned head, CI, mergeability, and preliminary findings in the host
conversation. Do not wait for CI or post this update to GitHub.

If the target changes while fetching, discard the evidence and retry once. If
it changes again, report the churn and wait for a stable target.

For a trusted PR head, materialize the pinned head in an isolated detached
worktree. A worktree freezes identity but is not a security sandbox. Treat fork
heads as untrusted unless the user and environment policy explicitly establish
otherwise: execute them only in a disposable, secret-free sandbox with restricted
filesystem, network, and resources; without one, use static SHA-addressed reads
and CI evidence only. For a local review, freeze the committed, index, worktree,
and NUL-safe in-scope untracked contents. Follow
[review-execution.md](references/process/review-execution.md) for trust gates,
state fingerprints, and byte-for-byte staleness checks.

### 2. Build the diff census

Group files into production code, tests, docs, configuration, build/CI, and
vendored artifacts (for example `afd_plugin/connectors/npu/bin/`). Map each
changed production file and test group to the PR goal. Compare the title/body
claims with the actual diff; use linked issues only when they define the
contract or reproduction.

Mark unrelated scope and unexplained generated artifacts. Do not infer behavior
from the PR description without tracing the live code.

### 3. Route from the live behavior

Trace each claimed behavior through the changed producer to its live consumer,
then use [design-contracts.md](references/process/design-contracts.md) and
[review-routing.md](references/process/review-routing.md) to select one primary
module contract, a second only for a real documented cross-boundary call path,
and every matching feature and evidence overlay. Treat titles and paths as
hints; live behavior and the frozen head's current design metadata are
authoritative. For docs-, tests-, or CI-only changes, route to the production
contract they protect or use only the applicable evidence checks.

### 4. Run the blocker scan

Apply every category in [general-checks.md](references/process/general-checks.md) before
lower-priority comments.

For each changed value or behavior, trace:

```text
vLLM engine/config ingress (--additional-config {"afd": ...})
  -> parse_afd_config / validate_afd_config -> platform/worker selection
  -> role worker/runner startup -> connector handshake
  -> attention<->FFN transfer -> model role forward (remote proxy)
  -> sampling/output in the owning role -> shutdown/teardown
```

Cover every applicable GPU/NPU, graph/eager, DBO on/off, 2A2F/2A1F,
runner V1/V2, colocation/disaggregation, and AFD-on/AFD-off path. Patched code
must stay transparent when AFD is inactive. Search bounded callers and sibling
implementations rather than assuming the changed hunk is the only path.

### 5. Apply module, feature, and patch contracts

Apply the reference set selected in step 3 and any matching repo-local skill.
Read the exact module and feature pages in the frozen head, including status,
ownership, `primary_code_paths`, dependencies, and `last_reviewed`. Candidate
or draft rules are questions, not blockers, unless current code, tests, or
AGENTS.md enforce them. Inspect both sides of every config, registration,
connector, rank-mapping, and patch boundary.

For every change under `afd_plugin/compat/`, enforce the AGENTS.md patch
contract: pinned upstream refs, full upstream copy with marked AFD differences,
identical signatures, `Patch reason`/`Patch functionality` comments, and
CPU-safe test coverage.

When a diff adds or expands a helper, class, fallback, compatibility branch, or
public behavior, run a subtraction pass: remove out-of-scope behavior and check
whether each new abstraction can be deleted, merged, moved, or inlined.

### 6. Verify the changed path

Before each validation group, verify the frozen SHA plus the tracked, index,
untracked, and ignored-file fingerprint, or recreate a pristine snapshot. On a
trusted head or inside the required sandbox, run an import/version preflight,
then the narrowest relevant tests and low-cost static checks. Bind every result
to the head SHA, snapshot fingerprint, and environment fingerprint. Never run
imports, tests, builds, hooks, or repo-configurable tooling from an untrusted
head on the reviewer host.

- Treat CI as status evidence; inspect only the first overlapping failure.
  Buildkite `test-ready` runs unit tests plus the DeepSeek-V2-Lite E2E gate;
  NPU has no in-repo CI, so NPU evidence is author-provided or manually run.
- For docs-only changes, use diff hygiene, links checks, and bounded live
  contract verification instead of dependency setup or pytest.
- For hardware-dependent paths, run available static/CPU checks (`pytest -m
  "not gpu and not vllm_runtime"`) and name the exact GPU/NPU gap. Never
  simulate device evidence.
- For performance or accuracy claims, require comparable base/head runs with
  the same environment, workload, warmup, repetitions, and graph/DBO mode.

Stop when each changed semantic path has a supported finding or an explicit
no-issue conclusion. Do not search further only to increase confidence.

### 7. Consolidate and deliver

Verify each finding against the current diff, deduplicate by root cause, and
order by severity.

Re-read the remote head and reverify or recreate the pristine validation
snapshot immediately before delivery. If either changed, mark the review stale
and restart from the new snapshot.

Return findings first. Use
[delivery-style.md](references/delivery/delivery-style.md) to keep them
direct and brief. Each finding must include an exact `path:line`, trigger or
call path, current behavior, impact, and smallest fix direction. If there are
no findings, say so briefly and name material validation gaps.

Keep the review read-only unless the user explicitly authorizes posting. Do not
submit `APPROVE`, `COMMENT`, or `REQUEST_CHANGES`, add labels, edit code, or push
commits as an implied part of review.

### 8. Optionally request focused owner reviews

Only when the user asks to identify or request reviewers, read
[review-requests.md](references/delivery/review-requests.md). Rank
path-matched CODEOWNERS against the frozen module contract's owners and
documented governance expertise; propose focused reviewers with an explicit
contract rationale.

Identifying or suggesting reviewers is read-only. Requesting reviewers or
posting `@mention` comments changes external state and requires explicit user
authorization. When authorized, recheck the head, deduplicate existing
requests, and post at most one consolidated comment. Do not infer this
permission from a request to review the code.

## Related skills

- `run-e2e` — execute the DeepSeek-V2-Lite / Qwen3 MoE / Qwen3.6 E2E gates; use
  when the review needs active GPU or NPU evidence.
- `adapt-model` — author-side workflow for adapting a new model family; suggest
  to contributors alongside the model-addition checklist.
- `upgrade-npu` / `upgrade-gpu-version` — the pinned-ref upgrade workflow; a PR
  that moves vLLM or vLLM-Ascend pins should follow them.
