---
name: maintain-vllm-main
description: >-
  Maintain afd-plugin's long-lived upgrade branch by merging AFD main,
  incrementally adapting to vLLM main, and recording validation and the next
  starting point. Use for manual rolling maintenance or resuming unfinished
  upgrades; qualify releases separately.
---

# Maintain AFD against vLLM main

Spread upgrade work across manually triggered maintenance rounds. Keep one
long-lived branch with completed adaptations, validation evidence, and a clear
starting point for the next round.

Read [lessons from previous upgrades](references/experience.md) on first use or
when a related problem arises. For a release, pin the release target and complete
the repository's final qualification.

## Working agreement

- Read the repository's `AGENTS.md` and follow its development and execution
  requirements.
- Reuse the branch named by the user or the maintenance record. If neither exists,
  `codex/vllm-main-sync` is a suitable default, based on this round's fixed AFD
  main. Use an isolated checkout, preserve other work and uncommitted changes,
  and maintain the same upgrade branch serially.
- Confirm the canonical AFD repository, `vllm-project/afd-plugin`, the writable
  result repository, and the separate `vllm-project/vllm` checkout from remote
  URLs. `origin` is not necessarily the canonical source.
- **Merge AFD main into the upgrade branch**, preserving published history.
  Adapt AFD to the selected vLLM source changes; do not merge vLLM Git history
  into the AFD repository.
- Establish the previous adaptation point from the maintenance record, code,
  and dependency declarations together. A version bump or one experimental
  patch does not prove the whole upstream interval was adapted. Investigate
  missing evidence before asking for an unknown starting point.
- Preserve the agreed support scope and backend version contracts. vLLM-Ascend
  has its own target. Tracking vLLM main does not automatically upgrade Ascend
  or establish NPU validation. Check affected consumers on the other backend
  when shared code changes.

## Each maintenance round

### 1. Fix the inputs and target

Fetch AFD main, the upgrade branch, and vLLM main; record their full SHAs. Read the
last completed vLLM adaptation point, existing evidence, and pending work. Leave
upstream changes arriving during the run for the next round.

Prefer this round's vLLM main HEAD and confirm that its runtime is available for
the execution platform. If a matching wheel is unavailable, make a bounded search
for a recent runnable commit and report its distance from main. Do not move
behind the completed adaptation point. Honor an exact user-selected target and
report any runtime gap.

The source, wheel or image, PyTorch/CUDA, and native extensions must correspond
to the selected target. Record actual provenance and build identity rather than
inferring them from version strings. If the runtime is unavailable, continue
source analysis and local implementation with validation marked incomplete.
Routine preflight does not require rebuilding vLLM or deploying a lasting service.

### 2. Merge AFD main first

Merge the fixed AFD main into the existing upgrade branch. Resolve conflicts
while preserving new main features and previous adaptation intent. Do not
recreate the result branch from main each round.

Check the semantics of newly merged code, tests, and configuration on the target
vLLM, including changes without textual conflicts. Avoid reverting newer
dependency adaptations to an older pin. If main itself advances the upstream
baseline, reconcile the two histories, retain newer work, and adjust this
round's target when necessary.

Continue when AFD main changed or pending work remains, even if vLLM did not
advance. Report no change only when both inputs are unchanged, no work is
pending, and existing evidence is still valid.

### 3. Adapt the actual differences

Analyze both the vLLM interval since the completed point and the newly merged AFD
main changes, together with pending work. Follow affected AFD code and its direct
consumers.

Trace imports, inheritance, patch assignments, upstream copies, and call chains.
Directory mappings only help locate code: renamed paths, outputs or scheduler
structures outside worker modules, and copies outside patch directories can
change the contract.

Make small, complete adaptations for the upstream change. Prefer target-version
extension points, inheritance, and composition. Refresh retained patches from
the exact target function and reapply necessary AFD differences, following
repository signature, comment, marker, and architectural review requirements.
Remove patches whose behavior upstream has taken over when the evidence supports
removal.

Align required dependency pins, version checks, and lockfiles with the target
source and runtime. Update affected tests and deployment entry points as well;
a version bump or source-only edit is insufficient.

For runner retirement, lifecycle redistribution, protocol changes, or removed
capabilities, explain the concrete contract and available choices. Do not
independently drop supported features or add broad compatibility layers.
Preserve explicitly accepted retirements.

Distinguish inherited main or upgrade work, necessary interface integration, and
new behavior fixes. Keep substantial or risky new fixes independently reviewable.
Before splitting, check dependencies so paired padding, counters, or graph-key
changes remain complete. Record unrelated issues separately. Only explicit user
acceptance can remove a required in-scope failure from qualification.

### 4. Validate and resolve failures

State the failure each check detects. Choose checks from the affected behavior:

- **Code or dependency changes:** run repository-required formatting and static
  checks, relevant regressions, and applicable CPU/unit tests. Read commands from
  this branch's CI and test entry points rather than fixing historical counts.
- **Target-version, import, or runtime boundaries:** verify vLLM, the AFD plugin,
  native extensions, and relevant interfaces in the real target environment.
  An older environment, mocks, or CPU-safe imports do not replace this evidence.
- **Deployment or test environments:** inspect pins, lockfiles, CI base images,
  build scripts, and actual test configuration. Respect justified overrides in
  upstream images; a generic `pip check` failure alone does not justify changing
  an official environment.
- **High-risk runtime changes:** use the smallest sufficient hardware validation
  within existing authorization and available resources, reusing repository E2E.
  For communication changes, choose meaningful unequal inputs or asymmetric
  topologies. For Graph/DBO, confirm actual capture/replay or two-stage execution
  and check real output.

Routine maintenance does not default to the full model and accuracy matrix. Use
[run-e2e](../run-e2e/SKILL.md) for GPU E2E entry points and operating requirements,
and follow those requirements when entering that workflow. Record missing
hardware evidence as unvalidated. A required check that is missing leaves this
round incomplete. Documentation-only changes need no additional model tests.

Classify failures as test configuration, environment, native upstream failure,
historical defect, or current regression before choosing the smallest fix.
Check actual chunked prefill, token budget, runner, and parallel settings first.
Keep inputs, sampling, and scoring comparable in controls. Avoiding a trigger
through configuration does not establish a production fix.

Rerun the failed check first, then affected checks; stop repeating retries that
produce no new information. Do not relax tolerances, omit agreed required cases,
or disable features to obtain a pass. Verify the tested source, loaded kernels,
and exit status. Report inference results, controller failures, and cleanup
failures separately. Verify process exit and resource release after every run.

### 5. Save the result and next starting point

Keep each cause and its necessary tests in clear local commits, with repository
sign-offs. Inspect the final diff and exclude temporary reports, raw logs, model
outputs, and unrelated changes. Retain necessary operating instructions and
evidence links.

Maintain a short state file, defaulting to `docs/upgrade-status.md`; reuse an
existing equivalent. Update current state rather than accumulating execution
logs. Include at least:

```text
Canonical AFD source and merged main SHA
Upgrade branch and result remote/branch
Observed vLLM main SHA, selected target, and reason for any lag
Completed vLLM adaptation SHA and pending interval or tasks
Tested AFD source commit, runtime identity, commands/results, and evidence links
Latest hardware-validated version, configuration, and results; unvalidated scope
Known limits, user-accepted scope decisions, and next action
```

Advance the completed vLLM starting point only after agreed adaptations and
required checks finish. Save partial code, failures, or environment gaps as work
in progress. A single probe does not certify the entire interval. On resumption,
inspect existing changes and evidence to avoid repeating completed work.

With existing push authorization for the specific maintenance branch, reuse it
and push normally. Otherwise finish locally and report the pending result.
Reconcile remote changes and revalidate affected behavior after a rejected push;
do not overwrite others with a force push. Preserve verified local work if a
push fails and report the actual remote state. Create or update PRs, mark them
ready, or merge only under the respective existing authorization.

## Final release upgrade

Pin the release vLLM tag/SHA, required Ascend target, and final AFD main or release
candidate. Reconcile their differences from the rolling branch. If the release
target predates the followed main, select an appropriate checkpoint and complete
the differences in a separate candidate rather than merging future adaptations.

Complete repository requirements and this release's agreed qualification. Reuse
comparable evidence; rerun affected validation when behavior-related code or the
runtime changed. GPU, NPU, model, feature, accuracy, and performance claims each
need matching evidence. Routine maintenance passing does not qualify a release.

Report the branch/commit and push status, merged AFD main, completed vLLM target
and gap to main, major changes, validation, and remaining work. Finish the
agreed round and wait for the next manual invocation.
