---
name: upgrade-gpu-version
description: Upgrade and align the AFD Plugin GPU backend across exact pinned vLLM revisions and CUDA runtime/toolchain environments. Use when Codex must plan, audit, implement, review, or finally validate a GPU vLLM version upgrade; rebase compatibility patches; adapt GPU models, workers, model runners, connectors, CUDA Graph, DBO, TP, profiling, packaging, or isolation contracts; or resolve GPU regressions caused by an upstream version change. Do not use for NPU-only upgrades, model-only adaptation, ordinary GPU bugs unrelated to an upstream upgrade, or E2E-only execution.
---

# Upgrade the AFD GPU backend

Run this workflow as a gated, version-neutral state machine. Resolve the exact
current and target vLLM revisions, inventory AFD behavior, and pass the
architecture gate before editing production code. Preserve supported AFD
behavior on the target upstream architecture; do not preserve an obsolete
implementation shape merely because an older release required it.

Read the repository `AGENTS.md` before acting. Read
[`references/upgrade-workbook.md`](references/upgrade-workbook.md) completely
for evidence tables, source-diff commands, handoff records, and report
templates. List and read every `references/*lessons.md` file before upstream
analysis. Treat lessons as historical failure patterns, never as target-version
instructions; re-derive every applicable fact from the exact current and target
sources. Read
[`references/documentation-refresh.md`](references/documentation-refresh.md)
only after the required GPU validation gates pass and before refreshing public
support claims.

## Task modes

Choose exactly one mode from the request. Do not silently combine modes.

- `planning-audit`: perform read-only identity, inventory, upstream diff,
  architecture-gate, and staged-plan work. Do not edit AFD or allocate GPUs.
- `implementation`: implement the approved upgrade in bounded adaptations,
  add focused tests, and pass the static/unit gate. Continue to GPU validation
  only when the request includes it and the required environment is available.
- `final-validation`: freeze an already selected AFD revision and run read-only
  native and AFD validation. Do not fix production code unless the user changes
  the mode to implementation.

Record the selected mode. If it is ambiguous and the stopping point changes
what may be edited or whether GPUs may be allocated, ask once before proceeding.

## Hard boundaries

- Treat current vLLM and target vLLM as immutable revisions. Resolve both refs
  to full commit SHAs before diffing or editing. If either ref, SHA, or exact
  source tree is unavailable, mark `FREEZE_IDENTITY` `BLOCKED` and every later
  phase `SKIPPED`; do not substitute a historical lesson or installed package.
- Discover the current ref from repository evidence when possible. Require the
  target vLLM ref and a local vLLM source repository. Never infer a target
  revision from the newest installed package or an unpinned branch.
- Define the AFD base revision as the immutable pre-upgrade AFD commit named by
  the issue/RFC or proven by branch ancestry; do not assume `origin/main` or the
  current merge-base. Define the target AFD mirrored version as the package or
  compatibility version the repository will publish for the target vLLM, not a
  branch name. Cite the repository policy; use `NOT_APPLICABLE` when no mirrored
  version policy exists.
- Do not modify, switch, or check out another revision in a user-provided vLLM
  source directory. If its HEAD is the target SHA, use it read-only. Otherwise
  use `git show` in read-only audits, or create a separate detached worktree
  only when workspace writes are in scope. Read a missing current tree the same
  way without disturbing the provided checkout.
- Record the intended target Python, PyTorch build CUDA and C++ ABI, vLLM
  artifact/build identity, CUDA toolkit/runtime, NVIDIA driver, host compiler,
  NCCL, device compute capability/count/topology, and model paths. Inventory
  plugin-owned GPU native sources separately. Planning may mark unavailable
  hardware evidence `UNKNOWN`; final validation must stop when the exact runtime
  or required hardware cannot be established.
- Do not edit AFD production code before the architecture gate passes. If a
  major architecture change is found, stop and report the required design
  decision. Do not silently remove a supported feature or build a broad
  compatibility layer around it.
- Preserve unrelated dirty changes. Record branch, HEAD, and status before
  work. Stop if user changes overlap planned files. Never reset, discard, stage,
  or commit unrelated paths.
- Never modify the vLLM source tree to make AFD pass. Keep target worktrees
  read-only after creation.
- Prefer native vLLM extension points, inheritance, and composition. Do not add
  a new public connector, runner, protocol, or runtime abstraction unless the
  exact target contract proves existing interfaces insufficient; treat a broad
  cross-component expansion as an architecture-gate decision.
- Follow `AGENTS.md` for every copied or patched function: exact target
  signature and return type, source/reason/functionality comments, narrowly
  marked AFD deltas, focused AFD and non-AFD tests, performance impact, and a
  removal or upstream plan.
- Access known upstream members directly. Do not hide drift with broad
  `getattr`, `hasattr`, exception swallowing, `Any`, or version-probing wrappers.
- In implementation mode, keep each coherent upstream cause and its focused
  regression tests together. Create local signed-off commits only when commits
  are in scope; never push, publish, merge, or open a PR unless separately
  requested.
- Run every required non-accuracy GPU E2E case before final accuracy. Never use
  a limited accuracy run as final qualification, and never hard-code a test
  count in this skill.

## Evidence authority

Use this order when evidence conflicts:

1. exact current and target vLLM source revisions;
2. current AFD source, tests, package pins, and normative repository contracts;
3. results from the frozen validation cell;
4. historical `*lessons.md` files as hypotheses to check, not facts to carry.

An old workaround, file path, runner generation, test count, topology, or CUDA
behavior remains applicable only after target-source verification.

## Subagent orchestration

Use distinct subagents when available to separate analysis, implementation,
testing, and review. The primary agent owns phase state, architecture decisions,
the integrated diff, commits, and the final report. Subagent conclusions are
evidence, not automatic gate decisions.

- Keep analysis agents read-only. Split by vLLM history or AFD surface and
  require exact SHAs, paths, symbols, classifications, invariants, and tests.
- Dispatch implementation agents only after the architecture gate. Give each
  one non-overlapping AFD file set and one upstream root cause. Prohibit branch
  changes, staging, commits, pushes, and upstream edits.
- Keep test agents read-only for tracked files. Make GPU E2E agents read
  `../run-e2e/SKILL.md` and `AGENTS.md`. Serialize GPU jobs unless reservations,
  devices, ports, writable caches, logs, processes, and cleanup ownership are
  explicitly isolated.
- Keep review agents independent and read-only. Require simplicity and
  minimality review first, then exact upstream signatures, patch markers,
  ownership, call order, performance risks, isolation, and test coverage.

Record every handoff using the workbook. Never test or review a worktree while
an implementation agent is writing it. Inspect returned diffs and command
results directly before accepting them.

Use this loop for each adaptation in implementation mode:

```text
ANALYZE -> PRIMARY GATE -> IMPLEMENT -> FOCUSED TEST
        -> INDEPENDENT REVIEW -> PRIMARY COMMIT -> REGRESSION GATE
```

## Required phase record

At every phase boundary record:

```text
phase: <name>
status: PASS | FAIL | BLOCKED | STOPPED | SKIPPED
evidence: <SHAs, files, commands, tests, logs, or reports>
blocker: <none or exact blocker>
next_allowed_action: <one action>
```

Do not enter a later required phase when an earlier phase is `FAIL`, `BLOCKED`,
or `STOPPED`.

```text
FREEZE_IDENTITY
  -> INVENTORY_AFD_CONTRACTS
  -> DIFF_VLLM
  -> ARCHITECTURE_GATE
  -> PRODUCE_STAGED_PLAN
  -> IMPLEMENT_ATOMIC_ADAPTATIONS
  -> STATIC_AND_UNIT_GATE
  -> TARGET_NATIVE_GPU_CONTROL
  -> GPU_BASIC_E2E_GATE
  -> GPU_ACCURACY_GATE
  -> DOCUMENTATION_REFRESH_GATE
  -> FINAL_AUDIT
  -> CAPTURE_LESSONS_AND_REPORT
```

## Phase 0 — Freeze identity

Record the task mode; AFD checkout, branch, HEAD, immutable pre-upgrade base
revision and its evidence, dirty status, and Python environment; vLLM source
path, HEAD, branch, and dirty status; current and target refs plus resolved SHAs;
and any detached read-only worktrees.

Discover the current vLLM ref in this order:

1. `pyproject.toml`, lockfile, and package metadata;
2. compatibility constants, patch annotations, installation/support references, GPU
   guides, recipes, CI, and active upgrade RFC;
3. installed package versions and adjacent checkout state as corroboration only.

Resolve annotated tags to commits. If authoritative sources disagree, mark the
identity phase `BLOCKED` rather than choosing the convenient value.

Record the target AFD published package/compatibility version and validate it
against the repository's mirrored-version policy, or record `NOT_APPLICABLE`
with evidence. Record the intended target stack: Python; PyTorch version, build
CUDA, and C++11 ABI; vLLM source/package/native-artifact identity; CUDA toolkit
and runtime; NVIDIA driver; `nvcc`; host compiler; NCCL; GPU model and compute
capability; device count and interconnect/topology; runner generation; and
model/checkpoint path. In final validation, record the effective values and
reconcile them with the intended tuple.

## Phase 1 — Inventory AFD contracts

Build the inventory before reading the upstream diff. Use source search as the
authority and documentation as supporting evidence. Cover at least:

- version pins, runtime checks, package build/install, plugin registration,
  CPU-safe imports, and non-AFD isolation;
- every compatibility patch and copied upstream method, including copies outside
  `afd_plugin/compat/patches/**`;
- GPU Attention and FFN workers, model runners, EngineCore integration, KV-cache
  and scheduler ownership, startup, failure propagation, and shutdown;
- model registration, role-aware construction, native lifecycle, forward paths,
  parameter ownership, quantization, and weight filtering/loading;
- GPU connectors, payloads, custom ops, process groups, rank/device/topology
  mapping, synchronization, ordering, errors, and cleanup;
- CUDA Graph capture/replay, compile behavior, dummy/warmup runs, DBO/uBatch,
  TP/DP/EP/PP, profiling, sampling, request output, and memory accounting;
- GPU native sources and build interfaces in AFD and target vLLM, including
  artifact ownership, build flags, supported compute capabilities, and ABI;
- focused tests, complete GPU E2E scenarios, recipes, documentation, CI, and
  release metadata;
- shared files whose change may require NPU regression evidence without making
  an NPU support claim.

For every patched or copied function, record AFD target, current upstream source
and symbol, exact signature, marker blocks, preserved invariant, state/lifecycle
owner, direct consumers, focused AFD and non-AFD tests, and removal/upstream
plan.

Freeze the current supported feature matrix. Discover all represented runner
generations, models, connectors, eager/graph modes, DBO/uBatch cells, gate
placements, TP/DP/EP/PP topologies, profiling modes, quantization/EPLB paths,
accuracy cases, and negative/isolation cases. An omitted feature is `UNKNOWN`,
not implicitly out of scope.

For `planning-audit`, the minimum sufficient inventory is every AFD dependency
on the changed upstream surface, every copied or patched upstream method, and
every currently supported feature cell affected by that surface. Prove an area
unaffected with source-search evidence. If a user-imposed path or time bound
leaves an affected surface `UNKNOWN`, block the architecture gate rather than
calling the inventory complete.

## Phase 2 — Diff vLLM

Compare current vLLM SHA to target vLLM SHA. Begin with rename-aware name
status and stats, first-parent history, and focused function-context diffs. Map
every relevant upstream change to AFD consumers and classify it:

- `MECHANICAL`: import move, rename, or exact signature change with equivalent
  ownership and behavior;
- `BEHAVIORAL`: defaults, schema, state lifetime, call order, device mapping,
  graph behavior, or ownership changed while the AFD invariant still maps;
- `ARCHITECTURAL`: an execution generation, stable seam, lifecycle owner,
  feature, protocol, or cross-component boundary was removed or conceptually
  replaced.

For copied upstream logic, reconstruct method by method:

1. use the exact target upstream function as the new skeleton;
2. use current AFD markers and focused tests as the local-difference ledger;
3. use current supported AFD behavior as the semantic contract;
4. replay only still-required AFD differences into the target skeleton;
5. verify decorators, signature, return type, call order, state ownership,
   resource lifetime, hot-path cost, graph/compile behavior, and shutdown.

Do not mechanically apply an upstream old-to-new diff onto an AFD copy. Check
whether target vLLM added an extension point that lets the patch be removed or
replaced by delegation.

## Phase 3 — Architecture gate

Stop without production edits when any of these is true:

- the target vLLM revision and requested CUDA/PyTorch/toolchain tuple are not a
  reproducible supported environment;
- the GPU worker/model-runner generation or lifecycle is replaced and AFD needs
  redesign rather than local adaptation;
- a required patch seam disappears without an equivalent stable target, or it
  can no longer be expressed as target upstream logic plus small AFD markers;
- upstream removes or disables a supported graph, DBO, parallelism, model,
  connector, or profiling contract and continuing would require silent feature
  deletion or validation bypass;
- model split, expert/gate ownership, scheduler/KV ownership, connector payload,
  topology, process group, sampling/output, or cleanup contracts change across
  multiple subsystems without a one-to-one invariant mapping;
- preserving behavior requires a new public protocol, broad core rewrite,
  extensive upstream copies, reflection-heavy compatibility, or hidden fallback;
- the selected PyTorch/vLLM native artifacts, CUDA runtime, C++ ABI, or target
  compute capability cannot form one executable target cell;
- the required CUDA/driver/compiler/NCCL transition cannot be built and tested
  as one exact target environment.

A moved implementation is not automatically architectural. Continue only when
inputs, outputs, ordering, ownership, and lifetime map cleanly and the adaptation
remains narrow. On stop, use the workbook's architecture report and leave AFD
production files unchanged.

## Phase 4 — Produce a staged plan

Create an upstream-to-AFD impact matrix and a reviewable implementation plan.
For each stage record upstream cause, expected files, preserved invariant,
patch disposition, focused and regression tests, graph/performance/distributed
risks, documentation impact, and removal or upstream contribution plan.

Classify each compatibility seam as `remove`, `delegate/inherit`, `port as an
exact-version adapter`, `propose upstream`, or `stop`. Do not use a scoring
system instead of making the architectural judgment.

In `planning-audit` mode, emit the plan, patch audit, feature matrix, exclusions,
and evidence gaps, then mark later phases `SKIPPED` and stop without writes. If
an earlier gate is blocked, emit only a blocker-resolution plan naming the exact
missing evidence and next allowed actions; label the implementation impact
matrix and staged plan unavailable rather than filling them with hypotheses.

## Phase 5 — Implement atomic adaptations

In implementation mode, create or use a dedicated upgrade branch only when
branch changes are in scope; otherwise record and preserve the current branch.
Before each edit batch record files, upstream cause, invariant, tests, risks,
and patch removal/upstream plan.

Implement in dependency order unless the impact graph proves another order:

1. version/package contract and CPU-safe imports;
2. compatibility patches and plugin initialization order;
3. model construction, native lifecycle, role ownership, and weight policy;
4. Attention/FFN workers and model runners, EngineCore, KV/scheduler ownership;
5. connectors, payloads, distributed topology, and device mapping;
6. CUDA Graph, DBO/uBatch, parallelism, profiling, warmup, and cleanup;
7. focused tests and repository-owned E2E scenarios needed for qualification.

Use existing connector and runner interfaces unless the architecture gate
explicitly approved a new boundary. Preserve native model forward, loading, and
runtime behavior wherever the target provides a usable seam. Do not change
shared runtime code to conceal an incomplete GPU adaptation.

For each retained patch, copy the target function, match its exact signature,
mark only AFD differences, test selected and non-AFD branches, and document
performance and removal/upstream plans. Remove an obsolete patch only after
proving target upstream absorbed the behavior.

After each coherent adaptation passes focused checks and independent review,
inspect the staged diff and create a signed-off local commit when commits are in
scope. Keep source changes and focused regression tests together.

## Phase 6 — Static and unit gate

Before allocating GPUs, inspect the repository's current CI and test entry
points and run their actual commands. Cover:

- diff, lint, format, compile, import, and package build/install checks;
- exact target runtime import and plugin registration;
- version pin/runtime check consistency;
- target vLLM native-module provenance/import plus PyTorch build-CUDA and C++11
  ABI consistency;
- a clean AFD package build and import for every plugin-owned compiled GPU
  extension that exists in the target tree;
- exact patch signatures, decorators, return types, comments, and markers;
- affected focused tests, then the complete CPU/unit suite;
- model construction and weight policy, worker/runner contracts, connector
  ordering/topology, graph/DBO helpers, cleanup, and non-AFD isolation;
- dependency consistency such as `python -m pip check` in the target runtime.

Treat collection or import failure as a version/environment failure until the
identity tuple is proven. Import, construction, dummy execution, or server
startup alone never proves GPU support.

If AFD has no plugin-owned compiled GPU source, record `NOT_APPLICABLE` with the
inspected build files instead of inventing a build requirement. Do not rebuild
vLLM merely for preflight when the repository requires an installed runtime;
verify that runtime's native artifact and reserve source builds for an approved
target-build qualification.

In implementation mode without requested hardware validation, report all GPU
phases `SKIPPED` and readiness `implemented, GPU-unverified`.

## Phase 7 — Target native GPU control

Before AFD E2E, prove the exact target vLLM runtime can load the selected model
and serve at least one real eager GPU request with AFD disabled. Keep model,
checkpoint, runner generation, quantization, prompt, sampling, and comparable
parallelism aligned with the first AFD cell. Record startup, weight loading,
readiness, output/correctness oracle, resource use, and cleanup.

On the reserved target GPU, execute the smallest target-native kernel or
operator smoke that proves the loaded artifact runs on the recorded compute
capability. When AFD owns a compiled GPU extension, run its smallest real
operator smoke before the request; otherwise record `NOT_APPLICABLE`.

Follow `AGENTS.md` for remote access and GPU scheduler ownership. Never treat a
free-looking device as authorization. If native construction, loading,
readiness, or request execution fails, preserve the first useful traceback and
stop AFD parity validation. A fallback setting creates a separate validation
cell and must not silently replace the requested target.

## Phase 8 — GPU basic E2E gate

Read and follow `../run-e2e/SKILL.md` for hardware detection, provisioning,
pytest selection, live output, skip reporting, and cleanup. Do not duplicate or
override its backend-selection logic in this skill. `AGENTS.md` remains the
authority for scheduler use and remote test artifacts.

Run the complete marker-based GPU non-accuracy suite before accuracy:

1. feature scenarios;
2. model scenarios;
3. any repository-owned GPU upgrade scenario not covered by those categories.

Use enough reserved GPUs to avoid capacity skips when claiming a full upgrade.
Every skip must map to a documented unsupported or explicitly excluded cell; a
new, unexplained, or hardware-capacity skip is not a full pass. Record actual
collected and executed counts from pytest instead of assuming a fixed suite
size.

Verify behavior, not only launch: real responses, eager/graph comparisons,
actual CUDA Graph capture/replay, actual DBO stages, TP/DP/EP collectives,
profiler traces, error propagation, and cleanup as represented by repository
tests. After every run verify the exact experiment PID tree, ports, GPU
processes, scheduler state, and IPC resources are clean.

For each failure, preserve the cell, command, environment, first useful root
traceback, logs, resource state, and cleanup. In final-validation mode classify
and report without source edits. In implementation mode trace one root cause,
add the narrowest regression test, implement one minimal fix, repeat focused
review, then rerun the failed case and complete basic gate.

## Phase 9 — GPU accuracy gate

Enter only after the complete basic E2E gate passes. Delegate all GPU accuracy
execution to `run-e2e`. Run every repository-owned GPU accuracy case, including
eager and graph variants, with no dataset limit for final qualification. A
limited run is diagnostic evidence only.

On failure, preserve metrics and artifacts and compare the exact native and AFD
cells. Never lower thresholds or widen tolerance to make an upgrade pass unless
the user explicitly approves a separately justified policy change. After a code
fix, rerun the complete basic gate before full accuracy.

## Phase 10 — Documentation refresh gate

Enter only after required basic and accuracy gates pass. Read
[`references/documentation-refresh.md`](references/documentation-refresh.md)
completely before editing upgrade documentation. Refresh exact runtime claims,
GPU guides, recipes, design/patch inventories, contributor templates, and
release metadata from the validation ledger.

Distinguish `hardware validated`, `unit validated`, `implemented but
unvalidated`, `unsupported`, and `shared-code only`. Preserve intentionally
historical versions and measurements. Never infer NPU support from a GPU result
or relabel an old recipe as target evidence.

In final-validation mode without documentation-write scope, audit and report
required changes instead of editing them.

## Phase 11 — Final audit

Audit version pins, source annotations, patch inventory, support matrices,
package/release metadata, recipes, GPU documentation, CI, test discovery, and
historical branch notes. Confirm every retained patch has focused AFD and
non-AFD coverage plus a removal/upstream plan. Reconcile every public support
claim with an exact validation cell and list all exclusions.

Report CPU-safe and GPU evidence separately. Shared-code unit evidence is not a
GPU hardware claim; GPU evidence is not an NPU claim. Performance claims require
matching measured evidence rather than successful functional E2E alone.

## Phase 12 — Capture lessons and report

After a completed implementation upgrade, create a new version-specific file
under `references/` using the workbook template. Name it
`<YYYYMMDD>-<current-vllm>-to-<target-vllm>-lessons.md`; sanitize refs and use a
short SHA when no immutable tag exists. Never overwrite an earlier lesson.

Record exact identities, upstream changes, adaptations, failures, misleading
attempts, useful validation, documentation conflicts, remaining debt, and the
next-upgrade checks. Separate reusable principles from version-specific facts.
Link raw evidence instead of copying large logs. Have an independent read-only
review when available, then commit the lesson only when repository writes and
commits are in scope.

In planning or validation-only mode, do not create a lesson file. Include the
same fields in the final report as proposed lessons instead.

Use the workbook completion template to report:

- mode, all refs and full SHAs, target AFD version, and exact CUDA stack;
- architecture decision, impact matrix, staged plan, patch dispositions, and
  feature matrix;
- changed components and each local commit/root cause;
- static/unit, native build/ABI, native control, basic E2E, and full accuracy
  commands/results;
- actual pass/fail/error/skip counts, metrics, logs, cleanup, and exclusions;
- documentation consistency and intentionally preserved history;
- remaining private seams, performance observations, release blockers, and
  branch/publish status;
- readiness separated into GPU-validated, shared-code, and unverified claims.

Declare the requested scope complete only when every required earlier phase
passes. A full GPU upgrade requires immutable compatible identity, architecture
approval, focused and complete CPU gates, native GPU control, complete basic GPU
E2E with no unexplained skips, full accuracy, documentation consistency, clean
resource teardown, and an evidence-backed final report.
