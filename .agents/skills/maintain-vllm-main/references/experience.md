# Lessons from maintenance and release upgrades

Use these records to inform decisions. Historical versions, test counts, and
one-time scope agreements are not defaults for the next round; use current source
and the active user agreement.

## Incremental upstream maintenance trials

The reusable sequence is: inspect the long-lived result branch, recommend a
coherent target and validation budget for approval, then merge AFD main, adapt
semantics, check and fix, and save the result and next starting point. Historical
trial targets and device counts do not prescribe a later round's scope or budget.

Observed problems and evidence limits:

- Static path mappings missed `vllm.v1.outputs` and
  `vllm.v1.core.sched.output`. Follow actual dependencies; no directory match
  does not establish that a consumer is unaffected.
- The latest main had no wheel, so a recent commit with a matching wheel was
  selected. Distinguish observed main from the adapted target and record the gap.
- An upstream SM context factory moved from a class method to a module function,
  bypassing AFD's override. Refreshing the target constructor with small AFD
  differences passed targeted CPU tests. Compatible signatures alone did not
  prove the extension point was still called.
- A single-point experiment without the target wheel, updated pins, or GPU
  execution could not certify a usable full upgrade or advance its completed
  adaptation record.
- Two-round fixtures for Git inheritance and publication did not establish a
  real end-to-end install, adapt, validate, publish, and resume cycle.

## v0.30 integration and qualification

These records describe that integration. They do not define later support scope
or current PR status.

- **A main merge does not guarantee semantic compatibility.** Preserve main
  features and fixes. Later main commits caused test conflicts again, and
  conflict-free changes still needed target-interface review.
- **The environment is part of an upgrade.** Dependencies targeted 0.30 while a
  CI base image still targeted 0.26, and installation did not automatically
  select the target extra. Checking pins alone missed the real test environment.
- **A check can reject a valid environment.** A new `uv pip check` gate rejected
  an intentional NCCL override for DeepEPv2 in the official image. Inspecting the
  exact target Dockerfile justified removing that generic gate while retaining
  actual version checks, without downgrading official dependencies or adding a
  filtering framework.
- **Check test conditions before declaring a production fix.** MRV2 DBO hung with
  chunked prefill disabled and a large token budget. Selected CI passed with
  appropriate test settings, and experimental production communication changes
  were withdrawn. The separate large-message deadlock was not declared fixed.
- **Real output exposes failures that signature tests miss.** NPU MRV2 2A1F
  started and returned garbled text. Asymmetric topology and unequal inputs
  exposed padding and FFN context issues. Graph and DBO also needed proof of
  actual execution.
- **An aggregate score does not replace a computation-contract review.** SP
  token splitting and shared-expert TP weight sharding could mix different
  tokens despite valid shapes. Accuracy from other configurations did not cover
  that contract.
- **Diagnose accuracy failures under the original assertions.**
  [Issue #429](https://github.com/vllm-project/afd-plugin/issues/429) traced intermittent
  errors to missing synchronization after SwiGLU writeback. Original FP32
  references and tolerances were retained; targeted device runs verified the
  fix and actual new kernel.
- **Make new fixes independently reviewable.**
  [PR #425](https://github.com/vllm-project/afd-plugin/pull/425) separated
  integration, interface adaptation, and new behavior fixes in its explanation,
  with larger fixes split out. Paired counters and graph-key changes needed by
  [PR #430](https://github.com/vllm-project/afd-plugin/pull/430) remained together.
  [PR #431](https://github.com/vllm-project/afd-plugin/pull/431) duplicated
  [PR #422](https://github.com/vllm-project/afd-plugin/pull/422) and was closed;
  check existing fixes before splitting.
- **Evidence applies to specific code and conditions.** After splitting a fix,
  older NPU results containing that fix could not certify the reduced branch for
  the same scenario. Unchanged GPU code could reuse matching evidence. Record
  useful inference data, controller errors, and final cleanup separately.
- **Use the active scope agreement.** The first-300-question limit, accepted
  retirements, and deferred fixes were explicit decisions for that release.
  They are not universal sample sizes or automatic waivers. Avoid mechanically
  rerunning the whole hardware matrix without new concerns.

## General GPU upgrade tradeoffs

Prefer native lifecycle hooks and existing connectors. Inspect upstream copies
outside patch directories, use meaningful asymmetric topologies, and retain
exact-version evidence. Apply these lessons when the architecture changes;
historical runners, models, and topologies do not constrain a new target.
