# General Review Checks

Use this reference for every review. It defines the repository-wide contract
trace, code-design rules, the blocking risk scan, and the finding bar.

## Contract and scope

Trace each changed value or behavior through the AFD call path:

```text
vLLM engine/config ingress (--additional-config {"afd": ...})
  -> parse_afd_config / validate_afd_config -> platform/worker selection
  -> role worker/runner startup -> connector handshake
  -> attention<->FFN transfer -> model role forward (remote proxy)
  -> sampling/output in the owning role -> shutdown/teardown
```

- Prove the path in the frozen head, not from the PR description.
- Cover the matrix the change can affect: GPU/NPU, graph/eager, DBO on/off,
  2A2F/2A1F, runner V1/V2, colocation/disaggregation, AFD-on/AFD-off.
- Patched or wrapper code must be transparent when AFD is inactive
  (`is_afd_active` false): a non-AFD request must keep upstream vLLM 0.26.0
  behavior exactly.
- Search bounded callers and sibling implementations (the GPU vs NPU twin of
  every worker/model file) rather than assuming the changed hunk is the only
  path; the two platforms drift independently.

## Module, API, and class design

These are the repo's own normative rules (AGENTS.md); enforce them as review
rules, not style nits:

1. Access vLLM / vLLM-Ascend attributes directly; `getattr`/`hasattr` guards
   and proactive custom exceptions hide upstream-compatibility breakage that
   upgrades must surface. Flag them unless the change is genuinely optional.
2. Keep parameter types concrete; `Any`/`object` parameters hide contract
   violations from mypy/pyright.
3. No new magic numbers; name constants (`MAX_CONTEXT_LENGTH`, not `2048`).
4. No new mutable global state; constants and immutable config objects only.
   New mutable globals require approval and must be justified in the PR.
5. Do not split simple functions into helper layers; extract only for real
   complexity, real duplication, or an established local pattern.
6. Keep imports CPU-safe at module level: importing `afd_plugin` (or any unit
   test module) must not import torch, torch_npu, or vLLM runtime modules.
   Defer or `importorskip` them inside functions/tests.
7. New public behavior needs a named owner for its design page and a docs
   impact statement in the PR template.
8. Performance-affecting changes need a comparable A/B claim or an explicit
   "no perf impact" rationale; see [perf-verification.md](../checks/perf-verification.md).

## Blocking risk scan

| Risk | Prove before reporting |
| --- | --- |
| Correctness | A reachable input reaches the changed code and produces wrong output, a hang, or a crash. |
| Patch-contract drift | A patched function's signature, return type, or upstream copy diverges from the pinned ref, or missing/incorrect `# ### PATCH` markers, `Patch reason`/`Patch functionality` comments, or `# Upstream source:` pointers break upgrade comparability. |
| Version/compat | The change breaks the vLLM 0.26.0 / vLLM-Ascend `80d8c194f` contract, or moves a pin without updating the version gate, docker base, docs, and design pages together. |
| Platform divergence | The GPU or NPU twin (worker/runner/model variant) is not updated and the change is not provably platform-neutral. |
| CPU-safe imports | A new module-level import of torch/torch_npu/vLLM runtime breaks `pytest -m "not gpu and not vllm_runtime"` or plugin import on a CPU host. |
| Concurrency/async | Async-DP busy loops, ubatching state machines, DBO yield, or connector transfer states can deadlock, drop, reorder, or leak work items on abort/timeout. |
| Distributed/topology | Rank mapping is not bijective/validated, process groups are created outside the AFD owner, or teardown leaks groups across restarts. |
| Lifecycle | Graph capture state leaks into eager paths, shutdown ordering between attention and FFN worlds hangs either side, or connector close leaves a transfer half-open. |
| Security/validation | Untrusted `--additional-config`, env flags, or connector metadata are consumed without validation; host/port/rank inputs reach device code unchecked. |
| Evidence | Perf/accuracy claims lack comparable base/head runs; NPU behavior claims lack manual evidence per `docs/npu/TESTING.md`. |
| User contract | `AFDConfig` fields, defaults, or recipe launch scripts change meaning without a docs update and migration note. |

## Finding bar

Report a finding only when you can name the changed `path:line`, the trigger or
call path, the current behavior, the user or maintainer impact, and the smallest
safe fix direction. Do not report:

- Style pre-commit already enforces (ruff, format, SPDX, markdownlint, DCO).
- Hypothetical issues with no reachable trigger in the supported matrix.
- Missing tests that would not protect the changed behavior.
- Debt the PR only modifies or removes.
