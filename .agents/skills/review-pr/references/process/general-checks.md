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
- Cover the matrix the change can affect:

  | Axis | Values in the current tree |
  | --- | --- |
  | Platform | CUDA, Ascend NPU |
  | Execution | eager, `FULL`, `FULL_DECODE_ONLY` graph |
  | Ubatching | DBO off, two native microbatches |
  | Topology | 1A1F, 2A1F, 2A2F, 4A2F, 4A4F, 2P1A1F, multi-pod |
  | Connector | `P2pNcclAFDConnector`, `DeepEPM2NAFDConnector`, `CAMP2pAFDConnector`, `CAMAsyncAFDConnector` |
  | FFN execution | control-plane-driven, routed-experts, async layered GMM, FFN two-stream |
  | Draft model | target-only, DSpark target + draft |
  | AFD | on, off (transparent passthrough) |

  Not every cell exists; claim only the combinations the code and design pages
  enumerate, and flag a change that silently widens the supported set.

- There is no V1 model-runner fallback. Both roles require
  `VLLM_USE_V2_MODEL_RUNNER=1`; Attention runs ModelRunnerV2 and FFN stays
  connector-driven. `validate_gpu_model_runner_v2_config` /
  `validate_npu_model_runner_v2_config` raise on a missing flag — a change that
  relaxes or bypasses that check is a contract change, not a convenience.
- Patched or wrapper code must be transparent when AFD is inactive
  (`is_afd_active` false): a non-AFD request must keep upstream vLLM behavior
  exactly, at the pinned ref.
- Search bounded callers and sibling implementations (the GPU vs NPU twin of
  every worker/model file, and the sync vs async NPU FFN runner pair) rather
  than assuming the changed hunk is the only path; the platforms drift
  independently.

## Module, API, and class design

These are the repo's own normative rules (AGENTS.md, docs/UT_MAINTENANCE.md);
enforce them as review rules, not style nits:

1. Access vLLM / vLLM-Ascend attributes directly; `getattr`/`hasattr` guards
   and proactive custom exceptions hide upstream-compatibility breakage that
   upgrades must surface. Flag them unless the change is genuinely optional.
2. Keep parameter types concrete; `Any`/`object` parameters hide contract
   violations from mypy/ty.
3. No new magic numbers; name constants (`MAX_CONTEXT_LENGTH`, not `2048`).
4. No new mutable global state; constants and immutable config objects only
   (`MappingProxyType` for the frozen registration maps is the house pattern).
   New mutable globals require approval and must be justified in the PR.
5. Do not split simple functions into helper layers; extract only for real
   complexity, real duplication, or an established local pattern.
6. Keep imports CPU-safe at module level: importing `afd_plugin` (or any unit
   test module) must not import torch, torch_npu, or vLLM runtime modules. The
   package uses `__getattr__` lazy export maps in `afd_plugin/__init__.py`,
   `afd_plugin/v1/worker/__init__.py`, and `afd_plugin/v1/worker/npu/__init__.py`;
   keep new runtime classes behind those maps, not as eager imports.
7. New public behavior needs a named owner for its design page and a docs
   impact statement in the PR template.
8. Performance-affecting changes need a comparable A/B claim or an explicit
   "no perf impact" rationale; see [perf-verification.md](../checks/perf-verification.md).

## Blocking risk scan

| Risk | Prove before reporting |
| --- | --- |
| Correctness | A reachable input reaches the changed code and produces wrong output, a hang, or a crash. |
| Patch-contract drift | A patched function's signature, return type, or upstream copy diverges from the pinned ref, or missing/incorrect `# ### PATCH START/END` markers, `Patch reason:` / `Patch functionality:` / `# Upstream source:` comments break upgrade comparability. Patched methods now live in `afd_plugin/v1/worker/**` as well as `afd_plugin/compat/`, and `tests/unit/compat/patches/test_signatures.py` owns the signature table for both. |
| Version/compat | The change breaks the current `TARGET_VLLM_VERSION` / vLLM-Ascend pin contract, or moves a pin without updating the version gate, `docker/Dockerfile.ci` base tag and its assertion, `pyproject.toml`, installation/support references, and design-page `upstream_refs` together. Read the pin from `afd_plugin/compat/vllm.py`, not from memory. |
| Platform divergence | The GPU or NPU twin (worker/runner/model variant, sync vs async NPU FFN runner) is not updated and the change is not provably platform-neutral. |
| CPU-safe imports | A new module-level import of torch/torch_npu/vLLM runtime breaks `pytest -m "not gpu and not vllm_runtime"` or plugin import on a CPU host. |
| Concurrency/async | Async-DP busy loops, ubatching state machines, DBO yield, async CAM work items, or two-stream slots can deadlock, drop, reorder, or leak work items on abort/timeout. |
| Distributed/topology | Rank mapping is not bijective/validated, process groups are created outside the AFD owner, or teardown leaks groups across restarts. |
| Lifecycle | Graph capture state leaks into eager paths, shutdown ordering between attention and FFN worlds hangs either side, or connector close leaves a transfer half-open. Known open failures (for example DeepSeek-V2-Lite DBO shutdown, NPU graceful exit) must be named as pre-existing context, not silently repaired by assertion changes. |
| Security/validation | Untrusted `--additional-config`, `connector_extra_config`, env flags, or connector metadata are consumed without validation; host/port/rank inputs reach device code unchecked. |
| Routed-expert contract | A routed-experts connector (`DeepEPM2NAFDConnector`) change assumes a weight layout the pinned vLLM MoE backend does not preserve. `ROUTED_MOE_BACKEND = "triton"` is pinned deliberately; a new backend needs an explicit layout proof, not a silent allow-list addition. |
| Quantization/layer contract | Async CAM layered FFN rejects incompatible weights at startup (`dynamicQuant=1`, homogeneous W4A8/W8A8, layering). A change that falls back to another FFN path instead of failing startup contradicts the design page. |
| E2E gate integrity | Case IDs, sample counts, device counts, or the required case set change without a design-page update; a gate case gains a `skip`/`xfail`; CI stops selecting a node ID it used to select. `docs/design/module/e2e_testing.md` is `normative`, so its `E2E-INV-*` items are blockers. |
| CI suppression | The diff is docs-only or skip-mark-only, so `skip_ci.py` suppresses all default CI. A green pipeline then proves nothing about the code; say so instead of treating it as validation. |
| Vendored artifacts | New or changed files under `afd_plugin/connectors/npu/bin/` (`.run`/`.whl`) are pre-commit-excluded by design; they still need source, version, and target CANN justification. |
| Agent/skill drift | A change to production paths, CI, or E2E gates that leaves `.agents/skills/**` stating stale behavior (pins, case counts, commands) is a P2 finding; those skills are executed instructions. |
| Evidence | Perf/accuracy claims lack comparable base/head runs; NPU behavior claims lack manual evidence per `tests/npu/README.md`. |
| User contract | `AFDConfig` fields, defaults, `connector_extra_config` keys, or recipe launch scripts change meaning without a docs update and migration note. |

## Finding bar

Report a finding only when you can name the changed `path:line`, the trigger or
call path, the current behavior, the user or maintainer impact, and the smallest
safe fix direction. Do not report:

- Style pre-commit already enforces (ruff, format, typos, SPDX, markdownlint,
  shellcheck, DCO).
- Hypothetical issues with no reachable trigger in the supported matrix.
- Missing tests that would not protect the changed behavior.
- Debt the PR only modifies or removes.
