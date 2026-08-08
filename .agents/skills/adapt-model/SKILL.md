---
name: adapt-model
description: Guide feasibility analysis, minimal implementation, review, and evidence validation for a new AFD model on GPU, NPU, or both. Use when Codex must adapt or review a new vLLM model, establish its native and AFD execution contract, add model-specific tests, or validate model support. Do not use for ordinary model explanations, existing-only E2E execution, generic bug fixes, or vLLM upgrades.
---

# Adapt a new AFD model

Use this skill as an executable, staged workflow. Treat every target model as
new: start from the repository baseline and exact native source. Do not assume
that a target adapter, successful runtime, recipe, or E2E evidence exists. Use
related AFD models only as implementation references; do not copy their model
assumptions without checking the target contract.

Keep the workflow model-specific but the public AFD runtime stable. Preserve
native model lifecycle, ordinary forward, residual/norm behavior, quantization,
parameter paths, weight mapping/loading, and supported parallelism unless the
exact contract proves that a narrow adapter extension is necessary.

## Non-negotiable boundaries

- Do not modify vLLM or vLLM-Ascend source trees.
- Default to zero common Runner, Connector, or runtime changes for a model-only
  adaptation. If the exact target contract proves an existing public interface
  insufficient, stop and report that gap before expanding scope.
- Do not modify native FusedMoE/MoERunner, routing, dispatch/combine,
  quantization, or kernel code for a model-only adaptation.
- Construct only the active role's large execution modules. Never construct
  the full opposite role and delete it afterward.
- Preserve non-AFD native registry resolution and behavior.
- Filter checkpoint paths once, preserve original names and tensor objects, and
  delegate the filtered iterator to the native loader.
- Do not reimplement native packed mapping, expert mapping, TP/EP sharding,
  quantization scales, or loader bookkeeping.
- Keep GPU and NPU evidence independent. A GPU result never proves NPU support.
- Do not commit, push, or publish unless the user explicitly requests it.
- Read and obey the repository `AGENTS.md`, applicable design documents, and
  backend validation instructions before changing code or allocating hardware.

## Task modes

Accept one of these modes. Do not silently combine them.

- `feasibility`: source inspection and a minimal implementation/validation
  plan; do not change production code or allocate hardware.
- `implementation`: implement only the explicitly requested adapter and its
  focused tests, then stop at the CPU gate unless hardware validation is also
  requested and authorized.
- `review`: inspect an existing adapter or patch against the exact native
  contract and AFD boundaries; do not redesign unrelated code.
- `validation`: validate an already selected implementation and exact model,
  checkpoint, backend, connector, topology, and mode; do not expand scope.

## Required inputs

Discover these before asking the user; required unknowns remain `UNKNOWN` and
must not be inferred from another architecture or backend:

```text
task_mode: feasibility | implementation | review | validation
model: family, exact architecture, checkpoint/config
backend_scope: gpu | npu | both
runtime:
  vllm: exact version/commit
  vllm_ascend: exact tuple when NPU is in scope
target:
  connector, topology, eager/graph, TP/DP/EP/PP, quantization, DBO/uBatch
oracle: native baseline and correctness/accuracy threshold when validating
```

## Phase state machine

Run these phases in order. At the end of every phase record:

```text
phase: <name>
status: PASS | FAIL | BLOCKED | UNKNOWN | SKIPPED
evidence: <files, commands, or logs>
blocker: <none or exact reason>
next_allowed_action: <one action>
```

Do not enter a later phase when a required earlier phase is `FAIL`, `BLOCKED`,
or a required fact is `UNKNOWN`.

```text
ESTABLISH_IDENTITY
  -> INSPECT_NATIVE_CONTRACT
  -> DESIGN_MINIMAL_ADAPTER
  -> IMPLEMENT_MINIMAL_ADAPTER
  -> CPU_CONTRACT_GATE
  -> NATIVE_EAGER_CONTROL
  -> AFD_EAGER_GATE
  -> OPTIONAL_GRAPH/DBO/PARALLELISM
  -> REPORT
```

The state machine is a superset. Task mode determines the stopping point;
mark phases outside the requested scope `SKIPPED`.

| Task mode | Required stopping point |
| --- | --- |
| `feasibility` | Phases 0–2, then report; do not implement or allocate hardware. |
| `review` | Phases 0–2 plus read-only contract checks as needed, then report. |
| `implementation` | Phases 0–4, then report; continue to hardware only when explicitly requested and authorized. |
| `validation` | Confirm Phases 0–2 read-only, skip implementation when no code change is required, then run only the applicable validation gates. |

## Phase 0 — Establish identity

Before reading or changing implementation code, record:

- AFD checkout, branch, commit, and dirty status;
- exact vLLM source path and commit; exact vLLM-Ascend source and software
  stack when NPU is in scope;
- model family, architecture class, checkpoint path, config, weight index, and
  quantization;
- requested role split, connector, topology, runner generation, and mode;
- available hardware and authorization for GPU/NPU use;
- relevant repository recipes, tests, and runbooks.

Do not require or assume a prior successful target-model checkout. Establish a
fresh native control in Phase 5.

Preserve unrelated dirty changes. Prefer an isolated branch/worktree for an
implementation. Never reset or delete user changes to make the identity clean.

## Validation cell identity

Before native or AFD execution, freeze one validation cell:

- repository/adapter SHA;
- model/checkpoint;
- vLLM/runtime and plugin state;
- runner generation and connector;
- Attention/FFN topology;
- TP/DP/EP/PP;
- quantization;
- eager/graph mode;
- multiprocessing/runtime launch mode;
- memory and `max_model_len` knobs;
- prompt and sampling parameters;
- relevant environment variables.

Keep model-semantic and non-AFD runtime settings identical to the native
control where applicable: checkpoint, prompt, sampling parameters,
quantization, runner generation, and native TP/DP/EP/PP dimensions. AFD-specific
role topology, connector settings, process layout, and necessary role-local
resource settings may differ; record those differences explicitly. Any other
change creates a new validation cell; do not compare results across cells as
one retry sequence.

## Evidence discipline

- Read the repository's local instructions and runbooks for execution details;
  do not duplicate backend- or machine-specific commands in this skill.
- Verify the effective runtime configuration and logs instead of treating an
  intended setting as proof that it was applied.
- If a runtime workaround is needed, create a separate validation cell, change
  one variable at a time, and report it as runtime configuration rather than an
  adapter change.
- Preserve the original failure artifacts, resource state, and cleanup evidence
  so that later results cannot silently replace the failed control.

## Phase 1 — Inspect the exact native contract

Read the pinned native implementation for the target architecture and record
the symbols and signatures that affect the requested scope:

- architecture registry and class hierarchy;
- CausalLM, Model, Decoder, and Layer constructors;
- ordinary `forward`, residual and norm order;
- layer factory or `decoder_layer_type` injection points;
- Attention, dense MLP, MoE, gate, shared expert, and experts boundaries;
- PP placeholders, embedding/final-output lifecycle, and sampling ownership;
- native parameter paths, packed projections, expert paths, FP8/quant scales,
  and `load_weights` delegation;
- TP/DP/EP/PP, graph/compile, DBO, and speculative/MTP behavior relevant to
  the requested first slice.

Record sampling/final-output ownership on the Attention side and any
connector-driven FFN daemon boundary; the FFN side does not acquire KV-cache,
scheduler, or final-output ownership merely because it computes FFN output.
Also record dummy-run, warmup, profiling, graph capture/replay, cleanup, and
exception behavior when those paths are part of the requested scope.

Inspect AFD's actual model API calls in the selected Attention/FFN runners,
ForwardContext metadata installation, connector metadata, DBO helpers, and
existing model-side proxy. Do not invent a model API because a related adapter
has one. The adapter must expose only methods the selected runner actually
calls, such as `compute_ffn_output` or `get_experts_layer_indices`.

Inspect the repository's documented test and launch contract before designing
new commands, and make the adapter's accepted scope match that contract.

Do not access an optional or dynamically installed context field directly just
because reflection is discouraged. Use the existing typed AFD metadata/helper
and preserve any proven fallback behavior; add a focused test for the context
variant.

## Phase 2 — Choose the smallest adapter

Apply this decision tree:

```text
Existing AFD wrapper already covers the native architecture?
  yes -> register/configure and add focused tests only
  no  -> continue

Native Model exposes a layer factory or decoder_layer_type hook?
  yes -> subclass/inject a role-aware Layer; do not copy the Model constructor
  no  -> continue

Native CausalLM exposes a model_cls hook?
  yes -> use it
  no  -> copy only the required constructor, with exact patch markers

Native Layer.forward calls mlp(hidden_states) at the desired boundary?
  yes -> reuse the existing parameter-free remote proxy
  no  -> add a native-compatible experts facade only when the call contract
         proves that the experts boundary is required
```

Default implementation surface:

- one model adapter file;
- registration/config rewrite only where the repository requires it;
- model-specific CPU contract tests;
- a model-specific E2E scenario only when hardware validation is requested.

Do not add a common proxy, runtime, protocol, coordinator, or capabilities
file unless the existing implementation demonstrably cannot satisfy the exact
target contract and at least one concrete consumer is identified.

Separate ownership into three questions:

| Ownership | Rule |
| --- | --- |
| Execution | Attention executes Attention; FFN executes dense MLP or native MoE. |
| Large parameters | Do not construct/load non-owner Attention projections, gates, routed/shared experts, or dense/MoE weights. |
| Lifecycle | Embedding, decoder norms, final norm, lm_head, PP placeholders, and tied weights follow native lifecycle needs; a documented dual-role exception is valid. |

Do not turn lifecycle-only modules into `PPMissingLayer` or make them
single-owner solely to satisfy an abstract ownership rule. Prove the native
construction, PP, loader, and runner requirements first.

For the model adapter:

- keep native Attention, dense MLP, MoE runner/kernel, and quantization;
- keep native ordinary forward whenever the role boundary permits it;
- implement only role-aware construction, model-specific runner entry
  delegation, stage/proxy integration, and path-to-role classification;
- keep gate-on-FFN as the default unless the target and requested scope prove
  an Attention-side gate contract;
- fail closed for unsupported gate, MTP, topology, or backend combinations.

When copying native code, add the required reason, source file, source commit,
exact signature/return type, and paired `PATCH START/END` markers immediately
around AFD-specific differences. Copy constructors only when a native hook is
unavailable. Do not copy native forward or loader loops when composition or
delegation is sufficient.

## Phase 3 — Implement the smallest change

Before editing, write a short ledger:

```text
files to add/modify:
new public abstractions:
copied native functions and source lines:
common runner/connector/runtime changes: default zero; any exception requires a
proven exact-contract gap and an identified consumer
unsupported combinations:
```

Implement in this order:

1. registration/config isolation;
2. role-aware layer/model construction;
3. existing proxy or the narrowest proven facade;
4. top-level FFN runner API delegation;
5. one-pass role-aware checkpoint filtering;
6. focused tests.

Do not change the common runtime to compensate for an incomplete adapter. If a
native contract cannot be expressed without a broad copy or common change,
stop and report the gap before expanding the patch.

Reuse the repository's documented validation entry point when it can express
the requested scope. Add model-specific test code only when the existing
contract cannot express the required behavior; do not create a parallel
execution harness to hide an adapter gap.

For repository E2E execution, read and delegate to
`.agents/skills/run-e2e/SKILL.md` whenever the existing E2E interface can
express the requested validation cell. Do not duplicate its hardware
detection, provisioning, pytest-marker execution, or cleanup workflow here.
If the target model lacks a repository-owned E2E scenario, add only the
model-specific scenario required by the requested scope, then hand execution
back to `run-e2e`.

## Phase 4 — CPU contract gate

Run compile/import/static checks and relevant CPU tests before hardware. Cover
only features used by the target model, including as applicable:

- registration isolation and native non-AFD behavior;
- exact constructor/forward/loader signatures and patch markers;
- native decorator/MRO/compile-support behavior;
- real layer schedule and role parameter manifests;
- absence of non-owner large parameters before checkpoint loading;
- stable native parameter paths for Attention, dense, MoE, gate, shared expert,
  and quantization scales;
- one-shot checkpoint filtering, unchanged names/tensors, native loader result;
- top-level FFN API and layer-index contract;
- proxy send → DBO yield → recv ordering, metadata, stage fallback, and error;
- PP/tied-weight/lifecycle exceptions when used;
- explicit negative tests for unsupported mode/backend/topology combinations.

A mock with a tiny invented model is not evidence for a large target model.
Use target config-derived layer counts and expert counts in static fixtures
when feasible. Mark unavailable dependencies as skipped or blocked; never call
them passed.

## Phase 5 — Native eager control gate

Establish a native vLLM eager control before starting AFD. The control must use
the exact target checkpoint and a runner/runtime comparable to the requested
AFD path. Record startup, weight loading, readiness, one correctness request,
and cleanup.

If native construction, weight loading, readiness, or the first correctness
request fails:

- preserve the original logs and resource state;
- identify the closest worker/root traceback, not only a serving-wrapper error;
- mark native execution `FAIL` or `BLOCKED`;
- mark AFD parity `SKIPPED`;
- do not continue expanding the adapter or use an AFD-only result as an oracle.

Diagnostics may create a separate validation cell only when necessary. Change
one variable at a time, state the hypothesis, and never merge its result into
the original control.

For a backend or kernel failure, first keep AFD disabled and run the smallest
single-variable diagnostic that can distinguish native runtime from adapter
contract. If the fallback passes, classify the original cell as a native
runtime failure and use the passing cell as the explicit AFD oracle; do not
silently change the adapter or call the workaround a model fix.

## Phase 6 — AFD eager gate

Only after the native control passes, run the narrowest requested AFD path:

- exact model and checkpoint;
- exact AFD checkout and runtime;
- same runner generation, comparable precision, and the same native
  TP/DP/EP/PP dimensions unless the claimed AFD scope explicitly changes them;
- one connector and one smallest supported role split;
- eager before graph, DBO, performance, or wider parallelism.

Reuse the repository's documented validation entry point and lifecycle helpers;
for repository-owned E2E, delegate to `.agents/skills/run-e2e/SKILL.md` rather
than duplicating hardware detection, provisioning, pytest-marker execution, or
cleanup here.
If a model-specific scenario is absent, add only the scenario needed for the
requested scope; do not create a second execution harness.

Keep model-semantic and non-AFD runtime settings identical to the native
control where applicable: checkpoint, prompt, sampling parameters,
quantization, runner generation, and native model-parallel dimensions.
AFD-specific role topology, connector settings, process layout, and necessary
role-local resource settings may differ; record those differences explicitly.
Any other change creates a new validation cell. Confirm the expected
native-to-AFD architecture mapping on both roles and that the FFN role reached
native dense/MoE computation before claiming a request pass.

Validate, as applicable:

- Attention and FFN construction and load completion;
- connector initialization and cleanup;
- send/receive order and layer/stage association;
- native FFN compute on the FFN role;
- native/AFD output parity using an explicit oracle;
- process, port, GPU/NPU, and reservation cleanup.

Text equality alone is a narrow oracle. Report when token IDs, hidden states,
logits, graph/DBO, or multi-rank behavior were not measured.

## Phase 7 — Optional extensions

Run graph, DBO, TP/DP/EP/PP, gate variants, MTP, quantization variants, NPU,
or performance only after the eager gate passes and only when the requested
scope includes them. Treat GPU and NPU as separate lanes with separate native
controls, contracts, commands, logs, and readiness results.

### GPU extension lane

For GPU claims, inspect and validate the exact GPU worker/runner, P2P
connector, CUDA eager/graph path, TP/EP/DBO settings when claimed, and
model-specific GPU E2E evidence.

### NPU extension lane

For NPU claims, independently inspect the exact vLLM-Ascend/software tuple,
NPU worker/runner, selected CAMP2P/CAM connector, ACL graph/uBatch behavior,
MoE/quantization/weight-loading overlays, and model-specific NPU E2E evidence.
Never derive one backend's support from the other.

## Failure and rollback protocol

On any failure:

1. stop the next phase;
2. save the exact source/runtime/command/environment/log/cleanup identity;
3. classify the failure as adapter contract, native runtime, connector,
   resource, or unknown;
4. preserve the diff and evidence. Revert only an isolated speculative change
   when the user authorizes it or the worktree is explicitly disposable;
   otherwise leave the change in place for review, preserving unrelated user
   changes;
5. report the next smallest diagnostic action.

Never change runner, topology, plugin state, memory limits, and adapter code in
one retry. Never claim model support from import success, construction alone,
FFN weight loading alone, or an AFD-only smoke when native control failed.

## Evidence report

Report these fields without omitting scope:

```text
Model: family, architecture, checkpoint, config
Source: AFD checkout/commit; vLLM and vLLM-Ascend source/runtime
Adapter: changed files, ownership, copied functions, common-code changes
Claimed scope: backend, connector, runner, topology, mode, quantization, gate
CPU: commands and pass/fail/skipped/blocked results
Native control: command, readiness, correctness oracle, cleanup
AFD eager: roles, load, connector, output/parity, cleanup
Extensions: graph/DBO/parallelism/NPU results or explicit exclusions
Execution: GPU/NPU not-run | passed | failed | blocked
Readiness: GPU/NPU validated | experimental | unverified | unsupported
Unknowns and blockers
Validator and date
```

Use `unsupported` only when the implementation explicitly rejects the
combination. Use `failed` for an executed failure and `blocked` when an
external dependency or environment prevents the test.

## Completion gate

Before declaring a new model adapted, verify:

- the target was treated as new and no prior success was assumed;
- exact native and AFD identities are recorded;
- the minimal adapter decision tree was followed;
- no unnecessary common abstraction or native forward/loader copy was added;
- role construction and lifecycle exceptions are evidenced;
- CPU contract gate passed;
- native eager control passed before AFD parity;
- AFD eager, connector, cleanup, and oracle evidence match the claimed scope;
- the adapter files deployed to the validation cell match the reviewed working
  tree by commit or content hash;
- any runtime workaround is listed separately from the adapter diff;
- all unrun, failed, blocked, and unsupported combinations are explicit.
