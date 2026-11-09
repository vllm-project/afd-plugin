# NPU upgrade documentation refresh

Read this reference completely after the target NPU basic and accuracy gates
pass and before editing upgrade-related documentation. Drive every support claim
from recorded validation evidence. Do not document intended support as validated
support.

## Contents

1. Documentation evidence
2. User-facing runtime documentation
3. Connector guides and recipes
4. Design and compatibility documentation
5. Contributor workflow and generated metadata
6. Conditional and historical artifacts
7. Consistency audit
8. Completion record

## Documentation evidence

Freeze this evidence before editing documentation:

```text
Target runtime:
  vLLM ref and resolved SHA:
  vLLM-Ascend ref and resolved SHA:
  Python / PyTorch / torch_npu / CANN / Triton Ascend:
  CAM, CAMP2P, or other operator packages:
  container image and NPU/SoC:

Validated cells:
  model / checkpoint / quantization:
  connector:
  TP / DP / EP / PCP topology:
  eager / graph / DBO / uBatch:
  native control result:
  basic E2E commands and counts:
  full accuracy commands and metrics:
  skips, exclusions, and cleanup:
```

Use backend-scoped claims. An NPU-only upgrade does not prove that the GPU
backend supports the new vLLM revision. Distinguish these evidence levels:

- `hardware validated`: the exact documented cell passed real NPU E2E;
- `unit validated`: static or unit coverage passed without an NPU support claim;
- `implemented, unvalidated`: code exists but target hardware qualification is
  absent;
- `unsupported`: the combination is rejected, removed upstream, or intentionally
  excluded.

## User-facing runtime documentation

### Installation, support, and configuration

Review `docs/installation.md`, `docs/support.md`, and `docs/configuration.md` for:

- target runtime wording and whether the version applies to NPU, GPU, or both;
- supported models, connector matrix, and validation limits;
- NPU hardware and full software-stack prerequisites;
- vLLM and vLLM-Ascend installation pins and resolved source revisions;
- build flags, `SOC_VERSION`, custom operator packages, and container guidance;
- plugin installation and import or environment verification commands;
- worker selection and required environment variables;
- AFD configuration keys, connector names and aliases, topology fields, graph,
  DBO, uBatch, PCP, quantization, and operator settings;
- links from `README.md` and `docs/README.md` to current guides and recipes.

Remove stale commands and unsupported examples. If a shortened launch example
is not a complete runnable Attention/FFN deployment, label it as a fragment and
link to the complete recipe.

### Connector package overview

Review `afd_plugin/connectors/README.md`. Keep its connector status matrix
consistent with `docs/support.md`, NPU guides, recipes, and
test evidence. State validation separately for CAMP2p and CAMAsync and for each
relevant eager, graph, DBO, PCP, or topology cell.

## Connector guides and recipes

### CAMP2p guide

Review `docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md` for:

- exact runtime, hardware, communication-library, and operator prerequisites;
- supported execution modes and known exclusions;
- Attention/FFN process topology, process groups, world ordering, physical
  device selection, and rank formulas under TP/DP/EP/PCP;
- connector configuration schema and derived ranks;
- DBO/uBatch behavior, supported microbatch count, execution order, and flags;
- graph compilation settings, model/operator fields, launch order, health
  checks, and cleanup;
- runnable commands matching the target recipe.

### CAMAsync guide

Review `docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md` for:

- one unambiguous target-version support statement;
- actual hardware-validation evidence or an explicit code-only/unvalidated
  status;
- MoE request, activation, routing, and output flow plus lifecycle ownership;
- process topology, process-group ownership, world ordering, rank/device
  mapping, and PCP semantics;
- connector schema, work-item and payload contracts, and control-plane ordering;
- native NPU DBO versus AFD asynchronous MoE ubatching;
- operator packages, environment variables, launch and cleanup instructions;
- current limitations and unsupported validation cells.

Do not leave a warning that says a path is unvalidated alongside a later claim
that the same target path is verified.

### Recipe index and runnable recipes

Review `recipe/README.md` and every target NPU recipe under `recipe/npu/`:

- make the index identify the exact runtime and evidence level for every recipe;
- update prerequisites, images, package versions, model paths, topology, CLI
  arguments, environment variables, and cleanup steps;
- inspect all referenced launch and baseline scripts, not only the recipe
  Markdown;
- rerun commands before presenting them as current runnable examples;
- attach benchmark or accuracy results only to the exact environment and command
  that produced them.

Prefer a new version-scoped recipe when a historical recipe contains old
measurements or a materially different topology. Do not relabel old results,
images, branches, or commands as target-version evidence. If a historical
recipe remains linked, label it historical and prevent the root index from
presenting it as current validation.

## Design and compatibility documentation

Refresh the internal source-of-truth documents when their contracts changed:

| Document | Required review |
| --- | --- |
| `docs/design/module/index.md` | Runtime pair, upstream references, validation entry points, code ownership, and links |
| `docs/design/module/compatibility_and_patches.md` | Exact target patch inventory, source symbols and signatures, patch reasons, tests, and upstream/removal plans |
| `docs/design/module/execution_platforms.md` | NPU Worker/ModelRunner initialization, ForwardContext, graph, DBO/uBatch, operator loading, and tested runtime matrix |
| `docs/design/module/connector_contracts.md` | Configuration schema, payload/work-item ownership, control plane, topology, lifecycle, and execution order |
| `docs/design/module/attention_runtime.md` | Attention worker/runner selection, metadata, forward context, connector handoff, graph, and ubatching |
| `docs/design/module/ffn_runtime.md` | FFN daemon/EngineCore lifecycle, MoE execution, connector steps, graph dispatch, and shutdown |
| `docs/design/module/model_integration.md` | Model registration, role-aware construction/loading, MoE routing/gate ownership, quantization, and weight loading |
| `docs/design/module/plugin_boundary.md` | Registration and patch order, worker classes, configuration aliases, environment variables, and upstream boundary |

For each reviewed design document, update `upstream_refs`,
`verified_platform_refs`, `validation_paths`, and `last_reviewed` only after
checking the corresponding content. Do not update only the review date.

Rebuild the patch inventory from source grep and exact target source. Remove a
documented workaround only after proving that target upstream absorbed its
behavior. Document any remaining patch's target symbol, invariant, focused
test, and long-term upstream or removal plan.

## Contributor workflow and generated metadata

Review these workflow documents when the repository-wide target or required
environment changes:

- `.github/ISSUE_TEMPLATE/100-bug-report.yml`: request vLLM and vLLM-Ascend
  refs, CANN, PyTorch/torch_npu, NPU/SoC, connector, topology, graph/DBO, and
  reproduction information;
- `.github/ISSUE_TEMPLATE/200-feature-request.yml`: remove obsolete version or
  extension-point assumptions;
- `.github/PULL_REQUEST_TEMPLATE.md`: require backend-scoped compatibility,
  exact upstream refs, NPU validation, skips, and documentation impact.

Audit package/release metadata and any generated documentation derived from the
README or project configuration. Regenerate tracked outputs through their
normal generator; do not hand-edit ignored or local `.egg-info` artifacts.

## Conditional and historical artifacts

Update these only when the corresponding behavior changed:

- architecture SVGs when component ownership, class relationships, or data flow
  changed;
- DBO/topology images when pipeline stages, rank mapping, or microbatch flow
  changed;
- design proposals when they are still active and their upstream assumptions
  changed.

Preserve intentional historical references in release notes, old recipes,
benchmarks, proposals, and branch-specific instructions. For an obsolete active
proposal, mark it `superseded` and link to the replacement decision instead of
rewriting its historical basis as though it described the target runtime.

## Consistency audit

Compare every public support claim across these sources:

| Claim | Sources that must agree |
| --- | --- |
| Runtime versions | Installation and support references, dependency pins, guides, recipes, design index, templates |
| Connector support | Support reference, connector README, both NPU guides, recipe index, E2E evidence |
| Models and quantization | Support reference, recipes, model integration design, model E2E evidence |
| DBO/graph/PCP/topology | Support reference, connector guides, recipes/scripts, execution and connector design docs, E2E evidence |
| Installation stack | Installation reference, guides, recipes, container/operator configuration, native-control evidence |

Search the whole documentation surface for stale versions and conflicting
support language. Adapt these commands to the current and target refs:

```bash
rg -n 'vLLM|vllm[-_ ]ascend|VLLM_TAG|VLLM_COMMIT|CANN|torch_npu' \
  README.md docs recipe .github afd_plugin/connectors/README.md
rg -n 'validated|verified|supported|unsupported|unvalidated|experimental' \
  README.md docs recipe afd_plugin/connectors/README.md
rg -n 'DBO|uBatch|PCP|CAMAsync|CAMP2p|ACL Graph|eager|graph' \
  README.md docs recipe afd_plugin/connectors/README.md
git diff --check
```

Review each search hit in context. A stale-looking old version can be correct
when it is explicitly historical.

## Completion record

Record the documentation phase as:

```text
Documentation refresh:
  evidence ledger:
  public docs changed:
  recipes and scripts changed:
  design docs changed:
  templates/metadata changed:
  historical references intentionally preserved:
  support claims reconciled:
  validation commands and results:
  documentation commit SHA/title:
  status: PASS | FAIL | BLOCKED
```

The phase passes only when target-version claims match exact validation
evidence, cross-document support statements agree, runnable examples use the
target stack, intentional history remains correctly scoped, and all related
documentation changes are included in one or more focused signed-off commits.
