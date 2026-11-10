# GPU Upgrade Documentation Refresh

Use this guide only after the target GPU runtime has passed the required
validation gates. Documentation is part of the upgrade contract: every claim
must describe the exact target revision and the evidence actually collected.

## Contents

1. Evidence ledger
2. Claim levels
3. Installation, support, and configuration
4. GPU connector and runtime guides
5. Recipes and launch scripts
6. Design and contributor documentation
7. Generated and release metadata
8. Historical artifacts
9. Consistency audit
10. Completion record

## 1. Evidence Ledger

Create one ledger before editing documentation. Record:

- AFD revision and package version;
- current and target vLLM revisions;
- Python, PyTorch, PyTorch build CUDA and C++11 ABI, vLLM native-artifact
  provenance, CUDA runtime/toolkit, driver, compiler, and NCCL used for
  validation;
- GPU model, count, memory, and topology;
- GPU compute capability, target-native kernel smoke, and any AFD compiled GPU
  extension build/import/operator smoke or `NOT_APPLICABLE` evidence;
- model and dataset paths;
- execution mode, connector, TP size, CUDA Graph state, DBO or uBatch state,
  and profiling state for each validated cell;
- exact commands, results, exclusions, and cleanup evidence.

When a value is unknown, write `not verified`. Do not infer a target environment
from a container tag, dependency constraint, or historical run.

## 2. Claim Levels

Label support using the strongest evidence available:

- **hardware validated**: exercised on the stated target GPU stack;
- **target-native control validated**: target vLLM ran without AFD for the same
  model and relevant environment;
- **CPU validated**: static, import, unit, packaging, or collection checks only;
- **implemented, GPU validation pending**: code exists but no target-hardware
  result is available;
- **shared-code only**: a common path changed, but this backend was not
  exercised;
- **unsupported**: deliberately excluded with a reason;
- **not evaluated**: outside the completed validation matrix.

Do not turn CPU-only evidence into a GPU support claim. Do not describe an
option as validated merely because it was accepted by the command line: confirm
that logs, counters, traces, or execution behavior show the path was used.

## 3. Installation, Support, and Configuration

Review `docs/installation.md`, the [support matrix](../../../../docs/support.md#tested-runtime-matrix), and `docs/configuration.md` for:

- the exact supported vLLM revision or range;
- the target Python, PyTorch, CUDA, driver, compiler, and NCCL requirements;
- supported GPU architectures and known hardware limits;
- installation commands and package extras;
- supported models, connectors, parallel modes, and execution modes;
- CUDA Graph, DBO or uBatch, TP, profiling, and isolation support;
- links from `README.md` and `docs/README.md` to current guides and recipes;
- explicit unsupported or unvalidated combinations.

Avoid broad phrases such as “GPU fully supported” when only selected models or
cells were validated. Prefer a concise matrix or a link to the validation
record.

## 4. GPU Connector and Runtime Guides

Review `afd_plugin/connectors/README.md` first. Keep its connector status matrix
consistent with `docs/support.md`, GPU guides, recipes, and
E2E evidence. State eager, CUDA Graph, DBO/uBatch, TP, asymmetric topology, and
profiling evidence separately where their support differs.

Review `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md` against the target source and
runtime. Confirm:

- process and rank topology;
- producer, router, and consumer responsibilities;
- tensor, metadata, token, and expert-routing ownership;
- send/receive ordering and synchronization assumptions;
- environment variables and configuration keys;
- launch order, health checks, shutdown, and cleanup;
- CUDA Graph and DBO or uBatch interactions;
- TP behavior and asymmetric topology behavior;
- profiling and observability behavior;
- version-specific limitations and failure signatures.

If the implementation delegates to a target-vLLM public API, document the
public contract rather than preserving the shape of an old copied patch.

## 5. Recipes and Launch Scripts

Review `recipe/README.md`, every target recipe under `recipe/gpu/**`, and each
referenced launch script. For each recipe, verify:

- model path and model family;
- number of processes and GPUs;
- TP and other parallel settings;
- connector and transport selection;
- CUDA Graph, DBO or uBatch, and profiling settings;
- ports, hostnames, environment variables, and required directories;
- target-runtime arguments and removed or renamed flags;
- expected output and shutdown procedure;
- whether the exact recipe was run, partially run, or only inspected.

Do not silently modernize a historical recipe and continue presenting it as the
original result. Preserve historical evidence and add a current recipe or a
clear version note.

## 6. Design and Contributor Documentation

Review every current design source of truth, not only documents whose names
appear related to the changed code:

| Document | Required GPU review |
|---|---|
| `docs/design/module/index.md` | target runtime, upstream refs, validation entry points, ownership, and links |
| `docs/design/module/compatibility_and_patches.md` | exact patch symbols/signatures, reasons, tests, dispositions, and removal/upstream plans |
| `docs/design/module/execution_platforms.md` | GPU worker/runner initialization, device selection, CUDA Graph, DBO/uBatch, and profiling |
| `docs/design/module/connector_contracts.md` | NCCL/P2P configuration, payload/metadata ownership, ordering, topology, lifecycle, and cleanup |
| `docs/design/module/attention_runtime.md` | Attention lifecycle, KV/scheduler ownership, forward context, connector handoff, graph, and ubatching |
| `docs/design/module/ffn_runtime.md` | FFN daemon/EngineCore lifecycle, MoE execution, graph dispatch, failure propagation, and shutdown |
| `docs/design/module/model_integration.md` | registration, role-aware construction/loading, gate/expert ownership, quantization, and weights |
| `docs/design/module/plugin_boundary.md` | registration/patch order, worker classes, aliases, environment variables, and upstream boundary |

For every reviewed design document, update `upstream_refs`, runtime/validation
references, and review metadata only after checking the corresponding content.
Rebuild the patch inventory from source search and exact target vLLM source.

Review native-build contracts in `csrc/gpu/README.md`, `setup.py`,
`pyproject.toml`, and package manifests. If AFD still has no compiled GPU
extension, keep that status explicit rather than documenting a build step. If
one exists in the target tree, document its prerequisites, build flags,
supported compute capabilities, ABI, import verification, and operator smoke.

Review the contributor workflow when the target runtime or evidence contract
changes:

- `.github/ISSUE_TEMPLATE/100-bug-report.yml`: request vLLM identity, PyTorch,
  CUDA/driver/compiler/NCCL, GPU/compute capability, connector, topology,
  graph/DBO, and reproduction evidence;
- `.github/ISSUE_TEMPLATE/200-feature-request.yml`: remove obsolete runtime or
  extension-point assumptions;
- `.github/PULL_REQUEST_TEMPLATE.md`: require backend-scoped compatibility,
  exact upstream refs, native build/ABI status, GPU validation, skips, cleanup,
  and documentation impact.

Contributor instructions must keep the exact-current-versus-exact-target
comparison rule and the repository patch policy visible. Remove examples that
teach an obsolete private API or stale copied implementation.

## 7. Generated and Release Metadata

Inspect metadata that can make a correct implementation appear stale or can
publish the wrong compatibility contract:

- package version and dependency constraints;
- `pyproject.toml`, `setup.py`, lockfiles, wheel/build metadata, and native
  artifact declarations;
- generated model or plugin registration tables;
- issue and pull-request templates;
- changelog or release notes;
- CI matrices and container references;
- comments or labels that state an old runtime version.

Update generated files through their documented generator when one exists.
Record the generator command and resulting diff. Do not hand-edit generated
output unless the repository explicitly permits it.

## 8. Historical Artifacts

Treat old RFCs, benchmark reports, logs, and upgrade lessons as historical
evidence. Preserve their original version context. If they are still useful:

- add a visible historical label;
- link to the current support statement;
- distinguish reusable principles from version-specific facts;
- do not rewrite old results to imply they were produced on the new target.

New lessons belong in a new version-pair file. Do not replace an earlier lesson
file with the latest upgrade story.

## 9. Consistency Audit

Search the full repository for the old version and changed public names. Start
with targeted searches such as:

```bash
rg -n '<old-vllm-version>|<old-cuda-version>|<old-pytorch-version>' \
  README.md docs recipe tests .github afd_plugin/connectors/README.md \
  csrc/gpu/README.md pyproject.toml setup.py uv.lock
rg -n '<removed-symbol>|<renamed-argument>|<old-environment-variable>' \
  README.md docs recipe tests .github afd_plugin csrc/gpu setup.py
rg -n 'supported|unsupported|validated|not verified|CUDA Graph|DBO|uBatch|TP' \
  README.md docs recipe afd_plugin/connectors/README.md
```

Classify each hit as one of:

- current and correct;
- historical and clearly labeled;
- stale and must be updated;
- generated and must be regenerated;
- unrelated to the GPU upgrade.

Then check that installation/support references, connector README, GPU guides, recipes, design modules,
tests, CI, package/build metadata, contributor templates, and release notes
agree on the same target identity, native-build status, and support scope.

## 10. Completion Record

Finish with a documentation record containing:

- files reviewed;
- files changed;
- exact evidence supporting each changed claim;
- historical files intentionally preserved;
- generated artifacts regenerated and commands used;
- native build/ABI and `NOT_APPLICABLE` claims reconciled with build files and
  target-GPU smoke evidence;
- stale references removed;
- known documentation gaps and owners;
- the final consistency-search commands and outcomes.

Documentation refresh is complete only when a new contributor can identify the
target environment, launch a supported configuration, understand the proven
support boundary, and find the evidence without consulting the upgrade author.
