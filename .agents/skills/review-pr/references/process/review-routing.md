# Module and Feature Review Routing

Use this reference after the diff census. Route from live behavior traced
through the changed producer to its consumer, not from file names alone; when
paths and behavior disagree, route on behavior and note the drift.

## Primary module contract

Select one primary contract; add a second only for a real cross-boundary call
path (for example a connector change that also moves rank mapping).

| Module | Live behavior signals | Read |
| --- | --- | --- |
| Plugin entry & config | `register_afd()`, `AFDConfig`, parse/validate, extra-config coercion, env flags | [plugin-config.md](../modules/plugin-config.md) |
| Compatibility patches | New or changed monkey patch under `afd_plugin/compat/patches/`, version gate, upstream copy | [compat-patches.md](../modules/compat-patches.md) |
| NPU runtime compat | `compat/npu/` deferred patch application, feature validation, Ascend ops loading | [npu-compat.md](../modules/npu-compat.md) |
| Connectors | Connector backend, control plane, transfer/metadata handshake | [connectors.md](../modules/connectors.md) |
| Distributed topology | AFD process groups, `AFDRankMapping`, topology validation | [distributed-topology.md](../modules/distributed-topology.md) |
| Engine orchestration | Async-DP engine loops, cross-role routing, DBO yield op, ubatch orchestration | [engine-orchestration.md](../modules/engine-orchestration.md) |
| Workers & runners | Role workers, model runners V1/V2, graph capture, attention/FFN metadata | [workers-runners.md](../modules/workers-runners.md) |
| Execution platforms | NPU/GPU worker selection, Ascend op build, `csrc/` kernels | [execution-platforms.md](../modules/execution-platforms.md) |
| Model integration | Model registrations, role-split wrappers, weight-role filtering | [model-integration.md](../modules/model-integration.md) |

Docs-, tests-, or CI-only changes route to the production contract they
protect; if none applies, use only the applicable evidence checks.

## Feature-design overlays

| Overlay | Signals | Read |
| --- | --- | --- |
| Disaggregation topologies | 2A2F/2A1F, colocation vs disaggregation, `recipe/` launch scripts | [disaggregation-topologies.md](../features/disaggregation-topologies.md) |
| Async CAM & ubatching | `CAMAsyncAFDConnector`, two-stage MoE ubatching, DSV3.2/DSV4 async CAM forwards | [async-cam-ubatching.md](../features/async-cam-ubatching.md) |
| Graphs & compilation | CUDA graph, ACL graph, capture/replay metadata, eager fallback | [graphs-and-compilation.md](../features/graphs-and-compilation.md) |

## Evidence and change overlays

| Signal | Read | Optional repo-local skill |
| --- | --- | --- |
| A model family or registration is added | [model-addition-checklist.md](../checks/model-addition-checklist.md) | `adapt-model` |
| Latency/throughput/memory/accuracy claim | [perf-verification.md](../checks/perf-verification.md) | `run-e2e` |
| Tests change or are missing for risky code | [test-quality-evaluation.md](../checks/test-quality-evaluation.md) | — |
| CI, docs, PR evidence, checklist | [tests-docs-checklist.md](../checks/tests-docs-checklist.md) | — |
| Hardware or runnable path available | [verification.md](../checks/verification.md) | `run-e2e` |
| vLLM/vLLM-Ascend pin moves | [compat-patches.md](../modules/compat-patches.md) + [execution-platforms.md](../modules/execution-platforms.md) | `upgrade-npu` / `upgrade-gpu-version` |

## Calibrate findings

- **P0:** security exposure, data corruption, an unusable plugin or engine
  start, a patch that silently changes non-AFD upstream behavior. Block merge.
- **P1:** reachable runtime failure, wrong output, platform divergence,
  patch-contract violation that breaks upgrade comparability, missing evidence
  for a blocking claim. Block merge until fixed or explicitly waived by an
  owner.
- **P2:** real defect with a concrete future failure mode, contract drift the
  code does not yet enforce, missing docs/test for changed behavior.
  Non-blocking; do not inflate to block.

Everything else is a question for the author, not a finding. One root cause
yields one finding even if it surfaces at several `path:line` sites.
