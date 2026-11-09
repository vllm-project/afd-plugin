# Multi-pod E2E execution

Each existing pod runs one slice of a model-declared GPU Case. The model owns
its topology and acceptance workload; `multi_pod` owns rank placement and peer
coordination; `deployment.py` owns this pod's service processes. Pods or
containers must already exist. See the
[Kubernetes walkthrough](../../../.agents/skills/run-e2e/resources/k8-multi-pod.md)
for provisioning, resources and completion handling.

## Entry and ownership

The seven existing GPU entries are unchanged: four DeepSeek-V2-Lite
scenario/layout pairs and three Qwen3 MoE DeepEP M2N pairs. Their pytest
entrypoints call `model_multi_pod_command()` and `run_case_entry()`:

```text
models/<model>/test_*_multi_pod.py
  -> python -m tests.e2e.models.<model>.cases multi-pod ...
  -> model-owned build_multi_pod_case()
  -> multi_pod/runner.case_main() / run_case()
       -> layout.py: logical topology -> this pod's RoleSlots
       -> rendezvous.py: TCPStore barriers, abort and verdict
       -> deployment.py: role commands, local start/readiness/cleanup
       -> declared Task: workload and result validation on the evaluator
```

For example, invoke the following in each of the two pre-existing pods, with
its own output directory and device visibility:

```bash
python -m tests.e2e.models.deepseek_v2_lite.cases multi-pod \
  --scenario afd-graph-2a2f --pod-layout 2A0F,0A2F \
  --run-id example-run --model /models/DeepSeek-V2-Lite \
  --device-backend gpu --output-path /work/results \
  --store-host afd-e2e-0.afd-e2e
```

The original NPU DSV4 multi-pod entries still use the retained
`python -m tests.e2e.multi_pod.runner` CLI and NPU launch helpers. The shared
coordination sequence is the same. NPU migration is separate work.

## Identity, placement and resources

`resolve_pod_index()` uses `AFD_E2E_POD_INDEX`, then `JOB_COMPLETION_INDEX`,
then a trailing index in `HOSTNAME`. Pod 0 owns the TCPStore; the Attention
leader owns evaluation. These need not be the same pod.

A layout such as `2A0F,0A2F` is independent of the Case's logical topology.
Per-role totals must match the Case, and each pod must hold a whole number of
TP groups: a TP group cannot span pods. Qwen3 DeepEP M2N retains its declared
role-split layout; its FFN-first world ordering and contiguous NVLink domains
are connector requirements, not a new placement capability of this migration.
Baseline placement across multiple pods remains unsupported.

`plan()` derives every pod's plan from the same topology, layout and addresses.
Each pod uses only its own `RoleSlot`s. A role spanning pods receives local DP
size, rendezvous address and RPC port. Only headless peers receive the DP start
rank; emitting it on the leader selects a different vLLM load-balancing mode
that rejects headless peers. Headless commands do not reserve HTTP ports.

Slot devices are positions within the pod's allocated device list. `--devices`
or `AFD_E2E_DEVICES` can bind that list explicitly; otherwise the GPU CLI uses
inherited `CUDA_VISIBLE_DEVICES`, or the pod's local ordinal list when unset.
The service launcher partitions that list by role.

Address resolution uses, in order:

1. `--pod-addresses` or `AFD_E2E_POD_ADDRESSES`;
2. `--pod-address-template` with an `{index}` field, without an exchange barrier;
3. each pod's published address and an address-exchange barrier.

The advertised local address prefers `HCCL_IF_IP`, then `POD_IP`, then hostname.
DNS resolution of the store host is bounded before TCPStore connection.
Pod 0 publishes the scenario ID and canonical layout; peers reject disagreement.
GPU execution binds this pod's allocated resources and cleans only its launched
groups. A different run marker alone does not prove stale ownership or block a
run on a shared host. The retained NPU path keeps its existing stale-run check
for its isolated pod environment; it does not clean up another run's resources.

## Coordination sequence

1. Read the model Case, validate placement, resolve identity and addresses.
2. Start local slots for the connector's first role; all pods pass its launched
   barrier before starting the second role. GPU connector order comes from
   `deployment.py`, not scenario-name branches in the coordinator.
3. Pass the launched barrier. This proves local children are alive, not ready.
4. The non-headless Attention leader waits for `/health`; other slots report
   liveness. FFN busy loops and headless workers do not expose an HTTP API.
   All pods then pass the serving barrier.
5. Exactly one evaluator executes the declared phases against its local API.
   It publishes the workload results, effective configuration and verdict.
   Peers consume those results rather than run the workload again.
6. Pass the verdict barrier and check local liveness once more.
7. Pass the teardown barrier before stopping any local role. It intentionally
   omits liveness polling: a peer may now legitimately start shutting down.
8. Clean up this pod's owned process groups, reap leaders and join log readers.
9. Non-master pods publish departure after their last store access. Pod 0 waits
   for departures with a bounded grace period, then releases the store.

Barrier and verdict waits poll peer aborts and local liveness. A local failure
publishes an abort so peers waiting on coordination can unwind. Cleanup uses
`cancellable_run()`: cancellation is deferred during cleanup and takes priority
on exit, followed by cleanup errors and the original execution error. Cleanup
status publication is best effort and cannot mask those errors.

Departure is different from arrival: an arrival barrier cannot prove that a
sleeping peer has finished its last store read. Pod 0 therefore outlives peers'
explicit departure messages; stragglers are reported after the bounded wait.
Do not automatically restart a pod into an already-advanced barrier sequence.

## Acceptance and records

Migration preserves existing GPU multi-pod acceptance:

| Suite | Workload | Threshold | Execution probes |
| --- | --- | --- | --- |
| V2 graph | 7 GSM8K samples | 0.27 | None |
| V2 graph + DBO | 24 GSM8K samples, concurrency 12 | 0.27 | None |
| Qwen3 DeepEP M2N | 100 GSM8K samples; DBO concurrency 12 | 0.27 | None |

These Cases explicitly declare no execution checks. Accuracy, local service
liveness, peer coordination and cleanup are checked; graph/DBO/Hash execution
across pods is not proved. The executor rejects a Case requesting unsupported
cross-pod execution checks instead of silently dropping them.

Each pod writes role logs and `run.json` under its output directory. The
evaluator writes task artifacts under the phase directory (currently
`evaluation/`). `AFD_E2E_GSM8K_OUTPUT` supplies that root to the pytest entry.
Records identify the local plan/devices, actual evaluator settings and results,
coverage scope, local execution and cleanup. A weakened evaluator configuration
produces a debug verdict on every pod. The CLI may complete that workload, but
`run_case_entry()` rejects it as acceptance.

The deployment passes only when every pod's pytest entry succeeds with
`acceptance_passed=true`. One pod's record does not prove peer cleanup. The
retained NPU CLI keeps its existing completion and scoped async-FFN cleanup
policy, including its longer process-group wait and marked-process cleanup.

## Extending and validating

For the migrated GPU Case path, add existing-capability combinations through
model declarations, IDs and layout selection. Do not add model/scenario branches
to shared coordination, commands, tasks or probes. Preserve the model's distributed acceptance settings explicitly;
single-deployment defaults are not automatically interchangeable.
The retained NPU executor is outside this extension contract until migrated.
GPU Case runs use deployment-owned cleanup; legacy NPU runs keep their existing
cleanup path. The coordinator does not run both for the same invocation.

`tests/e2e/test_runner.py` exercises two real runner processes and a real
TCPStore, with stub service/evaluator CLIs. These host integrations require
PyTorch for TCPStore, but do not load models or establish device correctness.
They cover role-split/interleaved placement, the Qwen workload, debug verdict
propagation, startup/evaluator failure, cancellation and owned cleanup.
Real-device validation is reported separately, including whether placement was
same-host or cross-node. This migration adds no cross-pod execution evidence.
