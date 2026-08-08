---
title: Attention runtime
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/v1/worker/attention_model_runner.py"
  - "afd_plugin/v1/worker/attention_worker.py"
  - "afd_plugin/v1/worker/ubatch_wrapper.py"
  - "afd_plugin/v1/worker/npu/attention_model_runner.py"
  - "afd_plugin/v1/worker/npu/attention_worker.py"
related_code_paths:
  - "afd_plugin/connectors/**"
  - "afd_plugin/model_executor/**"
  - "afd_plugin/v1/worker/npu/{forward_context,npu_ubatch_wrapper,ubatch_utils,ubatching}.py"
depends_on:
  - "plugin_boundary.md"
  - "connector_contracts.md"
  - "model_integration.md"
  - "execution_platforms.md"
  - "compatibility_and_patches.md"
validation_paths:
  - "tests/unit/v1/worker/test_attention_model_runner.py"
  - "tests/unit/v1/worker/test_npu_runtime.py"
  - "tests/e2e/features/test_serving_gpu.py"
  - "tests/e2e/features/test_serving_npu.py"
  - "tests/e2e/accuracy/**"
  - "tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py"
upstream_refs:
  - "vLLM vllm.v1.worker.gpu_worker.Worker"
  - "vLLM vllm.v1.worker.gpu_model_runner.GPUModelRunner"
  - "vLLM-Ascend vllm_ascend.worker.worker.NPUWorker (tested environment evidence only)"
  - "vLLM-Ascend vllm_ascend.worker.model_runner_v1.NPUModelRunner (tested environment evidence only)"
verified_platform_refs:
  - "CUDA paths marked gpu in tests/e2e"
  - "Ascend E2E environment recorded in the installation and NPU guides"
related_issues:
  - "#86"
  - "#88"
  - "#105"
  - "#129"
last_reviewed: 2026-08-06
---

# Attention runtime

## Purpose and boundary

This document is the primary design for Attention-side request and lifecycle
orchestration. Attention owns the external API request, vLLM scheduler
execution, KV cache, sampling/output path, AFD metadata installation, and the
handoff to FFN. Backend graph, stream, profiler, native-op, and build
mechanisms belong to [execution platforms](execution_platforms.md). Connector
wire semantics and process-group ownership belong to
[connector contracts](connector_contracts.md).

## Ownership and dependency direction

Attention consumes plugin configuration, connector transport, model-side AFD
execution, platform mechanisms, and upstream compatibility. Connector and
model modules must not depend on the Attention worker implementation.

## Runtime selection

AFD does not add an AFD-specific CLI flag. Configuration is read from vLLM
`additional_config["afd"]`, and config normalization selects the role-specific
worker for the active platform when `worker_cls="auto"`.

| Platform | Worker | Model runner | Current connectors |
| --- | --- | --- | --- |
| CUDA | `afd_plugin.v1.worker.AFDAttentionWorker` | `AFDAttentionModelRunner` | `P2pNcclAFDConnector` |
| NPU | `afd_plugin.v1.worker.npu.AFDNPUAttentionWorker` | `AFDNPUAttentionModelRunner` | `CAMP2pAFDConnector`, `CAMAsyncAFDConnector` |

CUDA launch shape:

```bash
vllm serve <model> \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

Ascend launch shape:

```bash
VLLM_PLUGINS=ascend,afd vllm serve <model> \
  --additional-config '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

Operational connector options and complete launch recipes remain in the
[NCCL P2P](../../gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md),
[CAM P2P](../../npu/CAM_P2P_CONNECTOR_USER_GUIDE.md), and
[CAM async](../../npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md) guides. Requests are
sent only to the Attention API process.

## Initialization and ownership

The common Attention initialization sequence is:

```text
vLLM worker construction
  -> validate AFD config, role, connector, and selected worker class
  -> reject model runner v2 and unsupported feature combinations
  -> initialize the matching upstream device worker
  -> construct the AFD Attention model runner
  -> derive the role-local AFD rank from DP/TP ranks when needed
  -> create and initialize the configured connector
  -> load the role-aware model
```

The worker retains upstream ownership of device/distributed initialization,
model loading entry points, KV-cache allocation, memory profiling, request
execution, sleep/wake, and normal output handling. The model runner owns its
connector, AFD profiler, pending request metadata, and graph/ubatch
coordination state. It closes the connector and profiler during shutdown.

CUDA calls the native `Worker.init_device()`, then replaces the native runner
with `AFDAttentionModelRunner`. NPU applies the AFD-scoped vLLM-Ascend patches,
fixes the all-to-all backend when required, initializes the vLLM workspace
manager, and constructs `AFDNPUAttentionModelRunner` directly. Their exact
inheritance and device mechanisms are specified in
[execution platforms](execution_platforms.md).

## Request and forward flow

```text
OpenAI request
  -> vLLM scheduler and Attention worker
  -> AFD Attention model runner builds native attention inputs/metadata
  -> install plugin-owned AFD metadata in ForwardContext
  -> optionally send an AFD control payload to FFN
  -> role-aware model executes one Attention layer
  -> model sends Attention output through the connector
  -> FFN computes and returns its output
  -> model continues with the returned hidden states
  -> native vLLM sampling and output path
```

```mermaid
sequenceDiagram
    participant Client
    participant Scheduler as vLLM scheduler
    participant Attention as Attention worker/runner
    participant Model as Role-aware model
    participant ControlPlane as Optional AFDControlPlane
    participant Connector
    participant FFN as FFN daemon/runner

    Client->>Scheduler: API request
    Scheduler->>Attention: Scheduled batch
    Attention->>Attention: Build native and AFD metadata
    opt connector.control_plane is not None
        Attention->>ControlPlane: AFDControlPayload
        ControlPlane->>FFN: Stage shapes and graph flags
    end
    Attention->>Model: Forward with afd_metadata
    loop Each split layer and stage
        Model->>Connector: Attention output and transfer metadata
        Connector->>FFN: FFN work
        FFN->>FFN: Role-aware FFN compute
        FFN->>Connector: FFN result
        Connector-->>Model: Matching hidden states
    end
    Model-->>Attention: Final hidden states
    Attention-->>Scheduler: Sampling/output data
    Scheduler-->>Client: Response
```

Attention remains request-driven even when the selected connector makes FFN
connector-driven. Idle DP ranks use the upstream dummy-batch path; the CUDA
runner installs AFD metadata lazily for that path because native dummy
execution bypasses `_model_forward()`.

## Forward-context metadata

The canonical metadata location on both CUDA and Ascend is:

```python
forward_context.additional_kwargs["afd_metadata"]
```

The installed object is `AFDForwardContextMetadata`. For the current model
path it carries stage token/request starts, padded and unpadded token lengths,
stage count/index, a transaction id, and the connector reference consumed by
plugin-owned model code. These fields describe the current implementation;
the open metadata issues prevent the object shape and live connector reference
from becoming a long-term API.

The runner creates pending AFD metadata while native attention metadata is
built and installs it immediately before model forward. For native ubatching,
each child forward context receives stage-local metadata. When vLLM does not
produce `DPMetadata` for DP size 1, the runner creates `AFDDPMetadata` from the
pending stage token count. DP greater than 1 requires native DP metadata.

The previous NPU design described a separate forward-context metadata mirror.
That is no longer the contract: current CUDA and Ascend model paths read the
canonical `additional_kwargs` entry.

## Control-plane coordination

When `connector.control_plane` is not `None`, the runner sends an
`AFDControlPayload` through that `AFDControlPlane` before model/data-plane
transfer. The payload contains a stage-indexed DP metadata map plus
`is_warmup` and `is_graph_capturing` flags.

- A non-ubatched request sends stage `0` and uses native or fallback DP
  metadata.
- Native ubatching sends one DP metadata item per stage.
- A padded full-graph request sends metadata for the padded capture token
  count.
- The control plane updates its owning connector's local state before sending
  the payload.

`P2pNcclAFDConnector` and `CAMP2pAFDConnector` install
`P2pNcclAFDControlPlane` and `CAMP2pAFDControlPlane`, respectively.
`CAMAsyncAFDConnector.control_plane` remains `None`, so the Attention runner
skips DP metadata coordination; CAM dispatch payloads carry the routing and
token metadata required to drive FFN work.

## Model handoff paths

### Control-plane-driven path

The standard model path iterates layers and stages. After Attention compute,
the plugin-owned model creates transfer metadata, sends hidden states, and
receives FFN output before continuing. Role-aware construction loads the
Attention modules and the shared components required by the vLLM lifecycle,
without loading FFN expert/MLP components on this role.

### CAM asynchronous path

The documented `CAMAsyncAFDConnector` configuration uses
`compute_gate_on_attention=true`. Attention computes MoE routing data and sends
hidden states together with `topk_weights`, `topk_ids`, and optional
`router_logits`. FFN receive is delayed until the next layer when possible,
which permits CAM dispatch/combine overlap. Dense layers remain local to the
Attention model path.

The optional `async_moe_ubatching` mode is distinct from vLLM native DBO. It:

- requires `compute_gate_on_attention=true`;
- uses exactly two request-boundary or token-balanced stages;
- keeps real token ranges separate from stage-local TP/SP padding;
- runs the dense prefix before the split;
- pipelines stage Attention send and FFN receive through the MoE layers;
- rejoins stage outputs after the pipeline;
- does not synchronize stage eligibility across asynchronous DP replicas;
- does not support prefill or decode context parallel metadata.

The Attention runner calls the device-independent planner in
`model_executor/npu/async_cam_ubatching.py`; the NPU model path applies the plan
through `models/npu/async_cam_layout.py`. FlashComm1/SP is Attention-local and
FFN topology validation is independent. See the CAM async user guide for the
supported topology and environment matrix.

CAM async requires eager execution and rejects vLLM native ubatching. Its
feature limits are recorded in the
[platform matrix](execution_platforms.md#tested-runtime-matrix).

## Native ubatching and graph orchestration

When native vLLM ubatching is enabled, AFD currently accepts exactly two
ubatches. The Attention runner chooses or normalizes slices, creates
stage-local AFD/DP metadata, and installs the platform wrapper during model
load. The role-level contract is that control-plane side effects complete
before replayable model work enters formal graph capture.

During graph warmup/capture, the Attention runner marks control payloads so the
FFN daemon can warm or capture the matching shape. A non-ubatched capture sends
the control payload explicitly before formal capture; an ubatched capture lets
the platform wrapper send the exact per-stage shape. CUDA Graph, ACL Graph,
stream, and wrapper implementation details are owned by
[execution platforms](execution_platforms.md).

## Failure and cleanup behavior

Initialization fails early for a missing or role-mismatched AFD config, an
unsupported connector/feature combination, an implicit or incorrect worker
class, model runner v2, invalid native ubatch count, or unsupported graph mode.
Missing DP metadata for DP greater than 1 and missing pending metadata for a
fallback are runtime errors rather than silently guessed shapes.

Connector initialization failures remain visible from runner construction.
Shutdown stops the platform profiler and closes connector-owned resources;
the Ascend runner also delegates to upstream shutdown.

## Candidate invariants

The following RFC candidate remains non-normative while this document is
draft:

- `ROLE-INV-001` (Attention part): Attention owns external requests,
  scheduler execution, and KV cache.

The canonical metadata location is tracked by `MODEL-INV-001` in
[model integration](model_integration.md); the long-term metadata payload shape
is not fixed here.

## Upstream relationship and validation requirements

Changes must be compared with the pinned vLLM worker and model-runner symbols.
Ascend claims must cite the tested environment and existing NPU
evidence. Run the role-specific unit tests plus the affected serving, model,
and accuracy E2E paths. Control-plane, graph, DBO, or async-CAM changes require
the matching platform and connector tests as well.

## Limitations and open issues

Current shared limits are the supported vLLM release, model runner v1, exactly
two native ubatches, and role-aware DeepSeek model integration.
Platform/connector limits are intentionally centralized in
[execution platforms](execution_platforms.md#tested-runtime-matrix).

This document does not decide the runtime refactor, connector metadata
ownership, or transfer metadata/state shape. See
[#86](https://github.com/JiusiServe/afd-plugin/issues/86),
[#88](https://github.com/JiusiServe/afd-plugin/issues/88), and
[#105](https://github.com/JiusiServe/afd-plugin/issues/105).
