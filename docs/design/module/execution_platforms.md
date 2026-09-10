---
title: Execution platforms
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/compat/profiler.py"
  - "afd_plugin/compat/npu/forward_context.py"
  - "afd_plugin/compat/npu/ops.py"
  - "afd_plugin/compat/npu/profiler.py"
  - "afd_plugin/v1/worker/cuda_graph.py"
  - "afd_plugin/v1/worker/dbo.py"
  - "afd_plugin/v1/worker/npu/forward_context.py"
  - "afd_plugin/v1/worker/npu/mla_graph.py"
  - "afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py"
  - "afd_plugin/v1/worker/npu/ubatch_utils.py"
  - "afd_plugin/v1/worker/npu/ubatching.py"
  - "csrc/**"
  - "setup.py"
  - "MANIFEST.in"
related_code_paths:
  - "afd_plugin/v1/worker/{attention_metadata,attention_model_runner,attention_model_runner_v2,ffn_model_runner}.py"
  - "afd_plugin/v1/worker/npu/{attention_model_runner,attention_model_runner_v2,ffn_model_runner}.py"
  - "afd_plugin/connectors/{gpu,npu}/**"
depends_on:
  - "plugin_boundary.md"
validation_paths:
  - "tests/unit/compat/test_profiler.py"
  - "tests/unit/compat/test_ascend_ops.py"
  - "tests/unit/compat/npu/test_profiler.py"
  - "tests/unit/package/test_ascend_build_files.py"
  - "tests/unit/v1/worker/test_cuda_graph.py"
  - "tests/unit/v1/worker/test_dbo.py"
  - "tests/unit/v1/worker/test_model_runner_v2.py"
  - "tests/unit/v1/worker/test_npu_device_contract.py"
  - "tests/unit/v1/worker/test_npu_mla_graph.py"
  - "tests/unit/v1/worker/test_npu_runtime.py"
  - "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py"
  - "tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py"
upstream_refs:
  - "vLLM vllm.compilation and vllm.v1.worker V1/V2 graph/ubatching APIs"
  - "vLLM-Ascend ACL graph, forward-context, and model-runner V1/V2 APIs in the tested environment"
  - "PyTorch CUDA, torch_npu, CMake, and Ascend CANN build interfaces used by the repository"
verified_platform_refs:
  - "CUDA eager, graph, DBO, and ModelRunnerV2 E2E paths; no canonical CUDA image is recorded"
  - "Ascend E2E environment recorded in the installation and NPU guides"
related_issues:
  - "#86"
  - "#129"
last_reviewed: 2026-08-27
---

# Execution platforms

## Purpose and boundary

This document is the primary design for CUDA and NPU mechanisms: runtime
class strategy, device graphs, native DBO/ubatching, streams, forward-context
adaptation, profilers, native operators, packaging, and the tested runtime
matrix. [Attention](attention_runtime.md) and [FFN](ffn_runtime.md) retain role
lifecycle orchestration; [connector contracts](connector_contracts.md) retain
transport and topology semantics.

## Ownership and dependency direction

Platform mechanisms consume the common plugin boundary and upstream device
runtimes. They may be used by role, connector, and model modules but must not
introduce a CUDA-to-Ascend or Ascend-to-CUDA inheritance dependency.

## Runtime class strategy

CUDA and Ascend use separate internal class paths and inherit the matching
upstream runtime classes:

| Role | CUDA | Ascend |
| --- | --- | --- |
| Attention worker | `AFDAttentionWorker(Worker)` | `AFDNPUAttentionWorker(NPUWorker)` |
| Attention runner V1 | `AFDAttentionModelRunner(GPUModelRunner)` | `AFDNPUAttentionModelRunner(NPUModelRunner)` |
| Attention runner V2 | `AFDAttentionModelRunnerV2(GPUModelRunnerV2)` | `AFDNPUAttentionModelRunnerV2(NPUModelRunnerV2)` |
| FFN worker | `AFDFFNWorker(Worker)` | `AFDNPUFFNWorker(NPUWorker)` |
| FFN runner for V1/V2 pairs | plugin-owned minimal `GPUFFNModelRunner` | `AFDNPUFFNModelRunner(NPUModelRunner)` |

The NPU classes do not inherit CUDA AFD classes. Shared behavior is carried
by configuration, connector payloads, forward-context metadata, graph-policy
helpers, and small role helpers. This keeps CUDA Graph assumptions out of ACL
Graph classes and keeps vLLM-Ascend lifecycle behavior visible through its own
upstream types.

Runtime modules import their real device dependencies. CPU safety applies to
the top-level package, common configuration/validation, version checks, and
the graph policy helper; importing a CUDA or Ascend runtime module requires
the corresponding runtime stack.

## Platform mechanism map

| Mechanism | CUDA owner | NPU owner |
| --- | --- | --- |
| Worker/device lifecycle | vLLM `Worker` plus AFD role worker | vLLM-Ascend `NPUWorker` plus AFD role worker |
| Attention execution | upstream V1 or V2 `GPUModelRunner` extension | upstream V1 or V2 `NPUModelRunner` extension |
| FFN execution | plugin minimal runner | upstream `NPUModelRunner` extension |
| Graph policy/keying | `v1/worker/cuda_graph.py` | shared policy/keying plus ACL/NPUGraph integration |
| Native ubatching | `AFDUBatchWrapper` and vLLM ubatching APIs | `AscendUBatchWrapper`, Ascend contexts, streams, and slice utilities |
| Profiling | `compat/profiler.py` | `compat/npu/profiler.py` |
| Native operators | PyTorch/vLLM CUDA runtime used by NCCL P2P | plugin CANN A2E/E2A ops or external CAM async ops |
| Build/packaging | no plugin CUDA extension | `setup.py`, `csrc/npu/**`, packaged `_cann_ops_custom` vendor tree |

```mermaid
flowchart TB
    COMMON["Shared plugin boundary and contracts"]
    COMMON --> CUDA["CUDA role workers"]
    COMMON --> NPU["Ascend role workers"]
    CUDA --> GPU_RUNNERS["GPUModelRunner extension / minimal FFN runner"]
    CUDA --> CUDA_GRAPH["CUDA Graph and AFDUBatchWrapper"]
    CUDA --> NCCL["NCCL P2P transport"]
    NPU --> NPU_RUNNERS["NPUModelRunner extensions"]
    NPU --> ACL["ACL/NPUGraph and AscendUBatchWrapper"]
    NPU --> CANN["CANN A2E/E2A or external CAM operators"]
    CUDA_GRAPH --> DEVICE["Device execution"]
    GPU_RUNNERS --> DEVICE
    NCCL --> DEVICE
    ACL --> DEVICE
    NPU_RUNNERS --> DEVICE
    CANN --> DEVICE
```

## CUDA mechanisms

### Worker and device setup

Both CUDA workers delegate device and distributed setup to native
`Worker.init_device()`. During that synchronous construction window, they
temporarily replace the selected upstream runner symbol with the AFD runner
and restore it in `finally`. V1 Attention selects `AFDAttentionModelRunner`;
V2 selects `AFDAttentionModelRunnerV2`; a V2-paired FFN still selects the
minimal `GPUFFNModelRunner`. `torch.accelerator.empty_cache()` releases
startup allocations. Attention retains the matching upstream request,
KV-cache, sampling, and output behavior; FFN remains connector-driven.

### CUDA Graph policy

`validate_cuda_graph_mode()` resolves the shared AFD policy without importing
torch or vLLM at module import time. Current behavior is:

- `enforce_eager=true` disables graph execution;
- when graph execution is enabled, only vLLM `FULL_DECODE_ONLY` is accepted;
- Attention may use the upstream full-decode graph path;
- FFN owns a graph cache keyed by stage-indexed token-count metadata;
- native ubatching with graphs is accepted only for exactly two ubatches;
- other graph modes fail before runtime execution.

Attention treats DP metadata transfer as a control-plane side effect. For a
single-stage capture it sends the padded capture shape before entering formal
CUDA Graph capture. For an ubatched capture, `AFDUBatchWrapper` supplies the
exact stage slices and sends per-stage metadata before the graph body.

ModelRunnerV2 rejects ubatching and reuses the native full-graph manager. Its
Attention wrapper observes each native capture descriptor at the exact input
preparation seam, publishes a warmup and capture payload outside the graph,
and suppresses duplicate context sends. Full-graph replay bypasses context
creation, so an instance-scoped `run_fullgraph` hook publishes the padded
shape immediately before replay. The wrapper verifies descriptor order and
call count and restores all temporary symbols and state in `finally`.

FFN uses the Attention payload's warmup/capture flags. It creates a shared
CUDA graph memory pool, uses `connector.control_plane` to update the owning
connector state before `torch.cuda.graph(...)`, captures only
model/data-plane work, and stores the graph by `make_ffn_graph_key()`. A
matching future payload replays the graph; a missing key runs eagerly.

### CUDA Graphs and the async GPU connector

`GpuAsyncAFDConnector` has no control plane, so the FFN worker takes the
connector-driven branch and never reaches the graph cache above: that side
stays eager. The Attention side captures, and its dispatch sits inside the
captured region, which constrains the window's flag protocol in two ways --
a replay runs no Python, and a captured stream wait compares against the value
recorded at capture time.

- The dispatch sequence number lives in a device tensor the graph increments,
  not in a host counter, so each replay still stamps a peer's flag with a
  number it has not seen; an FFN rank recognizes an arrival exactly by that
  change.
- A reply stamps the constant `FLAG_REPLY_READY`, and the Attention side calls
  `SymmWindow.clear_flag()` inside the graph once it has consumed the slot.
  A rising reply sequence cannot work here: the wait would be frozen at one
  value and fall straight through on every later replay.
- The header prefix a dispatch copies from the host is therefore fixed per
  `(layer, stage, token count)` and cached in pinned memory, since a graph
  records the source address.

A role either polls its flags or stream-waits and clears them, never both:
`poll()` recognizes an arrival by the flag differing from what it last saw,
which a reset would defeat.

### CUDA native ubatching

`AFDUBatchWrapper` replaces vLLM's GPU wrapper during Attention model load
when native ubatching is enabled. It:

- preserves vLLM's two-ubatch execution model;
- installs stage-local `AFDForwardContextMetadata`;
- builds the stage `additional_kwargs` and DP metadata list;
- uses padded token sizes for graph coordination while retaining unpadded
  lengths for transfer semantics;
- supports the DP-size-1 decision path that upstream normally coordinates
  only across multiple DP ranks;
- rejects a split that would produce an empty first or final stage.

The current AFD policy accepts exactly two native ubatches.

### CUDA profiling

Attention and FFN runners create separate optional `torch.profiler` instances.
They are controlled by `AFD_GPU_ATTENTION_PROFILER_*` and
`AFD_GPU_FFN_PROFILER_*` environment prefixes. Each runner advances its
profiler on execution and stops it during shutdown. `VLLM_TORCH_PROFILER_DIR`
is a fallback trace directory.

## NPU mechanisms

### NPU worker and runtime setup

NPU workers apply AFD-scoped vLLM-Ascend compatibility patches before
upstream construction. During device initialization they:

1. validate Ascend-specific feature combinations;
2. apply the non-sequence-parallel all-to-all backend correction when needed,
   including legacy explicit-worker launches;
3. validate the synchronous CAMP2P ModelRunnerV2 subset when V2 is selected;
4. call `NPUWorker._init_device()`;
5. initialize the vLLM workspace manager for one or two ubatches;
6. construct the matching V1 or V2 Attention runner, or the connector-driven
   FFN runner.

The all-to-all correction selects `flashinfer_all2allv` when sequence
parallelism is disabled. Automatic worker selection receives the matching
upstream default-worker rewrite during config normalization; the worker-side
correction remains as a fallback for legacy explicit-worker launches.

### NPU forward context

Attention extends the upstream vLLM-Ascend forward flow and installs AFD data
in `ForwardContext.additional_kwargs`. FFN uses
`ascend_forward_context()` to create the minimal upstream context needed for
connector-driven MoE compute, including token counts and ACL graph runtime
mode when applicable.

When ModelRunnerV2 is selected, `ascend_forward_context()` uses native
`set_forward_context()` plus vLLM-Ascend's MRV2 profile override. This places
Ascend-specific state, `model_instance`, and AFD metadata in
`additional_kwargs`, matching the proxy layout read by the V2 runtime. V1
continues to use `set_ascend_forward_context()`.

For native ubatching, `create_ascend_forward_context()` creates one context per
stage with stage attention/DP metadata, batch descriptor, and graph mode.
Sequence-parallel intermediate tensors and DP token counts are sliced or
reassembled to match the upstream Ascend layout.

### NPU native ubatching and DBO

`AscendUBatchWrapper` is plugin-owned and deliberately separate from the CUDA
wrapper. The current path:

- supports exactly two native ubatches;
- creates one forward context and execution thread per stage;
- coordinates the threads with a barrier and paired CPU events;
- records a thread-local current NPU stream and restores the correct forward
  context after a DBO yield;
- slices input ids, positions, embeddings, intermediate tensors, and Attention
  metadata per stage;
- merges final tensors or pipeline-parallel intermediate tensors in stage
  order;
- performs TP all-gather and removes stage padding when the upstream FlashComm
  path requires it.

`v1/worker/dbo.py` registers the model-side yield operation and dispatches to
the platform DBO implementation. The optional CAM async MoE pipeline is not
this native DBO path: it is an eager, two-stage pipeline owned by the
model/connector flow. It supports request-boundary and token-balanced stages,
with real-token metadata kept separate from physical TP/SP padding. FlashComm1
is Attention-local; without it, the CAM boundary shards replicated Attention
tokens and restores the FFN result with TP all-gather.

### ACL Graph and NPU Graph

NPU Attention follows the upstream ACL graph dispatcher while adding AFD
metadata and control-plane coordination. `AscendUBatchWrapper` can capture or
replay the two-stage model path, stores `NPUGraph` entries by total token
count, and keeps per-stage contexts with the captured entry.

For MLA DBO full graphs, each stage records its own upstream `GraphParams`.
`merge_mla_graph_params()` validates identical layer order and record counts,
requires the two stages to share one FIA workspace, and merges records in
layer-major/stage-minor order. During the upstream updater call, the merged
registry is exposed only through the active forward context under
`afd_mla_graph_params`; the compatibility resolver falls back to upstream
process-global state outside that scope.

The NPU V2 runner supports eager, `FULL`, and `FULL_DECODE_ONLY`. Like CUDA V2,
it publishes descriptor-matched warmup/capture control outside formal graph
capture and installs an instance-scoped pre-replay hook because native full
replay creates no `ForwardContext`. V2 does not use `AscendUBatchWrapper` and
rejects DBO/ubatching.

The FFN runner owns a separate ACL graph cache keyed by stage token counts and
A/F topology. Warmup runs the eager FFN path. Formal capture updates connector
state through `connector.control_plane` before entering `torch.npu.graph(...)`,
so replay contains only model and data-plane operations. An unknown key falls
back to eager execution. CAM async does not enter this path because validation
requires eager execution and `connector.control_plane` is `None`.

### Ascend native operators and packaging

`CAMP2pAFDConnector` uses plugin-owned A2E/E2A CANN operators. `setup.py`
builds the Ascend extension by default when `torch_npu`, Ascend environment
variables, or the default toolkit path identifies an Ascend environment.
`AFD_BUILD_ASCEND_OPS` explicitly enables or disables that selection, and
`AFD_SKIP_ACLNN_BUILD=1` skips the preceding ACLNN vendor build when the
artifacts already exist.

The build performs the CANN vendor build under `csrc/npu`, builds the PyTorch
CMake extension, and packages `_cann_ops_custom`. Loading remains lazy:
`ensure_cam_p2p_ops_available()` updates the vendor/library environment, imports
`afd_plugin._C_ascend`, and verifies `torch.ops.afd_ascend.a2e/e2a` only when
the connector initializes. The package can therefore be imported without the
NPU extension, but the CAMP2P data path cannot run without it.

`CAMAsyncAFDConnector` instead requires `torch_npu`, `umdk_cam_op_lib`, and the
real CAM dispatch/combine operator namespace. Its loader verifies
`async_dispatch_send`, `async_dispatch_recv`, `async_combine_send`, and
`async_combine_recv` when the connector initializes.

### NPU profiling

Attention and FFN use independent optional `torch_npu.profiler` instances,
controlled by `AFD_NPU_ATTENTION_PROFILER_*` and
`AFD_NPU_FFN_PROFILER_*`. The helper configures CPU/NPU activities, Level 2
experimental output, role-specific defaults, optional stacks/modules, and a
TensorBoard trace handler. Runners step the profiler on execution and stop it
during shutdown.

## Tested runtime matrix

This table records current validation gates and repository evidence. It is not
an expansion of the supported runtime contract.

| Platform/path | Execution | Ubatching | Routing/quantization limits | Evidence |
| --- | --- | --- | --- | --- |
| CUDA V1 + `P2pNcclAFDConnector` | Eager or `FULL_DECODE_ONLY` CUDA Graph | Native DBO, exactly two ubatches | Registered CUDA model boundaries; Attention-side or FFN-side gate where the model supports it | DeepSeek-V2-Lite eager/graph/DBO accuracy E2E; model, graph, connector, and profiler unit tests |
| CUDA V2 + `P2pNcclAFDConnector` | Eager or `FULL_DECODE_ONLY` native V2 CUDA Graph | DBO and ubatching rejected | `compute_gate_on_attention=false`; PP/CP, elastic EP, EPLB, SP MoE, and compile SP rejected; role ranks equal DP x TP | DeepSeek-V2-Lite eager/graph DP2 and TP2 accuracy E2E, plus focused V2 unit tests |
| Ascend V1 + `CAMP2pAFDConnector` | Eager or current ACL Graph path | Native DBO, exactly two ubatches | Common and connector-local `compute_gate_on_attention=false`; `connector_extra_config.quant_mode=0`; plugin CANN ops required | Backend-neutral DeepSeek-V2-Lite eager/graph/DBO accuracy cases plus NPU runtime, graph, ops, connector, and profiler unit tests |
| Ascend V2 + `CAMP2pAFDConnector` | Eager, `FULL`, or `FULL_DECODE_ONLY` native V2 ACL Graph | DBO and ubatching rejected | `compute_gate_on_attention=false`; PP/CP, elastic EP, EPLB, SP MoE, and compile SP rejected; role ranks equal DP x TP | Focused runner, context, validation, and device-contract unit tests; no repository hardware E2E case |
| Ascend + `CAMAsyncAFDConnector` | Eager only | Native DBO rejected; optional AFD-managed MoE ubatching uses exactly two request or token-balanced stages | Experimental v0.26 port; `async=true`; documented path uses common `compute_gate_on_attention=true`; token mode requires Attention TP > 1; model runner v1 PCP is unsupported; prefill and decode context parallelism are unsupported; `connector_extra_config.dynamicQuant` is 0 or 1; external CAM ops required | Focused unit coverage; pre-fix DP3TP2/EP2 six-case E2E matrix; post-fix full 61-layer DP2TP8+EP16 token-split run reached `0.9522` strict match on the complete GSM8K evaluation |

All rows target vLLM 0.26.0. Hardware validation exists for CUDA V1/V2 and the
recorded Ascend V1 paths; the Ascend V2 row is an implemented, unit-tested
contract rather than a hardware-validated claim. GPU/NPU rank topology and
connector resource rules remain owned by
[connector contracts](connector_contracts.md).

The repository does not record a canonical CUDA container or a released
vLLM-Ascend v0.26 container. The NPU implementation records source commit
`80d8c194f`; environment evidence is not an authoritative package tag.

## Failure and cleanup boundaries

Unsupported graph, ubatch, async, gate, or quantization combinations fail in
configuration/worker/runner initialization. Missing native operators fail at
connector initialization rather than package import. Device graph caches and
profilers are runner-owned; process groups and communication handles are
connector-owned; workspace/device teardown remains upstream-owned after the
AFD role releases its resources.

## Candidate invariants

The following RFC candidate is non-normative while this document is draft:

- `PLAT-INV-001`: CUDA and Ascend AFD classes do not inherit from each other;
  platform extensions use matching upstream classes, while
  `GPUFFNModelRunner` remains a plugin-owned minimal runner.

No cross-connector graph invariant is recorded until evidence is verified on
both platforms.

## Upstream relationship and validation requirements

CUDA behavior is developed against the pinned vLLM release. The recorded
Ascend source snapshot and environment are compatibility evidence, not a
released package/tag pin. Build,
graph, profiler, and native-op changes require the matching unit and hardware
E2E paths listed above.

## Limitations and open issues

The official vLLM-Ascend v0.26 tag/container and canonical CUDA/Ascend versus
GPU/NPU terminology are unresolved. This document uses CUDA/Ascend for backend
mechanisms and preserves GPU/NPU where it appears in public names, environment
variables, or test markers. See
[#129](https://github.com/JiusiServe/afd-plugin/issues/129).
