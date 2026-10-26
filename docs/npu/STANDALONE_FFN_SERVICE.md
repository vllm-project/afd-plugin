# Standalone CAM FFN service

The standalone entry point loads DeepSeek-V4 Flash per-channel ModelSlim W4A8
routed experts in an external F-only process group. After native loading,
sharding and packing complete, it retains the F-local compute environment and
native expert modules while lending their tensors to the existing Async CAM
runtime. It does not
construct a complete CausalLM, vLLM Worker, upstream ModelRunner, EngineCore,
request scheduler or KV cache.

This is a staged independent-service delivery under
[RFC #391](https://github.com/vllm-project/afd-plugin/issues/391). The megakernel
attachment remains an architectural reference for parameter ownership and
lifecycle. This revision does not implement a fused communication/compute
megakernel or qualify the attachment's broader session and bounded-drain
semantics. The earlier transport work remains available separately.

## Scope and dependencies

This entry point reuses `CAMAsyncAFDConnector`, `AsyncCAMW4A8Executor`,
`AsyncNPUFFNModelRunner`, `AsyncFFNTwoStream` and `AsyncCAMGraphStartup`. The
Attention model, router, shared experts, padding, DBO hooks and connector are
the existing CAM path. There is no FaaS-specific connector, wire ABI or
PUBLISH/PREPARE/FINISH control operator in this implementation.

The supported model is DeepSeek-V4 Flash with homogeneous per-channel W4A8
routed parameters on A3/910C, static expert parallelism and F TP=1. All main
model MoE layers are loaded; MTP layers, gates, shared experts and Attention
weights are not instantiated. The native ModelSlim factory, checkpoint loader
and quantizer own expert mapping, NZ packing, encoded scales and compensation.

Layered V4 execution requires the separate common SwiGLU clamp correctness
fix: gate is capped above, up is clamped symmetrically, and both clamps precede
SwiGLU and activation quantization. A zero limit preserves unlimited activation.
Rebuild the plugin extension and CANN package together after this API change;
an older binary cannot implement the new `swiglu_limit` argument. This
dependency applies to traditional CAM and standalone F equally. It does not
include adaptive tiling, W2 `.out` or a new transport. The original ACLNN entry
point keeps its unlimited-activation ABI; the new binding resolves a distinct
clamp-aware symbol rather than passing new arguments to an older library.

## Platform and ownership boundary

The CPU-only CLI, `FFNService` and host `FFNModelRunner` use the narrow
`FFNBackend` interface in `ffn_faas/interfaces.py`. The runner accepts a backend
factory, retains its result before native preparation, and owns the receiving
thread and failure boundary. It does not import Ascend, a V4 model or a CAM
runtime. Platform launch options and metadata are supplied by the selected
adapter; CAM readiness metadata belongs to the Ascend adapter.

`ffn_faas/platforms/npu/` contains the implemented `ascend-cam` adapter,
configuration, device/compute environment, V4 checkpoint description, native
parameter owner and CAM composition. This is a small selection seam, not a
GPU implementation, general plugin registry or synchronous ControlPlane.

Startup selects the currently qualified V4 Flash ModelSlim per-channel W4A8
adapter explicitly. The CAM backend receives a `CAMRoutedModel`, without a V4
checkpoint or model type. This resource owns native loading, local expert-map
validation, progress, storage accounting and cached-view lifetime. Its narrow
`RoutedModelAdapter` hooks provide source names and native owner construction;
`V4ModelAdapter` keeps V4 names, tensor-count checks and loading estimates in the
model integration. The original `V4RoutedWeightOwner` still constructs the same
native routed modules and delegates the same native V4 loader and quantizer.
Native module paths and V4's A-side scaling/clamp contract stay in that adapter.
CAM static-placement constraints belong to platform settings, not V4 metadata.

The CPU-only `v4_model.py` keeps `V4CheckpointDescription` and
`V4ModelAdapter` together. `v4_weights.py` holds the native parameter owner
and is imported only when loading begins. The shared `model.py` manages
loading and resource lifetime; V4 has no separate execution runtime.

Selection and loading hooks run before execution. The runtime still consumes
`get_experts_layer_indices()`, `get_async_cam_w4a8_layers()` and the existing
`W4A8LayerWeights`; no model dispatch, tensor transfer or synchronization is
added per step. The loading fence and CAM-before-compute-before-owner release
order are preserved. The unsupported legacy FFN-forward fallback still raises.
These boundaries do not qualify a second model, another quantization scheme or
arbitrary layer layouts. The existing CAM connector also continues to require
its native HF configuration fields; this change does not generalize it.

```text
CPU Service / host runner -> FFNBackend
  Ascend CAM backend
    AscendFFNEnvironment: device, native F-only compute groups, CPU affinity
    CAMRoutedModel: loading, native owner and cached views
      V4ModelAdapter / V4RoutedWeightOwner: V4 names and native loading hooks
    mature Async CAM runtime: A/F group, DS/DR/CS/CR, Graph and streams
```

`FFNBackend` here is a lifecycle adapter for that complete runtime. The public
host loop calls `backend.execute_step()`, while the mature Async runtime keeps
communication, compute, Graph and two-stream scheduling. This differs from
RFC #391's pure `compute(item)`/facts backend: a cross-platform,
cross-connector shared WorkItem/StepPlan/Graph Step Loop is not implemented.
Composing the existing runtime deliberately preserves the qualified CAM path.
A future GPU or synchronous connector integration should determine whether
and how to share that loop; an adapter interface alone does not unify it.
The environment, backend and host runner already own device/compute state,
composition and host-thread lifecycle, without another `FFNWorker` wrapper.

The environment initializes the selected device before native configuration
and establishes F-only vLLM/Ascend groups before parameter construction. Its EP
group and the original routed modules remain resident through readiness and
execution. CAM creates a distinct A/F process group; it does not require
destroying the F-only groups. The environment also reuses the native Worker's
CPU affinity policy after warmup/capture, before creating the receiving thread.
Its physical device mapping and Engram memory-migration guard follow the
pinned native API. Performance comparisons must record actual affinity rather
than infer success solely from a binding option.

After proven receiver completion, shutdown orders CAM/Graph release before
native compute-group teardown, and drops the routed owner last. Any failed or
pending native boundary retains all these owners until whole-pair isolation.
Group initialization and teardown are no longer weight-loader side effects.

Static CAM expert placement remains a constraint. Native `routed_experts`,
their expert-map manager and quantization state are retained, but EPLB, elastic
EP, redundant experts and dynamic EPLB views are rejected explicitly. Keeping
their context resident does not enable relocation or elastic membership. This
constraint belongs to the current CAM/layout integration; the removed custom
SessionSpec/window ABI is not its justification.

The parameter container declares the pinned native factory's execution-only
keyword contract and rejects unsupported execution modules. The V4 loader
adapter delegates to the original native loader without copying its body; its
marked adaptation documents the required HF config, layer range/container,
named parameters and redundant-expert count. Dependency upgrades need actual
factory/checkpoint-loading qualification, not just unchanged version strings.

Known checkpoint/topology checks run on the CPU before loading, but actual CAM
pairing still completes after native weights are loaded. A lightweight
handshake before loading, as suggested in review, remains follow-up work;
this stage retains the mature paired-startup sequence and signatures.

## Launch

Keep the existing Attention launch and its `CAMAsyncAFDConnector` configuration.
See the [connector guide](CAM_ASYNC_CONNECTOR_USER_GUIDE.md) and
[eager/FULL runtime guide](ASYNC_FFN_RUNTIME.md) for its existing contract.
It must use eager MRV2, Attention-side gating and dynamic quantization. Native
Attention DBO uses the existing integration contract; do not enable DBO on F.
A finite real-model integration run has qualified 12A/4F with A TP4/DP3 and
native DBO, plus F FULL/two-stream execution. See the validation scope below.

For a single-node 12A/4F deployment, with A using logical devices 0–11 and a
matching dispatch envelope of 2048 tokens:

```bash
# Activate the existing CANN environment and keep its Python/library paths.
# GE must see both native vendor paths before the first torch-npu import.
afd_cam_vendor=/path/to/afd_plugin/_cann_ops_custom/vendors/afd-plugin
ascend_native_vendor=/path/to/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$ascend_native_vendor:$afd_cam_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
# F-only setting: keep this unset or 0 in the Attention environment.
export AFD_ASYNC_CAM_LAYERED_GMM=1
export SOC_VERSION=910c
export VLLM_PLUGINS=ascend,afd,ascend_model,ascend_model_loader,ascend_kv_connector
# Keep BATCH_SIZE_FACTOR, HCCL/network settings and receive capacity consistent
# with the existing A launch and CAM runtime guide.
torchrun --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 \
  -m afd_plugin.ffn_faas serve /path/to/model --backend ascend-cam \
  --host 127.0.0.1 --port 29600 \
  --attention-ranks 12 --attention-tp 4 --ffn-ranks 4 \
  --devices 12,13,14,15 --max-num-batched-tokens 2048 \
  --full-graph --ffn-two-stream \
  --ffn-compute-aic-num 20 --ffn-compute-aiv-num 40 \
  --ffn-combine-aiv-num 8 --status-port 29700
```

`--devices` indexes the process's visible logical devices, not physical IDs.
When F has a restricted visibility list, use `--devices 0,1,2,3` instead. The
external launcher's `WORLD_SIZE` is F-only. CAM creates the separate
`[A0 ... A11, F0 ... F3]` world using the same host, port, group name, topology,
quantization, layer table and startup signatures as traditional F. Use a
different rendezvous port for the F-only compute world and CAM.

`--max-num-batched-tokens` must match A's dispatch envelope; it is not the
maximum request context length. F creates configuration objects for native
loading, but allocates no request KV cache. Omit `--full-graph` to use the
mature layered eager path. Two-stream execution supports both eager and FULL;
omitted core budgets use the existing CAM defaults. FULL preserves paired
warmup, private graph pools, device layer selection and FIFO slot ownership.

## Management and shutdown

Each local F rank exposes a management endpoint on `--status-port + LOCAL_RANK`.
`--status-port 0` chooses an ephemeral port. `/status` reports model loading,
CAM pairing, phase, completed submission batches and resource-release state.
These batches are not requests or complete model forwards. `/health` is host
liveness; `/ready` is available only after weights and paired CAM startup are
prepared. Neither endpoint proves progress of an outstanding device receive.

`POST /stop`, SIGINT and SIGTERM request a stop between completed batches.
The host joins its receive loop for `--shutdown-timeout` seconds. This timeout
does **not** cancel a CAM operator or prove device quiescence.
It applies after startup; native loading and rendezvous keep their existing
timeouts. An unresponsive startup also requires whole-deployment isolation.

- After a successful receiver exit and completion fence, release graphs,
  connector resources and finally the routed weight owner.
- If startup or execution fails, or a receive is still pending, report failure
  with `resources_released=false`. Do not destroy its communicator, drop
  captured pools or release weights still referenced by native work.
- The CLI retains owners until an immediate, nonzero process exit, bypassing
  Python destructors for that unsafe session. The operator must tear down the whole
  paired A/F deployment; independent hot restart or lossless bounded drain is
  not supported by the mature CAM contract.

Stop or restart policies must account for CAM's potentially unbounded DR. A
host signal, an HTTP response or a completed client request does not establish
that the next pre-submitted receive has finished.

Status output, status-file writes, traceback reporting and HTTP cleanup are
best effort. Their exceptions cannot bypass the final unsafe-session exit.
If resource release cannot be confirmed, the CLI fails closed and retains its
runner/backend owners until process isolation. Confirmed release permits
normal interpreter shutdown; diagnostic failures still return a nonzero code.

## Validation and performance reporting

The model-integration revision at `d3c6198` passes 79 CPU cases covering model
selection/rejection, source metadata, local placement, CPU-only first imports
and host ownership. This includes an independent metadata adapter contract and
12 real subprocess isolation cases. The metadata adapter is a test seam, not
evidence for a second model. CPU tests do not qualify native packing, HCCL,
Graph capture or pending device work.

Applicable lower-level evidence is reused: 24 clamp/unlimited and empty/tail
numerical cases, finite synthetic eager/FULL CAM execution, and the original
43-layer checkpoint with one A peer and four independent F processes. These
are earlier checks of the unchanged operators and execution contract, rather
than new candidate measurements. FULL/two-stream outputs matched eager exactly;
an extra transaction exercised pending-receive isolation.

The [native A fixture](../../tests/npu/standalone_cam_attention.py) drives these
independent layer inputs and can hold its communication owner for stop checks.
This fixture qualifies FFN/CAM behavior, not Attention, KV or HTTP inference.
Reuse applicable existing qualification when the execution path is unchanged.

The real-model regression at `d3c6198` used the original checkpoint, native
Attention HTTP service, A TP4/DP3 with eager MRV2 and two nonempty DBO
microbatches, and four independent F processes with FULL/two-stream execution
(20 AIC/40 AIV compute, 8 AIV combine). Prefix caching was explicitly disabled,
with a 2048-token dispatch envelope and runtime warmup. Three short warmups
preceded six frozen 14,399–14,935-token prompts at C6 and 17/31/257-token tails;
each request generated one token. All nine business token IDs, generated-token
log probabilities and top-5 log probabilities matched the existing mature CAM
reference at `d136157` exactly. That reference was reused; the mature service
was not rerun. The native binaries were also reused without rebuilding.

CPU-only functional observers confirmed real DBO on all 12 A workers and
43-layer FULL/two-stream execution on all four F workers, importing the new
`CAMRoutedModel` resource. F-only EP4 and separate CAM16 groups coexisted;
the same native routed owner, expert-map managers and quantizers remained
reachable through the failed/pending boundary, with 64 local experts per layer.

All four F ranks failed closed at stop with `resources_released=false` and
owners retained. Whole-pair isolation left no owned or NPU process. F exited
nonzero; Attention logged peer errors after successful requests and exited
zero. This proves paired-process teardown, not graceful drain, EOS or hot
restart. The finite next-token regression is not a whole-model accuracy or
throughput/TTFT benchmark. No GSM8K or performance matrix was rerun. The run
used another A3 node after resource migration; it is not mixed with the earlier
same-node performance comparison below.

Profiling-management APIs, cancellable receive, a uniform bounded-exit
contract, migration of other connectors and independent A/F dependency pins
remain separate work. The current management API exposes no profiling control.

Performance comparisons keep model/W4A8, topology, input, concurrency, output
length, explicit prefix-cache disablement and runtime warmup comparable.
Report batch elapsed time and mean/tail TTFT separately. Historical results
for the separate FaaS transport are not measurements of this backend. This
architecture change does not promise a speedup over the mature CAM backend.

## Clean whole-service prefill comparison

The candidate at `9151609` was compared with the mature `vllm serve` FFN
path on the same source/common clamp fix, native binaries, original W4A8 model
and single A3 node. Both used 12A/4F, A TP4/DP3 with native DBO, F FULL/two-stream
(20/40 compute and 8 combine), explicit prefix-cache disablement and natural
expert routing. Product CLIs loaded no functional observer or profiler.

Each measured round replayed the same 36 requests from six six-turn sessions
at C6, with one output token. Input lengths were 14,399–58,462 tokens
(median 27,951.5), totaling 1,071,401 prompt tokens per round. Frozen priorities,
dependencies, domain routing and request payloads were preserved; generated
answers did not enter later prompts. One full workload warmup per deployment
was excluded. The ordered blocks were A1/A2, B1/B2, B3/B4, A3/A4; both B blocks
shared one standalone deployment.

| Service / change | Batch (s) | Mean TTFT (s) | Mean of round p95 (s) |
| --- | ---: | ---: | ---: |
| Mature AFD | 51.301 | 7.839 | 11.814 |
| Standalone CAM F | 50.270 | 7.875 | 11.804 |
| Standalone vs mature | -2.01% | +0.46% | -0.09% |

| Round | Service | Batch (s) | Mean TTFT (s) | Round p95 TTFT (s) |
| --- | --- | ---: | ---: | ---: |
| A1 | mature | 51.838 | 7.861 | 11.817 |
| A2 | mature | 51.901 | 7.874 | 11.882 |
| B1 | standalone | 50.484 | 7.900 | 11.823 |
| B2 | standalone | 50.246 | 7.880 | 11.780 |
| B3 | standalone | 50.139 | 7.885 | 11.830 |
| B4 | standalone | 50.211 | 7.836 | 11.782 |
| A3 | mature | 50.812 | 7.810 | 11.777 |
| A4 | mature | 50.652 | 7.811 | 11.781 |

Means above average four round metrics per arm. The p95 column is the mean
of within-round 36-request linear percentiles, not a pooled p95. TTFT runs from
client send start to the first generated token; dependency/admission waits are
excluded. Batch time spans first send start to last completed stream. All
requests succeeded and reported zero cached prompt tokens.

Mature batch means drifted from 51.870 to
50.732 s between brackets
(-2.19%). The standalone batch advantage is therefore not a causal speedup
claim. All three mean metrics, and their observed cross-round upper comparison
bounds, remained below the 2% regression reference. This finite workload supports
no material regression; four correlated rounds on one node do not establish
universal statistical noninferiority. The measurement includes Attention and KV
work and must not be reported as isolated FFN latency.
