# MRV2 Async CAM FFN eager/FULL runtime

FULL captures one routed-expert transaction: dispatch receive (DR),
W4A8/W8A8 FFN, and combine send (CS). Attention remains MRV2 eager.

`AsyncNPUFFNModelRunner` owns Async CAM model loading, eager/FULL transactions
and receive-safe resource release independently of the synchronous CAMP2P
`AFDNPUFFNModelRunner`. The FFN worker selects the owner from the connector.

## Selection and startup

The target is the pinned vLLM 0.32 / vLLM-Ascend stack on A3/910C, with
homogeneous DeepSeek-V4 W4A8 or W8A8 layers, dynamicQuant=1, static expert placement,
F TP=1 and Attention-side gating. Both roles select
`VLLM_USE_V2_MODEL_RUNNER=1`. Attention keeps `--enforce-eager`.
Attention may enable native DBO under the
existing two-microbatch constraints. Enable native DBO only on Attention; FFN
consumes CAM work items without joining Attention's DBO thread schedule.
The [0.32 validation record](https://github.com/vllm-project/afd-plugin/pull/507) identifies the tested
W4A8 eager/FULL fixture and its limits, including graceful-exit failures.

On F, omit `--enforce-eager` and pass
`--compilation-config '{"cudagraph_mode":"FULL"}'`. To select eager explicitly,
use `--enforce-eager`. Unsupported FULL configurations and mismatched transport
parameters fail startup; there is no automatic eager fallback. Piecewise and
decode-only graph modes are rejected.

DeepSeek-V4 DSpark uses this same F graph for target and draft routed experts.
The combined layer table and startup agreement cover target0–42/draft43–45;
all layers must share the same W4A8 or W8A8 layout. Native P loads the complete
draft model. Target/draft Attention and the native speculative configuration
remain eager. DSpark excludes DBO; see the
[DSpark recipe](../../../recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash/README.md).

W8A8 reuses the loaded INT8 NZ weights, FP32 W13 scales and BF16 W2 scales.
It needs no W4A8 compensation. The W2 binding requires per-layer bias entries,
so the executor shares one zero INT32 bias across all layers; with 32 local
experts and hidden size 4096 this is 0.5 MiB per F rank. The executor performs
no host layer read or per-transaction weight conversion. W4A8 retains its
existing packed INT4 weights and compensation path. Mixed quantizations do
not form a single layered table.

`BATCH_SIZE_FACTOR` determines the existing native receive capacity. It must
cover the dispatched TP-group token envelope, and per-source indices must fit
the compact protocol's uint16 bound. All F ranks must agree on execution mode
and static executor layout through the rendezvous Store before HCCL creation.

FULL startup validates the configuration, runs a paired eager communication
warmup across all A/F ranks, captures one graph per F, then waits for the F
receive threads to announce readiness. Every A rank participates concurrently
in cold MC2 initialization. Output comparison belongs to tests. Startup failures
are published to peers, including F ranks that join after an A timeout.

Service readiness requires an available A API and loaded F workers with capture
and receive loops ready. The baseline F EngineCore runs a connector daemon;
F HTTP API initialization is not a readiness requirement.

## Capture boundary and ownership

By default, the runner owns capture/replay, weights, stable buffers, a private graph pool
and one execution stream. The connector owns HCCL, payload validation and
communication lifecycle coordination. Its Push session owns per-stage queues;
its Pull session owns a single window, native publish/combine, device status
and quiet/drain/stopped. The host-only `AsyncCAMGraphStartup`
coordinator owns Store negotiation, stage waits and failure propagation; it owns
no device resources or threads. A captured transaction receives activations, scales, expert counts
and the original return metadata from DR. Device-side layer selection uses
`batch_info[2]`, through the existing mapping or a one-element clone. The same
metadata goes to CS; host code does not select the layer or copy per-item inputs.

| Resource | Owner |
| --- | --- |
| DR outputs and original metadata | Runner's retained payload/context |
| Layer-index tensor, FFN intermediates and output | Runner |
| CS completion tensor | Retained transfer state |
| Weights and scale/compensation lists | Runner's model and executor |
| Captured native workspace | Graph's private allocator pool |
| HCCL group and final communication cleanup decision | Connector facade |
| Communication arguments/windows and pending routing references | Selected Push/Pull session |

Each business batch submits as many replays as there are loaded FFN layers,
then waits once for device completion. This is a bounded submission count,
not a request boundary: each DR independently selects available work. Replays
reuse stable buffers in stream order. There is no per-transaction Store poll,
host metadata read or cancellation protocol.

Changing weights, layout, capacity or topology requires fresh workers. There
is no graph cache, online weight replacement or partial-rank restart.

## Resource lifetime

With default Push transport, a host event stops further batch submissions. Graph reset and model/communicator
release are permitted only after the receive thread exits normally and device
completion is confirmed. Partial capture, partial launch, execution failure or
a still-live receive thread retains resources for process cleanup. A failure
cannot be cleared by a health check or receive-thread restart.

The original DR cannot be cancelled. An idle receive or incomplete batch can
remain blocked after requests finish; stopping the host event does not interrupt
already submitted work. Such services require process-level cleanup. This change
does not guarantee graceful idle shutdown or all A/F processes exiting with code
zero. Test reports distinguish successful inference from cleanup, and forced
termination must not be described as normal shutdown. Single-stream eager and synchronous
stopping policies retain their connector/process-level cleanup boundaries.

### Opt-in Pull execution session

`connector_extra_config.ffn_mega_moe=true` selects Pull; Push remains the default.
The connector owns its execution session and all Pull status/Store handshakes.
Attention fences pending dispatch at `attention_step()` exit. FFN keeps serving
until every Attention source is quiet, drains active work, then publishes stopped;
only then may Attention and FFN release their communication resources. Failed
handshakes retain those resources. Standalone `close()` also validates shutdown
status before release: this is an additional defensive shutdown check, not a
relocated hot-path fence.

## Computation semantics

For W4A8, FULL and layered eager forward the same model `swiglu_limit` to W13.
A positive `L` clamps gate above at `L` and up to `[-L, L]` before SiLU gating and output
quantization. Zero disables clipping. DeepSeek-V4's configured `10.0` therefore
applies on both execution paths. The limit is a host scalar fixed at graph
capture; layer selection remains controlled by the device `layer_index` tensor.
Layered and Mega W8A8 retain unclipped computation regardless of the model
`swiglu_limit`; the executors pass zero and the native W8A8 Plan skips clamp.
Native Mega W4A8 applies the limit in its MC2 ACT epilogue and bounded-column
fallback. Native W8A8 uses an INT8 GEMM with INT32 intermediate results and a
bounded-column activation; weight scales stay FP32 for W13 and BF16 for W2.
Both plans share Pull transport and the BF16 return path. W8A8 applies weight
and token scales in Vector before activation, preserving the Layered rounding
sequence. The mode is selected once through `config[11]` (1=W4A8, 2=W8A8).

The synthetic transport fixture accepts `--quantization W8A8 --swiglu-limit 0`
or `--swiglu-limit 10`; W8A8 ignores either model limit. Save matching Layered
eager outputs with `--save-outputs`,
then compare Mega eager and `--full-graph` using `--reference-outputs`; add
`--boundary` to exercise receive rounds, sparse experts and poisoned scratch.

## Validation

Use the pinned stack and native artifacts built from the same baseline sources.
The real-model fixture defaults to DeepSeek-V4 Flash per-channel W4A8 on
A3/910C, 4A/4F, A TP4, F TP1, receive capacity 128 and DBO disabled. Select
4A/8F with `--ffn-ranks 8` and twelve device IDs for a larger W8A8 checkpoint;
this topology still needs target-device validation. Synthetic per-group
coverage does not qualify loading real per-group checkpoints.

- Host contracts: `tests/unit/connectors/npu/test_async_cam.py`,
  `tests/unit/connectors/npu/test_async_cam_session.py`,
  `tests/unit/connectors/npu/test_async_cam_startup.py`,
  `tests/unit/v1/worker/test_async_ffn_graph.py`,
  `tests/unit/v1/worker/test_async_ffn_runtime.py`, and
  `tests/unit/connectors/npu/test_async_cam_startup_wiring.py`; fakes verify host ordering and
  ownership, not device execution.
- Finite native comparison: `tests/npu/async_cam_runtime.py`. Run eager with
  `--save-outputs PREFIX`, then FULL with `--full-graph --reference-outputs PREFIX`,
  keeping other arguments identical. `--boundary` covers chunking and poisoned
  unused capacity; `--attn-ranks 2` covers different Attention sources. The test
  supplies finite work counts, compares outputs and checks captured addresses.
  For Pull, add `--mega --idle-shutdown` to both eager and FULL runs so the
  fixture exercises the real worker drain/finish handshake. Compare FULL to its
  matching ON eager reference separately from the OFF-to-ON numerical check.
- Real MRV2 generation: the bounded 4A/4F fixture below. Its test observers add
  synchronization for functional evidence, so it is not a performance test.

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode eager --mega --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/on-eager-evidence
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode full --mega --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/on-full-evidence \
  --reference-dir /path/to/on-eager-evidence
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode eager --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/off-eager-evidence \
  --reference-dir /path/to/on-eager-evidence
```

The fixture saves inputs, generated tokens, actual runner identities and per-rank
execution evidence. FULL must capture once per F, replay during requests and
match eager tokens.
Batch-end metadata samples do not prove per-layer coverage.
Inference results and cleanup are recorded separately; remaining owned processes
or failed requests block acceptance. Validation evidence must identify the
source and native artifacts actually deployed.
