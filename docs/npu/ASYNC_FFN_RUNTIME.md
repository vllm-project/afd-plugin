# MRV2 Async CAM FFN FULL graph

AFD #449 captures one complete routed-expert transaction: original dispatch
receive (DR), existing layered W4A8 FFN, and original combine send (CS).
Attention remains MRV2 eager. Issue #451 retires the separate MRV1 paths while
preserving this runtime and its original transport/lifecycle boundary.

`AsyncNPUFFNModelRunner` owns Async CAM model loading, eager/FULL transactions
and receive-safe resource release independently of the synchronous CAMP2P
`AFDNPUFFNModelRunner`. The FFN worker selects the owner from the connector;
neither runner inherits MRV1 or owns Attention request/KV state.

## Selection and startup

The target is the pinned vLLM 0.30 / vLLM-Ascend stack on A3/910C, with
homogeneous DeepSeek-V4 W4A8 layers, dynamicQuant=1, static expert placement,
F TP=1 and Attention-side gating. Both roles select
`VLLM_USE_V2_MODEL_RUNNER=1`. Attention keeps `--enforce-eager`.
Attention may enable native DBO or its `async_moe_ubatching` alias under the
existing two-microbatch constraints. Enable native DBO only on Attention; FFN
consumes CAM work items without joining Attention's DBO thread schedule.

On F, set `AFD_ASYNC_CAM_LAYERED_GMM=1`, omit `--enforce-eager`, and pass
`--compilation-config '{"cudagraph_mode":"FULL"}'`. To select eager explicitly,
use `--enforce-eager`. Unsupported FULL configurations and mismatched transport
parameters fail startup; there is no automatic eager fallback. Piecewise and
decode-only graph modes are rejected.

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

The runner owns capture/replay, weights, stable buffers, a private graph pool
and one execution stream. The connector owns CAM communication. The host-only `AsyncCAMGraphStartup`
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
| HCCL group, communication arguments and windows | Connector |

Each business batch submits as many replays as there are loaded FFN layers,
then waits once for device completion. This is a bounded submission count,
not a request boundary: each DR independently selects available work. Replays
reuse stable buffers in stream order. There is no per-transaction Store poll,
host metadata read or cancellation protocol.

Changing weights, layout, capacity or topology requires fresh workers. There
is no graph cache, online weight replacement or partial-rank restart.

## Resource lifetime

A host event stops further batch submissions. Graph reset and model/communicator
release are permitted only after the receive thread exits normally and device
completion is confirmed. Partial capture, partial launch, execution failure or
a still-live receive thread retains resources for process cleanup. A failure
cannot be cleared by a health check or receive-thread restart.

The original DR cannot be cancelled. An idle receive or incomplete batch can
remain blocked after requests finish; stopping the host event does not interrupt
already submitted work. Such services require process-level cleanup. This change
does not guarantee graceful idle shutdown or all A/F processes exiting with code
zero. Test reports distinguish successful inference from cleanup, and forced
termination must not be described as normal shutdown. Eager and synchronous
stopping policies retain their connector/process-level cleanup boundaries.

## Computation semantics

FULL uses the corresponding layered eager computation without activation
clipping changes. `swiglu_limit` remains metadata; layered W13 preserves main's
existing behavior. Ordinary non-layered eager may apply the limit differently.
This change does not repair that difference, quantization behavior or checkpoint
loading. The original native operators and their metadata contracts are reused.

## Validation

Use the pinned stack and native artifacts built from the same baseline sources.
The real-model fixture targets DeepSeek-V4 Flash per-channel W4A8 on A3/910C,
4A/4F, A TP4, F TP1, receive capacity 128 and DBO disabled. Synthetic per-group
coverage does not qualify loading real per-group checkpoints.

- Host contracts: `tests/unit/connectors/test_async_cam_startup.py`,
  `tests/unit/v1/worker/test_async_ffn_graph.py`, and
  `tests/unit/v1/worker/test_async_ffn_runtime.py`.
- Finite native comparison: `tests/npu/async_cam_runtime.py`. Run eager with
  `--save-outputs PREFIX`, then FULL with `--full-graph --reference-outputs PREFIX`,
  keeping other arguments identical. `--boundary` covers chunking and poisoned
  unused capacity; `--attn-ranks 2` covers different Attention sources. The test
  supplies finite work counts, compares outputs and checks captured addresses.
- Real MRV2 generation: the bounded 4A/4F fixture below. Its test observers add
  synchronization for functional evidence, so it is not a performance test.

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode eager --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/eager-evidence
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode full --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/full-evidence \
  --reference-dir /path/to/eager-evidence
```

The fixture saves inputs, generated tokens, actual runner identities and per-rank
execution evidence. FULL must capture once per F, replay during requests and
match eager tokens. Batch-end metadata samples do not prove per-layer coverage.
Inference results and cleanup are recorded separately; remaining owned processes
or failed requests block acceptance. Validation evidence must identify the
source and native artifacts actually deployed.
