# MRV2 Async CAM FFN eager/FULL runtime

AFD #449 captures one complete routed-expert transaction: original dispatch
receive (DR), existing layered W4A8 FFN, and original combine send (CS).
Attention remains MRV2 eager. Issue #451 retires the separate MRV1 paths while
preserving this runtime and its original transport/lifecycle boundary.

`AsyncNPUFFNModelRunner` owns Async CAM model loading, eager/FULL transactions
and receive-safe resource release independently of the synchronous CAMP2P
`AFDNPUFFNModelRunner`. The FFN worker selects the owner from the connector;
neither runner inherits MRV1 or owns Attention request/KV state.

## Selection and startup

The target is the pinned vLLM 0.32 / vLLM-Ascend stack on A3/910C, with
homogeneous DeepSeek-V4 W4A8 layers, dynamicQuant=1, static expert placement,
F TP=1 and Attention-side gating. Both roles select
`VLLM_USE_V2_MODEL_RUNNER=1`. Attention keeps `--enforce-eager`.
Attention may enable native DBO under the
existing two-microbatch constraints. Enable native DBO only on Attention; FFN
consumes CAM work items without joining Attention's DBO thread schedule.
The experimental two-stream option below uses the same Attention DBO contract.
The [0.32 validation record](https://github.com/vllm-project/afd-plugin/pull/507) identifies the tested
W4A8 eager/FULL fixture and its limits, including graceful-exit failures.

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

By default, the runner owns capture/replay, weights, stable buffers, a private graph pool
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

## Experimental FFN two-stream execution

Set these fields in the **FFN** `afd.connector_extra_config` to opt in:

```json
{
  "ffn_two_stream": true,
  "ffn_compute_aic_num": 12,
  "ffn_compute_aiv_num": 24,
  "ffn_combine_aiv_num": 24
}
```

`ffn_two_stream` defaults to `false`. Both eager and FULL use exactly two
streams and two slots: receive plus the complete layered executor on the
compute stream, and FIFO combine-send on the send stream. Each send waits for
its matching compute-ready event; slot reuse waits for its previous send-done
event. The original batch metadata and CS return tensor remain owned by the
slot. The CS return tensor is an allocation anchor, not a completion flag.

Core counts must be positive integers. Compute Cube must fit the available
AIC budget, compute Vector must be at least twice compute Cube, and the sum
of compute/send Vector limits must fit available AIV. The send stream sets
only its Vector limit. Invalid budgets, inconsistent FFN layouts and unsupported
Attention capabilities fail through pre-HCCL Store negotiation. Configuration
hashing includes connector extras.

This option requires the layered MRV2 W4A8 prerequisites above. Keep PP/PCP/DCP
at 1, SP and FlashComm1 off on both roles, and speculative decoding off.

Attention DBO, when enabled, uses exactly two native ubatches and the existing
receive hook: the preceding raw receive is queued before the next ubatch sends
into CAM's single physical window. FFN slots remain independent work-item
buffers; they do not correspond to Attention ubatch IDs. Enable DBO only on
Attention, which remains eager, including when FFN uses FULL graph.
Streams are limited before warmup; FULL preserves
the existing matched warmup receive count. Enablement adds no startup receives.

Eager retains each payload, intermediate and output until send completion is
observed before reallocating that slot. FULL captures **four graphs**, one
compute and one send graph per slot, each with a private pool and retained
mutable addresses. Events stay outside graph capture. Both modes preserve
the existing bounded batch size of `len(layer_ids)` and one device fence at
batch completion. Four graph objects do not mean four business transactions:
FULL submits two graph replays per transaction.

Two-stream eager uses the same conservative shutdown rules as FULL: join the
receiver before releasing graph/tensor/model/communicator resources. Failed
capture, submission or device completion is terminal and retains resources.

Host protocol and lifecycle tests cover this implementation. **NPU validation
is pending**: stream limits reaching custom tiling, native same-group DR/CS
window safety, split-graph capture and actual overlap/performance have not been
qualified. Keep this experimental option off in validated deployments until
the target stack passes those checks.

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
termination must not be described as normal shutdown. Single-stream eager and synchronous
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
  `tests/unit/v1/worker/test_async_ffn_runtime.py`. CPU-safe two-stream coverage
  is in `tests/unit/v1/worker/test_async_ffn_two_stream.py` and
  `tests/unit/test_async_cam_startup_wiring.py`; fakes verify host ordering and
  ownership, not device execution.
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
match eager tokens. These repository fixtures exercise the single-stream path.
Two-stream native and real-model qualification uses independent drivers outside
`tests/e2e` and `tests/npu`, setting the FFN connector options above and recording
four captures per F in FULL mode. Preserve driver sources, launch configuration,
source hashes, inputs and raw outputs with the evidence.
Batch-end metadata samples do not prove per-layer coverage.
Inference results and cleanup are recorded separately; remaining owned processes
or failed requests block acceptance. Validation evidence must identify the
source and native artifacts actually deployed.
