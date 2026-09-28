# HCCL P2P Connector User Guide

## Overview

`P2pHcclAFDConnector` is the standard PyTorch distributed connector for
Attention-FFN Disaggregation (AFD) on Ascend NPU. It transfers hidden states,
DeepSeek-V4 input IDs, and FFN results with blocking
`torch.distributed.send`/`recv` calls over HCCL. A separate Gloo control plane
carries token layout and execution-mode metadata before the FFN side posts its
data receives.

Unlike `CAMP2pAFDConnector`, this connector does not call the CAMP2P A2E/E2A
custom operators. Use it for the pinned DeepSeek-V4 vLLM 0.23 path, for
integer-ratio Attention/FFN topologies, or for the Graph/U2 implementation
described below. Continue to use the CAM connector and its own recipe for
v0.26 DeepSeek-V3.2 deployments. `WindowAFDConnector` is a different v0.23
data path with Attention-side routing and Window operators; see its
[separate guide](WINDOW_AFD_CONNECTOR_USER_GUIDE.md).

The public single-host A5 recipe is
[`recipe/npu/P2pHcclAFDConnector/deepseek_v4`](../../recipe/npu/P2pHcclAFDConnector/deepseek_v4/README.md).

## Runtime baseline

The DeepSeek-V4 A5 recipe is tied to the following source baseline:

| Component | Baseline |
| --- | --- |
| vLLM | `releases/v0.23.0`, `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`, `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-phase1-delivery`, `57bcde14cc0da248bfcb8641476b8426811a64c5` |
| Hardware | One eight-device Atlas A5 host for the A4F2 recipe |
| CANN/HCCL | Exact versions from the validated image manifest |

Do not combine CANN paths or substitute another HCCL build without repeating
the functional gates. The A5 A4F2 evidence established functionality for its
captured image/runtime combination; it did not establish accuracy or
performance.

## Architecture

AFD splits each decoder layer around its remote MoE experts:

```text
Attention role                                  FFN role

attention/shared/router
        |
        | hidden states + input IDs   HCCL
        +-----------------------------> remote routed experts
                                      gate + dispatch + experts + combine
        | FFN output                  HCCL
        <-----------------------------+
        |
remaining attention-side layer work
```

The connector owns communication only. The DeepSeek-V4 model wrapper and role
runners decide which modules are instantiated and where the handoff occurs.
The gate remains on FFN for this NPU runtime.

### Communication groups

The connector creates these groups:

- one HCCL data process group for each batch or ubatch stage;
- one HCCL input-ID group for each stage when the model is DeepSeek-V4;
- one Gloo control group for DP token counts, stage identity, graph/eager
  selection, MTP phase headers, and shutdown control;
- the role-local expert-parallel groups created by vLLM/vLLM-Ascend.

Each U2 stage has distinct HCCL groups, so stage 0 cannot consume a stage 1
message. All peers must use the same model hidden size and dtype, rank counts,
AFD host/port, execution mode, and ubatch count.

## Rank topology

The connector world always orders FFN ranks before Attention ranks:

```text
[F0, F1, ..., A0, A1, ...]
```

One role count must be an integer multiple of the other. The larger side is
partitioned into consecutive subgroups, with one rank from the smaller side in
each subgroup. `P2pHcclAFDConnector` supports both directions:

- `A = k * F`: each FFN rank aggregates `k` Attention peers and splits the FFN
  output back to those peers;
- `F = k * A`: each Attention rank fans out to `k` FFN peers and combines the
  returned data according to the connector contract;
- `A = F`: one-to-one communication.

Other P2P connectors do not accept `A < F`.

### A4F2 mapping

The single-host A5 recipe uses four Attention ranks and two FFN ranks:

```text
AFD world rank:   0    1    2    3    4    5
role rank:       F0   F1   A0   A1   A2   A3

subgroup 0:      F0 <----> A0, A1     world ranks (0, 2, 3)
subgroup 1:      F1 <----> A2, A3     world ranks (1, 4, 5)
```

With TP1, the role rank is the DP rank. In the A4F2 recipe:

| Role | Physical devices | DP | TP | AFD role ranks |
| --- | --- | ---: | ---: | --- |
| Attention | NPU 0,1,2,3 | 4 | 1 | A0-A3 |
| FFN | NPU 4,5 | 2 | 1 | F0-F1 |

The FFN scheduler capacity must cover one complete Attention subgroup. With an
Attention limit of 4096 tokens and ratio 2:1, the FFN limit is therefore 8192
tokens.

## Transfer protocol

For each target decoder stage:

1. Attention prepares the DP token layout and input IDs.
2. The control plane sends the stage header and exact per-peer token counts.
3. FFN sizes its aggregate buffers and posts receives for its subgroup.
4. Attention sends each peer's hidden states and the required DeepSeek-V4 IDs.
5. FFN concatenates the subgroup input, computes remote MoE, and partitions
   the output back to the original peer slices.
6. Attention receives the matching FFN output and continues the decoder layer.

The public API is synchronous. Eager paths still use blocking `send`/`recv`;
there are no background transfer threads and no `isend`/`irecv` API contract.
NPU streams and events change device-side ordering without changing the wire
protocol.

## Execution modes

### Eager U1

Eager U1 creates one stage and performs each layer's A2F/FFN/F2A exchange in
order. Configure both roles with:

```bash
export EXECUTION_MODE=eager
export U_BATCHES=1
```

The recipe maps this to `--enforce-eager` without DBO.

### Eager U2

The implementation supports eager U2. It splits eligible work into exactly two
request-boundary stages and uses connector-owned send, receive, and compute
streams with events. The DeepSeek-V4 wrapper submits work from one host thread
in layer-major order:

```text
layer 0: stage 0 -> stage 1
layer 1: stage 0 -> stage 1
...
```

The next layer consumes a stage only after the prior layer's F2A receive event
for that stage is complete. If a request-boundary split cannot create two
non-empty stages, execution can fall back to U1 for that step.

Eager U2 is a code-supported mode, but it is not the public A4F2 default. The
checked-in A5 recipe exposes eager/U1 and the validated Graph/U2 combination.

### FULL_DECODE_ONLY Graph U2

Graph execution is limited to `FULL_DECODE_ONLY`. The A5 A4F2 mode uses two
stages, disables async scheduling, and enables these paths on both roles:

```text
AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1
AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1
AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
```

During graph capture, torch-npu lowers the graph-visible HCCL send/receive
operations into the ACL Graph. Input IDs and dynamically sized control metadata
remain outside the graph. Events connect Attention compute/A2F/F2A and FFN
receive/compute/send streams; FFN can release a completed stage without waiting
for unrelated later work.

Capture is synchronized between roles. A known startup shape may be captured
and replayed only when both sides select the same mode. A live shape with no
matching capture falls back to eager U2 for that complete target step on both
roles; one side must never capture while its peer has selected eager execution.

## Capability boundary

The table separates implementation support from the public A5 A4F2 evidence:

| Capability | Code boundary | A5 A4F2 publication boundary |
| --- | --- | --- |
| Connector | CAMP2P, HCCL P2P, or Window for DeepSeek-V4 | HCCL P2P only |
| A/F ratio | `A=F`, `A=kF`, or `F=kA` integer ratios for HCCL P2P | A4F2, fan-in 2:1 |
| TP | TP1 or TP2; TP2 requires HCCL P2P and equal A/F rank counts | TP1 |
| PP/PCP/DCP | Must each be 1 | 1 |
| Sequence-parallel MoE | Not supported | Disabled |
| Gate placement | FFN for HCCL P2P/CAMP2P; Attention for Window | FFN |
| Eager | U1 and U2 implemented | U1 is the public quick-start mode |
| Graph | `FULL_DECODE_ONLY`; Graph/U2 allows HCCL P2P or Window | HCCL P2P Graph/U2 functionally validated |
| MTP | HCCL P2P, one MTP layer, 1-3 speculative tokens | Disabled; not claimed |
| DSpark | Attention-only speculative config with checkpoint-matched block size | Disabled; not claimed |

The A4F2 Graph/U2 evidence consists of two independent cold starts in the
validated A5 environment. Batch sizes 1, 8, and 32, request cancellation and
recovery, real two-stage execution, coordinated shutdown, and NPU cleanup
passed. `golden_checked=false`; this evidence does not establish token-exact
accuracy or performance.

## Configuration

AFD is configured inside vLLM `--additional-config`:

```json
{
  "afd": {
    "role": "attention",
    "connector": "P2pHcclAFDConnector",
    "host": "127.0.0.1",
    "port": 29761,
    "num_attention_ranks": 4,
    "num_ffn_ranks": 2
  }
}
```

Use the same object on FFN with `role` changed to `ffn`. This connector rejects
non-empty `connector_extra_config`; its stream switches are environment
variables. Native DBO is configured with vLLM CLI options, not inside the AFD
object.

The A4F2 recipe exposes these primary environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | required | Native A5 DeepSeek-V4 checkpoint |
| `NIC_NAME` | required | Interface visible to Gloo/HCCL |
| `HCCL_IF_IP` | required | IPv4 address assigned to `NIC_NAME` |
| `AFD_HOST` | `127.0.0.1` | AFD rendezvous host |
| `AFD_PORT` | `29761` | Shared AFD rendezvous port |
| `ATTENTION_DEVICES` | `0,1,2,3` | Attention device set |
| `FFN_DEVICES` | `4,5` | FFN device set |
| `ATTENTION_RANKS` | `4` | Fixed Attention rank count |
| `FFN_RANKS` | `2` | Fixed FFN rank count |
| `EXECUTION_MODE` | `eager` | `eager` or `full-decode-only` |
| `U_BATCHES` | `1` | `1` for eager, `2` for graph |

The scripts reject MTP, DSpark, TP2, overlapping devices, an incompatible
model config, a mismatched vLLM version, an invalid NIC/IP pair, or occupied
role ports before starting model workers.

## Profiler rank selection

The plugin-owned NPU profiler now defaults to role rank 0 instead of creating a
profiler on every rank. Override the selection independently for each role:

```bash
export AFD_NPU_ATTENTION_PROFILER_ENABLE=1
export AFD_NPU_FFN_PROFILER_ENABLE=1
export AFD_NPU_ATTENTION_PROFILER_RANKS=0,1
export AFD_NPU_FFN_PROFILER_RANKS=0
```

Use `all` to profile every rank. Empty values, negative ranks, and non-integer
lists fail fast. Keep `AFD_NPU_ATTENTION_PROFILER_WITH_STACK=0` and
`AFD_NPU_FFN_PROFILER_WITH_STACK=0` unless Python stacks are the subject of the
experiment; profiler configuration is not a service-readiness signal.

## Startup and readiness

Start the FFN launcher and the Attention launcher back-to-back. FFN can block
while waiting for Attention and must not be awaited first. Send client traffic
only after:

- Attention `GET /v1/models` returns HTTP 200; and
- both FFN ranks log `AFD FFN EngineCore started; workers run connector loop.`

Do not probe the FFN launcher port as an HTTP health endpoint. Process liveness
alone is not readiness because model loading and graph warmup can continue for
minutes after launch.

## Shutdown

Stop Attention before FFN. Attention sends an explicit shutdown payload during
drain; keep FFN alive until both workers receive it. Then stop FFN and verify
that `npu-smi info` contains no deployment processes. The recipe launchers use
separate process groups so a launcher signal or abnormal vLLM exit cleans the
remaining processes for that role.

## Troubleshooting

### Import or operator errors

Check the active Python packages, driver, CANN, and HCCL before debugging AFD
logic. A Python import that resolves to a different source tree than the image
manifest is a runtime error. Do not source a second CANN installation.

### Rendezvous timeout

Verify both roles use identical `AFD_HOST`, `AFD_PORT`, rank counts, execution
mode, and ubatch count. Confirm FFN and Attention were launched back-to-back and
that a previous deployment does not still own the port.

### Rank or shape mismatch

For A4F2, require DP4/TP1 on Attention and DP2/TP1 on FFN. The FFN token limit
must be at least twice the Attention limit. Inspect both logs for the first
control payload or graph-key disagreement rather than relying on the final HCCL
timeout.

### Attention is healthy but requests do not finish

Attention health proves only the client-facing role is ready. Confirm both FFN
loop markers, then check for connector-loop failures, peer exits, HCCL errors,
or an execution-mode mismatch.

### FFN exits while idle

Set `HCCL_EXEC_TIMEOUT=0`, as the recipe does. An idle connector loop can wait
legitimately for the next Attention request.

### Graph/U2 failure

Confirm both roles use `FULL_DECODE_ONLY`, U2, async scheduling off, the same
capture sizes, and the same five Graph/U2 stream switches. Preserve both logs
and the exact image digest. Do not describe a fallback or a single successful
request as a completed functional gate.
