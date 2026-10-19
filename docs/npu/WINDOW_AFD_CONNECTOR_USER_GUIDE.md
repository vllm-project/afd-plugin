# Window AFD Connector User Guide

## Overview

`WindowAFDConnector` is the DeepSeek-V4 Ascend NPU data path that uses an
HCCL CommContext Window and CANN transformer operators instead of pairwise
`torch.distributed.send`/`recv`. Attention computes the MoE gate and top-k
routing, publishes fixed-capacity token records into the Window, and later
combines FFN results. FFN ranks consume ready records through a connector-driven
loop.

The connector can run in either lock-step or asynchronous data-parallel mode:

- `async=false`: every Attention rank and FFN rank advances in lock-step;
- `async=true`: each Attention rank is an independent session, and FFN batches
  whichever session/layer records are ready.

This guide describes the Window data path on `feat/dsv4-afd`. The
Atlas A5 A4F2 deployment recipe uses
[`P2pHcclAFDConnector`](HCCL_P2P_CONNECTOR_USER_GUIDE.md).

## Runtime baseline

The implementation belongs to the pinned DeepSeek-V4 vLLM 0.23 source line:

| Component | Baseline |
| --- | --- |
| vLLM | `releases/v0.23.0`, `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`, `11ee45653b199a097805b87011824a81ffa51b95` |
| afd-plugin | `feat/dsv4-afd`; record the exact commit with `git rev-parse HEAD` |

The runtime must provide compatible `torch_npu._afd` scheduling APIs,
`cann_ops_transformer` Window operators, HCCL, and the MXFP kernels required by
the model path. Record their exact versions in the image manifest and do not
mix CANN installations.

MXFP8 Window execution uses batching with `mx_output_uint8`, type-2 group
lists in `DeviceOperator.npu_grouped_matmul_gmm2`, and
`torch.ops._C_ascend.npu_swiglu_group_quant`. The multistream path uses
`torch.npu.get_stream_limit`/`set_stream_limit` in eager mode and
`torch.npu.npugraph_ex.scope.limit_core_num` during Graph capture.

## Architecture

The model is split at the routed and shared expert boundary:

```text
Attention role
  Attention / HC / residual
  MoE gate + top-k routing
       |
       | AttentionToFfn: hidden state, expert IDs, session/layer/slot
       v
HCCL CommContext Window
       |
       +--> FFN rank 0: shared expert for every decoder layer
       +--> FFN rank 1..N-1: fixed routed-expert ranges for every layer
       |
       | FfnToAttention result records
       v
AttentionWorkerCombine
  expert-scale combine / residual continuation
```

The connector allocates role-local Window memory, aligned to 2 MiB, and creates
one HCCL process group across all AFD ranks. The schedule context tracks
Attention session ID, model layer, Window slot, token ID, expert offset, and
ready state without a separate host control plane.

### Rank mapping

Window AFD uses an M-to-N topology rather than the integer-ratio subgroups used
by P2P HCCL. The communication world still places FFN ranks first:

```text
[F0, F1, ..., F(N-1), A0, A1, ..., A(M-1)]
```

Every Attention rank can publish work for every FFN rank, and every FFN rank
can return work to every Attention session. `F0` owns the single shared expert.
The remaining FFN ranks receive consecutive, balanced ranges of routed experts.
For `E` routed experts and `N` FFN ranks:

```text
routed FFN ranks = N - 1
base experts     = E // (N - 1)
remainder        = E %  (N - 1)
```

The first `remainder` routed ranks own one additional expert. Each routed FFN
rank must own at least one expert.

## Data path

For each Attention layer and Window slot:

1. Attention computes gate scores, routed expert IDs, and expert scales.
2. `attention_to_ffn` writes fixed-capacity input and routing records into the
   Window. An active mask distinguishes live rows from capacity padding.
3. Each FFN rank calls the batching operator and receives only records for its
   local shared or routed experts.
4. FFN passes the type-2 group list directly to grouped matrix multiplication
   and runs the ready expert groups.
5. FFN publishes its output with the original session, slot, token, and expert
   offsets.
6. Attention calls `npu_attention_worker_combine` and consumes the combined
   result before the same slot enters its dependent next-layer computation.

`quant_mode=4` is the default: A2F stores FP8 E4M3FN hidden states and E8M0
scales, one scale per 32 values. FFN consumes the quantized input directly;
batching exposes E8M0 scale bytes as uint8 for Graph execution.
`quant_mode=2` stores an INT8 hidden-state record plus one FP32 dynamic scale
for A2F. `quant_mode=0` stores FP16/BF16 A2F records. F2A records retain the
Attention continuation dtype.

## Lock-step and asynchronous modes

### Lock-step

With `async=false`, the connector requires global DP synchronization. A2F uses
the synchronization layer ID and notifies all FFN ranks, including ranks that
receive no token in the current transaction. FFN processes decoder layers in
model order.

### Asynchronous DP

With `async=true`, each Attention role rank becomes a distinct session and
publishes the real model layer ID. A2F notifies only FFN ranks that receive
tokens. The FFN batching operator can return work from different sessions and
layers in one transaction; the global FFN path keeps layer-major MXFP weights
and executes the ready groups without selecting Python layer modules or copying
the scheduling metadata to the host.

Asynchronous DP removes lock-step coordination between Attention ranks. It does
not make an individual request independent of its own A2F/F2A dependencies, and
it does not make FFN an HTTP service.

## U-batch and graph execution

`connector_extra_config.micro_batch_num` is a Window slot count. It must be:

- `1` when native vLLM DBO is not active;
- `2` when native DBO/U2 is active.

The connector rejects any other value or a mismatch with the runtime DBO
configuration. With async DP and two slots, Attention creates independent A2F
send and F2A receive streams. Events protect each `(layer, slot)` dependency so
the next layer consumes only its matching Combine result.

Graph execution is restricted to `FULL_DECODE_ONLY`. Graph/U2 uses the same two
Window slots and a side Attention compute stream. The existing synchronized
startup capture and live cache-miss fallback rules still apply. A Window launch
must not claim Graph/U2 readiness until both slots are observed in an actual
request and the service completes a clean restart.

## Configuration

The following shape illustrates the required connector fields; it is not a
published deployment topology:

```json
{
  "afd": {
    "role": "attention",
    "connector": "WindowAFDConnector",
    "async": true,
    "host": "127.0.0.1",
    "port": 29761,
    "num_attention_ranks": 8,
    "num_ffn_ranks": 16,
    "compute_gate_on_attention": true,
      "window_ffn_multistream_enabled": true,
    "connector_extra_config": {
      "micro_batch_num": 2,
      "quant_mode": 4
    }
  }
}
```

Use the same topology and connector options for FFN, changing only `role`.
For each service, its AFD role-rank count must equal that service's vLLM data
parallel size.

`window_ffn_multistream_enabled` defaults to `true`. For async Window U2,
Batching and expert compute run on the main stream while F2A runs on a side
stream. Communication operators use separate core limits; expert compute keeps
the full AIC budget. The three local pipeline slots are independent of the two
protocol micro-batch slots. Set this field to `false` on FFN to use serial
transactions; keep the same value in both roles' AFD configuration.

## Code-enforced limits

| Area | Current boundary |
| --- | --- |
| Model | DeepSeek-V4 on the pinned Ascend v0.23 path |
| TP | TP1; TP2 is restricted to `P2pHcclAFDConnector` |
| PP/PCP/DCP | Each must be 1 |
| Gate | `compute_gate_on_attention=true` is required |
| DP scheduling | `async` may be false or true |
| Window slots | `micro_batch_num` is 1 or 2 and must match native DBO |
| A2F quantization | `quant_mode` is 0, 2, or 4; default 4 |
| Token capacity | `max_num_batched_tokens <= 512` on each role |
| Routing | `num_experts_per_tok <= 16`; exactly one shared expert |
| FFN layout | At least two FFN ranks: one shared plus routed ranks |
| Expert placement | EPLB disabled; `mix_placement=false` |
| Sequence-parallel MoE | Not supported |
| Graph | `FULL_DECODE_ONLY`; Window Graph/U2 is code-supported |
| MTP | Not supported with Window; MTP requires HCCL P2P |
| DSpark | Speculative config belongs to Attention only; no Window release claim |
| PD | `MooncakeHybridConnector` on Attention as `kv_consumer`; A5 validation pending |

Window + PD is code-supported on the pinned runtime, including DSpark's
asynchronous KV allocation and per-layer KV layout negotiation. It does not
establish an A5 validation result for Window + PD, DSpark, or Graph/U2. Keep the
existing Window TP1, gate, rank, and capacity limits above. Before deployment,
verify the first token after remote Prefill, continued Decode, real U2 stages
and Graph replay where enabled, DSpark drafted/accepted counts, and normal
shutdown. Existing P2pHccl validation results do not certify this combination.

The global Window FFN additionally requires all decoder layers on each FFN rank,
W4A8MXFP routed experts, W8A8MXFP8 shared experts, and supported group counts.
The packed global weight path rejects more than 1024 layer/expert groups, while
the batching operator rejects more than 8192 ready expert groups. For example,
a 43-layer checkpoint with 256 routed experts can place no more than 23 routed
experts on one FFN rank under the 1024-group limit, so it needs at least 12
routed FFN ranks plus the dedicated shared-expert rank. Derive this constraint
from the actual checkpoint before selecting a topology.

## Profiler rank selection

When the corresponding `AFD_NPU_*_PROFILER_ENABLE=1` switch is set, the plugin
profiler is role-aware and defaults to rank 0 for both Attention and FFN. Set
`AFD_NPU_ATTENTION_PROFILER_RANKS` or
`AFD_NPU_FFN_PROFILER_RANKS` to a comma-separated non-negative rank list, or to
`all`. Keep stack collection disabled for performance traces unless Python
stacks are explicitly required.

Profiling starts on an explicit vLLM profile request and stops on the matching
stop request or runner shutdown. Setting the enable environment variable alone
does not start a trace at worker construction. FFN continues processing utility
requests while its workers run the connector loop.

## Startup and readiness

Create a launch recipe only after the target image passes its own dependency,
checkpoint, topology, and operator probes. Start FFN and Attention launchers
back-to-back because HCCL process-group creation waits for the complete Window
world.

Attention remains the only client API. FFN readiness must be derived from every
FFN worker entering its connector-driven loop and from one `[Window][init]`
record per configured role rank. Verify the log fields for role rank, world
rank, A/F sizes, slot count, capacity, expert count, FFN kind, and Window size.

Stop Attention before FFN and treat either role's unexpected exit as a failure
of the complete service. A restart must recreate the entire Window world; do
not restart an individual rank in place.

## Troubleshooting

### Import or initialization failure

Verify that `cann_ops_transformer`, `torch_npu._afd`, the HCCL communicator-name
API, and the expected MXFP methods come from the pinned image. A missing symbol
or mixed CANN path is a runtime compatibility problem, not a rank-mapping issue.

### Configuration fails before model loading

Check TP1, role rank count versus DP size, Attention-side gate, EPLB,
`mix_placement`, slot count, token capacity, top-k width, and the number of
shared experts. Do not relax these checks in a deployment script.

### HCCL rendezvous stalls

Both roles must use identical A/F counts, rendezvous host/port, async mode, slot
count, and quant mode. Confirm all processes were started back-to-back and no
stale process owns the rendezvous port.

### FFN weight initialization fails

Confirm that routed weights are W4A8MXFP, shared weights are W8A8MXFP8, every
decoder layer is present, and the fixed expert layout fits the group limits.
Do not substitute a generic checkpoint by editing only `config.json`.

### Attention stalls on Combine

Inspect the first Window scheduling or FFN failure. Check that the same slot is
not reused before its prior Combine dependency is consumed, all routed ranks
remain alive, and `micro_batch_num` matches native DBO on both roles.
