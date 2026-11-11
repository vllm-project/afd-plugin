# CAM P2P Connector User Guide

## Overview

`CAMP2pAFDConnector` is the synchronous Ascend NPU connector for
Attention-FFN Disaggregation (AFD). It exchanges hidden states and FFN outputs
through CAMP2p custom operators backed by HCCL, while a Gloo control group
carries the token-count metadata needed to size each transfer.

Use this connector when Attention and FFN workers run as separate synchronous
Ascend services. Use `P2pNcclAFDConnector` for CUDA deployments and
`CAMAsyncAFDConnector` for the asynchronous Ascend path.

`CAMP2pAFDConnector` supports prefill and decode in eager mode. The V2 path
also accepts graph execution and native DBO with exactly two microbatches
and Attention DP > 1. Plain MLA graph+DBO requires `FULL_DECODE_ONLY`.
See the [0.32 validation record](https://github.com/vllm-project/afd-plugin/pull/507) for tested models,
topologies and the known graceful-exit failures.

## Prerequisites

- vLLM `0.32.0` at `5df63a72` and an Ascend PyTorch/vLLM-Ascend environment
  based on source commit
  [`bd42eea0`](https://github.com/vllm-project/vllm-ascend/commit/bd42eea01d8312c4df469fabaed45f77a12df4d3).
- The AFD Ascend custom operators must be built and available at runtime.
- HCCL connectivity for the data path and Gloo connectivity for DP metadata.
- Identical model hidden size, model dtype, AFD topology, rendezvous address,
  and DBO settings on the Attention and FFN sides.
- A free AFD rendezvous port that every participating rank can reach.

## Topology

For a `4A2F` deployment:

```text
world rank:  0   1   2   3   4   5
member:      F0  F1  A0  A1  A2  A3
mapping:     F0 <-> A0,A2
             F1 <-> A1,A3
```

`num_attention_ranks` must be greater than or equal to `num_ffn_ranks`. For
the normal balanced mapping used by CAMP2p, the Attention rank count is an
integer multiple of the FFN rank count. FFN rank `f` receives Attention ranks
`f`, `f + num_ffn_ranks`, and so on. Transfer sizes follow this strided mapping.

The connector creates these communication groups:

- one HCCL AFD process group per batch or ubatch for CAMP2p transfers;
- one FFN-only HCCL group used by the MoE path;
- one Gloo group used to send DP metadata from participating Attention ranks
  to FFN ranks.

## DBO and ubatching

Set `VLLM_USE_V2_MODEL_RUNNER=1` on both roles. Attention extends native
Ascend MRV2; synchronous FFN is an independent connector-driven runner.
There is no MRV1 execution or fallback. Without DBO, the runner accepts
eager, `FULL`, or `FULL_DECODE_ONLY`. Plain MLA graph+DBO requires
`FULL_DECODE_ONLY`; `FULL` is rejected for that backend. SFA/DSA use the
generic graph path, whose accepted modes do not establish model qualification.
DBO requires Attention DP > 1, exactly two microbatches, and no speculative
decoding. The pair requires static EP, PP=PCP=DCP=1, and each
role's ranks equal to DP x TP. Sequence parallelism follows the model
contract; synchronous DeepSeek-V2/V3 rejects effective model SP. Native MRV2
owns request slicing, capture, graph selection, replay and output merge.
AFD supplies stage-specific Ascend metadata and CAMP2P control/communication. The [NPU testing guide](../../tests/npu/README.md) separates executor
checks from model correctness evidence.

DBO is configured with vLLM CLI flags, not inside the `afd` object:

| Parameter | CAMP2p behavior |
| --- | --- |
| `--enable-dbo` | Enables native vLLM Dual Batch Overlap and splits eligible work into two ubatches. |
| `--dbo-decode-token-threshold <N>` | Minimum decode token count at which vLLM splits the batch. |
| `--dbo-prefill-token-threshold <N>` | Minimum prefill token count at which vLLM splits the batch. |
| `--ubatch-size 2` | Configures the two ubatches required by the current synchronous NPU runtime when DBO is enabled. |

Without DBO, CAMP2p uses one batch and one HCCL AFD process group. With DBO
enabled, the current runtime requires exactly two ubatches and creates one HCCL
AFD process group for each ubatch.

Attention and FFN processes must use the same DBO enablement, ubatch count, and
thresholds. Thresholds determine when splitting occurs; choose them together
with the expected workload and, when using ACL graphs, the configured graph
capture sizes.

Example DBO settings:

```bash
--enable-dbo \
--dbo-decode-token-threshold 2 \
--dbo-prefill-token-threshold 12 \
--ubatch-size 2
```

### Two-ubatch pipeline

The following pipeline is a simplified view of how the two ubatches can
overlap. The exact start and finish times depend on the workload and runtime
scheduling.

![Two-ubatch DBO pipeline](dbo.png)

1. vLLM splits an eligible batch into ubatch 0 and ubatch 1.
2. CAMP2p uses a separate HCCL AFD group for each ubatch, so the two transfers
   do not use the same communication group.
3. After Attention sends one ubatch, work on the other ubatch can overlap with
   FFN computation or communication for the first ubatch.
4. Attention receives each FFN result through the HCCL group that belongs to
   the same ubatch, then continues processing that ubatch.

## AFD configuration

Pass AFD configuration through vLLM's `--additional-config` option under the
`afd` key. The presence of the `afd` object enables AFD; omit it to disable AFD.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "CAMP2pAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 4,
    "num_ffn_ranks": 2,
    "compute_gate_on_attention": false,
    "connector_extra_config": {
      "hccl_buffer_size": 2048
    }
  }
}
```

Use the [common configuration reference](../configuration.md) for defaults
and aliases. CAMP2P requires `async_dp=false`; `compute_gate_on_attention` may be `true` on
both roles for DeepSeek-V4 at TP=1 without DBO or ubatching. For multi-machine placement, use
the FFN machine's reachable IP as `host`.

### CAMP2P `connector_extra_config`

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `core_num` | `int` | `8` | Positive default AIV core count for both roles. |
| `attn_core_num` / `ffn_core_num` | `int` | unset | Positive role-specific override for `core_num`. |
| `quant_mode` | `int` | `0` | A2E transport quantization. `0` carries a BF16 activation; `4` carries an E4M3 activation with its E8M0 MX scale. Other values are rejected. `4` requires the preconditions in [MXFP8 transport](#mxfp8-transport-quant_mode4). |
| `compute_gate_on_attention` | `bool` | `false` | Kept for backwards compatibility with older configurations and ignored for routing placement: transport derives the gate from the payload it actually carries. Omit it when deploying and set the common `afd.compute_gate_on_attention` field instead. |
| `hccl_buffer_size` | `int` | unset | Positive CAMP2P HCCL buffer size in MB. The override applies to every connector-owned `afd*` communication domain and the FFN-side `afd_moe` domain; the Gloo control group is unaffected. When unset, HCCL uses `HCCL_BUFFSIZE`, then its built-in default. Use the same value on all members of each HCCL domain. |

The per-domain setting avoids increasing buffers for unrelated TP, DP, or EP
process groups. `HCCL_BUFFSIZE` remains a process-wide fallback.

### MXFP8 transport (`quant_mode=4`)

`quant_mode=4` moves a pre-quantized activation instead of BF16. Attention
quantizes to E4M3 with `torch_npu.npu_dynamic_mx_quant` and sends the activation
together with its E8M0 MX scale (`ceil(hidden_size / 32)` bytes per token); the
FFN side hands both to the experts without dequantizing.

Construction rejects the mode unless all of the following hold on every rank:

- the common `afd.compute_gate_on_attention` is `true`, because the FFN side
  consumes the routing and never runs the gate itself;
- Attention and FFN have the same rank count, each side between 1 and 32;
- `core_num` (or the role override) is at least 3;
- the installed `a2e` operator takes `x_scale` and `quant_mode` and returns the
  received scale. The check inspects the operator schema, and an unreadable
  schema is treated as unsupported.

On the FFN side the token-ids slot is not usable in this mode: the shared-expert
and router paths still need the model dtype, so the FFN rank must receive the
Attention-side routing (`recv_routing`). Expect the scale to reach the experts in
the layout the MoE path consumes: three-dimensional
`(rows, cols // 2, 2)` for ALLGATHER, two-dimensional for MC2, whose dispatcher
forwards the scale with `quant_mode=0` instead of quantizing again.

## Single-node `4A2F` example

This example uses six NPUs on one host. It is a configuration template: replace
`/path/to/model` and add any model-specific vLLM options required by the model.

This graph + DBO template selects native ModelRunnerV2 and loads both the
Ascend and AFD plugins on each role.

The FFN service uses devices `4,5`:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_PLUGINS=ascend,afd VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/model \
  --host 127.0.0.1 \
  --port 8001 \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-batched-tokens 16 \
  --max-num-seqs 16 \
  --compilation-config '{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [16]
  }' \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --ubatch-size 2 \
  --additional-config '{
    "afd": {
      "role": "ffn",
      "connector": "CAMP2pAFDConnector",
      "host": "127.0.0.1",
      "port": 6239,
      "num_attention_ranks": 4,
      "num_ffn_ranks": 2,
      "compute_gate_on_attention": false,
      "connector_extra_config": {
        "ffn_core_num": 8,
        "hccl_buffer_size": 2048,
        "quant_mode": 0
      }
    }
  }'
```

The Attention service uses devices `0,1,2,3`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=ascend,afd VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/model \
  --host 127.0.0.1 \
  --port 8000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-batched-tokens 8 \
  --max-num-seqs 8 \
  --compilation-config '{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [8]
  }' \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --ubatch-size 2 \
  --additional-config '{
    "afd": {
      "role": "attention",
      "connector": "CAMP2pAFDConnector",
      "host": "127.0.0.1",
      "port": 6239,
      "num_attention_ranks": 4,
      "num_ffn_ranks": 2,
      "compute_gate_on_attention": false,
      "connector_extra_config": {
        "attn_core_num": 8,
        "hccl_buffer_size": 2048,
        "quant_mode": 0
      }
    }
  }'
```

## Single-node `4A4F` synchronous gate-on-Attention example (DeepSeek-V4, TP=1)

This is the supported Attention-side gate deployment: synchronous CAMP2P, DeepSeek-V4, TP=1,
BF16 transport (`quant_mode: 0`), no DBO and no ubatching, four Attention ranks on devices `0-3`
and four FFN ranks on devices `4-7`.

Set `afd.compute_gate_on_attention` to `true` on **both** roles and leave the connector-local
`compute_gate_on_attention` key out of `connector_extra_config`; that key is kept for backwards
compatibility with older configurations and does not control routing.

The FFN service uses devices `4,5,6,7` and starts first:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_PLUGINS=ascend,afd VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/DeepSeek-V4 \
  --host 127.0.0.1 \
  --port 18010 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len 9216 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 3072 \
  --additional-config '{
    "afd": {
      "role": "ffn",
      "connector": "CAMP2pAFDConnector",
      "host": "127.0.0.1",
      "port": 6239,
      "num_attention_ranks": 4,
      "num_ffn_ranks": 4,
      "compute_gate_on_attention": true,
      "connector_extra_config": {
        "quant_mode": 0
      }
    }
  }'
```

The Attention service uses devices `0,1,2,3`:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=ascend,afd VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve /path/to/DeepSeek-V4 \
  --host 127.0.0.1 \
  --port 18000 \
  --data-parallel-size 4 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len 9216 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 3072 \
  --additional-config '{
    "afd": {
      "role": "attention",
      "connector": "CAMP2pAFDConnector",
      "host": "127.0.0.1",
      "port": 6239,
      "num_attention_ranks": 4,
      "num_ffn_ranks": 4,
      "compute_gate_on_attention": true,
      "connector_extra_config": {
        "quant_mode": 0
      }
    }
  }'
```

Drop `--enforce-eager` and add `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32, 64, 128]}'` to run the same pair with ACL graphs.
The configuration is rejected when `compute_gate_on_attention=true` is combined with DBO,
ubatching, TP > 1, a connector other than `CAMP2pAFDConnector`, or a target other than DeepSeek-V4.

For model-specific launch and validation cases, see the
[E2E guide](../../tests/e2e/README.md).

For CAM operator, HCCL, and device-memory failures, see the
[NPU troubleshooting guide](TROUBLESHOOTING.md).
