# CAM P2P Connector User Guide

## Overview

`CAMP2pAFDConnector` is the synchronous Ascend NPU connector for
Attention-FFN Disaggregation (AFD). It exchanges hidden states and FFN outputs
through CAMP2p custom operators backed by HCCL, while a Gloo control group
carries the token-count metadata needed to size each transfer.

Use this connector when Attention and FFN workers run as separate synchronous
Ascend services. Use `P2pNcclAFDConnector` for CUDA deployments and
`CAMAsyncAFDConnector` for the asynchronous Ascend path.

`CAMP2pAFDConnector` supports prefill and decode in eager mode. V1 ACL graph
uses `FULL_DECODE_ONLY`; the implemented V2 path also accepts `FULL`.
V2 accepts eager native DBO with exactly two microbatches and Attention
DP > 1. V2 DBO graphs remain unsupported; other models and topologies require
their own evidence.

## Prerequisites

- vLLM `0.30.0` at `ced6857a` and an Ascend PyTorch/vLLM-Ascend environment
  based on source commit
  [`8d4409d6`](https://github.com/vllm-project/vllm-ascend/commit/8d4409d6256d8a6729140ddcc0d1889e3f96cdd6).
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
AFD supplies stage-specific Ascend metadata and CAMP2P control/communication. The [NPU testing guide](TESTING.md) separates executor
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

### Fields

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Set to `CAMP2pAFDConnector` for this synchronous NPU path. |
| `host` | `str` | `"127.0.0.1"` | Non-empty rendezvous host shared by every rank. It must be reachable from all participating processes. If the Attention and FFN ranks run on different machines, set this field to the FFN machine's reachable IP address. |
| `port` | `int` | `1239` | AFD rendezvous port in `1..65535`. It is separate from the vLLM HTTP service ports. |
| `afd_process_group_timeout_s` | `int` | `120` | Positive AFD group rendezvous timeout in seconds; also used by the native NPU DBO startup barrier. It does not bound a polling device kernel. |
| `num_attention_ranks` | `int` | `1` | Total number of Attention worker ranks. Must be positive. |
| `num_ffn_ranks` | `int` | `1` | Total number of FFN worker ranks. Must be positive. |
| `compute_gate_on_attention` | `bool` | `false` | Controls whether the MoE gate is computed on the Attention side or the FFN side. Currently only `false` is supported. |
| `connector_extra_config` | `dict` | `{}` | CAMP2P-specific settings such as role-specific core counts and `quant_mode`. Unknown fields are rejected. |
| `async` / `async_dp` | `bool` | `false` | Must remain `false` for the current synchronous; Ascend async mode requires `CAMAsyncAFDConnector`. |

Compatibility aliases are accepted for `afd_role`, `afd_connector`,
`afd_host`, `afd_port`.

The connector factory derives each process's role rank from its global DP rank
and local PCP/TP coordinates before connector construction. Do not configure a
role rank.

### CAMP2P `connector_extra_config`

| Field | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `core_num` | `int` | `8` | Positive default AIV core count for both roles. |
| `attn_core_num` / `ffn_core_num` | `int` | unset | Positive role-specific override for `core_num`. |
| `quant_mode` | `int` | `0` | CAM quantization mode; the only switch for the A2E transport. `0` transports BF16 and lets the FFN quantize. `4` transports pre-quantized E4M3/E5M2 activations plus E8M0 MX scales, so the attention side quantizes once and the A2E payload halves; the FFN restores the model dtype before its MoE runs, because the router reads the activation itself and the shared experts quantize their own input. `4` additionally requires at least 3 AIV cores and a build whose A2E operator accepts the `x_scale` / `quant_mode` arguments; the connector raises at construction when either is missing. Both roles must use the same value. Measured on A5 4A4F, prefill-dominated: +34.7% total-token throughput versus `0` when concurrency saturates all DP ranks, at about −0.15 GSM8K points. |
| `compute_gate_on_attention` | `bool` | `false` | Connector-local operator option; must remain `false`, as must the common AFD field. |
| `hccl_buffer_size` | `int` | unset | Positive CAMP2P HCCL buffer size in MB. The override applies to every connector-owned `afd*` communication domain and the FFN-side `afd_moe` domain; the Gloo control group is unaffected. When unset, HCCL uses `HCCL_BUFFSIZE`, then its built-in default. Use the same value on all members of each HCCL domain. |

The per-domain setting avoids increasing buffers for unrelated TP, DP, or EP
process groups. `HCCL_BUFFSIZE` remains a process-wide fallback.

## A5 A2E MXFP8 transport contract

The standalone A2E operator accepts optional `x_scale=None` and `quant_mode=0`
arguments after `compute_gate`. Existing positional calls remain valid. Mode
`0` retains the five-output BF16 contract and the unchanged BF16 E2A path.

Both modes use the same native `aclnnA2eGetWorkspaceSize` / `aclnnA2e`
interface. The workspace-size signature includes `xScale`, `quantMode`, and
`xScaleOut`; mode `0` passes null scale tensors. Rebuild the custom operator
library, Torch extension, and any external native callers together: existing
Python calls remain compatible, but the expanded C signature changes the native
binary ABI.

Mode `4` is an Ascend 950/A5 transport-only path, initially restricted to equal
positive Attention and FFN rank counts, with at most 64 total ranks (`E + A <= 64`).
`compute_gate` must be `0` or `1`; gate `1` requires at least three AIV cores
(`aiv_num >= 3`). The standalone sample uses four AIV cores for both gates.
The caller supplies contiguous E4M3 or E5M2 activation
storage (`torch.float8_e4m3fn` or `torch.float8_e5m2`) and separate contiguous E8M0 MX scales
(`torch.float8_e8m0fnu`), with shape `[tokens, ceil(hidden_size / 32)]`.
A2E neither quantizes nor dequantizes either tensor. The activation output keeps
the input dtype; paired ranks must use the same activation dtype. Activations and MX scales
occupy separate contiguous packet regions; the existing FP32 `scales` argument
still carries expert-routing weights when `compute_gate=1`.

Mode `4` returns six tensors: the original five outputs followed by received MX
scales. On FFN ranks, activation and scale shapes are `[batch_size, hidden_size]`
and `[batch_size, ceil(hidden_size / 32)]`; Attention ranks receive `[1, 1]`
placeholders for both. An FFN caller may instead supply an empty contiguous
`[0, hidden_size]` FP8 input and omit `x_scale`; A2E receives both payloads
from its paired Attention rank. Supplying full FFN-side activation and scale
tensors remains valid for callers that retain those buffers. Do not pass the
received FP8 tensor to BF16 E2A.
Packet capacity is checked against the runtime HCCL window before shared-window
writes, requiring
`IPC_DATA_OFFSET + (rankSize - 1) * 1024 + packetBytes <= winSize`.
Capacity violations fail closed; no universal maximum batch or hidden
size is implied by this contract.

The synchronous serving connector accepts `quant_mode=4` on this branch. The FFN
consumer restores the model dtype from the received pair before its MoE runs: the
router reads the activation itself and the shared experts quantize their own
input, so a model-dtype copy is needed regardless, and the transport is the only
stage that quantizes. The connector raises at construction when the A2E operator
does not accept the `x_scale` / `quant_mode` arguments or when fewer than 3 AIV
cores are configured, so a misconfigured deployment fails fast rather than
producing wrong results.

Measured on A5 single-host 4A4F with 128 concurrent 8192-token requests:
318,519 total token/s versus 236,378 for `quant_mode=0` (+34.7%), TTFT median
2905 ms versus 3748 ms, GSM8K 93.71% versus 93.86%. The gain depends on the
concurrency actually saturating the FFN ranks: with 8 concurrent requests the
same configuration measured −0.3%, because each layer's round-trip wait is then
dominated by its fixed latency rather than by the activation payload.

On A5 hardware, the explicit standalone byte-transport sample is:

```bash
AFD_TEST_DEVICES=2,3 AFD_TEST_QUANT_MODE=4 \
  python tests/unit/compat/npu/a2e_e2a_a5.py
```

It checks both activation and scale bytes for both FP8 formats and `compute_gate=0` and `1`, without
calling E2A. Default `AFD_TEST_QUANT_MODE=0` retains the BF16 roundtrip. Runtime
checks require real A5 devices and are not validated by compilation or Meta
tests alone.

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

For model-specific launch and validation cases, see the
[E2E guide](../../tests/e2e/README.md).

For CAM operator, HCCL, and device-memory failures, see the
[NPU troubleshooting guide](TROUBLESHOOTING.md).
