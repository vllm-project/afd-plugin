# CAM P2P Connector User Guide

## Overview

`CAMP2pAFDConnector` is the synchronous Ascend NPU connector for
Attention-FFN Disaggregation (AFD). It exchanges hidden states and FFN outputs
through CAMP2p custom operators backed by HCCL, while a Gloo control group
carries the token-count metadata needed to size each transfer.

Use this connector when Attention and FFN workers run as separate synchronous
Ascend services. Use `P2pNcclAFDConnector` for CUDA deployments and
`CAMAsyncAFDConnector` for the asynchronous Ascend path.

`CAMP2pAFDConnector` supports prefill and decode in eager mode. ACL graph use
is limited to `FULL_DECODE_ONLY`.

## Prerequisites

- vLLM `0.26.0` (v0.26 NPU baseline; the current release gates vLLM `0.28.0`,
  so this pairing is unsupported until the NPU upgrade) and an Ascend
  PyTorch/vLLM-Ascend environment based on source
  commit [`80d8c194f`](https://github.com/vllm-project/vllm-ascend/commit/80d8c194f7584b17fe08065ea99a130916f6b0e7).
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
mapping:     F0 <-> A0,A1
             F1 <-> A2,A3
```

`num_attention_ranks` must be greater than or equal to `num_ffn_ranks`. For
the normal balanced mapping used by CAMP2p, the Attention rank count is an
integer multiple of the FFN rank count. Each FFN rank handles the consecutive
Attention ranks assigned to it.

The connector creates these communication groups:

- one HCCL AFD process group per batch or ubatch for CAMP2p transfers;
- one FFN-only HCCL group used by the MoE path;
- one Gloo group used to send DP metadata from participating Attention ranks
  to FFN ranks.

## DBO and ubatching

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
| `quant_mode` | `int` | `0` | CAM quantization mode. The current runtime supports only `0`. |
| `hccl_buffer_size` | `int` | unset | Positive CAMP2P HCCL buffer size in MB. The override applies to every connector-owned `afd*` communication domain and the FFN-side `afd_moe` domain; the Gloo control group is unaffected. When unset, HCCL uses `HCCL_BUFFSIZE`, then its built-in default. Use the same value on all members of each HCCL domain. |

The per-domain setting avoids increasing buffers for unrelated TP, DP, or EP
process groups. `HCCL_BUFFSIZE` remains a process-wide fallback.

## Single-node `4A2F` example

This example uses six NPUs on one host. It is a configuration template: replace
`/path/to/model` and add any model-specific vLLM options required by the model.
The checked-in DeepSeek-V3.2 recipe linked below provides a concrete,
model-specific deployment.

The FFN service uses devices `4,5`:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V1=1 \
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
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1 \
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

For a complete DeepSeek-V3.2 deployment, see
[`recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/`](../../recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/).

For CAM operator, HCCL, and device-memory failures, see the
[NPU troubleshooting guide](TROUBLESHOOTING.md).
