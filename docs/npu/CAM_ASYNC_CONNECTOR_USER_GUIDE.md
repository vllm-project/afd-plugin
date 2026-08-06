# CAM Async Connector User Guide

`CAMAsyncAFDConnector` is the Ascend CAM-backed asynchronous connector for AFD
Attention/FFN disaggregation. It lets Attention workers compute MoE routing and
exchange routed and shared-expert activations with independent FFN expert ranks
through CAM async dispatch/combine operators.

This guide describes the supported deployment shape, configuration contract,
rank mapping, data flow, startup requirements, and current limitations. The
[DeepSeek-V3.2 recipe](../../recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md)
contains the complete validated multi-node launch commands.

## When to use this connector

Use `CAMAsyncAFDConnector` for the currently supported asynchronous Ascend NPU
prefill path when all of the following are true:

- CAM operator packages are installed on every node;
- Attention performs MoE gating before dispatch to FFN ranks;
- execution is eager and AFD async-DP is enabled; **`async=true` is required**;
- the service is the prefill stage of a prefill/decode-disaggregated deployment;
- optional MoE ubatching is managed by AFD as two stages: request boundaries
  for PCP, or token-balanced non-PCP DP+TP/SP stages.

CAM async currently does not support decode, ACL graph execution, or vLLM
native DBO.

## CAM async data flow

CAM async removes synchronization between data-parallel replicas and avoids the
rank-wide synchronization imposed by dispatch/combine all-to-all communication
at large expert-parallel sizes. When requests are unevenly distributed across
DP replicas, each replica can continue its CAM dispatch/combine work without
waiting for every other replica to reach the same communication point. This
eliminates unnecessary idle time caused by DP request imbalance.

One MoE layer follows this sequence:

1. Attention computes top-k expert IDs and weights.
2. `async_dispatch_send` sends hidden states and routing IDs into the CAM group.
3. Each FFN rank calls `async_dispatch_recv` and receives its routed expert
   tokens, shared-expert tokens, token counts, and optional dynamic-quant scales.
4. The FFN worker executes its local routed and shared experts.
5. `async_combine_send` returns those outputs with the dispatch metadata.
6. Attention calls `async_combine_recv`; CAM routes, weights, and combines the
   expert results for the original tokens.

CAM dispatch payloads carry the token-count and routing metadata. Consequently,
this connector does not use the separate Gloo DP-metadata control plane used by
the synchronous connectors.

## Topology and rank derivation

The connector creates one HCCL world with all Attention ranks first and all FFN
ranks second:

```text
world rank:  0    1   ...  A-1   A    A+1  ...  A+F-1
member:      A0   A1  ...  A_    F0   F1   ...  F_
```

For `A = num_attention_ranks` and `F = num_ffn_ranks`:

- Attention role rank `i` has world rank `i`;
- FFN role rank `j` has world rank `A + j`;
- world size is `A + F`;
- each role rank must be unique and within its role's configured rank count.

Attention ranks are normally `DP x PCP`. `attn_ranks_per_dp` is the PCP width
and is also passed to CAM as its Attention TP width. Before connector
initialization, the connector factory resolves the effective role rank as:

```text
role_rank =
    (global_dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
```

The role rank is runtime state, not public configuration. vLLM's global DP
rank already includes `data_parallel_start_rank`, and the connector factory
uses one shared resolver before constructing any connector.
The DeepSeek-V3.2 recipe uses `DP3PCP8 + EP8`:

```text
num_attention_ranks = 3 * 8 = 24
num_ffn_ranks = 8
attn_ranks_per_dp = 8

No role-rank field is configured.

Attention node 0, global DP ranks 0..1: effective role ranks 0..15
Attention node 1, global DP rank 2:     effective role ranks 16..23
FFN EP8 process, global DP ranks 0..7:  effective role ranks 0..7

CAM world ranks: A0..A23 = 0..23, F0..F7 = 24..31
```

FFN ranks follow expert parallel placement. The runtime derives experts per rank
from the model routed-expert count and `num_ffn_ranks`; use a model/topology in
which routed experts divide evenly across FFN ranks. All roles must use the same
model, routed-expert layout, quantization, HCCL address, rank counts, CAM
settings, and ubatching settings. In an async deployment, every Attention (A)
and FFN (F) process must also set `--max-num-batched-tokens` to the same value.

## AFD configuration

Pass AFD configuration through vLLM's `--additional-config` under the `afd`
key. The presence of the `afd` object enables AFD; omit it to disable AFD.
There is no separate `--afd-config` option.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "host": "10.0.0.1",
    "port": 6239,
    "num_attention_ranks": 24,
    "num_ffn_ranks": 8,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 8,
      "async_moe_ubatching": true,
      "async_moe_num_ubatches": 2,
      "async_moe_split": "request"
    }
  }
}
```

### Common fields

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Must be `CAMAsyncAFDConnector`. |
| `async` / `async_dp` | `bool` | `false` | Must be `true`. `async` is the accepted compatibility alias for canonical `async_dp`. |
| `host` | `str` | `"127.0.0.1"` | **Must be the IP address of the node that owns Attention rank 0.** Every rank must use the same reachable value. |
| `port` | `int` | `1239` | HCCL rendezvous port in `1..65535`; it must be free and reachable. |
| `num_attention_ranks` | `int` | `1` | Total Attention ranks, including all DP/PCP-derived ranks. |
| `num_ffn_ranks` | `int` | `1` | Total FFN expert ranks. |
| `compute_gate_on_attention` | `bool` | `false` | Must be `true`; CAM async runs MoE routing on Attention before dispatching to FFN ranks. |
| `connector_extra_config` | `dict` | `{}` | Connector-specific settings. Unknown top-level AFD fields are rejected. |

Compatibility aliases `afd_role`, `afd_connector`, `afd_host`, and `afd_port`
are also accepted. New configurations should use the canonical names shown
above, except `async`, which is retained as the documented compatibility
spelling used by the recipes.

### CAM async `connector_extra_config`

| Field | Type | Default | Meaning and constraint |
| --- | --- | --- | --- |
| `dynamicQuant` | `int` | `0` | Enables CAM dispatch/combine dynamic-quant metadata. Only `0` and `1` are accepted. With `1`, FFN receives quantized routed activations plus scale tensors and must return output compatible with combine-send. |
| `attn_ranks_per_dp` | `int` | `1` | Positive Attention rank count per DP replica (`PCP x TP`), passed to CAM as its Attention grouping width. It is independent of the FFN process's local TP size. Runtime role-rank derivation uses vLLM's DP/PCP/TP placement directly. |
| `async_moe_ubatching` | `bool` | `false` | Enables AFD-managed asynchronous MoE-only ubatching. |
| `async_moe_num_ubatches` | `int` | `2` | Number of asynchronous MoE stages. Only `2` is supported. |
| `async_moe_split` | `str` | `"request"` | `"request"` preserves request boundaries and supports PCP, plain TP, and FlashComm1/SP; it requires at least two scheduled requests. `"token"` balances the real flattened token extent for non-PCP DP+TP/SP and requires Attention TP greater than one with no context parallelism. With FlashComm1/SP, each stage gets only its minimum trailing TP padding and the parent padded layout is restored afterward. The FFN side consumes CAM work items and may independently use TP1. |

For a DP+SP deployment such as DP3TP2 Attention + DP2TP1/EP2 FFN, set
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only on Attention and explicitly set it to
`0` on FFN. Attention then holds contiguous TP-local sequence shards. For plain
DP+TP Attention, leave FlashComm1 disabled on both roles. In either case, FFN
consumes expert-routed CAM work items and may independently use TP1.

## Native DBO and async MoE ubatching are different

### vLLM native DBO

Do not pass any of these options to a CAM async process:

```bash
--enable-dbo
--dbo-decode-token-threshold <N>
--dbo-prefill-token-threshold <N>
```

They enable vLLM's native dual-batch overlap/ubatching. Runtime validation
rejects native DBO with `CAMAsyncAFDConnector`; those flags belong to supported
synchronous connector deployments.

### AFD-managed asynchronous MoE ubatching

`async_moe_ubatching` pipelines only the MoE portion of CAM async execution.
Work is divided into exactly two stages. Request splitting requires at least
two requests and keeps every request wholly within one stage. Token splitting
supports a single long request and balances the real token count even when DP
padding is much larger. With SP, the full shard layout is transposed once into
TP-local stage shards and restored once after all MoE layers. Without SP, the
model keeps its replicated TP token layout; the CAM boundary alone sends one
contiguous token shard per TP rank and all-gathers the FFN result before the
next Attention layer. If a DP replica cannot form two stages containing real
tokens, that step runs without MoE stage pipelining; Async CAM does not add a
DP synchronization for this decision. Each stage keeps its own pending
Attention routing metadata so dispatch and combine remain paired while
Attention and FFN work overlap. This does not enable vLLM native DBO and does
not use the DBO threshold flags.

For shape-only runtime diagnostics, set
`AFD_ASYNC_MOE_LAYOUT_LOG=1` on Attention. The log reports the parent token
extent, CAM-local slice, padding, and whether the FFN result is TP all-gathered.
It does not inspect tensor values or force a device synchronization.

### E2E validation matrix

The opt-in
`tests/e2e/accuracy/test_gsm8k_npu_async_cam.py::test_gsm8k_lm_eval_async_cam_dp3tp2_ep2`
test covers one parameterized DP3TP2 Attention + DP2TP1/EP2 FFN matrix:

- token and request split modes when async MoE ubatching is enabled;
- Attention FlashComm1/SP enabled and disabled;
- one non-ubatched case for each SP setting.

This produces six distinct cases. When ubatching is disabled,
`async_moe_split` is not sent to the server because it has no runtime effect.
Each case checks the configured GSM8K accuracy threshold independently; the
suite does not launch a second deployment for an automatic baseline
comparison.

Run the full matrix with:

```bash
AFD_NPU_ASYNC_CAM_RUN_DP3TP2=1 \
pytest -svv \
  tests/e2e/accuracy/test_gsm8k_npu_async_cam.py::test_gsm8k_lm_eval_async_cam_dp3tp2_ep2
```

When `async_moe_ubatching=true`, all roles must set:

```json
{
  "compute_gate_on_attention": true,
  "connector_extra_config": {
    "async_moe_ubatching": true,
    "async_moe_num_ubatches": 2,
    "async_moe_split": "request"
  }
}
```

Decode context parallel size greater than one is also rejected because the
current async MoE metadata path does not support it.

## Requirements

The checked-in recipe has been verified with:

- Ascend 910C;
- `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`;
- the included `CAM_ascend910_93_openEuler_aarch64.run` installer;
- `umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl`.

Install the CAM packages from the repository root inside the container:

```bash
bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
pip install afd_plugin/connectors/npu/bin/umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
```

Every CAM async process needs the CAM operator library on its loader path and
the Ascend plugin enabled. The complete recipe includes all tuning variables;
the essential setup is:

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

At initialization, the runtime verifies that `torch`, `torch_npu`,
`umdk_cam_op_lib`, and the four real `torch.ops.umdk_cam_op_lib` operators are
available: `async_dispatch_send`, `async_dispatch_recv`,
`async_combine_send`, and `async_combine_recv`.

## Current limitations

- Eager execution only; ACL graph mode is unsupported.
- Prefill stage only in a prefill/decode-disaggregated deployment.
- vLLM native DBO/ubatching is unsupported.
- AFD-managed MoE ubatching supports exactly two PCP request or non-PCP
  token-balanced DP+TP/SP stages.
- Decode context parallel metadata is unsupported with async MoE ubatching.
- Routed experts should divide evenly across FFN ranks.
- Other Ascend hardware, full unmodified DeepSeek-V3.2, different model
  families, CAM/CANN/container versions, cross-version combinations, and
  topologies other than the recipe should be treated as unverified.
