# CAM Async Connector User Guide

`CAMAsyncAFDConnector` is the Ascend CAM-backed asynchronous connector for AFD
Attention/FFN disaggregation. It lets Attention workers compute MoE routing and
exchange routed and shared-expert activations with independent FFN expert ranks
through CAM async dispatch/combine operators.

This guide describes the supported deployment shape, configuration contract,
rank mapping, data flow, startup requirements, and current limitations. The
[DeepSeek-V3.2 recipe](../../recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md)
contains the historical multi-node launch commands and measurements.

> [!WARNING]
> The vLLM 0.26 upgrade did not revalidate CAM async. The linked PCP8 recipe and
> its measurements were produced with the former vLLM/vLLM-Ascend 0.19.1
> environment. vLLM-Ascend 0.26 removes PCP from model runner v1, so those
> commands are retained as historical experiment records, not as a supported
> v0.26 deployment recipe. Current v0.26 support claims cover the synchronous
> `CAMP2pAFDConnector` path.

## When to use this connector

The retained implementation describes an asynchronous Ascend NPU prefill path
with the following constraints. These are code-level constraints, not a v0.26
hardware support claim:

- CAM operator packages are installed on every node;
- Attention performs MoE gating before dispatch to FFN ranks;
- execution is eager and AFD async-DP is enabled; **`async=true` is required**;
- the service is the prefill stage of a prefill/decode-disaggregated deployment;
- optional MoE ubatching is managed by AFD as two stages: request boundaries
  or token-balanced DP+TP/SP stages.

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

Attention ranks are `DP x TP`. `attn_ranks_per_dp` must equal the Attention TP
width passed to CAM. Before connector initialization, the connector factory
resolves the effective role rank as:

```text
role_rank =
    global_dp_rank * tp_size + tp_rank
```

The role rank is runtime state, not public configuration. vLLM's global DP
rank already includes `data_parallel_start_rank`, and the connector factory
uses one shared resolver before constructing any connector.

For example, the DP3TP2 Attention + DP2TP1/EP2 test topology uses:

```text
num_attention_ranks = 3 * 2 = 6
num_ffn_ranks = 2
attn_ranks_per_dp = 2

No role-rank field is configured. Attention runtime role ranks are `0..5`, and
FFN runtime role ranks are `0..1`.

CAM world ranks: A0..A5 = 0..5, F0..F1 = 6..7
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
    "num_attention_ranks": 6,
    "num_ffn_ranks": 2,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "attn_ranks_per_dp": 2,
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
| `num_attention_ranks` | `int` | `1` | Total Attention ranks across all DP and TP groups. |
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
| `attn_ranks_per_dp` | `int` | `1` | Positive Attention TP rank count per DP replica. It is the CAM Attention grouping width and is independent of the FFN process's local TP size. |
| `async_moe_ubatching` | `bool` | `false` | Enables AFD-managed asynchronous MoE-only ubatching. |
| `async_moe_num_ubatches` | `int` | `2` | Number of asynchronous MoE stages. Only `2` is supported. |
| `async_moe_split` | `str` | `"request"` | `"request"` requires two scheduled requests and preserves their boundaries. `"token"` balances flattened real tokens and requires Attention TP greater than one. Both modes reject context parallelism; FFN may independently use TP1. |

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
supports a single long request and balances real tokens independently of parent
padding. SP uses stage-local TP shards; plain TP shards only at the CAM
boundary and all-gathers each FFN result. A DP replica that cannot form two
non-empty stages runs that step without stage pipelining. This decision is not
synchronized across DP replicas. The feature does not enable native DBO or use
DBO threshold flags.

For shape-only runtime diagnostics, set
`AFD_ASYNC_MOE_LAYOUT_LOG=1` on Attention. The log reports the parent token
extent, CAM-local slice, padding, and whether the FFN result is TP all-gathered.
It does not inspect tensor values or force a device synchronization.

### E2E validation matrix

The opt-in DP3TP2 Attention + DP2TP1/EP2 FFN test covers:

- token and request split modes when async MoE ubatching is enabled;
- Attention FlashComm1/SP enabled and disabled;
- one non-ubatched case for each SP setting.

This produces six cases. Each checks its GSM8K threshold independently; no
automatic baseline deployment is launched.

Run the full matrix with:

```bash
AFD_NPU_ASYNC_CAM_RUN_DP3TP2=1 \
pytest -svv \
  tests/e2e/accuracy/test_gsm8k_npu_async_cam.py::test_gsm8k_lm_eval_async_cam_dp3tp2_ep2
```

The reviewer-requested full DeepSeek-V3.2 accuracy deployment uses two
16-NPU nodes: Attention DP2TP8 on one node and FFN EP16 on the other. It runs
token-balanced ubatching with Attention FlashComm1/SP enabled and forced expert
load balancing disabled. Because pytest does not own a cross-node process
manager, launch both roles through the cluster job system with the checked-in
[v0.26 accuracy recipe](../../recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy/README.md),
then run:

```bash
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V3.2-W8A8 \
AFD_NPU_ASYNC_CAM_RUN_V3_2_DP2TP8_EP16=1 \
python -m pytest -svv \
  tests/e2e/accuracy/test_gsm8k_npu_async_cam.py::test_gsm8k_lm_eval_async_cam_v3_2_dp2tp8_ep16
```

The test rejects reduced checkpoints by requiring all 61 transformer layers.
Leave `AFD_GSM8K_LIMIT` unset for the full accuracy run.

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

Prefill or decode context parallel size greater than one is rejected because
the current async MoE metadata path supports DP+TP/SP, not PCP/DCP.

## Requirements

The target CAM async v0.26 NPU validation baseline is:

- Ascend 910C;
- Python 3.12;
- CANN 9.0.1;
- runtime image build `nightly-main-a3-openeuler-20260801230444_aarch64`;
- vLLM v0.26.0 at commit `568afb3a1`;
- vLLM-Ascend branch `releases/v0.26.0rc` at commit `80d8c194f`;
- the included `CAM_ascend910_93_openEuler_aarch64.run` installer;
- `umdk_cam_op_lib-209.0.0b1-cp312-cp312-linux_aarch64.whl`.

The nightly image identifier records the intended validation environment; it
is not a promise of a stable public pull tag. Some development package metadata
in that image still reports a `0.19.1rc2.dev1327` version. The source commits
above are the compatibility baseline for this port. The DP3TP2 NPU E2E matrix
and the full-model DeepSeek-V3.2 DP2TP8+EP16 accuracy test must pass on this
baseline before the v0.26 path is described as hardware validated.

Install the CAM packages from the repository root inside the container:

```bash
bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
pip install afd_plugin/connectors/npu/bin/umdk_cam_op_lib-209.0.0b1-cp312-cp312-linux_aarch64.whl
```

Every CAM async process needs the CAM operator library on its loader path and
the Ascend plugin enabled. The complete recipe includes all tuning variables;
the essential setup is:

```bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH}
export HCCL_BUFFSIZE=4096
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

For the experimental FlashComm1/SP cases, set
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only on Attention. Set it to `0` for plain TP
Attention and on every FFN process.

At initialization, the runtime verifies that `torch`, `torch_npu`,
`umdk_cam_op_lib`, and the four real `torch.ops.umdk_cam_op_lib` operators are
available: `async_dispatch_send`, `async_dispatch_recv`,
`async_combine_send`, and `async_combine_recv`.

## Current limitations

- Eager execution only; ACL graph mode is unsupported.
- Prefill stage only in a prefill/decode-disaggregated deployment.
- vLLM native DBO/ubatching is unsupported.
- AFD-managed MoE ubatching supports exactly two request-boundary or
  token-balanced DP+TP/SP stages.
- PCP is unsupported by vLLM-Ascend 0.26 model runner v1.
- Prefill and decode context parallelism are unsupported with async MoE
  ubatching.
- Routed experts should divide evenly across FFN ranks.
- Other Ascend hardware, full unmodified DeepSeek-V3.2, different model
  families, CAM/CANN/container versions, cross-version combinations, and
  topologies other than the recipe should be treated as unverified.
