# A5 DeepSeek-V4-Flash asynchronous AFD development

## Source baseline

This branch starts at AFD `update/v0.32.0`, commit `2917805`, and targets the
[#537](https://github.com/vllm-project/afd-plugin/issues/537) software environment:

| Component | Target |
| --- | --- |
| vLLM | `5df63a7203cdb9328d824a27b682e77c7b9928be` (0.32.0) |
| vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |
| CANN | 9.2.1 |
| Model | DeepSeek-V4-Flash, original FP8 checkpoint with MXFP4 routed experts |

The model target is V4 Flash, not V4.1. The existing A3 validation record uses an
older Ascend commit and does not establish A5 support. This branch removes the
DSV4 wrapper's calls to the DSA-CP interfaces deleted in the target Ascend source.
The native loader, attention, HC, hash routing and shared-expert arithmetic remain
responsible for the model semantics.

## Two execution paths

| | `CAMAsyncAFDConnector` | `CAMAsyncMegamoeAFDConnector` |
| --- | --- | --- |
| Attention | Routing and shared experts; asynchronous dispatch/combine | Same model-side ownership |
| Transfer input | BF16 (`dynamicQuant=0`) | Dispatch quantizes BF16 to MXFP8 inside the operator |
| FFN | CAM receive, native MXFP4 expert MLP, CAM send | One fused receive + expert MLP + send operator |
| Weights | Native transposed FP4 NZ tensors and E8M0 scales | Output-major FP4 C0=32 storage and E8M0 scales, prepared once after loading |
| Scheduling | Existing CAM device work selection | Bounded per-DP/per-FFN Store descriptors select layer and generation; tensors travel through MTE |

Both use `afd.async=true` and `afd.compute_gate_on_attention=true`. Attention
DPs can progress independently. Both require eager ModelRunnerV2 for this A5
path. Use `enable_fused_mc2=0` so native loading retains the tensors consumed by
these backends. The A3 layered INT4/INT8 executor is not the A5 MXFP4 executor;
leave `AFD_ASYNC_CAM_LAYERED_GMM` disabled.

For ordinary CAM, the BF16 receive scale output is a placeholder. The connector
now passes `None` to the FFN MLP, causing the native MX method to perform group-32
MXFP8 activation quantization. Passing the placeholder would incorrectly suppress
that quantization. The FFN MLP consumes the native weights directly and does not
perform another EP dispatch. Top-k scaling remains applied once by combine.

For MegaMoE, each ubatch stage owns a communication window and dispatch ticket.
Within each window, each Attention DP advances the UMDK ring sequence from 1 through 32 and back to 1.
Every FFN rank consumes the descriptor, including ranks with no selected expert.
The kernel's PUBLISHED/TAKEN/READY/CONSUMED protocol protects input/output reuse.
Each FFN rank admits at most one fused submission per window at a time.
The validated recipe uses 2A2F. Other A/F counts use DP x TP per role and contiguous static FFN EP in one
MTE supernode; no graphs, MTP or EPLB. Close the roles together after draining
requests: communication-window destruction requires the union-group barrier.

## Operator dependencies and current limits

**Plugin development is not an A5 runtime validation result.** Remote deployment,
accuracy, long-run communication and performance checks have not been performed.

- Ordinary CAM's four bundled operators now include an A5 HCCL-window path,
  AIV communication-engine selection, A5 registration and Python bindings.
  Rebuild both OPP and extension, then follow [A5 verification](A5_ASYNC_CAM_VALIDATION.md).
  This source adaptation has not yet been compiled or executed on A5.
- MegaMoE needs the Ascend950 OPP and `umdk_cam_megamoe` wheel from UMDK
  [`793d2f02`](https://gitcode.com/lqy16/umdk/commit/793d2f02c2564eda1e0f8971040f8626d89295d0),
  including the ring-sequence changes. This is separate from the ordinary CAM
  bundle. Its three Python entry points are `megamoe_dispatch_send`,
  `megamoe_fused_moe`, and `megamoe_combine_recv`.
- MegaMoE's one-time FP4 storage conversion preserves packed nibbles; its
  format-cast and CANN interpretation still need device verification. Local CPU
  checks establish byte reordering and queue behavior, not numerical support.

## Configuration contract

Supply role, rendezvous and rank counts as usual. The relevant additional-config
fields for ordinary CAM are:

```json
{
  "enable_fused_mc2": 0,
  "afd": {
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "compute_gate_on_attention": true,
    "connector_extra_config": {"dynamicQuant": 0, "attn_ranks_per_dp": 1}
  }
}
```

For MegaMoE, select `CAMAsyncMegamoeAFDConnector`, keep both booleans true,
set the same A/F counts on both roles, and optionally set `connector_extra_config.attn_ranks_per_dp` to the Attention TP width.
Each role sets DP x TP to its own configured rank count and uses static EP, BF16 model activations and eager execution. These are
configuration fragments, not complete deployment commands or verified recipes.

For Attention DP3TP2 + FFN DP2TP1/EP2, configure 6 Attention ranks and
2 FFN ranks on both roles. Set Attention max-num-batched-tokens to a multiple
of 2. The three independent DP groups dispatch as [0,1], [2,3], [4,5];
each group submits one FFN work item per layer. Plain TP partitions replicated
token rows and gathers the combined output. Attention SP uses native local
shards. Disable SP on FFN. FFN TP may differ from Attention TP when static EP
assigns each rank complete local experts. This eager topology extension needs
A5 qualification; the existing 2A2F test recipe uses TP1.


For eager MegaMoE DBO, enable `--enable-dbo` on Attention only, with two
ubatches and native decode/prefill thresholds of at least two tokens.
For example, use `--dbo-prefill-token-threshold 8 --dbo-decode-token-threshold 2`
to exercise splitting with eligible requests. Attention dispatches each shard
into its own registered window and combines it with its own ticket after the
DBO yield. Each FFN rank polls both stage queues. While one fused kernel is in
flight it can submit work for the other stage on a second NPU stream. An event
on each stream protects its window until the previous fused kernel completes.
Both stages use the same expert weights.

DBO allocates two full-capacity windows per rank and two HCCL groups at startup.
Batches below the split threshold use stage 0. With Attention DBO disabled,
the connector allocates one window and uses the original serial FFN loop.
To verify hardware overlap, profile eligible batches after warmup and compare
the two FFN streams' fused kernel intervals. Concurrent host submission alone
does not establish simultaneous AIC/AIV execution or a throughput gain.
