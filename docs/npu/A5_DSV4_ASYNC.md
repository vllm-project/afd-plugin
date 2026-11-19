# A5 DeepSeek-V4-Flash asynchronous AFD development

## Source baseline

The PR branch `feat/a5-async-with-megamoe` starts at upstream `main`, commit
`0d6ec9e`, and carries the A5 work from `feature/a5-async-megamoe` (`9dcb69b`).
The branch also merges upstream `main@3dc2347` (#600 and #601). It targets the
[#537](https://github.com/vllm-project/afd-plugin/issues/537) software environment:

| Component | Target |
| --- | --- |
| vLLM | `5df63a7203cdb9328d824a27b682e77c7b9928be` (0.32.0) |
| vLLM-Ascend | `f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e` |
| CANN | 9.2.1 |
| Model | DeepSeek-V4-Flash, original FP8 checkpoint with MXFP4 routed experts |

The model target is V4 Flash, not V4.1. The existing A3 validation record uses an
older Ascend commit and does not establish A5 support. On A5, the wrapper omits the DSA-CP interfaces deleted in the target Ascend
source. The existing A3 constructor path is preserved.
The native loader, attention, HC, hash routing and shared-expert arithmetic remain
responsible for the model semantics.

## Scope after PR #603 review

This development branch contains only A5 `CAMAsyncMegaAFDConnector`, its three
split operators and the DSV4 integration needed by that path. A5 support for the
existing `CAMAsyncAFDConnector` is retained on `feature/a5-async-cam-followup` for a
separate PR: its four operator changes, BF16 receive handling, MXFP executor,
validation and tests are not included here. Existing A3 Async CAM behavior is
preserved. Shared model TP layout and DBO helpers remain because Mega uses
them; their existing CAM defaults are unchanged.

## Mega execution path

Attention owns routing and shared experts. Dispatch converts BF16 activations to
MXFP8; the FFN fused operator receives, computes the MXFP4 experts and sends results.
`connectors/npu/mega_weights.py` performs the one-time FP4 C0=32 storage
conversion and E8M0 scale preparation after loading. Native checkpoint loading
remains in the DSV4 model wrapper.

The path requires eager ModelRunnerV2, `afd.async=true`,
`afd.compute_gate_on_attention=true` and `enable_fused_mc2=0`. The current optional
FFN device-layer-selection switch is documented in [MEGA_EAGER.md](MEGA_EAGER.md).
Its independent configuration name is a separate review follow-up; this scope
change removes the public validation of the old switch for other connectors/roles.

For Mega, each ubatch stage owns a communication window and dispatch ticket.
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

- Mega is bundled in AFD: build the extension and private Ascend950 OPP
  together with `AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=950 pip install --no-build-isolation .`.
  No external `umdk_cam_mega` wheel is required. Import the three wrappers
  from `afd_plugin.connectors.npu.mega_runtime`; they use `afd_plugin._C_afd_async_mega`,
  `torch.ops.afd_ascend.afd_async_*`, and AFD's private `libcust_opapi.so`.
  The loader configures `AFD_CUST_OPAPI_LIB_PATH` automatically.
  Rebuild both the extension and OPP after protocol changes; mixing old kernels
  with the ring-sequence Python wrappers is unsupported.
- Mega's one-time FP4 storage conversion preserves packed nibbles; its
  format-cast and CANN interpretation still need device verification. Local CPU
  checks establish byte reordering and queue behavior, not numerical support.

## Configuration contract

Supply role, rendezvous and rank counts as usual. Select
`CAMAsyncMegaAFDConnector`, set `afd.async=true` and
`afd.compute_gate_on_attention=true`, and use `enable_fused_mc2=0`.
Set identical A/F counts on both roles and optionally set
`connector_extra_config.attn_ranks_per_dp` to the Attention TP width. Each role's
DP x TP must match its configured rank count. Use static EP, BF16 activations and
eager execution. These are fragments, not a complete deployment command.

For Attention DP3TP2 + FFN DP2TP1/EP2, configure 6 Attention ranks and
2 FFN ranks on both roles. Set Attention max-num-batched-tokens to a multiple
of 2. The three independent DP groups dispatch as [0,1], [2,3], [4,5];
each group submits one FFN work item per layer. Plain TP partitions replicated
token rows and gathers the combined output. Attention SP uses native local
shards. Disable SP on FFN. FFN TP may differ from Attention TP when static EP
assigns each rank complete local experts. This eager topology extension needs
A5 qualification; the existing 2A2F test recipe uses TP1.


For eager Mega DBO, enable `--enable-dbo` on Attention only, with two
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
