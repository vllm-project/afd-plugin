# A5 DeepSeek-V4-Flash asynchronous AFD development

## Source baseline

The working branch `feature/a5-async-megamoe` is rebased onto upstream
`main@e4ddab1`. Its runtime target follows [installation.md](../installation.md):

| Component | Target |
| --- | --- |
| vLLM | `0.34.0`, `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| vLLM-Ascend | `7dcdb690caf879a17ab07382880d5773f5202e20` |
| CANN | 9.3.0 |
| torch / torch-npu | `2.10.0+cpu` / `2.10.0.post7` |
| Model | DeepSeek-V4-Flash, original checkpoint with MXFP4 routed experts |

These are the current development targets. The installation guide's recorded
hardware validation is on A3; earlier A5 results apply to their recorded
commits and environments. This rebased 0.34.0 Mega path still requires A5 build,
accuracy and end-to-end validation. The model admission remains V4 Flash.

The DSV4 wrapper follows the pinned Ascend constructor: it restores missing
`compress_ratios` through vLLM's configuration helper and uses the current
Attention constructor without legacy version or DSA-CP arguments. The native
loader, attention, HC, hash routing and shared-expert arithmetic own model semantics.

## Scope after PR #603 review

This branch adds A5 `CAMAsyncMegaAFDConnector`, its three split operators and
model integration. Existing CAMAsync/CAMP2P operators and connector behavior
follow the rebased main branch. Shared entry points select Mega explicitly;
Mega model admission, routing and native weight extraction live in the DSV4
adapter. See [MEGA_EAGER.md](MEGA_EAGER.md#model-adapter-boundary).

## Mega execution path

Attention owns routing and shared experts. Dispatch converts BF16 activations to
MXFP8; the FFN fused operator receives, computes the MXFP4 experts and sends results.
`connectors/npu/async_mega/mega_weights.py` performs the one-time FP4 C0=32 storage
conversion and E8M0 scale preparation after loading. Native checkpoint loading
remains in the DSV4 model wrapper.

The path requires eager ModelRunnerV2, `afd.async=true`,
`afd.compute_gate_on_attention=true` and `enable_fused_mc2=0`. FFN selects the DP and layer on device as described in [MEGA_EAGER.md](MEGA_EAGER.md).

All ubatches share one HCCL group and one registered window per rank.
Each Attention DP advances one sequence ring from 1 through 32 and back to 1
across its ubatches. Python retains a ticket/result per ubatch. The native DBO
receive hook submits the previous receive before the next send so the data slot
is reused in stream order.

Each FFN rank has one lane, one stream and at most one fused call in flight.
The kernel scans all DPs and claims a task when all Attention TP members have
published its next generation, including when no expert on this FFN rank was
selected. PUBLISHED/TAKEN/READY/CONSUMED protect input/output reuse. The next
call is submitted after the previous completion event becomes ready.

The 2A2F recipe is retained as a device-validation entry. Other A/F counts follow
DP x TP per role and contiguous static FFN EP in one MTE supernode; no graphs,
MTP or EPLB. Idle fused kernels wait for new work, so stopping a run may require
terminating the workers.

## Operator dependencies and current limits

**The rebased 0.34.0 target needs fresh A5 validation.** Host checks do not
establish NPU accuracy, long-run communication progress or performance.

- The Mega build requires CANN 9.2+; the current target above uses CANN 9.3.0.
  Mega is bundled in AFD: build the extension and private Ascend950 OPP
  together with `AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=950 pip install --no-build-isolation .`.
  No external `umdk_cam_mega` wheel is required. Import the three wrappers
  from `afd_plugin.connectors.npu.async_mega.mega_runtime`; they use `afd_plugin._C_afd_async_mega`,
  `torch.ops.afd_ascend.afd_async_*`, and AFD's private `libcust_opapi.so`.
  The loader configures `AFD_CUST_OPAPI_LIB_PATH` automatically.
  Rebuild both the extension and OPP after protocol changes; mixing old kernels
  with the ring-sequence Python wrappers is unsupported.
- Mega's one-time FP4 storage conversion preserves packed nibbles; its
  format-cast and CANN interpretation still need device verification. Local CPU
  checks establish byte reordering and layer mapping, not numerical support.

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
to exercise splitting with eligible requests. Both ubatches share the same
window and FFN execution lane; batches below the threshold use stage 0.
Disabling DBO also uses that single lane.

The intended overlap is ubatch 0's FFN work with ubatch 1's Attention work.
The next dispatch follows the previous combine receive on the Attention stream,
so this implementation does not queue two same-DP payloads in separate slots.
Profile Attention and FFN together after warmup to verify that overlap and
measure the wait at the receive/send boundary. There are no two FFN streams
to compare in the current implementation.

## Host regression checks

```bash
python -m pytest tests/unit/model_executor/models/npu/test_deepseek_v4.py \
  tests/unit/connectors/npu/test_mega_topology.py \
  tests/unit/connectors/npu/test_mega_layers.py \
  tests/unit/config/test_config.py tests/unit/config/test_validation.py \
  tests/unit/package/test_mega_build_files.py
```

The DSV4 construction tests expose the current native interface without a
legacy version helper, cover both roles, and check restoration of missing
compression metadata. Their configuration-conversion dependency is isolated so
they remain runnable on CPU with PyTorch and without vLLM/Ascend installed.
