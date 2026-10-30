# Ascend validation for vLLM 0.32

The single-Pod validation in [PR #507](https://github.com/vllm-project/afd-plugin/pull/507)
used AFD production source `84c2ebaac3890d993deaa1876af7457c2aa2ea8d`,
vLLM `5df63a7203cdb9328d824a27b682e77c7b9928be`, and vLLM-Ascend
`bd42eea01d8312c4df469fabaed45f77a12df4d3`. The Ascend revision is a
source pin, not a published release tag.

The measured A3/910C environment used Python 3.12.13, PyTorch 2.10.0+cpu,
torch_npu 2.10.0.post7, CANN 9.2, Triton Ascend 3.6, and memfabric-hybrid /
memcache-hybrid 1.3. The image and target AFD native artifacts were identified;
complete model-weight contents and system-library mappings were not verified.
Six dependency conflicts were accepted for this environment; `pip check`
remained nonzero. This acceptance does not apply to other environments.

## DeepSeek-V2-Lite

All entries used the same first 100 GSM8K documents and prompts, eight-shot
prompting, `max_gen_toks=512`, and strict-match threshold 0.27.

| Entry | Correct / 100 | Inference and path | Graceful exit |
| --- | ---: | --- | --- |
| Native Graph DP4 | 31 | Pass | Pass |
| CAMP2p eager 2A2F | 31 | Pass | Fail |
| CAMP2p Graph 2A2F | 31 | Pass | Fail |
| CAMP2p Graph+DBO 2A2F | 32 | Pass | Fail |
| CAMP2p eager 2A1F | 31 | Pass | Fail |
| CAMP2p Graph 2A1F | 28 | Pass | Fail |
| CAMP2p Graph+DBO 2A1F | 34 | Pass | Fail |
| CAM async uBatch | 32 | Pass | Fail |

Graph/DBO entries included actual replay and nonempty-microbatch checks.
All allocations were released after physical cleanup. The AFD exit failures,
including force-kill results, remain failures.

## DeepSeek-V4 Flash W4A8

All entries used the same checkpoint, documents, prompts and targets, with the
first 100 GSM8K documents, eight-shot prompting, `max_gen_toks=512`, concurrency
12, and threshold 0.27. Attention was eager. The AFD topology was Attention
DP2/TP4 and FFN DP8/TP1, using 16 DIE. Strict and flexible scores were equal.

| Entry | Correct / 100 | Attention DBO | Graceful exit |
| --- | ---: | --- | --- |
| Native eager DP2/TP4 | 96 | Off | Pass |
| CAM async, layered GMM off | 95 | Off | Fail |
| CAM async, layered GMM on | 94 | Off | Fail |
| CAM async, ordinary GMM | 95 | On | Fail |

The DBO entry recorded two nonempty microbatches during the evaluation window
on all eight Attention ranks. FFN used ordinary GMM without DBO. This evidence
does not establish splitting of every question or per-transaction A/F freshness.
The native and AFD DBO entries shared 95 correct answers, with one answer
correct only in native. Physical cleanup and allocation release completed;
AFD graceful-exit failures remain unresolved. Native DBO was outside acceptance
because the pinned Ascend runtime does not support that configuration.

## Additional coverage and limits

The target-runtime unit suite reported 1843 passes, six skips and 180 passing
subtests. Single-device coverage reported 67 passes; three additional GMM
nodes, synchronous two-rank bit-exact roundtrip, Async CAM roundtrip and routed
numerical checks also passed. W8A8 serving was a terminal/nonempty smoke check;
checkpoint-specific shared/routed/final comparisons covered Hash layer 0 and
ordinary layer 3. W4A8 FFN eager/FULL replay produced matching token IDs for
the measured fixture; this does not qualify synchronous V4 end-to-end Graph/DBO.

- Accuracy scope is 100 samples, not the full 1319-question dataset or general
  per-sample equivalence. No performance qualification is claimed.
- Multi-Pod qualification remains open, including the observed cross-node
  507034 startup failure. Main's multi-Pod tooling does not establish target
  runtime validation.
- Long-prefill diagnostics retain answer failures on both AFD and the
  same-input native control; an AFD-specific root cause was not established.
- Separate V4 position/KV, padding-only, indexer, workspace and A/F freshness
  boundaries were not qualified. CUDA DBO workarounds do not cover Ascend's
  separate DSA/CANN implementations.
- Shared control-plane close/codec changes do not implement the complete NPU
  receive-cancellation and receiver-join guarantee. Inference and physical
  cleanup passes must not be reported as graceful-exit passes.
