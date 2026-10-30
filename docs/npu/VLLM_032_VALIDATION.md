# Ascend validation for vLLM 0.32

This record scopes the 2026-10-29 upgrade validation. It is not a claim for
other model, topology, device, or software combinations. The AFD production
source was `84c2ebaac3890d993deaa1876af7457c2aa2ea8d`, with vLLM
`5df63a7203cdb9328d824a27b682e77c7b9928be` and vLLM-Ascend
`bd42eea01d8312c4df469fabaed45f77a12df4d3`. The Ascend commit is a source
pin, not a claim about a published release tag.

The measured environment used Ascend A3/910C, Python 3.12.13,
PyTorch 2.10.0+cpu, torch_npu 2.10.0.post7, CANN 9.2,
Triton Ascend 3.6, memfabric-hybrid and memcache-hybrid 1.3. The tested image
was identified by digest; the source tree and target AFD native artifacts were
checked independently. The checks did not cover complete model-weight contents
or every system-library mapping. `torch`'s `+cpu` distribution label did not
prevent NPU execution through torch_npu. The recorded environment's six dependency
conflicts were explicitly accepted for this validation; `pip check` remained
nonzero and this is not a general dependency waiver.

## DeepSeek-V2-Lite single-Pod accuracy

All eight entries used the same first 100 GSM8K test documents, eight-shot
prompts, `max_gen_toks=512`, and strict-match threshold 0.27. Prompt and
sample identities were compared across entries. These results are a
100-sample validation, not a 1,319-sample full-dataset result.

| Entry | Correct / 100 | Strict match | Request and path | Graceful exit |
| --- | ---: | ---: | --- | --- |
| Native Graph DP4 | 31 | 0.31 | Pass | Pass |
| CAMP2p eager 2A2F | 31 | 0.31 | Pass | Fail |
| CAMP2p Graph 2A2F | 31 | 0.31 | Pass | Fail |
| CAMP2p Graph+DBO 2A2F | 32 | 0.32 | Pass | Fail |
| CAMP2p eager 2A1F | 31 | 0.31 | Pass | Fail |
| CAMP2p Graph 2A1F | 28 | 0.28 | Pass | Fail |
| CAMP2p Graph+DBO 2A1F | 34 | 0.34 | Pass | Fail |
| CAM async uBatch | 32 | 0.32 | Pass | Fail |

The two 2A2F Graph entries were verified offline against their original
archived outputs after correcting the document-hash serialization to match
lm-eval. Their original controller failures remain in the evidence; the
requests and tests were not rerun for that correction.

The Graph/DBO cases retain the original test's actual replay and nonempty
microbatch checks. All eight completed physical cleanup and their allocations
were released. The seven AFD graceful-exit failures, including force-kill
records, remain failures; successful requests or physical cleanup do not
establish graceful shutdown. The native baseline used the recorded 30/60/90 s
worker/server/outer shutdown budgets and passed its separate strict check.

## DeepSeek-V4 Flash W4A8 single-Pod accuracy

The native control and three AFD entries used the same checkpoint and first
100 GSM8K documents, eight-shot prompts, `max_gen_toks=512`, concurrency 12,
and strict-match threshold 0.27. Document, rendered-prompt and target hashes
matched for every sample. Attention was eager throughout; the AFD topology
was Attention DP2/TP4 and FFN DP8/TP1, using 16 DIE.

| Entry | Strict / flexible correct out of 100 | Attention DBO | Graceful exit |
| --- | ---: | --- | --- |
| Native eager DP2/TP4 | 96 / 96 | Off | Pass |
| CAM async, layered GMM off | 95 / 95 | Off | Fail |
| CAM async, layered GMM on | 94 / 94 | Off | Fail |
| CAM async, ordinary GMM | 95 / 95 | On | Fail |

The DBO entry recorded completed steps with two nonempty microbatches in
the actual evaluation window on every Attention rank: 42 on each DP0 TP0–3
rank and 44 on each DP1 TP0–3 rank. FFN used ordinary GMM without DBO.
This is rank-aggregate request-window evidence; it does not establish that
every question split into two microbatches or that each A/F transaction was
paired. Relative to native, both filters had 95 samples correct in both runs,
one correct only in native, none correct only in AFD, and four wrong in both.
Passing the agreed threshold does not imply per-sample parity or full-dataset
qualification. Native DBO was not required because the pinned vLLM-Ascend
runtime does not support that native configuration.

All 16 A/F workers in the DBO entry mapped the frozen target AFD native artifact.
System-library mapping verification remained incomplete. The AFD post-request exit failures
remain failures; device processes and ports were separately verified clear,
and the dedicated allocation was stopped with persistent storage retained.

## Other measured coverage

The target-runtime unit run at `84c2eba` reported 1843 passed, 6 skipped,
and 180 subtests passed. Four device-invocation skips have separate device
results; two absence tests skipped because the Ascend extension was installed.
The single-device suite had 67 passes including one Python-only shape test;
three additional GMM device nodes and the synchronous two-rank bit-exact
roundtrip passed. Async CAM roundtrip, routed numerical and eager/FULL
synthetic checks covered 12, 9 and 8 cells respectively.

DSV4 Flash W4A8 async's original ten short requests passed their answer,
stop and overlap checks; they do not establish actual two-stage DBO. W8A8
synchronous serving was a nonempty/terminal smoke test, not an answer
accuracy test. W8A8 checkpoint-local shared/routed/final comparisons covered
Hash layer 0 and ordinary layer 3. The W4A8 FFN eager/FULL fixture compared
five prompt shapes, six generated sequences and eight tokens per sequence;
all token IDs matched and all four FFN ranks captured and replayed. These
counts do not establish per-request execution of all 43 layers. Relevant AFD
post-request shutdown failures remain recorded separately.

Ascend multi-Pod qualification is deferred. The observed cross-node 507034
startup failure remains unresolved; [PR #385](https://github.com/vllm-project/afd-plugin/pull/385)
and [issue #370](https://github.com/vllm-project/afd-plugin/issues/370) define
the original validation scope, not a proven fix for that failure. V4 position/KV,
padding-only, indexer, workspace and per-transaction A/F freshness boundaries
were not separately qualified. The request-window DBO result above does not
close those boundaries. The earlier AFD long-prefill experiment retained four
answer-oracle failures among ten requests; the same-input 1501-token native
control also failed its answer check, with 8/10 correct. No common GPU/NPU
root cause or AFD-specific attribution was established. These diagnostic
failures remain recorded; broader long-prefill qualification was outside the
agreed 100-sample and request-window acceptance. V4 synchronous end-to-end
Graph/DBO was not part of this validation; FFN-only FULL replay does not
establish that combination.

## PR #502 applicability

The measured upgrade originally included the fixed GPU combination `04be3c9`
from [PR #502](https://github.com/vllm-project/afd-plugin/pull/502). The final
NPU-only PR is based on `update/v0.32.0` at
`60d0178d1ea14e2938e763bd0b4a385aee24f9b9`, where
[PR #508](https://github.com/vllm-project/afd-plugin/pull/508) already carries
final #502 commit `17f6b364d657a3004e89f8a1cce80da5d968f634`.
Those GPU fixes are baseline content; the six fixes do not all execute on NPU. Shared control-plane
close/codec groundwork is present; the NPU paths do not implement the complete
GPU receive-cancellation and receiver-join-before-destruction guarantee. NPU
graceful-exit failures remain unresolved, without a proven single root cause.
The current NPU launch paths already separate the Attention and FFN HTTP ports.

The four V4 CUDA fixes cover positions/KV staging alignment, padding-only
Attention, indexer SM budgets and per-owner FlashMLA workspace reservation.
They are CUDA-scoped and are not directly reused on NPU. Ascend DSA/CANN uses
separate metadata, empty-token handling and workspace lifetimes. Those source
differences neither prove a shared CUDA/NPU bug nor qualify the untested V4
boundaries listed above. See the [compatibility inventory](../design/module/compatibility_and_patches.md)
for the target patch contracts.

GPU and older NPU results retain their own version, model, sample and hardware
identities. This record does not claim GPU requalification or a performance
improvement.
