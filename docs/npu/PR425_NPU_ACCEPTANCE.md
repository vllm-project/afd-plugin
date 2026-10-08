# PR #425 NPU hardware acceptance

Completed on 2026-10-08: five representative configurations, **300 GSM8K
questions each**, plus functional and operator checks on Guian Ascend A3.
Accuracy is close to the corresponding native references. Two necessary
repairs have hardware regression evidence. Legacy DBO 2A1F remains a local-only
failing configuration; the supported Legacy 2A2F configuration passes.

This acceptance covers NPU execution and accuracy. GPU validation, repository
CI, inherited DCO issues, and final merge remain separate PR requirements.

## Source and environment

- Initial tested PR head: `3ee2e812656a0d88baf6487b02fcf6f15a88ea2e`.
- Repair base: `e63d0048a36461194cc2339d6cd232b34ee20bba`; intervening
  changes did not change the tested runtime files.
- vLLM 0.30.0: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- vLLM-Ascend: `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`.
- CANN 9.1.0; torch 2.10.0+cpu; torch-npu 2.10.0.post4, commit
  `5dd8ef3f9b375b5ae4a83538d5785754148c3302`; driver 25.5.1.1.
- Image tag: `nightly-main-a3-openeuler-20260928172624_aarch64`.
- Complete checkpoints: V2-Lite, 27 layers / 64 routed experts / four shards;
  DSV4 Flash W4A8, 43 layers / 256 experts / 42 shards, per-channel quantization.
- Imports, packaged operators, actual library paths, upstream commits, and
  AFD-disabled native requests were checked on device. The final runtime's
  506 tracked source files match the reviewed worktree; upstream trees are clean.

The pybind extension was rebuilt for the workspace repair; vendor kernels and
the opapi library are unchanged. SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| Final `_C_ascend.so` | `188e4296ef2fe4f8d5dd7625c8cc6196f313b7d0b6271452dc39edc47d9a8bc2` |
| Vendor `libcust_opapi.so` | `9bbb64e9bd2dead81f99bf4b53b85ff1b98f79ccc431efb75fd7533f3a28e1e4` |
| Eight-file repair patch against the base above | `5ed2a7d041f0812a4efb97981652ed56a24f86110ad7aa82b3ec569875ce2d56` |

## Accuracy

All runs use the same frozen first 300 GSM8K test questions, doc IDs 0–299,
8-shot prompts, temperature 0, maximum generation 512, and concurrency 12.
Question, prompt, and target hashes match the corresponding reference.
Scoring uses lm_eval's existing **flexible-extract** metric. No strict-match,
output identity, or additional accuracy-loss threshold is imposed.

| Configuration | Correct | Accuracy | Difference from native |
| --- | ---: | ---: | ---: |
| V2-Lite native reference | 106/300 | 35.33% | — |
| V2-Lite MRV1 graph, 2A2F | 106/300 | 35.33% | 0.00 pp |
| V2-Lite MRV2 FULL, 2A2F | 113/300 | 37.67% | +2.33 pp |
| V2-Lite ordinary Async CAM | 106/300 | 35.33% | 0.00 pp |
| DSV4 native reference | 286/300 | 95.33% | — |
| DSV4 layered off | 287/300 | 95.67% | +0.33 pp |
| DSV4 layered on | 286/300 | 95.33% | 0.00 pp |

The five AFD runs total **1500 questions**. Each run has 300 nonempty responses;
sample scores agree with the aggregate. Native-correct/current-wrong and the
reverse changes are retained. Layered on differs from off by one question;
this is not an output-equivalence claim or a measured accuracy improvement.

The first four AFD 300-question results are reused from the frozen runtime.
Affected CAMP paths were then rechecked, including a same-question 128-question
MRV2 FULL comparison: native **39/128**, original **38/128**, repair **41/128**.
The final common extension ABI was exercised by three additional functional
checks below. Layered-on 300 ran with the final repaired sources and extension.

Selected-question SHA256:
`48bca369ecdd428784d36b12329ba2ccdd6c472f2935af5b7a4829e203dff935`.
Client versions: lm_eval 0.4.12, datasets 5.0.1, transformers 5.14.1.

## Functional and execution evidence

Functional probes use one serial request followed by 24 requests at concurrency
12. Four actual prompt lengths are **305, 369, 433, 497 tokens**. Their content
checks reuse the existing smoke expectations, separate from GSM8K accuracy.

| Path | Result and actual execution |
| --- | --- |
| MRV1 2A2F / 2A1F eager and graph | Covered by the corresponding 300-question run or 25-response short tests; graph and fallback records retained. |
| Legacy MRV1 DBO 2A2F | Final ABI: 25/25. Live two-stage splits execute eager under the existing MLA fallback; unsplit decode retains graph replay. |
| MRV2 2A2F eager / FULL_DECODE_ONLY / FULL | Covered by shorts and FULL 300; live FULL replay confirmed. |
| MRV2 2A1F eager / FULL_DECODE_ONLY / FULL | Original paths returned garbled content. CAMP repair: each mode 25/25; final ABI FULL also 25/25. |
| Many-A to one-F communication | Live preparation records actual counts 305/1 padded to 305/305 before input preparation, `profile=False`; successful FULL replay also recorded. The count of 1 is the idle A's dummy input. |
| Ordinary Async CAM | 300 questions plus concurrent functional requests. |
| Async MoE ubatching | Final ABI: 25/25. Actual FFN work items alternate routed counts 924/912, consistent with the two-stage TP2 padded split; stage indices were not independently logged in this final short. |
| DSV4 layered off / on | Both complete 300. Device traces confirm gate, quantization, dispatch/combine, and the respective GMM kernels inside the real request window. |

The final common ABI shorts cover MRV2 2A1F FULL, Legacy 2A2F DBO, and Async
MoE ubatching: **75/75 responses, all three driver exits 0**. Related CPU
contracts: **148 passed, no skips** on the target environment. Changed Python
files pass Ruff 0.15.13 lint/format, mypy 1.11.1 for Python 3.10, and SPDX checks;
the patch passes `git diff --check`. Final test-only formatting preserves ASTs.

For layered on, one real FFN trace contains **43 dispatch-recv, 43 combine-send,
43 layered W13, and 43 layered W2 kernels**. Inputs include INT8 activations and
INT4 weights; W2 emits BF16. Attention traces separately contain gate, dynamic
quantization, dispatch-send, and combine-recv. These are device kernel records,
not conclusions drawn only from environment switches; no throughput benchmark
is claimed.

## Necessary repairs and origin

**CAMP count contract:** physically pad eager/fallback inputs equally within
each receiving group before native input preparation; use the kernel's strided
Attention-to-FFN mapping for receive counts and graph keys; retain native
FFN-level DP context. Native graph selection and collectives are preserved.
The missing many-A eager adaptation already exists in main's initial NPU MRV2
[PR #257](https://github.com/vllm-project/afd-plugin/pull/257); the upgrade
retained it. This historical conclusion is based on source, not an old-version
hardware replay.

**Workspace lifetime:** the queued handler must retain its workspace until
ACLNN submission, then release its Tensor capture. Otherwise the completed
handler remains in torch-npu's execution ring and holds the workspace.
The measured residual growth was approximately **192 MiB per work item**;
after the repair, allocated memory stays at **20098992128 bytes** through
896 work items in the causal probe. The original 43-layer loop remains intact;
there is no added synchronization or numeric/tiling change. The capture was
introduced by upgrade [PR #412](https://github.com/vllm-project/afd-plugin/pull/412),
commit `d7229fc9c3b0e5faef7d15c4ca8b3ff4516506bf`. Its pre-submission ownership
fix remains necessary; this repair shortens ownership after submission.

## Remaining limits

- **Legacy DBO 2A1F:** historical-recipe repeat passes only 12/25; concurrent
  responses can be garbled. Upgrade [PR #407](https://github.com/vllm-project/afd-plugin/pull/407),
  commit `18ef89d48992a7d2cb59f45b3a437bddd272e507`, added local request-boundary
  splitting to synchronous DBO, permitting unequal per-stage lengths at the
  fixed-stride CAMP boundary. This local-only combination is not fixed here;
  supported Legacy 2A2F passes. Stage padding or an explicit unsplit fallback
  needs its own implementation and verification.
- **Shutdown:** cleanup warnings remain. Earlier layered-off and the repaired
  short report the same AIV/MTE signature after SIGTERM. Final layered-on 300
  has no OOM/AIV/MTE/507035 in the request window; EngineDeadError and
  KeyboardInterrupt occur after shutdown. Clean device exit is not claimed.
- **Per-group toy operator tolerance:** one existing comparison failed its
  hard-coded `rtol=0.04`, `atol=0.05`. The user deferred this as a release gate;
  the accepted full-model checkpoint uses per-channel quantization. This
  acceptance does not establish general per-group numerical equivalence.
- No new NPU MRV2 DBO, general FFN TP topology support, or performance guarantee
  is added. PR #422 was not merged or taken over.

## Evidence and reproduction

Original logs, per-question outputs, command records, source/library manifests,
and trace summaries are retained in the task workspace
`npu-v030-acceptance-1008/`. Full device traces remain on persistent task storage.
Both validation jobs were stopped after collection; persistent storage remains.

The task launchers reuse the existing E2E runner through a small private adapter
for NPU MRV2 selection and the agreed evaluation protocol. They are included
with the reproducibility bundle, together with the frozen question IDs and
runtime manifests. Run in the recorded image with matching complete checkpoints
and the task's `ITASK_WORKDIR`/environment:

```bash
export ACCEPTANCE_PLUGIN_ROOT="$ITASK_WORKDIR/runtime/afd-plugin-npu-acceptance-final"
python3 "$ITASK_WORKDIR/scripts/run_final_v2lite.py"
python3 "$ITASK_WORKDIR/scripts/run_final_repair_checks.py"
ACCEPTANCE_START_CASE=dsv4-layered-1 \
    BATCH_SIZE_FACTOR=0.125 ACCEPTANCE_FFN_PROFILER_ACTIVE=1 \
    python3 "$ITASK_WORKDIR/scripts/run_dsv4_matrix.py"
python3 "$ITASK_WORKDIR/scripts/summarize_accuracy.py"
```

DSV4 uses its existing DP2TP4 / FFN EP8 topology, full 43-layer loop, scheduler
batch 8192, context 4096, and the documented capacity factor 0.125 (32768 rows).
The layered-on reproduction reuses the retained native and layered-off runs.
The bounded FFN profiling window records one group of all 43 layers. No
checkpoint subset, expert subset, arbitrary batch reduction, or dependency
replacement is used for final acceptance.

Key raw archive SHA256 identities:

| Archive in task `config/` | SHA256 |
| --- | --- |
| `dsv4-layered-final-300-evidence.tar.gz` | `ad781e11e78ac80111c38967b3b932714acc211783b77abfdf7237af4460f852` |
| `final-common-abi-all-evidence.tar.gz` | `43c73aac4df2a6968d4f5e36b03c9b81f5dca3ad538c4431386005c7ad007904` |
| `final-camp-common-abi-evidence.tar.gz` | `67af8d63f6fbcf757e1bb0c3953067342d722920bc248ff3d253b099c81c24cf` |
| `repair-128-evidence.tar.gz` | `75be7680ad3508eea249d1fa7c827ac789de031bab4b26b411a9ce2174263696` |
