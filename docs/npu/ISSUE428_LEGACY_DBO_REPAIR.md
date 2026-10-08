# Issue #428: Legacy DBO 2A1F repair

This follow-up to [PR #425](https://github.com/vllm-project/afd-plugin/pull/425)
repairs the unequal-stage CAMP layout in NPU MRV1 Legacy DBO reported in
[Issue #428](https://github.com/vllm-project/afd-plugin/issues/428).
Validation uses the complete DeepSeek-V2-Lite checkpoint on Guian Ascend A3.

## Diagnosis

Request-boundary DBO splitting can turn an equally padded whole batch into
unequal stage lengths. CAMP requires equal physical chunks within each FFN
receiving group. Native A2E uses the actual input tensor's first dimension
for its payload offsets; changing only an integer argument is insufficient.
FFN control counts, received rows, calculation context, and return chunks must
describe the same physical layout.

The earlier, uncommitted prototype did pad actual tensors in Python
`send_attn_output()`. Its 20 unit checks passed, but the model check passed
only 10/25. A real three-rank native probe of that prototype passed 72 layout
and content checks, so repeating its eager-only tests would miss the failure.

An observer placed directly around native A2E identified the compiled-model
boundary that remained wrong:

```text
rank=2 stage=0 rows=738 argument=738 physical=802 mismatch=True
rank=2 stage=1 rows=433 argument=433 physical=994 mismatch=True
```

The outer Python padding branch was traced with unsplit profile metadata;
the compiled model did not reevaluate that branch for live split requests.
The first mismatch is therefore present before native A2E transmission.

## Repair

The production change is confined to `afd_plugin/connectors/npu/camp2p.py`:

- Convert split-stage control counts into independent physical metadata,
  using the maximum count within each existing strided FFN receiving group.
  Expand DP counts to Attention TP peers before grouping.
- Pad the actual payload inside the opaque send custom op, which reads the
  current stage metadata at runtime. Optional token IDs and scales follow the
  same physical layout.
- Keep original Attention metadata and tensors intact. The existing E2A
  binding receives the valid prefix using the original reference length.

For the issue's count pair, the native probe verifies the following layout:

| Stage | Real Attention rows | Wire rows | FFN rows | Returned rows |
| --- | --- | --- | ---: | --- |
| 0 | 802 / 433 | 802 / 802 | 1604 | 802 / 433 |
| 1 | 866 / 674 | 866 / 866 | 1732 | 866 / 674 |

The repair retains request-boundary splitting and `send → yield → recv`.
Two-Attention/two-FFN groups need no additional padding. No runner, binding,
vendor kernel, dependency, or fallback policy changes are introduced.
Padding adds work for unequal stages; throughput was not benchmarked.

## Environment and source identity

- Repair base: `605ba33ac27a4fc667d5d4ef93732d9c5a4f626c`, PR #425 head.
- PR #425 subsequently advanced to `acf876107d62fda553e1f07eec08e1c3c41bcc0d`.
  Its two changed layered W4A8 files are outside this Legacy BF16 CAMP path;
  the tested production and regression files remain identical.
- vLLM 0.30.0: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- vLLM-Ascend: `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`.
- CANN 9.1.0; torch 2.10.0+cpu; torch-npu 2.10.0.post4.
- Image: `nightly-main-a3-openeuler-20260928172624_aarch64`.
- Checkpoint: complete BF16 V2-Lite, 27 layers, 64 routed experts, four shards.
- Dedicated task: `afd-issue428-legacy-dbo-1008`, four NPU devices.

All 506 tracked files present in the tested runtime match the repair worktree.
The omitted file is the existing PR #425 acceptance Markdown report, which
does not participate in execution. The native extension is unchanged from
the accepted PR #425 runtime:

```text
_C_ascend.so SHA256:
188e4296ef2fe4f8d5dd7625c8cc6196f313b7d0b6271452dc39edc47d9a8bc2
```

## Functional and native verification

The two model probes each use one serial request and 24 concurrent requests,
concurrency 12, temperature 0, generation limit 64, and actual prompt lengths
305 / 369 / 433 / 497 tokens. Configuration: batch tokens 4096, sequences 8,
context 4096, block size 128, memory utilization 0.75, DBO thresholds 2/8,
prefix caching off, chunked prefill off.

| Check | Result |
| --- | --- |
| Related connector and token-ID unit tests | 24 passed, no skips |
| Native three-rank A2E/E2A probe | 72/72 layout and content checks |
| Original Legacy DBO 2A1F counterexample | 25/25 content checks |
| Legacy DBO 2A2F regression | 25/25 content checks |

The native probe uses recognizable BF16 data, three unequal count pairs,
two cooperative Attention stages, four layers, and real CAMP bindings. FFN
receives complete physical chunks; Attention receives its original valid
prefix. Original metadata remains intact. Python mocks do not establish this
native data-path result.

The model observer also checks every live native A2E call against physical
control counts and asserts FFN `num_tokens == output rows == transfer rows`.
The 2A1F log contains 248 live split records, including eight unequal records,
such as `[372, 803]` / `[305, 433]`. The 2A2F log contains 250 live split records.
These records exclude warmup, capture, and graph replay. Live split execution
uses the existing MLA eager fallback; unsplit decode retains graph support.

The 2A2F E2E scenario and all content checks completed successfully. Its shell
wrapper subsequently exited 2 because the launcher file had been edited in
place while waiting for the E2E command. Saved responses and live records were
independently audited, and the faulty wrapper status is retained alongside
that audit. The launcher was then replaced atomically and syntax-checked
before accuracy evaluation; the completed model check was not repeated.

## Bounded accuracy comparison

Both runs completed successfully with driver exit 0. They use the first
128 GSM8K test questions, identical 8-shot prompts, temperature 0, generation
limit 512, concurrency 12, and lm_eval's existing flexible answer extraction.
Question, prompt, target hashes, and evaluation protocols match. All 128
responses in each run are nonempty; sample scores match the aggregate.

| Repaired 2A1F configuration | Correct | Accuracy |
| --- | ---: | ---: |
| DBO disabled reference | 40/128 | 31.25% |
| Legacy DBO enabled | 41/128 | 32.03% |

The difference is one correct answer, **+0.78 percentage points** with DBO.
Accuracy is close under the agreed bounded comparison. There is no output
identity requirement or additional absolute-score gate. The comparison does
not reuse the historical 2A2F reference or repeat the five 300-question runs.

The frozen first-128-question file's SHA256 is:

```text
5d01b545a62d879f883a4166d17ad6e304680f11600729d26692a8473d31c3e0
```

## Evidence and reproduction

The task workspace retains the native probe, observer, launchers, source
identities, raw logs, all 50 functional responses, and per-question accuracy
outputs. Local artifacts are under `npu-v030-acceptance-1008/issue428/`.
The complete evidence archive is `issue428-repair-evidence.tar.gz`, SHA256:

```text
fb4cc863258f3d195e5a5a596f7e77510cde3f9b2ce85d1825417a3979f80820
```

The existing issue reproduction uses the same functional workload; replace
`afd-graph-dbo-2a1f` with `afd-graph-dbo-2a2f` for its regression.

The runtime bundle's accuracy launcher uses the established E2E runner and
recorded image, model, and environment:

```bash
python3 "$ITASK_WORKDIR/scripts/run_issue428_accuracy.py"
```

This repair adds no NPU MRV2 DBO, general FFN TP support, or performance claim.
Issue #429's operator tolerance is outside its scope. Earlier teardown
warnings remain separate from successful request-window results.
