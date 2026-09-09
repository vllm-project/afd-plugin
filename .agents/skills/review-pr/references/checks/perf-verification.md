# Performance and Accuracy Verification

Use when a PR claims latency, throughput, memory, or accuracy improvement —
including "[PERF]" titles, ubatching/CAM control-path optimizations, graph
changes, and routing changes.

## Comparable A/B contract

Accept a claim only with base and head runs that hold constant:

- Hardware pool and placement (CI presets: `l4_1`–`l4_4`, `h100_4`; NPU:
  documented manual environment per `docs/npu/TESTING.md`).
- Container image and dependency versions (CI image pins
  `VLLM_BASE_TAG=v0.26.0`).
- Workload and datasets (`tools/benchmarks/decode_bench.py` or the named E2E
  scenario; GSM8K-7 for accuracy gates).
- Graph/eager mode, DBO on/off, ubatching mode, and topology (2A2F/2A1F,
  colocation/disaggregation).
- Warmup, repetitions, and the metric definition (mean vs median, percentiles,
  tokenization boundaries).

## Isolate per claim

One claim, one comparison. A PR claiming several improvements (transfer
overlap, MoE split, control-path overhead) needs evidence per mechanism or a
staged ablation; a single end-to-end delta cannot attribute multiple causes.

## Evidence table

| Dimension | Typical evidence |
| --- | --- |
| Latency | Decode bench per-step or E2E per-request timing, same batch mix |
| Throughput | Tokens/s at stated concurrency, steady-state window |
| Memory | Peak allocator/device memory, same graph/capture settings |
| Accuracy | GSM8K-7 gate plus the model suite's accuracy scenario; state pass criteria |

## Reviewer verification ladder

1. Check the A/B contract above against the PR's Test Result section; missing
   constants make the claim unproven, not merely weak.
2. When a runnable GPU path exists, offer to rerun the named benchmark or E2E
   scenario via the `run-e2e` skill; otherwise mark the claim unverified and
   name the exact missing run.
3. For NPU claims, require the manual-run evidence (environment, command,
   numbers) in the PR; there is no NPU CI, and unit tests never substitute for
   device measurements.

An unproven perf claim on a hot path is a P1 finding (missing blocking
evidence), stated as the specific missing comparison — not as distrust.
