# Performance and Accuracy Verification

Use when a PR claims latency, throughput, memory, or accuracy improvement —
including `[PERF]` titles, layered GMM / CAM control-path optimizations, graph
changes, TP/SP scheduling changes, and routing changes.

## Comparable A/B contract

Accept a claim only with base and head runs that hold constant:

- Hardware pool and placement (CI presets: `l4_1`–`l4_4`, `h100_4`; NPU:
  documented manual environment per `tests/npu/README.md`).
- Container image and dependency versions (`docker/Dockerfile.ci` pins
  `VLLM_BASE_TAG` and asserts the installed vLLM version).
- Workload and datasets (`tools/benchmarks/decode_bench.py`,
  `tools/benchmarks/request_generator.sh`, or the named E2E scenario; GSM8K
  for accuracy gates).
- Graph/eager mode, DBO on/off, ubatching mode, FFN execution path, and
  topology (1A1F/2A1F/2A2F/4A2F/4A4F, colocation/disaggregation, multi-pod).
- Warmup, repetitions, and the metric definition (mean vs median, percentiles,
  tokenization boundaries). The NPU benchmark harnesses deliberately do not
  pool or interpolate samples — preserve that when comparing.

## Isolate per claim

One claim, one comparison. A PR claiming several improvements (transfer
overlap, MoE split, layered scheduling, control-path overhead) needs evidence
per mechanism or a staged ablation; a single end-to-end delta cannot attribute
multiple causes.

## Evidence table

| Dimension | Typical evidence |
| --- | --- |
| Latency | Decode bench per-step or E2E per-request timing, same batch mix |
| Throughput | Tokens/s at stated concurrency, steady-state window |
| Memory | Peak allocator/device memory, same graph/capture settings |
| Accuracy | GSM8K gate plus the model suite's accuracy scenario; state pass criteria and sample count |

## Reviewer verification ladder

1. Check the A/B contract above against the PR's Test Result section; missing
   constants make the claim unproven, not merely weak.
2. When a runnable GPU path exists, offer to rerun the named benchmark or E2E
   scenario via the `run-e2e` skill (or `run-vllm-bench-k8s` for a deployed
   recipe); otherwise mark the claim unverified and name the exact missing run.
3. For NPU claims, require the manual-run evidence (environment, command,
   numbers) in the PR; there is no NPU CI, and unit tests never substitute for
   device measurements.
4. Watch for numbers carried over from a previous pin: evidence measured on an
   older vLLM/vLLM-Ascend pair does not support a claim about the current pair.
   The design pages mark such rows historical for the same reason.

An unproven perf claim on a hot path is a P1 finding (missing blocking
evidence), stated as the specific missing comparison — not as distrust.
