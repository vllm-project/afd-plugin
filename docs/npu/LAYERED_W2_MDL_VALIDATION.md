# Layered W13/W2 graph validation

The old W2 `MSD_API_DEQUANT` route used `GetNormalConfig()` despite its
configuration constant's MDL name. The change gives only this route an explicit
MDL configuration while retaining `enUnitFlag=true`. Tile sizes, quantization,
workspace requests, input layout and other dispatch routes are unchanged.
The SDK configuration bundle changes `doNorm`, `doMultiDataLoad` and `batchMode`;
batch-specific branches are gated out for the ordinary, non-batched Matmul type.

## Environment and method

Measured on Ascend A3 on 2026-10-20 with CANN 9.1.0, driver 25.5.1.1,
Python 3.12.13, torch 2.10.0+cpu and torch_npu 2.10.0.post4.
Baseline source: `300fb2017d3f9eba032dd699b37cfe05f54ffd5f`.
Measured candidate: `fdc61626083e9a51e1f546da5a17f7b924129a62`.
The PR cherry-picks their fixture/kernel changes onto current main; the relevant
kernel, executor, operator loader and NPU test sources are byte-identical to the
measured candidate. A fresh device build of the rebased PR was not repeated.

The benchmark captures one actual `AsyncCAMW4A8Executor` W13→W2 invocation,
including the device selector clone, with its four retained tensors and eight
weight layers resident. Geometry: E=64, H=4096, I=2048, C=262144, seed=1234.
Input restoration and routing metadata updates happen outside timing. W2-only
measurements also restore its in-place input before every call.

Five independent baseline processes establish per-case variability. Three
independent ABBA groups each use four processes, ten graph warm replays after
capture/shape changes and 50 samples per cell. Speedup is the geometric mean
of the three paired group ratios. The 95% interval resamples the three groups,
not the 50 within-process samples. Formal timing has profiling disabled.

## Results

Times are geometric aggregates of six baseline and six candidate process medians.

| Graph workload | Baseline ms | MDL ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Actual chain, balanced M=98304 | 14.013778 | 12.710788 | 9.297920% |
| Actual chain, balanced M=16384 | 2.544288 | 2.342758 | 7.920894% |
| Actual chain, skewed M=98304 | 14.273416 | 12.960874 | 9.195716% |
| W2 single, balanced M=98304 | 5.915060 | 4.611928 | 22.030746% |

Main-chain speedup is 1.10251055, with group-bootstrap 95% interval
[1.10189365, 1.10365339], above the frozen 0.250595% baseline-noise gate.
None of 28 event/host median/p95 regression checks triggered. Chain M=512
has no established gain. These are operator results from three independent
groups, not full-model throughput or concurrent-deployment measurements.

Main-chain event medians (ms/call), retained for direct paired recomputation:

| Group | A1 | B1 | B2 | A2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 14.028779984 | 12.702919960 | 12.664860249 | 13.968490124 |
| 2 | 14.000279903 | 12.722589970 | 12.708100319 | 14.021639824 |
| 3 | 14.020480156 | 12.737730026 | 12.728660107 | 14.043119907 |

## Precision and resource checks

The existing device suite passed 31 tests with zero skips:

```bash
SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 python -m pytest -q \
  tests/npu/test_async_cam_layered_w4a8.py \
  tests/npu/test_gmm_layered_device_group_list.py \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py
```

Additional differential probes passed ND N=288 FP16/BF16 tails; group-list
encodings 0/1; quantization groups 1/2; ten dynamic M values with balanced/skewed
and empty-expert routing; and selectors 0→7→0 in one graph. W13 int8 outputs,
FP32 scales and final chain outputs matched baseline bytes, including M=98304.
Three retained-tensor graph replays had bounded allocations; releasing the graph
returned allocated memory to input/weight residency. Long-running/multi-graph
memory behavior was not tested. GPU E2E and model-level throughput were not run.

## Profiling and lessons

Independent graph profiles show approximately 3 us between W13 and W2, favoring
kernel-internal work over launch-gap changes. After MDL, W2 MTE2 busy time changes
from 4675.414 to 3033.624 us while MAC busy time stays 2616.889 versus 2616.828 us;
W13 activities remain approximately unchanged. This supports reduced W2 transfer
and scheduling overhead, not a source-line or whole-device bandwidth claim.

Prior synchronization-removal candidates had no established graph gain and were
reverted. Increasing W13 workspace regressed M=16384 chain latency by 4.596% and
was reverted. None of those changes are in this PR.

Two validation-harness corrections were made before formal candidate timing:
the builtin cumulative-list reference was normalized to counts/type1 while
layered type0 remained tested, and the first timed graph gained explicit warmup.
The old five A/A blocks were excluded and fresh thresholds collected. Failed
and superseded runs remain in the local experiment archive, alongside complete
raw profiles, hashes and per-sample timing records. This report contains the
review-facing results; it does not claim that remote-only artifacts are public.

## Follow-up W13 tile experiment (rejected)

A separate experiment changed only W13 A8W4 MSD host tiling from
128×256×256 to 256×128×256, using retained W2 MDL as its baseline.
The intent was to rebalance A/B L1-to-L0 instruction cost; it does not reduce
ideal total packed transfer bytes. CANN source selects direct LoadData3DPro
for A and repeated LoadDataWithTranspose for B. These source observations are
a hypothesis, not an instruction-level measured bottleneck.

The task restarted on another A3 node before this experiment. The 44 retained
baseline build artifacts matched their sealed hashes, and fresh A/A blocks
established new noise thresholds. Independent candidate build, 31 native tests
(zero skips), and exact boundary W13 q/FP32-scale/chain comparisons for both
group-list encodings passed. Runtime logs confirmed the candidate host library
and 256×128 tile, unchanged mLimit=1638 and ACLNN workspace=83,870,208 bytes.
Three graph replays with four retained tensors showed no allocated-memory growth.

A 12-shape diagnostic pilot triggered the predeclared regression confirmation
at M=3277. Five fresh A/A processes and three independent ABBA groups confirmed
**6.338645% higher actual single-chain graph event-median latency**, from
0.858917405 to 0.913361128 ms. All three B/A group ratios exceeded the 1.420935%
noise threshold. Speedup was 0.940391898 with three-group bootstrap 95% interval
[0.928349916, 0.950339727]. Host median and event/host p95 also regressed beyond
their own thresholds in every group.

| Group | A1 ms | B1 ms | B2 ms | A2 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.876879990 | 0.907660007 | 0.910479993 | 0.851160020 |
| 2 | 0.880030006 | 0.909600019 | 0.915769994 | 0.841030002 |
| 3 | 0.854310006 | 0.909180015 | 0.927630007 | 0.850809991 |

Candidate `f15ae6e` was reverted by `525a9d0` in the local experiment branch;
neither commit is part of this PR. Under the frozen early-rejection rule, the
M98304 benefit matrix and conditional candidate PMU were canceled. M16384 was
4.406384% faster in the pilot only; this is not a confirmed gain and cannot be
extrapolated to M98304. W2's retained result is unaffected.

The lesson is that a fixed tile can favor larger expert fragments while hurting
small fragments and workspace boundaries. A possible separate experiment is
tile selection based on actual device-side expert/slice rows, with consistent
Matmul resources and task partitioning. Host allocation capacity cannot safely
make that choice; switching cost and any adaptive benefit remain unverified.

## Device-adaptive W13 tiles (retained)

After rejecting the fixed tile, candidate `463ff4e` selects between 128×256 and
256×128 from each actual expert/workspace-slice intersection on device. Host
capacity does not determine the choice. It compares M16/N64-aligned packed
A+B traffic and task counts, selecting the alternative only when neither grows
and B traffic decreases. K256, workspace, quantization and cross-core pipeline
synchronization are preserved. The cost model is not a latency guarantee.

Two complete host-generated Matmul configurations are initialized once. All
AICs make the same choice, including cores without an assigned task. Empty
intersections do not switch; active selection persists across workspace slices.
On a change, the previous object calls End, then public SetMMRowMajor restores
the shared layout state cleared by End. Task grids, tails and offsets use the
selected configuration. Repeated Init was rejected because it accumulates buffer
handles; resetting TPipe would affect event state. Installed CANN 9.1 source
supports shared hardware buffers and a derived bound of 16 handles for the two
objects, while device tests provide the concrete reuse evidence.

This experiment compares against the **retained W2 MDL baseline**, on the new
A3 node used by the rejected fixed-tile follow-up. Five fresh A/A processes
preceded three ABBA groups (12 processes), each measuring the same eight graph
cases with ten warm replays and 50 samples. Times below are geometric aggregates
of six baseline and six candidate process medians. Do not combine these timings
with the earlier node's W2 results to claim an end-to-end cumulative speedup.

| Actual single-chain graph | W2-MDL baseline ms | Adaptive W13 ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Balanced M=98304 | 12.661287 | 12.072201 | 4.652655% |
| Balanced M=16384 | 2.329549 | 2.204595 | 5.363888% |
| Balanced M=3277 | 0.855156 | 0.856919 | -0.206142% |
| Balanced M=512 | 0.686131 | 0.685848 | 0.041170% |
| Skewed M=98304 | 12.911829 | 11.898706 | 7.846475% |

Main latency reduction is 4.652655%, above its frozen 0.293774% noise threshold;
the three-group speedup 95% interval is [1.047238244, 1.051524820]. Main host
median and event/host p95 also improve. None of 32 event/host median/p95 checks
meets the predeclared reproducible-regression rule. M3277's +0.206142% latency
and M512's −0.041170% latency are within their approximately 1.82% noise gates;
no improvement is established for those small cases. Only three independent
groups were used, and samples within a process are not independent replicates.

Switching controls keep total M=516 and the count multiset (four experts with
64 rows, four with 65, the others empty) constant. Clustered versus alternating
order produces one versus seven source-modeled tile transitions. The event
latency difference-in-differences, (candidate alternating−clustered) minus
(baseline alternating−clustered), is +3.3637, +1.7306 and +5.5648 microseconds
across the three groups; host values are +2.2364, +1.3357 and +3.6524 microseconds.
This indicates an ordering/switching-associated net cost, not an isolated API
cost: count-to-weight mapping and access locality also change. Both controls
and same-graph alternating metadata pass the frozen no-regression gates. Their
larger noise thresholds remain a limitation, not evidence that switching is free.

Independent build and 31 native tests passed with zero skips. The first pressure
gate changes M98304→3277→98304 in one graph with selectors 0/7/0 and eager
references; q, finite FP32 scale and actual-chain output fingerprints exactly
match baseline. The main workload models 94 transitions, 47 in each direction;
these are source-derived counts, not hardware measurements. Two group-list
encodings, 13-M boundaries and 25 logical same-graph scenarios (including empty,
skewed and alternating controls) pass exact baseline and cross-encoding checks.

Fresh capture and three replays allocate 11,341,407,744 bytes on both sides;
release returns to 8,654,953,984 bytes of resident inputs/weights. Actual loading
uses the candidate W13 object and host library. All 13 W2 kernel objects are
byte-identical to baseline. Long-running/multi-graph behavior and model throughput
are not established. The PR cherry-pick has byte-identical relevant sources;
its rebased checkout was not rebuilt separately on device.
