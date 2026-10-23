> Main performance acceptance was corrected to DeepSeek-V4 on 2026-10-23.
> The V2-Lite results below are regression/diagnostic evidence only.
>
> Provenance: this report records V2-Lite evidence collected on the
> `codex/ffn-faas-megakernel` branch before the V2 implementation was merged
> back into the framework branch. The `output/` archives it cites are local
> experiment artifacts and are not part of this checkout.

# FFN FaaS optimization validation, 2026-10-23

**Performance acceptance remains pending.** The latest measured product, v34,
uses an independent FFN service and one fused receive/select/group/Cube MLP/
writeback device entry. F has no vLLM EngineCore, request scheduler or KV cache.
A keeps routing and shared experts. A bounded model scope queues at most four
serial exchanges and validates every completion before returning a model result.

## Current fixed-workload evidence

The model remains BF16 DeepSeek-V2-Lite, eager, A DP1/TP2 and F EP2/TP1 on the
same four DIE. All original four prompts, maxseq2/batched32, 16 generated tokens
per prompt, seed0 and disabled prefix cache are retained. Transport capacity is
eight rows per TP member with two slots and a 64 MiB window. Each mode has one
load, three warmups and seven measured synchronous LLM.generate calls.

| Version | Mean generate | Median | Std. dev. | Original model numerics |
| --- | ---: | ---: | ---: | --- |
| Initial FaaS, historical | 7.419419 s | 7.419259 s | 0.060457 s | 10/10 |
| v32 old AFD | 1.633316 s | 1.648323 s | 0.048795 s | 10/10 |
| v34 C++ submission/context | 1.715721 s | 1.699538 s | 0.061354 s | 10/10 |
| Latest old AFD after v34 | 1.570169 s | 1.551502 s | 0.050752 s | 10/10 |

v34 ran at 00:45:17–00:46:56 UTC; the subsequent old AFD run was at
01:13:18–01:15:11. The interval is recorded and these are not simultaneous or
strictly adjacent measurements. v34 is 9.2698% slower in mean and 9.5414% in
median against that latest old AFD run. No numerical performance tolerance was
invented to accept this gap. Old AFD is the performance reference; the original
non-AFD/native model remains the numerical reference only.

All ten v34 samples retain 64 greedy IDs and pass the 95 original logprob values
(maximum absolute error 0.096002579, NRMSE 0.010515318). The bounds remain 0.15
absolute error and 0.02 NRMSE. Both drivers exit 0/0 with no cleanup failures;
FaaS ranks stop and release resources. Earlier faster/slower old samples remain
archived and are not substituted to make the candidate pass.

## Affected execution and failure checks

v34 normal and scoped 13-call four-rank probes pass all 52 saved tensor records
at the original expert/combine bounds: atol=0.06, rtol=0.02 and NRMSE<=0.02.
They cover independent sources, zero work, repeated slots, a partial scope tail
and actual queued clone consumers. These are functional checks, not allocator
pressure or hardware-fault proofs. The F thread affinity fix is verified in the
actual model process and uses the existing Ascend CPU planner without IRQ,
shared memory or page-migration changes.

A separate v33b actual-worker test first observes four OK receipts and changes
the third to INVALID. Both A workers discard model state and return no model
result; F fails and releases resources. Cleanup has no process leftovers. This
is completed-receipt corruption, not a hardware or partial-submit injection.
That evidence is reused for v34's unchanged failure path. Unknown submission
failures retain C++/Python owners until the existing bounded abort/unsafe-exit
boundary; those exact queue/allocation failures have not been device-injected.

## Current candidate and reference correction

v35d is the last V2-Lite build here. It retains the connector's first BF16
rounding, moves only the FP32 expansion into combine and keeps the generic FP32
input path. Its initial BF16 submission was rejected by stale generated dtype
support metadata; that was a registration/build failure, not a BF16 kernel
numerical result. After the generated interface was refreshed, the fixed
workload measured above gave mean **1.617635 s**, median **1.540660 s** and
standard deviation 0.135426 against the same-resource old AFD mean 1.570169 s
(ratio 1.03023: 3.02% slower on the mean, 0.70% faster on the median). The two
runs were not interleaved, so this is not a non-inferiority result; the
independent check records `goal_met: false`. All ten full-model samples pass the
unchanged limits (greedy IDs equal, maximum logprob error 0.0960 < 0.15, NRMSE
0.0105 < 0.02), and 208 tensor comparisons were recomputed on CPU.

Expanded E64/top-k6 synthetic checks exposed a false expert comparison on both
exact v34 and v35. The CPU FP32 dot product crossed a BF16 midpoint. The exact
saved-BF16 dot is 12287997/2097152, below 5.859375 by 1.430511474609375e-6;
the CPU FP32 approximation rounded on the other side. FP64 projections with
all original BF16 boundaries and FP32 SwiGLU/combine reproduce the device result.
The native probe therefore uses FP64 only for its independent CPU projections
and records reference_projection_dtype. All 19 retained successful/failed tensor
files pass the unchanged bounds with that uniform reference, as independently
reviewed. The original full-model reference and thresholds are unchanged. The
old reference and first-failure tensors remain in the evidence archive.

Evidence under output/optimization-20261022 includes all original samples,
source/native manifests, actual loaded-worker maps, saved tensors, failures and
reproducible scripts. Supervisor audits independently recompute these records.
Profiler/CPU-plan runs and warm GMM diagnostics are not performance acceptance;
F work totals include startup. Withdrawn candidates and their negative results
remain archived. No multi-op GMM fallback is used by the service.

The historical validation below describes its stated earlier versions only.

---

# Historical FFN FaaS validation, 2026-10-22

The fused native path, independent FFN service, asynchronous transport, repeated
slots, lifecycle and full-model numerical checks passed for the configuration
below. After matching the native BF16 combine-probability boundary, the final
uninstrumented run has 64 identical greedy token IDs and 95 compared log
probabilities with maximum error **0.148594856≤0.15**, NRMSE
**0.016621949≤0.02**. No tolerance was relaxed and no failed prompt was removed.
Earlier failures remain archived.

**Current overall acceptance: performance pending.** These device results
describe the functional version through `e54c409`; they do not validate the
later local padding change or establish acceptable performance. Two saved
uninstrumented runs measured native TP2 at 2.83369 seconds and FaaS 2A2F at
7.66470 seconds, an adverse signal with different device placement and no
process-local warmup/repeats. A still earlier native run took 11.19033 seconds;
all are retained. None is a same-topology comparison against old AFD.
See the [prepared benchmark](FFN_FAAS_BENCHMARK.md). No new NPU run has been
performed for this local follow-up; the original numerical bounds remain.

## Implementation and environment

The V2 evidence branch was `codex/ffn-faas-megakernel`, based on
`6759d07db329ea3d21498069c04ff1f3aa44ccdb`. The device evidence below uses
production Python through `e54c40911be69ce588b4c94cd7b0acd580352018`: the connector's
combine-probability rounding. Diagnostic tests are at `5877a09`. Native code
last changed at `1118135cf9bbc91b116d387383a5fd3683d25a7f`.

- Ascend 910C/A3, eight allocated DIE; checks use four or six logical devices.
- Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0.
- Clean vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend
  `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`; no historical CWS patches.
- Local DeepSeek-V2-Lite BF16 checkpoint: H=2048, I=1408, 64 routed experts,
  top-k=6, routed layers 1–26, two shared experts, greedy top-k, no probability
  renormalization, scaling factor 1.0. The config-and-index fingerprint is in
  each rank's saved session specification; it is not a hash of all checkpoint
  tensor bytes. Both launchers used the same local mounted checkpoint.
- Full model: eager A DP1/TP2 and F EP2/TP1. Asynchronous service: A DP2/TP2 and
  F EP2/TP1. Transport capacity eight rows per TP contribution, two reusable
  slots; larger model calls are chunked.

`afd_plugin/ffn_faas/` owns the CPU protocol, FFN-only checkpoint loader,
executor and service. `afd_plugin/connectors/npu/ffn_faas.py` connects Attention.
`csrc/npu/ascend_kernels/afd_ffn_faas/` implements one F EXECUTE entry containing
input reception, device selection of source/layer, expert SwiGLU and remote
result publication. The pybind binding calls this operator directly; F does not
launch separate recv, GMM or send operators and has no F–F collective in its
work loop. A retains routing, shared experts and dense layers.

FFNModelRunner also performs device binding; a separate empty Worker class was
unnecessary. F loads only its contiguous routed shard: 832 layer/expert entries,
14,394,851,328 parameter bytes per rank. It creates no EngineCore, request
scheduler, Attention model or KV cache. Computation uses AIV FP32 accumulation
with BF16 boundaries; Cube optimization is outside this validation.

FastAFD's CUDA finite-lane/host-plan design informed weight descriptors but was
not transplanted as an Ascend asynchronous protocol. ASAP informed readiness,
backpressure and device layer selection; its communication overlap alone does
not establish fused communication and computation. Existing Async CAM remains
a separate path. These distinctions are reflected in the implemented ABI.

## Results

All paths below are relative to this checkout's local, untracked `output/`.
They identify saved evidence, not files required at runtime.

| Check | Result | Evidence and scope |
| --- | --- | --- |
| Native build and attribution | Pass | `build-1118135-clean.log`, `1118135-artifact-attribution.json`: source, staged and installed dynamic kernel hashes agree. `runtime-maps-ae11c2a-fullnormal-v1.json` records the actual A/F process mappings and extension/vendor hashes. |
| Expert and combine numerical checks | Pass | `f3c00af-device-evidence.tar.gz`: 24 synthetic 2A2F, 48 synthetic 4A2F and 12 real-checkpoint layer-1/2 input cases, with 84 saved tensor records. Independent CPU FP32/BF16 expert and combine references; real expert max NRMSE 0.000418136, combine 0.000492934. |
| Independent F/source progress | Pass | The same native archive includes unequal idle launch counts, different initial F source choices, multiple A sources at different layers/speeds, slow consumers and backpressure. These are direct protocol checks rather than throughput inferences. |
| Sequence/EOS admission | Pass | `lifecycle-da37a1b-evidence.tar.gz`: replay/skip rejection, EOS(0), nonempty EOS and final generation boundary. The maximum-boundary test seeds private counters; it does not run billions of rounds. |
| Full FFN-only service and reuse | Pass | `service-a9bee8f-normal-v1/`: four A clients × 156 CPU comparisons across all 26 routed layers; two F ranks each complete 364 work items. Includes unequal TP rows, exhausted-member zero work and all-zero calls. Store samples stay within 32–35 keys, including finite lifecycle keys. |
| Production aligned TP chunks | Pass | `service-7f364b7-aligned-v1/`: four A × 32 comparisons, tokens 0/1/8/9/16/17, capacity 8, two slots and independently phased DP sources. Every A reports `shape_cells_created=0`; max NRMSE 0.000792745. A/F exit zero and report released resources. |
| Idle/pending stop and killed F | Pass, bounded failure | `service-da37a1b-{idle-stop,inflight-stop,killed-peer}-v1/`. The pending-stop probe proves published, unconsumed A work; it does not establish that the native compute body was active at the instant of stopping. |
| Store owner killed | Pass, bounded failure | `service-ae11c2a-lost-store-v1/`: A0 SIGKILL after readiness/arming; the remaining three A clients report the specified peer/Store failure. |
| SIGTERM | Pass, bounded failure | `service-18045aa-signal-stop-v1/`: real SIGTERM to F0. CPU subprocess tests also inject signals while service/Event locks are held. Handlers only set a plain pending flag. |
| Actual vLLM normal drain | Pass | `fullmodel-e54c409-normal-v1/`: A/F exit zero, both F ranks stopped with resources released and 964 completed work items. The driver validates lifecycle, separately from numerical comparison. Earlier `18045aa` also passed drain. |
| Actual vLLM killed F | Pass, bounded failure | `fullmodel-18045aa-killed-peer-v1/`: the driver waits for actual generate to start and F work to increase before killing F0. A/F fail within bounds; post-run device checks show no worker residue. |
| Full-model numerical comparison | Pass | `fullmodel-e54c409-comparison.json` and the separately recomputed `fullmodel-e54c409-offline-comparison.json`: 64/64 greedy IDs agree, all 95 prompt/generated log probabilities pass; max absolute error 0.148594856, NRMSE 0.016621949, against the original uninstrumented `18045aa` baseline. No hooks are enabled. |
| Question probability boundary | Pass | `question-5877a09-evidence.tar.gz`: actual native routing and 78 raw expert outputs reconstructed from the sorted input, expert map and counts. Native expert outputs plus BF16-rounded probabilities reproduce all 26,624 native routed values exactly. The resulting connector fix is isolated to A. |
| Native baseline repeat | Pass | `baseline-aa-18045aa-analysis.json`: two baseline runs have identical input IDs, 64 generated IDs and all 95 compared log probabilities. NRMSE is zero. |
| Affected regression seams | Pass within recorded scope | `runtime-unit-18045aa-v2.log`: 475 passed, zero skips in the matched runtime, covering FFN/config/model construction and weight policies, connector factory/old CAM seams and NPU model-runner shutdown. This is not an old CAM hardware rerun. The final connector is exercised by the original full-model comparison; changed source and test files pass mypy/Ruff/diff checks. |

The native expert/combine bounds were fixed before execution: atol=0.06,
rtol=0.02 and NRMSE≤0.02. Full-model bounds were also fixed before execution:
all greedy IDs equal, every compared log probability within 0.15 absolute error,
overall NRMSE≤0.02. Before the fix, the sole absolute-bound violation was prompt index 1
(`Question: What is 7 plus 5? Answer:`), prompt position 6, token 8109:
baseline −7.4773645401 versus candidate −7.2232799530. Identical repeated
candidate results and the exact baseline repeat establish reproducibility of
the original discrepancy. Its combine-probability cause and the affected full
model's passing regression are documented below. The corrected Question value
is −7.3287696838; its difference from the original native value is 0.148594856.

Native protocol, F expert computation, generic client probabilities and
lifecycle code did not change in `e54c409`; their earlier device evidence is
reused. The new full-model run covers the changed production connector and
normal drain. Fault hardware checks were not mechanically repeated.

## Bounded numerical diagnosis

`boundary-4c58e8a-{baseline,normal}-v1/` samples only the first actual prefill's
first MoE layer. It is the five-token Paris prompt, not the failing Question
position. Diagnostic hooks copy tensors to CPU and synchronize devices; these
runs cannot support production performance claims.

`first-moe-4c58e8a-analysis.json` and
`first-moe-4c58e8a-reference.json` show:

- Input IDs, positions, MoE input, gate input and FP32 gate logits are bitwise
  identical. Both actual gate weights are FP32. No gate precision fix follows
  from this evidence.
- Shared gate/up and activation are identical. The final global shared output
  is identical, and shared+routed equals the recorded MoE output in both runs.
- Candidate versus native routed output: max error 0.001953125, NRMSE 0.00235913.
  Against an independent CPU FP32/BF16 reference using the captured input,
  candidate route IDs/probabilities and checkpoint weights, candidate NRMSE is
  0.000370259 and native NRMSE 0.00237458. Both fit the local numerical bounds.
- Native route IDs were not captured because its fused path bypasses the hooked
  method. Exact native/candidate route identity is therefore unverified.

The early `first-moe-4c58e8a-regroup.json` records a discarded hypothesis:
already globally reduced native values were summed again. Its 0.5 NRMSE is
double counting, not a model discrepancy. Direct global comparisons above
supersede that analysis; the raw record is retained.

This Paris sample did not establish the Question discrepancy's cause. The user
subsequently authorized one baseline/candidate Question comparison and one
necessary regression, keeping the original model, topology, batch and criteria.

### Question comparison and minimal fix

`question-5877a09-{baseline,normal}-v1/` retains the original four-prompt workload
and observes only the prefill containing Question's exact 12 input IDs. Both
runs capture 13 rows: one Paris decode row, followed by Question. Metadata
records `target_row_start=1`. Prompt logprob position 6/token 8109 is predicted
from Question position 5, hence global row 6. Both original failing logprob
values reproduce exactly at that position. Native routing chunks contain seven
and six rows; candidate transport padding gives seven and seven, cropped back
to 13 for comparison.

`question-5877a09-analysis.json` records identical MoE inputs, actual route IDs,
probabilities, FP32 gate logits and global shared output. The native/candidate
expert-output NRMSE is 0.000121745. Native activation equals the FP32 SwiGLU
reference with BF16 output exactly; gate or expert-precision changes are not
justified. `final_norm` differs at the causal row, but no intermediate model
layers were traced.

`question-5877a09-combine-dtype.json` isolates the combine boundary. With the
saved native expert outputs and original FP32 probabilities, CPU combine
differs from native by NRMSE 0.001974132. Rounding those same probabilities to
BF16, then accumulating in FP32 and casting the result, restores every native
routed value exactly. Candidate expert outputs with that boundary give NRMSE
0.000117169 relative to native. The pinned Ascend token dispatcher's AllGather
and All2All token-unpermute paths explicitly cast probabilities to the hidden
dtype. Independent supervisor reconstruction from raw sorted input, counts,
expert maps and GMM2 output confirms this result.

`e54c409` therefore rounds probabilities in the FaaS model connector immediately
before `AttentionSession.send`, then retains FP32 storage and accumulation.
Route selection, shared/dense experts, F math, native ABI, ownership, generations
and credit are unchanged. This matches the model reference; no threshold was
changed and no upstream patch was added.

The two diagnostic runs took 78.80 and 89.45 seconds including startup/shutdown,
within the 30-minute limit. Some other prompt logprobs in the observed baseline
differ from its earlier unobserved run; the observed candidate matches its
earlier outputs. Diagnostic results do not replace the original baseline.
The sole post-fix full-model run disables all hooks and passes the original
comparison against that original baseline. No additional device experiments
were performed.

## Cost and failure boundaries

Recorded elapsed times include launch/initialization and shutdown: full 156-call
service 57.59 s; aligned 32-call service 33.47 s; full-model baseline 75.12 s;
full-model FaaS after the fix 89.48 s. For the latter pair, saved generation time
for 64 tokens is 2.834 s and 7.665 s respectively (aggregate 22.59 and 8.35
tokens/s). Earlier
baseline initialization/cache effects produced different elapsed times. These
small correctness workloads establish neither acceleration nor a general
throughput ratio. FFN parameter storage is 13.40625 GiB/rank; raw client memory
samples and NPU health/process snapshots are retained in each run directory.

Normal release requires local completion plus peer quiescence. Some deliberate
peer-kill cases cannot prove quiescence and correctly record
`resources_released=false`; owned objects stay alive until process exit. Device
runtime/process reclamation is the observed cleanup boundary in those cases,
not a successful destructor-based release. Intentional faults can emit IPC
resource-tracker warnings. No shared device reset or system-stack change was
used. Earlier build, startup and shutdown failures remain archived and are not
reclassified as passes.

The owned ordinary task `afd451-mrv2-1015` was stopped after each final evidence
transfer; persistent storage was retained. The final follow-up's
`resource-return-e54c409.json` and `final-e54c409-task-describe.log` record the
verified Stopped state. The final workspace process scan is empty and all eight
DIE report OK with no NPU processes in `final-e54c409-resource-health.log`.

## Reproduction

Use an existing authorized allocation with at least six logical NPU devices for
4A2F, the matched dependencies above and a local checkpoint. Preserve the
allocation's device visibility mask. From this checkout:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
export HCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com HF_HOME="$PWD/output/huggingface"
# Build once using FFN_FAAS_SERVICE.md; never reuse an unverified stale vendor.

timeout --signal=TERM --kill-after=20s 240s python -m \
  tests.npu.ffn_faas_service_driver --model /home/admin/model \
  --case normal --attention-ranks 4 --rounds 156 \
  --output-dir output/reproduce-service

timeout --signal=TERM --kill-after=20s 240s python -m \
  tests.npu.ffn_faas_service_driver --model /home/admin/model \
  --case normal --attention-ranks 4 --rounds 32 --tp-rows-aligned \
  --output-dir output/reproduce-aligned

timeout --signal=TERM --kill-after=20s 600s python -m \
  tests.e2e.models.deepseek_v2_lite.ffn_faas_driver --case baseline \
  --model /home/admin/model --output-dir output/reproduce-baseline
timeout --signal=TERM --kill-after=20s 600s python -m \
  tests.e2e.models.deepseek_v2_lite.ffn_faas_driver --case normal \
  --model /home/admin/model --output-dir output/reproduce-faas
python -m tests.e2e.models.deepseek_v2_lite.ffn_faas_inference compare \
  --baseline output/reproduce-baseline/inference.json \
  --candidate output/reproduce-faas/inference.json \
  --output output/reproduce-comparison.json
```

The final command applies the original numerical criterion; the recorded
post-fix run passes it. Driver `--case killed-peer` reproduces the full-model fault check;
service driver cases are `idle-stop`, `inflight-stop`, `killed-peer`,
`lost-store` and `signal-stop`. Use fresh output directories and run serially.
Commands, fault markers, rank status, pre/post health and logs are saved by the
drivers. `--boundaries` enables the diagnostic sampling described above.
Add `--boundary-prompt-index 1` to observe Question while keeping all four
prompts and their original batching configuration.

The archive also preserves `output/reproduce-first-moe-reference.py`, using the
same independent `CheckpointOracle.reference` as the saved CPU calculation, and
`output/analyze-saved-fullmodel.py`, which records every numerical error before
returning a nonzero exit status. Neither script starts inference or submits NPU
work. To reproduce the CPU reference from the already saved boundary tensors:

```bash
python output/reproduce-first-moe-reference.py --model /home/admin/model \
  --baseline-dir output/boundary-4c58e8a-baseline-v1/boundaries \
  --candidate-dir output/boundary-4c58e8a-normal-v1/boundaries \
  --output-prefix output/reproduced-first-moe-reference
```

It requires the recorded runtime because the reused probe module imports
torch-npu; all reference tensors and arithmetic remain on CPU. Original
reference files are retained; the script refuses to overwrite them.

## Local evidence archive

The local `output/` retains these original archives, expanded run directories
and per-overlay source manifests. No model checkpoint is included:

- `f3c00af-device-evidence.tar.gz`: native prototype and real first-two-layer
  tensor comparisons.
- `lifecycle-da37a1b-evidence.tar.gz`: clean native build attribution, normal
  service/reuse, EOS boundaries, early lifecycle and baseline evidence.
- `model-lifecycle-18045aa-evidence.tar.gz`: Store failure, SIGTERM, full-model
  normal/fault runs, baseline repeat, runtime regression and process mappings.
- `boundary-4c58e8a-evidence.tar.gz`: aligned chunks, first-MoE raw tensors,
  initial diagnostic failure and corrected runs.
- `first-moe-4c58e8a-reference-evidence.tar.gz`: independent CPU reference and
  selected reference/candidate/native tensors.
- `question-5877a09-evidence.tar.gz`: the two authorized Question diagnostic
  runs, actual native routing/expert boundaries and independent CPU analysis.
- `fullmodel-e54c409-evidence.tar.gz`: the single post-fix, uninstrumented
  full-model numerical pass, normal drain and final process/health snapshots.

The module and lifecycle milestones received independent supervisor review.
The three targeted fixes and the bounded numerical diagnosis were also reviewed;
the Question evidence and minimal connector fix were independently checked
before its sole full-model regression. Final delivery metadata identifies the
reviewed source commit and review result. No PR, external comment, merge or
deployment is part of this delivery.
