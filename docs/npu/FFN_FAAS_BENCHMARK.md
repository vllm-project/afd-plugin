# Fixed BF16/eager FFN comparison

> This is the V2-Lite comparison launcher recorded on the
> `codex/ffn-faas-megakernel` branch. It is reference material for the BF16
> backend and makes no V4 fusion or performance claim; its `output/` paths are
> local experiment artifacts, not files in this checkout.

This launcher is prepared for review. It has **not run on NPU**. Performance
acceptance remains pending; earlier functional acceptance does not establish
that the implementation meets the user's performance requirement.

The primary comparison is the existing `CAMAsyncAFDConnector` versus
`FaaSAsyncAFDConnector`, both on the same four DIE: A DP1/TP2, F EP2/TP1.
The optional native TP2/EP2 complete model uses two DIE and is an auxiliary
reference. It cannot isolate service or kernel overhead.

## Fixed workload and timing

All modes use the original local DeepSeek-V2-Lite unquantized checkpoint,
BF16, eager execution, EP enabled, SP/DBO off, max model length 256,
`max_num_seqs=2`, `max_num_batched_tokens=32`, memory utilization 0.6,
prefix caching off and seed 0. The four original prompts are imported from
`ffn_faas_inference.py`, including the Question input. Each call uses greedy
generation, max 16 output tokens and prompt/generated logprobs=1.

Each mode loads once, then performs **3 warmups and 7 measured calls in that
same LLM process**, with the same prompts and parameters. No model reload,
worker restart or cache flush occurs between samples. Prefix caching remains
off. All ten samples, outputs and log probabilities are retained. The driver
checks 64 output tokens and equal greedy IDs across samples; it does not
silently drop an incomplete or slow iteration.

`generate_seconds` measures `LLM.generate` through its synchronous return,
including request processing, prefill, decode and vLLM's output construction.
It excludes the harness's record conversion, JSON writes, model load and
shutdown. It is not kernel latency or decode-only TPOT. The older
`inference_seconds` field also includes record conversion; both timings are
saved so that the historical boundary remains explicit. Diagnostic boundary
hooks are unavailable through this launcher.

The report retains ordered durations, min/max, mean, median and sample standard
deviation. No performance pass percentage has been chosen. Compare all samples
and their variation, with the user's requirement of no obvious regression.

## Static verification of the old path

| Item | Existing path used by the launcher |
| --- | --- |
| Old A | Same offline LLM arguments as FaaS; `legacy-attention` selects CAM async, Attention-owned gate/shared experts and `dynamicQuant=0`. |
| Old F | Pinned vLLM CLI, `data_parallel_size=2`, `tensor_parallel_size=1`, EP enabled; AFD FFN worker/connector daemon with eager `AsyncNPUFFNModelRunner`. No quantization or Graph shortcut. |
| F routed arithmetic | `deepseek_v2_attention_gate.py` explicitly supports `QuantType.NONE`, passes BF16 `w13_weight`/`w2_weight` into the Ascend MoE MLP. The layered W4A8 and two-stream paths are off. |
| Rank mapping | A role ranks 0/1 are TP0/1 in DP0. Old F role ranks 0/1 are DP0/1, TP0; `build_async_topology` maps them to CAM world ranks 2/3 and 32 contiguous routed experts each. FaaS torchrun ranks 0/1 likewise map to session world ranks 2/3. |
| Device mapping | Each role gets its explicitly selected runtime visibility mask. Inside its two-device mask FaaS binds logical 0/1; the old vLLM F workers also bind role-local 0/1. The masks must come from the actual allocation, never guessed physical IDs. |
| Communication buffer | Both use a 64 MiB domain and `HCCL_BUFFSIZE=64`. The latter is required by old CAM host tiling for the window-tail address, independently of the HCCL per-group override. The old kernels check required bytes against this size. |

The old CAM sender passes max sequence capacity 32 and local row count to
tiling. Its FFN receive capacity is 32/TP2=16 rows from each A rank; with H2048,
K6, 32 local experts the token/count/address regions are far below 64 MiB.
This is a static size check, not hardware liveness or performance evidence.

## Prepare and inspect

Run from this checkout with the matched Python interpreter. Set `A_VISIBLE`
and `F_VISIBLE` from the allocated runtime IDs after checking the allocation's
logical/physical mapping. They are values for `ASCEND_RT_VISIBLE_DEVICES`,
not logical indices to reinterpret inside the inherited mask. If that variable
already restricts the process, the launcher checks that selected IDs are a
subset. It keeps container/device allocation variables inherited.

```bash
python -m tests.e2e.models.deepseek_v2_lite.ffn_faas_benchmark \
  --mode legacy --model /path/to/DeepSeek-V2-Lite \
  --attention-visible-devices "$A_VISIBLE" --ffn-visible-devices "$F_VISIBLE" \
  --output-dir output/benchmark-legacy-plan --plan-only
```

Use `--mode faas` and a different new output directory to inspect FaaS. Native
uses `--mode native` and only `--attention-visible-devices`. `--plan-only`
imports no torch/vLLM, starts no process and touches no device; it writes
`plan.json` with the commands, controlled environment, workload and topology.

Before a later approved device run, verify the pinned vLLM/Ascend source,
checkpoint identity, package import path, actual loaded extension and vendor
paths. Build **both** FaaS and the four old CAM operators in the same private
checkout/package. A previous `AFD_BUILD_OPS=afd_ffn_faas` build alone is not
proof that the old four installed kernels match the source. The normal
910C registry build includes all five; remove a stale subset override for that
build. Archive the repo/staged/installed source hashes and extension hash/maps.
Do not modify the shared stack or inherit CWS/checkpoint experiments.

The launcher aligns `HCCL_BUFFSIZE`, disables layout/debug logging and removes
launch-blocking, FlashComm1 and server-dev observer environment switches. It
explicitly sets `AFD_FORCE_BALANCED_TOPK_IDS=0` and
`AFD_ASYNC_CAM_LAYERED_GMM=0` in each child: inherited experiments must not
change routing or select/reject the old CAM path. The parent environment is
preserved. Verify that no worker extension or other observer was injected;
log silence alone does not prove absence of synchronization overhead.

## Later device execution and results

After review authorizes device work, omit `--plan-only` and use a **new** output
directory per mode. Run legacy and FaaS serially on the same selected DIE; record
their order. Normal progress has no fixed overall run deadline; investigate a
stalled run from its logs, samples and owned processes before cancelling it.
Communication/drain and owned-process cleanup retain their technical deadlines.
Failure retains any completed warmup/repeat rows. The old F HTTP server
is stopped by owned-process cleanup after A finishes; that is not evidence of
FaaS-style normal release. FaaS additionally checks both rank reports for normal
drain and released resources. Cleanup targets a unique inherited run ID.

Output files are `plan.json`, `samples.jsonl`, `inference.json`, A/F logs,
`driver.json`, and FaaS rank status reports. Compare original numerical bounds
before interpreting performance, using the existing comparison command with
the native report as auxiliary numerical oracle:

```bash
python -m tests.e2e.models.deepseek_v2_lite.ffn_faas_inference compare \
  --baseline output/benchmark-native/inference.json \
  --candidate output/benchmark-faas/inference.json \
  --output output/benchmark-faas/comparison.json
```

That command compares the final sample; all other sample records remain in
JSON/JSONL for numerical and variation review. It must not be described as
having compared every sample. No device execution, speed improvement or final
performance acceptance follows from preparing these files.
