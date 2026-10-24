# Appendix: comparing multiple runs

The main skill produces one report per `RUN_ID`. This appendix covers
comparing several of them -- most often an AFD recipe against its
GPU-equivalent non-AFD baseline, but equally two AFD configurations
(eager vs graph, DP2TP1 vs DP1TP2) or the same recipe across images.

**Both recipes must be named explicitly by the user up front.** Never
derive, construct, or guess a counterpart recipe (e.g. a non-AFD baseline)
on the AFD recipe's behalf -- if the user asks for a comparison but gives
only one recipe, stop and ask them for the second `RECIPE_SCRIPT_PATH`
rather than building one.

## 1. Confirm the runs are comparable

Read every `run.json` first. A comparison is only meaningful when the runs
agree on **model, GPU count, and workload** (same dataset, ISL/OSL,
`NUM_PROMPTS`, and the same `SWEEP` points). Anything else being equal is a
bonus, not a requirement.

```bash
for d in ./reports/*/; do
  echo "--- $d"; cat "$d/run.json"
done
```

If a run you want is missing locally -- it was never copied out, or predates
the current session -- fetch it from the reports PVC first, per
[fetching-past-reports.md](fetching-past-reports.md).

Refuse to produce a delta if the workloads differ -- rerun instead. Two runs
at different offered rates or prompt shapes are not a comparison, and a
percentage between them is worse than no number at all. If the images or
nodes differ, the comparison is still valid but say so in the report: node
class and image are confounds worth naming.

Also confirm each run's report directory is genuinely its own -- one
`RUN_ID`, one directory. If two runs somehow share a path, discard both: the
sweep writes one file per point, so a shorter sweep leaves the earlier run's
files in place and the delta is computed against stale data.

## 2. Run the set

Run the main skill once per recipe, **sequentially** -- each run tears down
`vllm-pod`/`vllm-service` in its own step 9, so a second run cannot start
until the first has finished and the previous recipe's pod/service are gone.
Keep `MODEL_ID`, `PVC_NAME`, `GPU_COUNT`, the image, and the whole workload
(dataset, ISL/OSL, `NUM_PROMPTS`, `SWEEP`) fixed; change only
`RECIPE_SCRIPT_PATH` (and therefore `RUN_ID`).

Reuse the same model PVC throughout so weights download once. Prefer running
back to back on the same node: a node change between runs is a confound.

## 3. Compare

Compare **per sweep point**. At low offered rate variants usually look
alike; the interesting result is the rate at which one starts queueing and
the other doesn't.

For each sweep point present in all runs -- matched by `rate:concurrency`,
never across different sweeps -- tabulate the same metrics the single-run
report uses. **TTFT (`mean_ttft_ms`, `p99_ttft_ms`) and TPOT
(`mean_tpot_ms`, `p99_tpot_ms`) are required**, alongside `mean_e2el_ms`,
`mean_itl_ms`, `output_throughput`, and `completed` against `num_prompts`.

TTFT and TPOT are what make an AFD-vs-baseline delta interpretable: splitting
attention from FFN changes prefill and decode by different amounts, so a
single end-to-end number can stay flat while TTFT improves and TPOT
regresses (or the reverse). Report the two separately and say which moved.

Give both absolute delta and percentage, always naming which run is the
reference:

```
Reference: <baseline RUN_ID>
Compared:  <afd RUN_ID>
Model <MODEL_ID>, <n> GPUs, workload <name> (identical across runs)

| Rate | Conc | Metric | Reference | Compared | Delta | % |
|------|------|--------|-----------|----------|-------|---|
```

Call out, in this order:

1. **Saturation point per run** -- the first sweep point where TTFT departs
   from flat (it moves before end-to-end latency) or `completed` drops below
   `num_prompts`. A recipe that saturates two points later is the headline
   result, more than any single-point percentage.
2. **Which of TTFT / TPOT moved**, and in which direction. "Faster" with no
   split between prefill and decode is not a usable finding.
3. **Sweep points where the runs diverge**, with the direction named.
4. **Sweep points where they are within noise** -- say so explicitly rather than
   reporting a small percentage as a finding.
5. **Any run where no point saturated** -- the ramp measured headroom, so
   the comparison bounds the difference from below and nothing more.

A single averaged percentage as the sole result is not an acceptable
report: it depends entirely on how much of the sweep sat past saturation.
