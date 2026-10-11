# Async combine send: resolve peers only when sending

## Change

`AfdAsyncCombineSend::CombineSend` previously resolved a destination before the token loop and again for every visited `(expert, TP rank)` cell. This includes cells skipped while locating the current AIV's interval and empty cells that send no data.

Resolve the destination inside the actual send branch instead. A local `resolvedTpIndex`, initialized to the invalid value `tpSize_`, caches the TP rank currently bound to `dstTokenStoreGM`. Rebind only when that rank changes. Within one invocation, the peer base depends on the TP rank and fixed batch metadata, so it can be reused across expert cells and successive UB-sized transfers. The cache is reset on every invocation.

No payload arithmetic, token ordering, per-TP send counters, UB allocation, DMA size, synchronization, or completion publication changes. This is a single optimization in the plugin-owned kernel, independent of dispatch-send PR #439 and dispatch-recv experiments.

## Environment and build

Baseline: main `913c9d35453fdf6b20553929ced9d5af1da7c0a3`. Validation used `jcz-afd1`, Ascend910_9382, CANN 9.0.1, torch 2.10.0+cpu, torch-npu 2.10.0.post2, and devices 0–3 (two Attention ranks, two MoE ranks). Both isolated builds selected the four async CAM operators from the registry. The host extension was identical and only combine-send device objects changed. This was not a complete operator-registry build.

Header SHA-256:

- Baseline: `df2abebcdc356a2976de5b8475e03a530230573002eb24ffc8d36576d645deda`
- Candidate: `e13773885b22d3e7d95336d0cf58b2864a51eb6b5278a9a1d3f83a4599e73892`

Build each isolated checkout with CANN configured and the same registry selection:

```bash
source /usr/local/Ascend/cann-9.0.1/set_env.sh
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 AFD_BUILD_JOBS=8 MAX_JOBS=8 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 python setup.py build_ext --inplace
```

## Performance

`msprof` wraps the unchanged `async_cam_performance.py --measure-mode send`, with one precision preflight, five warmups, and five measured iterations. Sort `AfdAsyncCombineSend` records in each MoE device's `op_summary` CSV by task start time. Sum any chunk tasks within each iteration and then take the slower of MoE ranks 2 and 3. The reported number is the mean of those iteration values. All performance cases below produced one combine-send task per rank per iteration.

| Tokens | Total experts | Hidden | Baseline (μs) | Candidate (μs) | Time reduction | Samples per version |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7/5 | 8 | 256 | 21.001 | 18.738 | 10.8% | 10 |
| 97/65 | 256 | 256 | 51.381 | 40.852 | 20.5% | 10 |
| 7/5 | 8 | 7168 | 25.936 | 18.084 | 30.3% | 5 |

Small and many-expert configurations were tested in both baseline/candidate and candidate/baseline order. Many-expert reductions were 15.5% and 25.2% in the two pairs. The small case was effectively flat in the first pair (+0.24% time) and improved in the second (−19.8%); its aggregate reduction should not be interpreted as a stable per-run improvement. Hidden 7168 has only one acquisition. These are operator device times, not model throughput. Per-rank and per-iteration values are in [combine_send_peer_cache_samples.json](combine_send_peer_cache_samples.json).

Run from each separately built checkout, using unique output paths for every version and repeat:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD"
msprof --output=/tmp/combine-send-profile --application="torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_performance.py --measure-mode send --warmup 5 --iterations 5 --dtype bf16 --dynamic-quant 1 --lengths 7,5 --experts-per-rank 4 --top-k 2 --capacity 16 --timeout-seconds 60 --output /tmp/combine-send.json"
```

For many experts use `--lengths 97,65 --experts-per-rank 128 --top-k 3 --capacity 512`; for wide hidden use the small configuration plus `--hidden-size 7168`. All performance preflights passed.

## Precision

All 11 cases passed the unchanged four-operator CPU-oracle test, with zero mismatches and clean teardown. Launch with `torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py --dtype <dtype> --dynamic-quant <quant> --iterations <rounds> --timeout-seconds 60 --output <unique-json>` plus the arguments below. Defaults are hidden 256, lengths 7/5, top-k 2, and TP size 2.

| dtype | Quant | Additional arguments | Rounds |
| --- | ---: | --- | ---: |
| fp16 | 0 | `--capacity 4` | 2 |
| bf16 | 1 | `--capacity 4` | 2 |
| fp16 | 1 | `--experts-per-rank 24 --top-k 3 --lengths 49,48 --capacity 128` | 2 |
| bf16 | 0 | `--experts-per-rank 25 --top-k 3 --lengths 97,65 --capacity 256` | 2 |
| bf16 | 1 | `--experts-per-rank 25 --top-k 3 --lengths 49,48 --capacity 128` | 2 |
| bf16 | 1 | `--routing sparse --capacity 16` | 2 |
| bf16 | 1 | `--experts-per-rank 3 --lengths 49,48 --capacity 128` | 2 |
| bf16 | 1 | `--experts-per-rank 192 --lengths 49,48 --top-k 3 --capacity 128` | 2 |
| fp16 | 0 | `--experts-per-rank 128 --lengths 1025,1024 --capacity 2048` | 1 |
| bf16 | 1 | `--tp-size 1 --lengths 97,65 --capacity 256` | 2 |
| bf16 | 1 | `--hidden-size 7168 --lengths 1025,1024 --capacity 2048` | 1 |

The final case exercises large cells requiring multiple UB-sized sends; TP1 tests reuse of one destination, while TP2 tests changing destinations. Sparse routing includes empty cells. No tolerances were changed. Full-model throughput and other hardware generations were not tested.

Raw CSV/JSON/log archive SHA-256: `4a86f264c30092202fa2dce5f7e7200c547a1d602278484d14a73e5e0eca01dd`.
