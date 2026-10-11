# Async dispatch send: batched prefix writeback

`PreProcess` previously wrote each inclusive-prefix value with a separate four-byte DMA and an S/MTE3 wait pair. With 48 AIV cores, that is 47 writes per owned column. The candidate assigns contiguous columns evenly, stages their prefixes in 32-byte-aligned UB rows, and writes up to 32 rows per DMA. Each active core therefore issues two writes for 48 AIVs and drains them once.

The compact GM prefix-table layout, routing counts, token payloads and inter-core barriers remain unchanged. All cores use the same maximum scratch-size check; insufficient UB selects the original implementation on every core. This avoids mixing incompatible column ownership schemes. The temporary scratch is drained before subsequent initialization reuses that UB region.

## Measurement

Validation ran on `jcz-afd1`, Ascend910_9382, CANN 9.0.1, torch 2.10.0+cpu and torch-npu 2.10.0.post2, on devices 0–3 (two Attention and two MoE ranks). The original measurement baseline was `d982430fec66ec97fe502555944c71ed612e82a4`. The PR is rebased onto main `913c9d3`; the four async operator sources and the three async CAM test scripts are identical between those revisions. The newer main has a separate pybind workspace-lifetime fix, so the historical figures below are not measurements of that updated host extension.

Baseline dispatch-send header SHA-256: `af16bd9a664e041ddce7ff400f341bb3eff0e3b3b778f5306b94f227d1ae813b`.
Candidate header SHA-256: `8e06441ddade4b54dd0e23dc856893b4d6c4244248f4d11be3b0d44d7acabc9f`.

Both isolated builds selected the same four async CAM operators from the operator registry. Device binaries changed only for dispatch send; other async device binaries and the host tiling library matched. This was a four-operator build, not a complete operator-registry build.

`msprof` wrapped `async_cam_performance.py --measure-mode send`, with a full-chain precision preflight, five warmups and five measured iterations. Sort `AfdAsyncDispatchSend` rows in each device's `op_summary` CSV by `Task Start Time(us)`, take the final five, and compute the maximum across Attention ranks for each iteration before averaging. Small and many-expert runs used baseline/candidate followed by candidate/baseline.

| Tokens | Total experts | Hidden | Baseline (μs) | Candidate (μs) | Time reduction | Samples per version |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7/5 | 8 | 256 | 57.955 | 42.507 | 26.7% | 10 |
| 97/65 | 256 | 256 | 118.654 | 77.512 | 34.7% | 10 |
| 7/5 | 8 | 7168 | 50.587 | 43.767 | 13.5% | 5 |

All performance cases used BF16 and dynamic quantization. These are device-task times including the kernel's communication/waits, not model throughput. The hidden-7168 case has only one acquisition and does not establish repeatability. Per-run, per-rank samples are in [dispatch_send_prefix_samples.json](dispatch_send_prefix_samples.json).

Example small-case command, run from each separately built checkout with its own `PYTHONPATH`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD"
msprof --output=/tmp/dispatch-send-profile --application="torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_performance.py --measure-mode send --warmup 5 --iterations 5 --dtype bf16 --dynamic-quant 1 --lengths 7,5 --experts-per-rank 4 --top-k 2 --capacity 16 --timeout-seconds 60 --output /tmp/dispatch-send.json"
```

For the many-expert case use `--lengths 97,65 --experts-per-rank 128 --top-k 3 --capacity 512`; for the wide-hidden case add `--hidden-size 7168` to the small configuration. Use unique output paths for every acquisition.

## Precision

The unmodified `tests/npu/async_cam_precision.py` checks the complete four-operator chain against its CPU oracle. All eight candidate cases passed on all four ranks, with zero mismatches and successful cleanup:

| dtype | Quant | Additional arguments | Iterations |
| --- | ---: | --- | ---: |
| fp16 | 0 | `--capacity 4` | 2 |
| bf16 | 1 | `--capacity 4` | 2 |
| fp16 | 1 | `--experts-per-rank 24 --top-k 3 --lengths 49,48 --capacity 128` | 2 |
| bf16 | 0 | `--experts-per-rank 25 --top-k 3 --lengths 97,65 --capacity 256` | 2 |
| bf16 | 1 | `--experts-per-rank 25 --top-k 3 --lengths 49,48 --capacity 128` | 2 |
| bf16 | 1 | `--routing sparse --capacity 16` | 2 |
| bf16 | 1 | `--experts-per-rank 192 --lengths 49,48 --top-k 3 --capacity 128` | 2 |
| fp16 | 0 | `--experts-per-rank 128 --lengths 1025,1024 --capacity 2048` | 1 |

Launch each with `torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py --dtype <dtype> --dynamic-quant <quant> --iterations <iterations> --timeout-seconds 60 --output <unique-json>` plus the listed arguments. Defaults are hidden 256 and lengths 7/5. The 192-experts-per-rank case exercises 386 columns: two cores stage nine columns with 64-byte row pitch, while the other cores stage eight with 32-byte pitch.

Both variants also passed default smoke tests and every performance preflight. UB-insufficient fallback was reviewed but not forced on hardware. No model-level throughput test was run. The first baseline many-expert shell wrapper had an EOF error after profiling completed; its profiler completion log, complete CSVs and successful harness JSON were verified, and reverse-order repetition confirmed the result.
