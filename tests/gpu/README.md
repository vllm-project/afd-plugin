# CUDA connector device tests

Opt-in device tests for `DeepEPM2NAFDConnector`. They run the connector's
transport and expert compute on real GPUs, without model weights or a vLLM
server, and run outside CI.

| File | Needs | Purpose |
| --- | --- | --- |
| `test_deepep_m2n_gpu.py` | pytest, up to 6 GPUs on one node | Dispatch, expert compute and combine checked against references, in eager mode, with two lanes (DBO), under `torch.compile`, and after CUDA-graph replay |
| `deepep_m2n_harness.py` | - | Multi-process harness: rank spawning, deterministic inputs, references, checks, scenarios, and benchmark scenarios |
| `__init__.py` | - | Package marker |

## Requirements

- Hopper (SM90) or newer GPUs on one node. Most cases need 4 GPUs; the two
  `2F-2DPx2TP` Attention TP cases need 6. Cases that need more GPUs than are
  visible skip, and every case skips below SM90.
- vLLM with the plugin installed, afdEP-comm (the `afdep` extra), and DeepGEMM
  with the elfutils headers. See
  [Installation](../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#installation)
  in the connector guide.
- The full suite has 34 cases and takes about 53 minutes on six H200 GPUs with
  cold JIT caches. DeepEP and DeepGEMM compile kernels on first use, so a
  rerun with warm caches is much faster.

## Run

```bash
# Full suite
pytest -sv -m gpu tests/gpu

# One payload format
pytest -sv -m gpu tests/gpu -k bf16
pytest -sv -m gpu tests/gpu -k fp8

# DBO (two lanes) or CUDA graph cases
pytest -sv -m gpu tests/gpu -k "lanes or dbo"
pytest -sv -m gpu tests/gpu -k "graph"

# One case
pytest -sv "tests/gpu/test_deepep_m2n_gpu.py::test_transport_and_deepgemm_experts[2F2A-fp8]"
```

Use `CUDA_VISIBLE_DEVICES` to choose the GPUs. The CPU-only unit gate
(`pytest tests/unit -m "not gpu and not vllm_runtime"`) does not collect these
tests.

## What the harness checks

`run_ranks` spawns one process per AFD rank (FFN ranks first, then Attention
ranks), each on its own GPU. Every rank builds the connector through
`AFDConnectorFactory` from a fake `vllm_config` (128 experts, top-k 8, hidden
size 2048), initializes it, runs a scenario, and writes its result to
`rank<N>.json` under the test's temporary directory.

Inputs and routing are deterministic, so every rank can regenerate every
Attention rank's tokens:

- FFN ranks check the received rows exactly: per-expert counts, then every row
  and top-k weight bit for bit.
- FFN ranks check the expert output against an fp32 reference and against
  vLLM `fused_experts`.
- Attention ranks check the combined output against a dense MoE reference.
- Any non-finite output or error fails the check.

The tolerances in `deepep_m2n_harness.py` are about 1.3 times the worst values
seen on H100. FP8 experts are checked against references that use the same FP8
inputs and weights; the combined FP8 output is checked end to end against the
BF16 model.

| Test | Covers |
| --- | --- |
| `test_transport_and_deepgemm_experts` | Standard cases (even, uneven, decode, empty, mixed; with and without host sync) for `1F1A`, `2F2A` and `1F3A`, BF16 and FP8 |
| `test_attention_tp_peers_dispatch_disjoint_slices` | Attention TP2: each TP peer dispatches its slice, every route arrives exactly once |
| `test_transport_with_fused_experts_backend` | The `fused_experts` expert backend |
| `test_hot_ffn_rank_routing` | Every top-k choice on FFN rank 0's experts (the worst-case receive size) |
| `test_fp8_without_host_sync_for_small_buckets` | FP8 without the host sync (`cpu_sync_min_tokens` above the capacity) |
| `test_graph_and_eager_plans_mixed_across_ranks` | Attention plans a step as graph while FFN plans it eager |
| `test_two_lanes_dbo_order`, `test_dbo_pipeline_*` | Two lanes in DBO order, eager and under CUDA-graph capture and replay |
| `test_cuda_graph_capture_and_replay` | One graph per bucket (1, 8, 64, 256), several MoE layers per graph |
| `test_torch_compile_attention_side` | Dispatch and combine as custom ops under `torch.compile` |
| `test_failed_init_reports_and_closes` | Mismatched payload formats fail the handshake, and every rank reports and closes |
| `test_peer_exit_fails_within_timeout` | A rank exiting mid-step makes the others fail through the transport timeouts |
| `test_non_finite_output_fails_the_numerical_checks` | The harness itself rejects NaN and infinite errors (no GPU needed) |

## How these tests differ from E2E

These tests exercise the connector alone with synthetic weights and a fake
vLLM config. They do not load a model, start `vllm serve`, or check accuracy.
The model-level E2E cases (`afd-deepep-m2n-*`, Qwen3-30B-A3B with GSM8K) are in
[`tests/e2e`](../e2e/README.md#deepep-m2n-scenarios).

## Results, skips and timeouts

- A failing rank's traceback appears in the assertion message, prefixed with
  its world rank and role. With `-s`, each rank also prints its metrics
  (worst errors and timings).
- A skip means too few visible GPUs or a GPU older than SM90.
- Each run is killed after 900 seconds (`RANK_TIMEOUT_S`); ranks still alive
  then are reported as timed out. A timeout usually means a rank did not reach
  a collective, for example after a failed peer. The first run on a node can
  be slow while kernels compile.

## Benchmark scenarios

`deepep_m2n_harness.py` also provides `benchmark_scenario` (eager per-layer
time, with a dispatch, experts and combine breakdown for one lane),
`graph_benchmark_scenario` (per-layer time of a replayed CUDA graph) and
`local_moe_baseline` (single-GPU vLLM `fused_experts` with all experts local).
No test calls them; pass them to `run_ranks` from a script. `bench_output_path()`
returns the output file, `deepep_m2n_bench.json` by default, or the path in
`AFD_DEEPEP_M2N_BENCH_OUT`.
