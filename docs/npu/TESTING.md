# NPU validation

Configuration and source-contract tests can run on CPU; runtime contracts
depend on vLLM/vLLM-Ascend, and device tests also need the matching Ascend
runtime and hardware. Use vLLM 0.30.0 and the vLLM-Ascend source revision in
the [installation guide](../../README.md#ascend-npu-installation).

From the repository root, run:

```bash
python3 -m pytest -q -ra tests/unit -m "not gpu and not vllm_runtime"
```

The suite covers the NPU worker and model-runner contracts, CAM connector
behavior, and module isolation. Tests that require unavailable NPU dependencies
are skipped in CPU-only CI, so a passing CPU run does not replace this check.

`tests/conftest.py` establishes the vLLM-Ascend import order required by the
test suite. No manual module preloading is needed.

When diagnosing a failure, run the failing pytest node by itself first. A test
that passes alone but fails in the full suite usually indicates leaked module
or monkeypatch state. Test fixtures and mocks must also follow the concrete
vLLM types and signatures used by the pinned runtime.

## Native DBO contracts and device execution

Run the focused contract checks in the matching runtime environment:

```bash
python3 -m pytest -q -ra \
  tests/unit/config/test_async_cam_dbo.py \
  tests/unit/config/test_validation.py \
  tests/unit/model_executor/test_async_cam_dbo.py \
  tests/unit/model_executor/test_async_cam_layout.py \
  tests/unit/v1/worker/test_npu_model_runner_v2.py \
  tests/unit/v1/worker/test_npu_ubatch_outputs.py \
  tests/unit/v1/worker/test_npu_mla_graph.py
```

The single-device executor test uses real NPU streams/events with isolated
subprocesses for failure injection:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest -q -ra tests/npu/test_mrv2_dbo.py
```

It checks native child handoff, stream/context restoration and peer failure
wakeup. It does not validate CAMP2P/CAM transport, checkpoint output, or model
accuracy. Before qualifying the current native-only Async CAM DBO path, run
model comparisons at Attention DP1 and DP > 1, plain TP and model SP, with
uneven prefill lengths and requests crossing child boundaries. Enable DBO on
Attention only and enable chunked prefill for DeepSeek-V2/V3. Keep two ubatches
and eager execution.

For synchronous MRV2 CAMP2P, include eager two-stage execution at Attention
DP > 1, many-to-one padding, and graph+DBO. Plain MLA graph+DBO requires
`FULL_DECODE_ONLY`. Both existing Lite graph+DBO cases must report completed
joint two-stage replays during real requests on every participating A/F rank.
Dummy/profile/capture or padded-only stages do not qualify that gate.
`tests/npu/test_mrv2_graph_inputs.py` separately checks persistent input
addresses with real NPU graph reads and modeled builders; model correctness
still belongs to E2E. Record execution, response correctness, revisions and
teardown outcomes separately.

The original DSV4 Async concurrent case keeps native DBO thresholds. Its ten
short prompts can remain unsplit while satisfying its exact-answer, `stop`,
overlap and cleanup contract. Qualify DSV4 DSA two-stage execution separately
with an above-threshold workload, retaining native defaults and requiring
completed nonempty stages on every Attention rank.

## Operator and model gates

Use [the operator harness](../../tests/npu/README.md) for isolated routed CAM
precision/performance and [model E2E cases](../design/module/e2e_testing.md)
for serving and accuracy. DSV4 synchronous A5/A3 profiles currently check
completion/liveness with `check_answer=False`; they are not correctness gates.
Any skipped native registration or numerical case means that capability was
not validated by that run.

## Layered W4A8 validation

The layered routed-expert operators are `gmm_swiglu_quant_v2_layered` (W13) and
`grouped_matmul_layered` (W2). Both select their layer on device, and their
`all_*` lists hold one element per layer rather than one per expert.

Run the CPU contract and path-selection tests first:

```bash
pytest -q tests/unit/test_envs.py \
  tests/unit/model_executor/test_async_cam_w4a8.py
```

In an environment with matching CANN and torch-npu versions, rebuild the AFD
extension and run the device suites. All three need one device and issue no
collective:

```bash
SOC_VERSION=910c pytest -q tests/unit/compat/npu/test_gmm_layered.py
SOC_VERSION=910c pytest -q tests/unit/compat/npu/test_gmm_swiglu_quant_v2_layered.py
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c LOCAL_RANK=0 \
  pytest -o addopts= -v -ra -s \
  tests/npu/test_async_cam_layered_w4a8.py \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py \
  tests/npu/test_gmm_layered_device_group_list.py
```

The three device suites answer different questions and none is redundant:

| Suite | Operator | Covers |
| --- | --- | --- |
| `test_async_cam_layered_w4a8.py` | W13 then W2 | The `AsyncCAMW4A8Executor` chain, 16 cases |
| `test_gmm_swiglu_quant_v2_layered_precision.py` | W13 | Both quantization scenarios against a reference, 11 cases |
| `test_gmm_layered_device_group_list.py` | W2 | A device-produced group list, both scenarios, 4 cases |

`test_async_cam_layered_w4a8.py` uses two layers with different nonzero INT4
weights, nonzero compensation, per-channel and per-group scales, interleaved
nonconsecutive layer IDs, and zero, single-row, uneven, and full-capacity expert
counts. It compares valid rows with an FP32 reference and perturbs the capacity
tail. It exercises only the `swiglu_limit=0` operator chain; nonzero limits are
temporarily ignored by the layered path and are not precision-equivalent to the
model. It does not check distributed CAM completion or compare actual checkpoint
results against the non-layered path. Recheck the tolerance and supported
configurations against the project's precision standard when the operator
changes.

The two precision suites are described under "Layered operator precision" in
[`tests/npu/README.md`](../../tests/npu/README.md), including why A8W4 cannot use
the fused built-in as its reference and why each attribute combination needs its
own activation shape. Read that section before changing a tolerance or adding a
case: the tiling cache is keyed on shape and tiling key, and the two attributes
reach the tiling only as data.

For end-to-end validation, toggle `AFD_ASYNC_CAM_LAYERED_GMM=0/1` with the same
W4A8 checkpoint and deployment configuration. Record the processed parameter
shapes, dtypes, and formats; software versions; actual-path startup logs;
accuracy; peak memory; and profiler traces. Cover completion on empty ranks,
multiple chunks of one layer, interleaved DP work, and async CAM ubatching.
Check that no layer-ID or count metadata D2H occurs between receive and send,
and measure the worker group synchronization separately. Performance gains
require a controlled A/B comparison.

For operator-level performance, use
[`tests/npu/gmm_layered_performance.py`](../../tests/npu/gmm_layered_performance.py)
rather than an end-to-end comparison. It measures both launch paths - eager and
ACL graph - and reports the fusion contribution and the graph contribution
separately, which an end-to-end run cannot separate. Anchors measured on Ascend
910C at `M = 98304`, `C = 262144`: W13 reaches about 1.80x the concatenated
baseline on the steady-state per-layer figure, `chain` about 1.46x, and W2 sits
at parity once the graph removes its launch overhead. Re-measure rather than
reuse these numbers; they are a starting point for a regression check, not a
claim about a different shape, topology, or CANN version.

The current ACLNN entry validates `layer_index` by its one-element int64
contract, including a slice with nonzero storage offset. The GMM path accepts
the logical 3D A8W8 weight view backed by 5D NZ storage as well as the W4A8
layout. Rebuild the extension and vendor library together after these native
changes; Meta-only success does not exercise the ACLNN entry or kernel.
Layered GMM is allowed with MRV2 pairing, but it still runs in the independent
eager FFN runner. The current shared-expert layout also has focused coverage
in `tests/unit/model_executor/models/test_deepseek_v4_shared_experts.py` and
`test_deepseek_v2_sequence_parallel.py`.
