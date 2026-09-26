# Running NPU unit tests

NPU unit tests depend on vLLM and vLLM-Ascend. Run them in an Ascend development
environment that uses the versions pinned by this repository.

From the repository root, run:

```bash
python3 -m pytest -q tests/unit -m "not gpu and not vllm_runtime"
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

## Layered W4A8 validation

Run the CPU contract and path-selection tests first:

```bash
pytest -q tests/unit/test_envs.py \
  tests/unit/model_executor/test_async_cam_w4a8.py
```

In an environment with matching CANN and torch-npu versions, rebuild the AFD
extension and check the device Tensor `group_list` interface from #384 and the
new W4A8 operator chain:

```bash
SOC_VERSION=910c pytest -q tests/unit/compat/npu/test_gmm_layered.py
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c \
  pytest -q tests/npu/test_async_cam_layered_w4a8.py
```

The NPU test uses two layers with different nonzero INT4 weights, nonzero
compensation, per-channel and per-group scales, interleaved nonconsecutive
layer IDs, and zero, single-row, uneven, and full-capacity expert counts. It
compares valid rows with an FP32 reference and perturbs the capacity tail. It
exercises only the `swiglu_limit=0` operator chain; nonzero limits are
temporarily ignored by the layered path and are not precision-equivalent to
the model. It does not check distributed CAM completion or compare actual
checkpoint results against the legacy path. All eight parameterized cases
passed on Ascend 910C after the NZ weight layout fix. Recheck the tolerance
and supported configurations against the project's precision standard when
the operator changes.

For end-to-end validation, toggle `AFD_ASYNC_CAM_LAYERED_GMM=0/1` with the same
W4A8 checkpoint and deployment configuration. Record the processed parameter
shapes, dtypes, and formats; software versions; actual-path startup logs;
accuracy; peak memory; and profiler traces. Cover completion on empty ranks,
multiple chunks of one layer, interleaved DP work, and async CAM ubatching.
Check that no layer-ID or count metadata D2H occurs between receive and send,
and measure the retained worker group synchronization separately. No
performance improvement has been measured for this change.

The device Tensor `group_list` interface for the second GMM was merged in
PR #384. This change uses it directly. The W13 layered GMM op_api also accepts
the logical 3D view with 5D NZ storage produced by the W4A8 loader; the kernel
math and PyTorch binding are unchanged.
