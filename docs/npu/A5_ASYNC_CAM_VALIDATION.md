# A5 asynchronous CAM: build and device verification

Target: vLLM `5df63a7203cdb9328d824a27b682e77c7b9928be`, vLLM-Ascend
`f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e`, CANN 9.2.1 (issue #537).
This change targets single-host A5, BF16 transport and DSV4 Flash MXFP4 experts.
Device compilation and numerical qualification must be performed on A5.

## Build

From the plugin checkout in the target environment, with the CANN environment
loaded and the target torch/torch_npu installed:

```bash
SOC_VERSION=ascend950 AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

Do not set `AFD_SKIP_ACLNN_BUILD=1`: both the OPP and Python extension must be
rebuilt. The A5 registry now includes all four `afd_async_*` operators and their
Python bindings. No separate UMDK wheel is required for this connector.

The kernels use the A5 HCCL flat `windowsIn` array, matching the existing A5
A2E/E2A implementation, and tiling selects the AIV MTE communication engine.
Async CAM retains its own control/payload layout. AIV count is capped at 48 to
fit that layout's reserved rows on both platforms. INT8 transport and layered
INT4/INT8 compute are outside the DSV4 A5 qualification target.

## Reuse the existing four-operator tests

Reuse `tests/npu/async_cam_precision.py` for both the one-round runtime smoke
and repeated precision checks. It invokes the four operators directly and checks
dispatch ordering and weighted combine
against a CPU reference using a deterministic expert transform. It does not
validate the MXFP4 expert GMM or whole-model accuracy.

Use four visible A5 devices and a fresh process group for each command:

```bash
torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py --iterations 1 \
  --attn-ranks 2 --moe-ranks 2 --tp-size 1 --dtype bf16 --dynamic-quant 0 \
  --hidden-size 4096 --experts-per-rank 128 --top-k 6 \
  --capacity 128 --window-mb 512 --timeout-seconds 180

torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py \
  --attn-ranks 2 --moe-ranks 2 --tp-size 1 --dtype bf16 --dynamic-quant 0 \
  --hidden-size 4096 --experts-per-rank 128 --top-k 6 \
  --capacity 128 --window-mb 512 --timeout-seconds 180 \
  --lengths 7,5 --routing balanced --iterations 100 --output /tmp/a5-cam-balanced.json

torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py \
  --attn-ranks 2 --moe-ranks 2 --tp-size 1 --dtype bf16 --dynamic-quant 0 \
  --hidden-size 4096 --experts-per-rank 128 --top-k 6 \
  --capacity 128 --window-mb 512 --timeout-seconds 180 \
  --lengths 1,13 --routing sparse --iterations 100 --output /tmp/a5-cam-sparse.json
```

Each precision run must exit successfully with no mismatches. Preserve the JSON
report, build log and device logs if a run fails or times out. The script also
accepts `--dry-run` to inspect configuration without initializing devices.

The existing `tests/npu/async_cam_runtime.py` exercises the A3 layered W4A8
runner and is not an A5 MXFP4 test; do not use it for this qualification.

## DSV4 Flash E2E configuration changes

The existing `afd-dsv4-flash-async-cam-dp2tp4-ep8` scenario contains A3-specific
quantization and topology defaults. Do not run it unchanged on A5. For a manual
2A2F deployment use the original DSV4 Flash FP8 checkpoint with FP4 routed experts,
BF16 model dtype, ModelRunnerV2, eager execution, Attention DP2/TP1 and FFN
DP2/TP1 with static EP2. Disable DBO, MTP, EPLB and layered GMM for this target.
Use the following additional configuration on both roles, setting `role` to
`attention` or `ffn` and supplying the same reachable rendezvous host/port:

```json
{
  "enable_fused_mc2": 0,
  "afd": {
    "role": "attention",
    "host": "127.0.0.1",
    "port": 6456,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 2,
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "compute_gate_on_attention": true,
    "connector_extra_config": {"dynamicQuant": 0, "attn_ranks_per_dp": 1}
  }
}
```

Set `VLLM_USE_V2_MODEL_RUNNER=1` and `AFD_ASYNC_CAM_LAYERED_GMM=0`.
Remove the A3 recipe's `--quantization ascend` override so the native checkpoint
quantization selects MXFP4 expert weights. Use `--enforce-eager --dtype bfloat16`
and `--enable-expert-parallel`. Size HCCL buffers for the selected token capacity
using the existing CAM guide; the small operator-test window is not an inference
sizing recommendation. Compare generated outputs/accuracy with native A5
inference using the same checkpoint, sampling parameters and evaluation inputs.
A passing communication test alone does not establish whole-model accuracy.
