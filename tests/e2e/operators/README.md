# Async CAM source migration numerical checks

Build the 910C AFD package first. Run these checks serially on otherwise idle
NPUs; retain stdout, the software/checkpoint manifest and generated artifacts.
No operator source modification is required.

For each `TP` in `1 2 4`, dtype in `float16 bfloat16`, and quantization in `0 1`:

```bash
torchrun --standalone --nproc-per-node=$((TP + 2)) \
  -m tests.e2e.operators.async_cam_roundtrip \
  --tp "$TP" --dtype "$DTYPE" --dynamic-quant "$QUANT"
```

This uses two FFN ranks and forces multi-chunk reception by reducing receive
capacity. An independent CPU oracle checks routed weighted sums; each run
includes empty ranks, sparse routes, first/last experts, a single-token batch,
and repeated window use. Tolerances are fixed in the script before execution.

The retired non-layered checkpoint probe remains available on earlier
revisions for a before/after comparison. Validate the current layered
implementation using the numerical and real-model fixtures in the
[NPU testing guide](../../../docs/design/npu/async_ffn_runtime.md#validation).
