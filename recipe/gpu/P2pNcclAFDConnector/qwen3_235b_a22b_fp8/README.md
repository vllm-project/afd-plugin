# Qwen3-235B-A22B-FP8 AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/4a2f_graph_dp4tp1.sh` runs the shared
Qwen3-235B-A22B-FP8 4A2F recipe with `AFD_CONNECTOR=P2pNcclAFDConnector`.
Topology, environment knobs, and the decode-only benchmark are documented with
the script itself in
[`../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md`](../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
bash recipe/gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this
connector's FFN: the FFN hangs on its first step.
