# Qwen3-235B-A22B-FP8 AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/4a2f_graph_dp4tp1.sh` runs the shared
Qwen3-235B-A22B-FP8 graph recipe
(`DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/graph.sh`)
in the 4A2F layout (`LAYOUT=4A2F`, one 6-GPU node) with
`AFD_CONNECTOR=P2pNcclAFDConnector`. Its layouts, settings and the
decode-only benchmark are documented in the
[`DeepEPM2NAFDConnector` recipes README](../../DeepEPM2NAFDConnector/README.md)
and the [model README](../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
bash recipe/gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this
connector's FFN: the FFN hangs on its first step.
