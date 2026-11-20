# Qwen3-235B-A22B-FP8 AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/2a2f_graph_dp2tp1.sh` runs Qwen3-235B-A22B-FP8 with
`FULL_DECODE_ONLY` CUDA graphs on `P2pNcclAFDConnector`, with the same
settings as the
[`DeepEPM2NAFDConnector` 2A2F graph recipe](../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md),
in the 2A2F layout (FFN DP2 x TP1, Attention DP2 x TP1; 4 GPUs on one host).
The layout, settings and running in a pod are documented in the
[colocation launcher README](../../common/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
bash recipe/gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this
connector's FFN: the FFN hangs on its first step.
