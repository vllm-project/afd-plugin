# Qwen3-235B-A22B-FP8 AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/graph.sh` runs Qwen3-235B-A22B-FP8 with
`FULL_DECODE_ONLY` CUDA graphs on `P2pNcclAFDConnector`, with the same
settings as the
[`DeepEPM2NAFDConnector` graph recipe](../../DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md),
in the 4A2F layout by default (one 6-GPU node). The layout, settings and
running in a pod are documented in the
[colocation launcher README](../../common/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
bash recipe/gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/graph.sh
```

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this
connector's FFN: the FFN hangs on its first step.
