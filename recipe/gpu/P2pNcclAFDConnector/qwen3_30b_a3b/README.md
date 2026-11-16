# Qwen3-30B-A3B AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/2a2f_graph_dp2tp1.sh` runs Qwen3-30B-A3B with
`FULL_DECODE_ONLY` CUDA graphs on `P2pNcclAFDConnector`, with the same
settings as the
[`DeepEPM2NAFDConnector` 2A2F graph recipe](../../DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md)
(FFN DP2 x TP1, Attention DP2 x TP1). The layout, settings and running in a
pod are documented in the
[colocation launcher README](../../common/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/P2pNcclAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```
