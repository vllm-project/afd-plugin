# Qwen3-30B-A3B AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/2a2f_graph_dp2tp1.sh` runs the shared Qwen3-30B-A3B
graph recipe (`DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh`,
default layout 2A2F) with `AFD_CONNECTOR=P2pNcclAFDConnector`. Its layouts,
settings and the decode-only benchmark are documented in the
[`DeepEPM2NAFDConnector` recipes README](../../DeepEPM2NAFDConnector/README.md)
and the [model README](../../DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/P2pNcclAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```
