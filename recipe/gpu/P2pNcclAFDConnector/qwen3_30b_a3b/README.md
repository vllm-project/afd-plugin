# Qwen3-30B-A3B AFD Recipe (P2pNcclAFDConnector)

`prefill_decode_colocation/2a2f_graph_dp2tp1.sh` runs the shared Qwen3-30B-A3B
2A2F recipe with `AFD_CONNECTOR=P2pNcclAFDConnector`. Topology, environment
knobs, and the decode-only benchmark are documented with the script itself in
[`../../DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md`](../../DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md).

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/P2pNcclAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```
