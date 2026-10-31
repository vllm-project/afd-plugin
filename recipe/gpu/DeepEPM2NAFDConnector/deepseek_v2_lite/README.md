# DeepSeek-V2-Lite with DeepEPM2NAFDConnector

Launch scripts for DeepSeek-V2-Lite (BF16) on `DeepEPM2NAFDConnector` with
vLLM `v0.32.0`. They use the same layouts as the DeepSeek-V2-Lite DeepEP M2N
E2E cases (`test_deepseek_v2_lite_deepep_m2n` in
[`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios)).

## Prerequisites

- 4 SM90 or newer GPUs on one node.
- The environment described in the
  [connector recipes README](../README.md#prerequisites).
- DeepSeek-V2-Lite weights on disk. The scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN).

## Layout

`2A2F`: Attention on GPUs 0,1 (DP2, or DP1 x TP2) and FFN DP2 on GPUs 2,3.
FFN rank 0 owns routed experts 0-31 and FFN rank 1 owns routed experts 32-63.

```text
prefill_decode_colocation/
├── 2a2f_eager_dp2tp1.sh       # Attention DP2 x TP1, eager
├── 2a2f_graph_dp2tp1.sh       # Attention DP2 x TP1, FULL_DECODE_ONLY graph
├── 2a2f_graph_dbo_dp2tp1.sh   # Attention DP2 x TP1, graph + DBO
└── 2a2f_graph_dp1tp2.sh       # Attention DP1 x TP2, FULL_DECODE_ONLY graph
```

## Configuration notes

- Both roles set `compute_gate_on_attention: true`, use the same rank counts,
  and run with `--enable-expert-parallel`. The FFN runs TP 1.
- Attention runs the gate, the router, the dense first layer and the two
  shared experts; the FFN runs only the routed experts.
- With Attention TP 2 each TP rank routes and dispatches its slice of the
  tokens and runs the shared experts on that slice with replicated weights;
  the TP group all-gathers the result.
- `connector_extra_config` is left empty, so the payload is BF16 and the FFN
  runs DeepGEMM experts.
- The graph and DBO settings match the
  [Qwen3-30B-A3B recipes](../qwen3_30b_a3b/README.md#configuration-notes).

## Limitations

- DeepSeek-V2 (`DeepseekV2ForCausalLM`) only; DeepSeek-V3 and GLM-DSA are
  rejected with this connector.
