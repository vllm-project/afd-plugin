# DeepSeek-V2-Lite with DeepEPM2NAFDConnector

DeepSeek-V2-Lite (BF16) recipes. The scripts, layouts, settings, multi-pod
runs and the decode-only benchmark are described once in the
[connector recipes README](../README.md); this page lists what is specific to
the model.

```text
prefill_decode_colocation/
├── 2a2f_eager_dp2tp1.sh       # eager, FFN DP2/TP1, Attention DP2/TP1
├── 2a2f_graph_dp2tp1.sh       # graph, FFN DP2/TP1, Attention DP2/TP1
├── 2a2f_graph_dbo_dp2tp1.sh   # graph + DBO, FFN DP2/TP1, Attention DP2/TP1
└── 2a2f_eager_dp1tp2.sh       # eager, FFN DP2/TP1, Attention DP1/TP2
```

## Defaults

| Setting | Value |
| --- | --- |
| `LAYOUT` | `2A2F`: FFN DP2 on GPUs 0,1, Attention DP2 on GPUs 2,3 |
| `--max-num-seqs` | 64 per Attention DP rank |
| `--max-num-batched-tokens` | 4096 |
| `--max-model-len` | 4096 |
| `CUDAGRAPH_CAPTURE_SIZES` | 8,16,32,64 |
| Payload | BF16 (no `connector_extra_config`); the FFN runs DeepGEMM experts |

These match the DeepSeek-V2-Lite DeepEP M2N E2E cases
(`test_deepseek_v2_lite_deepep_m2n` in
[`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios));
`ATTN_TP=2` matches the Attention TP2 case.

## Notes

- Attention runs the gate, the router, the dense first layer and the two
  shared experts; the FFN runs only the routed experts. With two FFN ranks,
  FFN rank 0 owns routed experts 0-31 and FFN rank 1 owns 32-63.
- With Attention TP each TP rank routes and dispatches its slice of the
  tokens and runs the shared experts on that slice with replicated weights;
  the TP group all-gathers the result.

## Limitations

- DeepSeek-V2 (`DeepseekV2ForCausalLM`) only; DeepSeek-V3 and GLM-DSA are
  rejected with this connector.
