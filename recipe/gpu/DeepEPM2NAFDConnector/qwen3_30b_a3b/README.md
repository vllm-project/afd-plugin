# Qwen3-30B-A3B with DeepEPM2NAFDConnector

Qwen3-30B-A3B (BF16) recipes. The scripts, layouts, settings, multi-pod runs
and the decode-only benchmark are described once in the
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

These match the Qwen3 MoE DeepEP M2N E2E cases (`afd-deepep-m2n-*-2a2f` and
the two-pod `2A0F,0A2F` and `1A1F,1A1F` cases in
[`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios));
`ATTN_TP=2` matches the Attention TP2 cases.

## Notes

- 4 GPUs of 80 GB (H100) or more fit `2A2F`. With two FFN ranks, FFN rank 0
  owns experts 0-63 and FFN rank 1 owns experts 64-127; the FFN count must
  divide the 128 experts.
- Decode-only benchmark pool sizing: Qwen3-30B-A3B holds about 96 KiB of BF16
  KV per token, so 500 prompts of 1025 tokens need about 47 GiB per Attention
  rank and fit an H100; 1000 prompts need about 94 GiB and need an H200.

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
